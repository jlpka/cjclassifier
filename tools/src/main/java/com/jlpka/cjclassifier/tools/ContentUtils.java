/*
 * Copyright 2026 Jeremy Lilley (jeremy@jlilley.net)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.jlpka.cjclassifier.tools;

import com.github.houbb.opencc4j.util.ZhConverterUtil;
import com.jlpka.cjclassifier.CJLanguage;
import java.io.*;
import java.util.function.BiConsumer;
import java.util.zip.GZIPInputStream;
import javax.xml.stream.*;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;

public class ContentUtils {

  // ========================================================================
  // I/O utilities
  // ========================================================================

  /**
   * Opens a decompression stream based on file extension (.bz2 or .gz). Returns raw buffered stream
   * for other extensions.
   */
  public static InputStream openCompressed(String filename) throws IOException {
    InputStream fileIn = new BufferedInputStream(new FileInputStream(filename), 1024 * 1024);
    if (filename.endsWith(".bz2")) {
      return new BZip2CompressorInputStream(fileIn, true);
    } else if (filename.endsWith(".gz")) {
      return new GZIPInputStream(fileIn, 1024 * 1024);
    }
    return fileIn;
  }

  /** Helper to extract char sequences from wikipedia */
  public static void extract(
      InputStream input, CJLanguage taggedLanguage, BiConsumer<CharSequence, CJLanguage> consumer)
      throws XMLStreamException {
    XMLInputFactory factory = XMLInputFactory.newInstance();
    factory.setProperty(XMLInputFactory.IS_SUPPORTING_EXTERNAL_ENTITIES, false);
    factory.setProperty(XMLInputFactory.SUPPORT_DTD, false);
    // Wikipedia dumps can exceed the default 50M entity accumulation limit
    factory.setProperty(
        "http://www.oracle.com/xml/jaxp/properties/totalEntitySizeLimit", Integer.MAX_VALUE);

    XMLStreamReader reader = factory.createXMLStreamReader(input, "UTF-8");

    StringBuilder textContent = new StringBuilder(64 * 1024);
    boolean inText = false;

    while (reader.hasNext()) {
      int event = reader.next();

      switch (event) {
        case XMLStreamConstants.START_ELEMENT:
          if ("text".equals(reader.getLocalName())) {
            inText = true;
            textContent.setLength(0);
          }
          break;

        case XMLStreamConstants.CHARACTERS:
        case XMLStreamConstants.CDATA:
          if (inText) {
            textContent.append(reader.getText());
          }
          break;

        case XMLStreamConstants.END_ELEMENT:
          if ("text".equals(reader.getLocalName())) {
            inText = false;
            if (textContent.length() > 0) {
              consumer.accept(textContent, taggedLanguage);
            }
          }
          break;
      }
    }
    reader.close();
  }

  public static BiConsumer<CharSequence, CJLanguage> wrapEmitJustCJRuns(
      BiConsumer<CharSequence, CJLanguage> inner) {
    return (text, lang) -> {
      int rangeFrom = -1;
      final int len = text.length();
      for (int i = 0; i < len; i++) {
        char c = text.charAt(i);
        if (inMainCJRange(c)) {
          if (rangeFrom < 0) {
            rangeFrom = i;
          }
        } else if (rangeFrom >= 0) {
          inner.accept(text.subSequence(rangeFrom, i), lang);
          rangeFrom = -1;
        }
      }
      if (rangeFrom >= 0) {
        inner.accept(text.subSequence(rangeFrom, len), lang);
      }
    };
  }

  static final int CJ_RANGE_START = 0x3400;
  static final int CJ_RANGE_END = 0x9FFF;
  static final int CJ_RANGE_SIZE = CJ_RANGE_END - CJ_RANGE_START + 1;

  public static boolean inMainCJRange(char c) {
    return c >= CJ_RANGE_START && c <= CJ_RANGE_END;
  }

  public static boolean hasAnyInMainCJRange(CharSequence cs) {
    for (int i = 0; i < cs.length(); ++i) {
      if (inMainCJRange(cs.charAt(i))) {
        return true;
      }
    }
    return false;
  }

  public static boolean isKana(char c) {
    if (c >= 0x3040 && c <= 0x30FF) return true;
    if (c >= 0x31F0 && c <= 0x31FF) return true;
    if (c >= 0xFF65 && c <= 0xFF9F) return true;
    return false;
  }

  // ========================================================================
  // XML extraction
  // ========================================================================

  // Wikipedia is a mix of simplified and traditional Chinese.
  // It is stored in its original format and then relies on load-time rewriting from one format to
  // the other.
  // (e.g. see ZhConversion.php in the WikiMedia sources; this is GPL so it can't be imported, but
  // we use an alternate opensource lib to do the conversino).
  public static BiConsumer<CharSequence, CJLanguage> wrapTryBothFormsForChinese(
      BiConsumer<CharSequence, CJLanguage> inner) {
    long[] simpOnly = {0};
    long[] tradOnly = {0};
    long[] both = {0};
    long[] both_same = {0};
    long[] total = {0};
    long[] nextLog = {1_000_000};
    return (text, lang) -> {
      if (lang.isChinese() && hasAnyInMainCJRange(text)) {
        String s = text.toString();
        boolean hasSimp = ZhConverterUtil.containsSimple(s);
        boolean hasTrad = ZhConverterUtil.containsTraditional(s);
        if (hasSimp && !hasTrad) {
          simpOnly[0]++;
          inner.accept(s, CJLanguage.CHINESE_SIMPLIFIED);
          inner.accept(ZhConverterUtil.toTraditional(s), CJLanguage.CHINESE_TRADITIONAL);
        } else if (!hasSimp && hasTrad) {
          tradOnly[0]++;
          inner.accept(s, CJLanguage.CHINESE_TRADITIONAL);
          inner.accept(ZhConverterUtil.toSimple(s), CJLanguage.CHINESE_SIMPLIFIED);
        } else {
          both[0]++;
          String trad = ZhConverterUtil.toTraditional(s);
          String simpl = ZhConverterUtil.toSimple(s);
          if (trad.equals(simpl) || s.equals(simpl) || s.equals(trad)) {
            inner.accept(trad, CJLanguage.CHINESE_TRADITIONAL);
            inner.accept(simpl, CJLanguage.CHINESE_SIMPLIFIED);
          } else {
            inner.accept(s, CJLanguage.CHINESE_TRADITIONAL);
            inner.accept(s, CJLanguage.CHINESE_SIMPLIFIED);
          }
          if (trad.equals(simpl)) {
            both_same[0]++;
          }
        }
        if (++total[0] >= nextLog[0]) {
          System.err.printf(
              "Chinese form stats @ %dM: simpOnly=%d tradOnly=%d both/neither=%d both_same=%d%n",
              total[0] / 1_000_000, simpOnly[0], tradOnly[0], both[0], both_same[0]);
          nextLog[0] += 1_000_000;
        }
      } else {
        inner.accept(text, lang);
      }
    };
  }
}
