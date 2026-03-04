package com.jlpka.cjclassifier.tools;

import static org.junit.jupiter.api.Assertions.*;

import com.jlpka.cjclassifier.CJLanguage;
import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BiConsumer;
import javax.xml.stream.XMLStreamException;
import org.junit.jupiter.api.Test;

class ContentUtilsTest {

  // ========================================================================
  // inMainCJRange / hasAnyInMainCJRange
  // ========================================================================

  @Test
  void inMainCJRange() {
    assertTrue(ContentUtils.inMainCJRange('\u4e00')); // CJK unified ideograph
    assertTrue(ContentUtils.inMainCJRange('\u9fff')); // end of range
    assertTrue(ContentUtils.inMainCJRange('\u3400')); // start of range
    assertFalse(ContentUtils.inMainCJRange('A'));
    assertFalse(ContentUtils.inMainCJRange('\u33ff')); // just below range
    assertFalse(ContentUtils.inMainCJRange('\ua000')); // just above range
  }

  @Test
  void hasAnyInMainCJRange() {
    assertTrue(ContentUtils.hasAnyInMainCJRange("hello\u4e16\u754c"));
    assertFalse(ContentUtils.hasAnyInMainCJRange("hello world"));
    assertFalse(ContentUtils.hasAnyInMainCJRange(""));
  }

  // ========================================================================
  // wrapEmitJustCJRuns
  // ========================================================================

  @Test
  void wrapEmitJustCJRunsSplitsRuns() {
    List<String> results = new ArrayList<>();
    BiConsumer<CharSequence, CJLanguage> wrapped =
        ContentUtils.wrapEmitJustCJRuns((text, lang) -> results.add(text.toString()));
    // \u4e16\u754c = "世界", \u4eba = "人"
    wrapped.accept("hello\u4e16\u754cworld\u4eba!", CJLanguage.JAPANESE);
    assertEquals(2, results.size());
    assertEquals("\u4e16\u754c", results.get(0));
    assertEquals("\u4eba", results.get(1));
  }

  @Test
  void wrapEmitJustCJRunsNoCJ() {
    List<String> results = new ArrayList<>();
    BiConsumer<CharSequence, CJLanguage> wrapped =
        ContentUtils.wrapEmitJustCJRuns((text, lang) -> results.add(text.toString()));
    wrapped.accept("hello world", CJLanguage.JAPANESE);
    assertTrue(results.isEmpty());
  }

  // ========================================================================
  // extract (XML)
  // ========================================================================

  @Test
  void extractParsesTextElements() throws XMLStreamException {
    String xml = "<page><text>ホッキョクグマが北海道に来た</text></page>";
    ByteArrayInputStream in = new ByteArrayInputStream(xml.getBytes(StandardCharsets.UTF_8));
    List<String> texts = new ArrayList<>();
    ContentUtils.extract(in, CJLanguage.JAPANESE, (text, lang) -> texts.add(text.toString()));
    assertEquals(1, texts.size());
    assertEquals("ホッキョクグマが北海道に来た", texts.get(0));
  }

  @Test
  void extractMultipleTextElements() throws XMLStreamException {
    String xml = "<root><page><text>こんにちは</text></page><page><text>もっと</text></page></root>";
    ByteArrayInputStream in = new ByteArrayInputStream(xml.getBytes(StandardCharsets.UTF_8));
    List<String> texts = new ArrayList<>();
    ContentUtils.extract(
        in,
        CJLanguage.JAPANESE,
        (text, lang) -> {
          texts.add(text.toString());
          assertEquals(CJLanguage.JAPANESE, lang);
        });
    assertEquals(List.of("こんにちは", "もっと"), texts);
  }

  @Test
  void extractSkipsEmptyText() throws XMLStreamException {
    String xml = "<page><text></text><text>こんにちは</text></page>";
    ByteArrayInputStream in = new ByteArrayInputStream(xml.getBytes(StandardCharsets.UTF_8));
    List<String> texts = new ArrayList<>();
    ContentUtils.extract(in, CJLanguage.JAPANESE, (text, lang) -> texts.add(text.toString()));
    assertEquals(1, texts.size());
    assertEquals("こんにちは", texts.get(0));
  }
}
