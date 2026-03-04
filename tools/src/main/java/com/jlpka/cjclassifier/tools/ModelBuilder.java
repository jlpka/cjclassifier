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

import com.jlpka.cjclassifier.CJLanguage;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;
import java.util.TreeMap;
import java.util.function.BiConsumer;
import java.util.zip.GZIPOutputStream;
import javax.xml.stream.*;

/**
 * Extracts text content from Wikipedia XML dump files (bzip2 compressed) to generate statistical
 * models for language classification.
 *
 * Using this module isn't necessary for classifiying text, this is for generating a new
 * model from Wikipedia (or similar) data.
 *
 * Our model consists of frequency data for both:
 *   - Single characters in the main CJ ideograph range [0x3400..0x9FFF]  (unigrams)
 *   - Adjacent characters in the main CJ ideograph range [0x3400..0x9FFF] (bigrams)
 *
 * The here idea is to compute the log(probability) of each relevant character unigram and bigram
 * based on the observed text.
 *
 * We use log(probability) rather than raw probability because we need to multiply probabilities
 * to get the total probability. In log space, this is just a simple addition. It's less costly
 * and it preserves floating point range.
 *
 * We treat this in a couple of phases:
 *
 * The first is computing raw stats over each wikipedia dump with the "statscj" command:
 * <pre>
 * export INVOKEBUILDER="java -cp tools/target/cjclassifier-tools-1.0.jar com.jlpka.cjclassifier.tools.ModelBuilder"
 * export WIKISRC="../wikipedia/orig"
 * export WIKIDERIVED="../wikipedia/derived"
 *
 * Run on Japanese and Chinese wikipedia dumps:
 * Note that these can be found in locations like:  https://dumps.wikimedia.org/other/mediawiki_content_history/jawiki/2026-02-01/xml/bzip2/
 * Don't download these gratuitiously, e.g. for 2026-02-01, Chinese is 32GB bzipped, and Japanese is 45GB.
 *
 *   $INVOKEBUILDER statscj --infiles $WIKISRC/jawiki-20260201-pages-articles.xml.bz2:ja --outfile $WIKIDERIVED/statsja.txt
 *   $INVOKEBUILDER statscj --infiles $WIKISRC/zhwiki-20260201-pages-articles.xml.bz2:zh --outfile $WIKIDERIVED/statszh.txt
 *</pre>
 *
 * These stats files are text files that look like "[char(s)] count(s)"
 *
 * The second phase is to combine stats files into a single logprob model file, with combinecj.
 * It's typically gzipped for disk space savings. The minlogprob is the minimum log(probability)
 * to emit - it helps trade off memory usage for accuracy. After a certain point, the expected
 * model probabilities are somewhat ambiguous.
 *
 * <pre>
 *  $INVOKEBUILDER combinecj --infiles $WIKIDERIVED/statsja.txt,$WIKIDERIVED/statszh.txt --outfile $WIKIDERIVED/cjlogprobs.gz --gzip --aslogprob --minlogprob -16.0
 * </pre>
 *
 * We also have a showwiki command to test out the wiki display function:
 * <pre>
 *  $INVOKEBUILDER showwiki --infile $WIKISRC/jawiki-20260201-pages-articles.xml.bz2 | less
 * </pre>
 */
public class ModelBuilder {

  private static void showWikiCommand(String infile, String outfile) throws Exception {
    PrintWriter out =
        outfile != null
            ? new PrintWriter(
                new BufferedWriter(new FileWriter(outfile, StandardCharsets.UTF_8), 1024 * 1024))
            : new PrintWriter(
                new BufferedWriter(
                    new OutputStreamWriter(System.out, StandardCharsets.UTF_8), 1024 * 1024));

    long[] pageCount = {0};
    long startTime = System.currentTimeMillis();

    BiConsumer<CharSequence, CJLanguage> printer =
        (text, lang) -> {
          if (pageCount[0] > 0) {
            out.print("\n\n");
          }
          out.print(text);
          pageCount[0]++;

          if (pageCount[0] % 10000 == 0) {
            System.err.printf("Processed %d pages...%n", pageCount[0]);
          }
        };

    try (InputStream in = ContentUtils.openCompressed(infile)) {
      ContentUtils.extract(in, null, printer);
    }

    out.flush();
    if (outfile != null) {
      out.close();
    }

    long elapsed = System.currentTimeMillis() - startTime;
    System.err.printf("Done. Extracted %d pages in %.2f seconds%n", pageCount[0], elapsed / 1000.0);
  }

  // ========================================================================
  // CJ stats primitives
  // ========================================================================

  // We take input wikpedia files tagged by language.
  //
  // We emit a text output file with a header that looks like:
  //   Unigrams: [num]  Bigrams: [num] Languages: [ja,zh-hans,zh-hant]
  // And then each line:
  //   unigram1 [count_lang1] [count_lang2] [count_lang3]
  //   ...
  //   bigram1 [count_lang1] [count_lang2] [count_lang3]
  //   ...
  //
  // We can later combine multiple of these into a single file that contains
  //   unigram1 log(prob_lang1), log(prob_lang2) log(prob_lang3)
  //   ...
  //   bigram1 log(prob_lang1), log(prob_lang2) log(prob_lang3)
  //   ...
  private static boolean cjAnyNonZero(
      long[][] unigramCounts, int cjCharIdx, List<Integer> activeLangIndices) {
    for (int li : activeLangIndices) {
      if (unigramCounts[li][cjCharIdx] > 0) return true;
    }
    return false;
  }

  private static void cjStatsAppendValue(
      StringBuilder sb, long count, long total, boolean asLogProb, double minLogProb) {
    sb.append(' ');
    if (asLogProb) {
      if (count == 0) {
        sb.append("0");
      } else {
        double logprob = Math.log((double) count / total);
        if (logprob >= minLogProb) {
          sb.append(String.format("%.4f", logprob));
        } else {
          sb.append("0");
        }
      }
    } else {
      sb.append(count);
    }
  }

  private static final CJLanguage[] CJ_LANGUAGES = {
    CJLanguage.CHINESE_SIMPLIFIED, CJLanguage.CHINESE_TRADITIONAL, CJLanguage.JAPANESE
  };

  private static int cjLangIndex(CJLanguage lang) {
    switch (lang) {
      case CHINESE_SIMPLIFIED:
        return 0;
      case CHINESE_TRADITIONAL:
        return 1;
      case JAPANESE:
        return 2;
      default:
        return -1;
    }
  }

  private static int cjBigramKey(char c1, char c2) {
    return ((int) c1 << 16) | (int) c2;
  }

  private static void writeCJStats(
      long[][] unigramCounts,
      long[] totalChars,
      Map<Integer, long[]> bigramCounts,
      long[] totalBigrams,
      String outfile,
      boolean asLogProb,
      double minLogProb,
      boolean gzip)
      throws Exception {
    List<Integer> activeLangIndices = new ArrayList<>();
    for (int li = 0; li < CJ_LANGUAGES.length; li++) {
      if (totalChars[li] > 0) {
        activeLangIndices.add(li);
      }
    }

    OutputStream fileOut = new FileOutputStream(outfile);
    if (gzip) {
      fileOut = new GZIPOutputStream(fileOut);
    }
    // Build header
    StringJoiner langJoiner = new StringJoiner(",");
    StringJoiner uniTotalJoiner = new StringJoiner(",");
    StringJoiner biTotalJoiner = new StringJoiner(",");
    for (int li : activeLangIndices) {
      langJoiner.add(CJ_LANGUAGES[li].isoCode());
      uniTotalJoiner.add(Long.toString(totalChars[li]));
      biTotalJoiner.add(Long.toString(totalBigrams[li]));
    }
    String header;
    if (asLogProb) {
      header =
          String.format(
              "Languages: %s UnigramTotals: %s BigramTotals: %s MinLogProb: %.1f",
              langJoiner, uniTotalJoiner, biTotalJoiner, minLogProb);
    } else {
      header =
          String.format(
              "Languages: %s UnigramTotals: %s BigramTotals: %s",
              langJoiner, uniTotalJoiner, biTotalJoiner);
    }

    // Build data lines
    List<String> unigramLines = new ArrayList<>();
    List<String> bigramLines = new ArrayList<>();
    StringBuilder sb = new StringBuilder();

    // Unigrams
    for (int i = 0; i < ContentUtils.CJ_RANGE_SIZE; i++) {
      if (!cjAnyNonZero(unigramCounts, i, activeLangIndices)) continue;
      if (asLogProb) {
        boolean anyMetMin = false;
        for (int li : activeLangIndices) {
          if (unigramCounts[li][i] > 0
              && Math.log((double) unigramCounts[li][i] / totalChars[li]) >= minLogProb) {
            anyMetMin = true;
            break;
          }
        }
        if (!anyMetMin) {
          continue;
        }
      }
      char c = (char) (ContentUtils.CJ_RANGE_START + i);
      sb.setLength(0);
      sb.append(c);
      for (int li : activeLangIndices) {
        cjStatsAppendValue(sb, unigramCounts[li][i], totalChars[li], asLogProb, minLogProb);
      }
      unigramLines.add(sb.toString());
    }

    // Bigrams
    for (Map.Entry<Integer, long[]> entry : bigramCounts.entrySet()) {
      long[] counts = entry.getValue();
      if (asLogProb) {
        boolean anyMetMin = false;
        for (int li : activeLangIndices) {
          // See comment about totalChars vs totalBigrams below.
          if (counts[li] > 0 && Math.log((double) counts[li] / totalBigrams[li]) >= minLogProb) {
            anyMetMin = true;
            break;
          }
        }
        if (!anyMetMin) {
          continue;
        }
      }
      int key = entry.getKey();
      char c1 = (char) (key >> 16);
      char c2 = (char) (key & 0xFFFF);
      sb.setLength(0);
      sb.append(c1).append(c2);
      for (int li : activeLangIndices) {
        // Non-obvious: using totalChars as the logprob denominator rather than totalBigrams.
        // This is because the classifier treat the bigram as the joint probability of the
        // 2-character sequence appearing at all.
        // We're including the UnigramTotals & BigramTotals in the header line, so it's
        // totally possible to re-normalize by adding log(totalBigrams[i]/totalChars[i])
        cjStatsAppendValue(sb, counts[li], totalBigrams[li], asLogProb, minLogProb);
      }
      bigramLines.add(sb.toString());
    }

    // Sort data lines lexicographically (more predictable and compresses better)
    // unigrams are effectively already sorted.
    Collections.sort(bigramLines);

    // Write
    try (PrintWriter out =
        new PrintWriter(
            new BufferedWriter(
                new OutputStreamWriter(fileOut, StandardCharsets.UTF_8), 1024 * 1024))) {
      out.println(header);
      for (String line : unigramLines) {
        out.println(line);
      }
      for (String line : bigramLines) {
        out.println(line);
      }
    }
    System.err.printf(
        "Wrote %d unigrams and %d bigrams to %s%n",
        unigramLines.size(), bigramLines.size(), outfile);
  }

  private static void statsCJ(List<Pair<String, CJLanguage>> infiles, String outfile)
      throws Exception {
    // unigramCounts[langIndex][charIndex]
    long[][] unigramCounts = new long[CJ_LANGUAGES.length][ContentUtils.CJ_RANGE_SIZE];
    long[] totalChars = new long[CJ_LANGUAGES.length];
    // bigramCounts: bigramKey -> long[CJ_LANGUAGES.length]
    Map<Integer, long[]> bigramCounts = new HashMap<>();
    long[] totalBigrams = new long[CJ_LANGUAGES.length];
    long startTime = System.currentTimeMillis();

    long[] totalAll = {0};
    long[] nextLog = {1_000_000};
    BiConsumer<CharSequence, CJLanguage> counter =
        (text, lang) -> {
          int li = cjLangIndex(lang);
          if (li < 0) return;
          char prev = 0;
          boolean prevInRange = false;
          for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            boolean inRange = ContentUtils.inMainCJRange(c);
            if (inRange) {
              unigramCounts[li][c - ContentUtils.CJ_RANGE_START]++;
              totalChars[li]++;
              if (prevInRange) {
                int key = cjBigramKey(prev, c);
                bigramCounts.computeIfAbsent(key, k -> new long[CJ_LANGUAGES.length])[li]++;
                totalBigrams[li]++;
              }
              if (++totalAll[0] >= nextLog[0]) {
                System.err.printf(
                    "Processed %dM unigrams (%.1fs)...%n",
                    totalAll[0] / 1_000_000, (System.currentTimeMillis() - startTime) / 1000.0);
                nextLog[0] += 1_000_000;
              }
            }
            prev = c;
            prevInRange = inRange;
          }
        };
    for (Pair<String, CJLanguage> lf : infiles) {
      try (InputStream in = ContentUtils.openCompressed(lf.first())) {
        BiConsumer<CharSequence, CJLanguage> cb =
            lf.second().isChinese() ? ContentUtils.wrapTryBothFormsForChinese(counter) : counter;
        ContentUtils.extract(in, lf.second(), ContentUtils.wrapEmitJustCJRuns(cb));
      }
    }

    long elapsed = System.currentTimeMillis() - startTime;
    long totalAllChars = 0, totalAllBigrams = 0;
    for (int li = 0; li < CJ_LANGUAGES.length; li++) {
      totalAllChars += totalChars[li];
      totalAllBigrams += totalBigrams[li];
    }
    System.err.printf(
        "Scanned %d CJ characters, %d bigrams in %.2f seconds%n",
        totalAllChars, totalAllBigrams, elapsed / 1000.0);

    writeCJStats(
        unigramCounts,
        totalChars,
        bigramCounts,
        totalBigrams,
        outfile,
        false,
        Double.NEGATIVE_INFINITY,
        false);
  }

  private static void combineCJ(
      List<String> infiles, String outfile, boolean asLogProb, double minLogProb, boolean gzip)
      throws Exception {
    long[][] unigramCounts = new long[CJ_LANGUAGES.length][ContentUtils.CJ_RANGE_SIZE];
    long[] totalChars = new long[CJ_LANGUAGES.length];
    Map<Integer, long[]> bigramCounts = new TreeMap<>();
    long[] totalBigrams = new long[CJ_LANGUAGES.length];

    for (String infile : infiles) {
      try (BufferedReader reader =
          new BufferedReader(
              new InputStreamReader(ContentUtils.openCompressed(infile), StandardCharsets.UTF_8),
              1024 * 1024)) {
        // Parse header: Languages: ja,zh-hans,zh-hant UnigramTotals: ... BigramTotals: ...
        String header = reader.readLine();
        if (header == null || !header.startsWith("Languages: ")) {
          throw new IllegalArgumentException("Invalid stats file (bad header): " + infile);
        }
        String[] headerParts = header.split(" ");
        // headerParts: [Languages:, ja,zh-hans, UnigramTotals:, 1,2,3, BigramTotals:, 4,5,6]
        String[] langCodes = headerParts[1].split(",");
        int[] langMap = new int[langCodes.length]; // maps file column -> CJ_LANGUAGES index
        for (int i = 0; i < langCodes.length; i++) {
          CJLanguage lang = CJLanguage.fromString(langCodes[i]);
          int li = cjLangIndex(lang);
          if (li < 0) {
            throw new IllegalArgumentException(
                "Unknown CJ language in header: " + langCodes[i] + " in " + infile);
          }
          langMap[i] = li;
        }

        // Read unigram and bigram lines
        String line;
        while ((line = reader.readLine()) != null) {
          if (line.isEmpty()) continue;
          String[] parts = line.split(" ");
          String key = parts[0];
          if (parts.length != langMap.length + 1) {
            throw new IllegalArgumentException(
                "Column count mismatch on line: " + line + " in " + infile);
          }
          if (key.length() == 1) {
            // Unigram
            char c = key.charAt(0);
            int idx = c - ContentUtils.CJ_RANGE_START;
            for (int col = 0; col < langMap.length; col++) {
              long count = Long.parseLong(parts[col + 1]);
              unigramCounts[langMap[col]][idx] += count;
              totalChars[langMap[col]] += count;
            }
          } else if (key.length() == 2) {
            // Bigram
            char c1 = key.charAt(0);
            char c2 = key.charAt(1);
            int bk = cjBigramKey(c1, c2);
            long[] counts = bigramCounts.computeIfAbsent(bk, k -> new long[CJ_LANGUAGES.length]);
            for (int col = 0; col < langMap.length; col++) {
              long count = Long.parseLong(parts[col + 1]);
              counts[langMap[col]] += count;
              totalBigrams[langMap[col]] += count;
            }
          }
        }
      }
    }

    System.err.printf("Combined %d stats files%n", infiles.size());
    writeCJStats(
        unigramCounts,
        totalChars,
        bigramCounts,
        totalBigrams,
        outfile,
        asLogProb,
        minLogProb,
        gzip);
  }

  // ========================================================================
  // CLI utilities
  // ========================================================================

  private static String getArg(String[] args, String name) {
    String flag = "--" + name;
    for (int i = 0; i < args.length - 1; i++) {
      if (flag.equals(args[i])) {
        return args[i + 1];
      }
    }
    return null;
  }

  private static boolean hasFlag(String[] args, String name) {
    String flag = "--" + name;
    for (String arg : args) {
      if (flag.equals(arg)) return true;
    }
    return false;
  }

  /**
   * Loads skip words from a comma-separated list of files. Each file contains one word per line.
   * Lines starting with '#' and empty lines are ignored. Only the text before the first space is
   * used as the word (to support topwords format).
   */
  private static Set<String> loadSkipWords(String skipwordsArg) throws IOException {
    Set<String> skipWords = new HashSet<>();
    if (skipwordsArg == null) {
      return skipWords;
    }
    for (String entry : skipwordsArg.split(",")) {
      entry = entry.trim();
      if (entry.isEmpty()) {
        continue;
      }
      // Optional /N suffix to limit to first N words from the file.
      String filename;
      int maxWords = Integer.MAX_VALUE;
      int slashIdx = entry.lastIndexOf('/');
      if (slashIdx > 0) {
        String suffix = entry.substring(slashIdx + 1);
        try {
          maxWords = Integer.parseInt(suffix);
          filename = entry.substring(0, slashIdx);
        } catch (NumberFormatException e) {
          // Not a number after slash — treat whole entry as filename.
          filename = entry;
        }
      } else {
        filename = entry;
      }
      int count = 0;
      try (BufferedReader reader =
          new BufferedReader(
              new InputStreamReader(new FileInputStream(filename), StandardCharsets.UTF_8))) {
        String line;
        while ((line = reader.readLine()) != null) {
          if (line.isEmpty() || line.startsWith("#")) {
            continue;
          }
          int space = line.indexOf(' ');
          String word = space >= 0 ? line.substring(0, space) : line;
          if (!word.isEmpty()) {
            skipWords.add(word);
            if (++count >= maxWords) {
              break;
            }
          }
        }
      }
    }
    if (!skipWords.isEmpty()) {
      System.err.printf("Loaded %d skip words%n", skipWords.size());
    }
    return skipWords;
  }

  private static void checkUnrecognizedArgs(String[] args, Set<String> validFlags) {
    Set<String> valuedFlags = Set.of("infile", "outfile", "minlogprob", "infiles", "modelfile");
    for (int i = 1; i < args.length; i++) {
      if (args[i].startsWith("--")) {
        String name = args[i].substring(2);
        if (!validFlags.contains(name)) {
          System.err.println("Error: unrecognized option: " + args[i]);
          printUsage();
          System.exit(1);
        }
        if (valuedFlags.contains(name) && i + 1 < args.length) {
          i++;
        }
      } else {
        System.err.println("Error: unexpected argument: " + args[i]);
        printUsage();
        System.exit(1);
      }
    }
  }

  private static void printUsage() {
    System.err.println(
        "Usage: java -cp target/cjclassifier-tools-1.0.jar"
            + " com.jlpka.cjclassifier.tools.ModelBuilder <command> [options]");
    System.err.println();
    System.err.println("Commands:");
    System.err.println("  statscj - Compute per-language CJ character/bigram stats");
    System.err.println("            --infiles <list>   Comma-delimited file:lang pairs (required)");
    System.err.println("                               e.g. wiki-zh.bz2:zh,wiki-ja.bz2:ja");
    System.err.println("            --outfile <file>   Output file (required)");
    System.err.println();
    System.err.println("  combinecj - Combine multiple CJ stats output files");
    System.err.println(
        "            --infiles <list>   Comma-delimited list of stats files (required)");
    System.err.println("                               e.g. stats-zh.txt,stats-ja.txt");
    System.err.println("            --outfile <file>   Output file (required)");
    System.err.println(
        "            --aslogprob        Emit log(count/total) instead of raw counts");
    System.err.println(
        "            --minlogprob <val> Min log-prob threshold (default -16, only with"
            + " --aslogprob)");
    System.err.println("            --gzip             Gzip the output file");
    System.err.println();
    System.err.println("  showwiki  - Extract and display text from Wikipedia XML dump");
    System.err.println("            --infile <file>   Input file (required)");
    System.err.println("            --outfile <file>  Output file (optional, defaults to stdout)");
  }

  // ========================================================================
  // Main
  // ========================================================================

  public static void main(String[] args) throws Exception {
    if (args.length < 1) {
      printUsage();
      System.exit(1);
    }

    String command = args[0];
    if (command.equals("showwiki")) {
      checkUnrecognizedArgs(args, Set.of("infile", "outfile"));
      String infile = getArg(args, "infile");
      String outfile = getArg(args, "outfile");
      if (infile == null) {
        System.err.println("Error: show requires --infile");
        System.exit(1);
      }
      showWikiCommand(infile, outfile);
    } else if (command.equals("statscj")) {
      checkUnrecognizedArgs(args, Set.of("infiles", "outfile"));
      String infilesArg = getArg(args, "infiles");
      String outfile = getArg(args, "outfile");
      if (infilesArg == null || outfile == null) {
        System.err.println("Error: stats requires --infiles and --outfile");
        System.err.println(
            "  --infiles format: file1:lang1,file2:lang2  (e.g. wiki-zh.bz2:zh,wiki-ja.bz2:ja)");
        System.exit(1);
      }
      List<Pair<String, CJLanguage>> langFiles = new ArrayList<>();
      for (String entry : infilesArg.split(",")) {
        int colonIdx = entry.lastIndexOf(':');
        if (colonIdx < 0) {
          System.err.println(
              "Error: each infile must be followed by :language_code, got: " + entry);
          System.exit(1);
        }
        String path = entry.substring(0, colonIdx);
        CJLanguage lang = CJLanguage.fromString(entry.substring(colonIdx + 1));
        if (lang == CJLanguage.UNKNOWN) {
          System.err.println("Error: unknown language code in: " + entry);
          System.exit(1);
        }
        langFiles.add(new Pair<>(path, lang));
      }
      statsCJ(langFiles, outfile);
    } else if (command.equals("combinecj")) {
      checkUnrecognizedArgs(args, Set.of("infiles", "outfile", "aslogprob", "minlogprob", "gzip"));
      String infilesArg = getArg(args, "infiles");
      String outfile = getArg(args, "outfile");
      if (infilesArg == null || outfile == null) {
        System.err.println("Error: combine requires --infiles and --outfile");
        System.err.println("  --infiles format: file1,file2");
        System.exit(1);
      }
      List<String> files = List.of(infilesArg.split(","));
      boolean asLogProb = hasFlag(args, "aslogprob");
      String minProbStr = getArg(args, "minlogprob");
      double minLogProb = minProbStr != null ? Double.parseDouble(minProbStr) : -16.0;
      boolean gzip = hasFlag(args, "gzip");
      combineCJ(files, outfile, asLogProb, minLogProb, gzip);
    } else {
      System.err.println("Unknown command: " + command);
      printUsage();
      System.exit(1);
    }
  }
}
