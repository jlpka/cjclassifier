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

import com.jlpka.cjclassifier.CJClassifier;
import com.jlpka.cjclassifier.CJLanguage;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.BiConsumer;
import javax.xml.stream.*;

/**
 * Utility to run evals on self-generated models. This is not necessary for running the
 * CJClassifier, which has its own model.
 *
 * <pre>
 * export INVOKEEVAL="java -cp tools/target/cjclassifier-tools-1.0.jar com.jlpka.cjclassifier.tools.EvalTool"
 * export WIKIDIR="../wikidata/orig/"
 * export MODELFILE="../wikidata/derived/cjlogprobs.gz"  # empty is ok
 *
 * Adhoc:
 * Runs an ad hoc query:
 *
 *   $INVOKEEVAL adhoc --phrase "羊驼是一种很好的动物。"
 *   $INVOKEEVAL adhoc --modelfile $MODELFILE --phrase "羊驼是一种很好的动物。"
 *   $INVOKEEVAL adhoc --modelfile $MODELFILE --phrase " 羊駝是一種很好的動物。"
 *   $INVOKEEVAL adhoc --modelfile $MODELFILE --phrase "東京皇居"
 *
 *
 * PhraseEval:
 * (Evaluates over a file of line-by-line phrases)
 *
 *   $INVOKEEVAL phraseeval --modelfile $MODELFILE --infile $PHRASES_DIR/zh.txt --language zh --misses
 *
 *
 * WikiEval:
 * Opens an xml wiki dump and iterates over phrases, running detection against an expected language.
 *
 *   $INVOKEEVAL wikieval --modelfile $MODELFILE --infiles $WIKIDIR/zhwiki-20260201-pages-articles.xml.bz2:zh
 *   $INVOKEEVAL wikieval --modelfile $MODELFILE --infiles $WIKIDIR/jawiki-20260201-pages-articles.xml.bz2:ja
 *
 *
 * loadmodelspeed:
 * Help benchmark the model loading speed, and prints memory as well:
 *
 *   $INVOKEEVAL loadmodelspeed --modelfile $MODELFILE
 *
 * </pre>
 */
public class EvalTool {

  /** Command for a single adhoc query */
  private static void adhocCommand(CJClassifier detector, String phrase) throws Exception {
    CJClassifier.Results results = new CJClassifier.Results();
    CJLanguage result = detector.detect(phrase, results);
    System.out.println(
        "Result: " + (result != null ? result.isoCode() + " (" + result + ")" : "unknown"));
    System.out.println(results.toString());
  }

  /** Command to evaluate vs a test file with a language per line */
  private static void phraseEvalCommand(
      CJClassifier detector, String testfile, CJLanguage expectedLanguage, boolean showMisses)
      throws Exception {
    int correct = 0;
    int total = 0;

    try (BufferedReader reader =
        new BufferedReader(
            new InputStreamReader(
                new FileInputStream(new File(testfile)), StandardCharsets.UTF_8))) {
      String line;
      while ((line = reader.readLine()) != null) {
        if (line.isBlank()) continue;
        total++;
        CJClassifier.Results results = new CJClassifier.Results();
        detector.detect(line, results);
        if (isEquivalent(results.result, expectedLanguage)) {
          correct++;
        } else if (showMisses) {
          String phrase = line.length() > 80 ? line.substring(0, 80) + "..." : line;
          String detected = results.result != null ? results.result.isoCode() : "null";
          System.err.println(
              String.format(
                  "  MISS [expected=%s detected=%s] scores={%s} \"%s\"",
                  expectedLanguage.isoCode(), detected, results.toString(), phrase));
        }
      }
    }
    double pct = total > 0 ? 100.0 * correct / total : 0;
    System.err.printf("Overall: %d/%d correct (%.1f%%)%n", correct, total, pct);
  }

  /** Poison pill for the producer-consumer queue (identity-compared). */
  @SuppressWarnings("StringOperationCanBeSimplified")
  private static final String POISON = new String("");

  // In Wikipedia and other corpora, simplified and traditional Chinese are mixed togther.
  // Hence call it equivalent if they're equal or both Chinese.
  private static boolean isEquivalent(CJLanguage a, CJLanguage b) {
    return a == b || (a.isChinese() && b.isChinese());
  }

  interface IsOkChar {
    boolean isOk(char ch);
  }

  /**
   * Tests classifier against an xml wikipedia dump. - minRun is a minimum number of characters to
   * eval, - fullDoc specifies that we should wait for a full document - justIdeographs specifies
   * that we should not include Japanese Kana (if present) - it has the effect of making the eval
   * "harder" by just using ideographs.
   */
  private static void wikiEvalCommand(
      List<Pair<String, CJLanguage>> infiles,
      String modelFile,
      int minRun,
      boolean fullDoc,
      boolean justIdeographs)
      throws Exception {
    CJClassifier detector = CJClassifier.load(modelFile);
    CJClassifier.Results detectStats = new CJClassifier.Results();
    long startTime = System.currentTimeMillis();

    long[] jaS = new long[ES.SIZE];
    long[] zhS = new long[ES.SIZE];
    long[] runCounter = {0};
    final long logInc = fullDoc ? 1_000 : 1_000_000;
    long[] nextLogHolder = {logInc};

    String[] missExample = {""};
    for (Pair<String, CJLanguage> lf : infiles) {
      CJLanguage expectedLang = lf.second();

      BiConsumer<CharSequence, CJLanguage> evalBlob =
          (text, lang) -> {
            int runLen = text.length();
            if (runLen < minRun) {
              return;
            }
            detectStats.clear();
            CJLanguage detected = detector.detect(text, detectStats);
            if (detected == CJLanguage.UNKNOWN) {
              return; // This means no CJ characters, just avoid counting it.
            }
            long[] s =
                expectedLang == CJLanguage.JAPANESE ? jaS : expectedLang.isChinese() ? zhS : null;
            if (s != null) {
              s[ES.RUNS.ordinal()]++;
              s[ES.CHARS.ordinal()] += runLen;
              if (detected == CJLanguage.JAPANESE) {
                s[ES.AS_JA.ordinal()]++;
                s[ES.AS_JA_CHARS.ordinal()] += runLen;
              } else if (detected == CJLanguage.CHINESE_SIMPLIFIED) {
                s[ES.AS_ZH_HANS.ordinal()]++;
                s[ES.AS_ZH_HANS_CHARS.ordinal()] += runLen;
              } else if (detected == CJLanguage.CHINESE_TRADITIONAL) {
                s[ES.AS_ZH_HANT.ordinal()]++;
                s[ES.AS_ZH_HANT_CHARS.ordinal()] += runLen;
              }
              // Track miss examples: wrong-language detections
              boolean isMiss =
                  (expectedLang == CJLanguage.JAPANESE && detected != CJLanguage.JAPANESE)
                      || (expectedLang.isChinese() && detected == CJLanguage.JAPANESE);
              if (isMiss) {
                missExample[0] =
                    text.subSequence(0, Math.min(text.length(), 100))
                        + " "
                        + detectStats.toString();
              }
            }
            runCounter[0]++;
            if (runCounter[0] >= nextLogHolder[0]) {
              printEvalStats("progress", runCounter[0], startTime, jaS, zhS);
              System.err.println("  Miss example: " + missExample[0]);
              nextLogHolder[0] += logInc;
            }
          };

      // justIdeographs causes us to just look at ideographic characters and
      // to not consider Hiragana/Katakana, making the eval more difficult.
      IsOkChar isOkChar =
          justIdeographs
              ? ContentUtils::inMainCJRange
              : (char ch) -> ContentUtils.inMainCJRange(ch) || ContentUtils.isKana(ch);

      BiConsumer<CharSequence, CJLanguage> evalConsumer;
      if (fullDoc) {
        evalConsumer =
            (text, lang) -> {
              StringBuilder sb = new StringBuilder();
              boolean sepAdded = true;
              for (int i = 0; i < text.length(); i++) {
                if (isOkChar.isOk(text.charAt(i))) {
                  sb.append(text.charAt(i));
                  sepAdded = false;
                } else if (!sepAdded) {
                  sb.append(' ');
                  sepAdded = true;
                }
              }
              evalBlob.accept(sb, lang);
            };
      } else {
        evalConsumer =
            (text, lang) -> {
              // Scan for runs of CJ ideographs >= minRun
              int runStart = -1;
              for (int i = 0; i <= text.length(); i++) {
                boolean inRange = i < text.length() && isOkChar.isOk(text.charAt(i));
                if (inRange && runStart < 0) {
                  runStart = i;
                } else if (!inRange && runStart >= 0) {
                  int runLen = i - runStart;
                  if (runLen >= minRun) {
                    evalBlob.accept(text.subSequence(runStart, i), lang);
                  }
                  runStart = -1;
                }
              }
            };
      }

      try (InputStream in = ContentUtils.openCompressed(lf.first())) {
        ContentUtils.extract(in, lf.second(), evalConsumer);
      }
    }

    printEvalStats("FINAL", runCounter[0], startTime, jaS, zhS);
  }

  private static void printEvalStats(
      String label, long totalRuns, long startTime, long[] jaS, long[] zhS) {
    double elapsed = (System.currentTimeMillis() - startTime) / 1000.0;
    System.err.printf("=== %s (%d runs, %.1fs) ===%n", label, totalRuns, elapsed);
    printEvalGroup("JA", jaS);
    printEvalGroup("ZH", zhS);
  }

  private static void printEvalGroup(String tag, long[] s) {
    long runs = s[ES.RUNS.ordinal()], chars = s[ES.CHARS.ordinal()];
    long asJa = s[ES.AS_JA.ordinal()], asJaCh = s[ES.AS_JA_CHARS.ordinal()];
    long asHans = s[ES.AS_ZH_HANS.ordinal()], asHansCh = s[ES.AS_ZH_HANS_CHARS.ordinal()];
    long asHant = s[ES.AS_ZH_HANT.ordinal()], asHantCh = s[ES.AS_ZH_HANT_CHARS.ordinal()];
    if (runs == 0) return;
    long zhEither = asHans + asHant;
    long zhEitherCh = asHansCh + asHantCh;
    System.err.printf(
        "  %s-tagged: %d runs, %d chars | ja=%d (%.1f%%) zh-hans=%d (%.1f%%) zh-hant=%d (%.1f%%)"
            + " zh-either=%d (%.1f%%)%n",
        tag,
        runs,
        chars,
        asJa,
        100.0 * asJa / runs,
        asHans,
        100.0 * asHans / runs,
        asHant,
        100.0 * asHant / runs,
        zhEither,
        100.0 * zhEither / runs);
    System.err.printf(
        "    by chars: ja=%d (%.1f%%) zh-hans=%d (%.1f%%) zh-hant=%d (%.1f%%) zh-either=%d"
            + " (%.1f%%)%n",
        asJaCh,
        100.0 * asJaCh / chars,
        asHansCh,
        100.0 * asHansCh / chars,
        asHantCh,
        100.0 * asHantCh / chars,
        zhEitherCh,
        100.0 * zhEitherCh / chars);
  }

  // Stats for wikiEval
  private enum ES {
    RUNS,
    CHARS,
    AS_JA,
    AS_JA_CHARS,
    AS_ZH_HANS,
    AS_ZH_HANS_CHARS,
    AS_ZH_HANT,
    AS_ZH_HANT_CHARS;
    static final int SIZE = values().length;
  }

  // "loadmodelspeed" command to benchmark loading
  private static void modelLoadingBenchmarkCommand(String[] args, int durationSec)
      throws Exception {
    // Warmup run with memory measurement
    System.err.println("Warmup...");
    Runtime rt = Runtime.getRuntime();
    System.gc();
    long heapBefore = rt.totalMemory() - rt.freeMemory();
    buildDetector(args);
    System.gc();
    long heapAfter = rt.totalMemory() - rt.freeMemory();
    double modelMB = (heapAfter - heapBefore) / (1024.0 * 1024.0);
    System.err.printf("** MODEL MEMORY: %.1fMB%n", modelMB);
    CJClassifier.clearCachedModels();

    long deadline = System.currentTimeMillis() + durationSec * 1000L;
    int iteration = 0;
    double totalSec = 0;
    double minSec = Double.MAX_VALUE;
    double maxSec = 0;

    while (System.currentTimeMillis() < deadline) {
      iteration++;
      long t0 = System.nanoTime();
      buildDetector(args);
      double sec = (System.nanoTime() - t0) / 1_000_000_000.0;
      CJClassifier.clearCachedModels();
      totalSec += sec;
      minSec = Math.min(minSec, sec);
      maxSec = Math.max(maxSec, sec);
      System.err.printf("  iteration %d: %.3fs%n", iteration, sec);
    }

    if (iteration > 0) {
      double avgSec = totalSec / iteration;
      System.err.printf("%nBenchmark results: %d iterations in %.1fs%n", iteration, totalSec);
      System.err.printf("  avg=%.3fs  min=%.3fs  max=%.3fs%n", avgSec, minSec, maxSec);
    }
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

  private static CJClassifier buildDetector(String[] args) throws Exception {
    String modelFile = getArg(args, "modelfile");
    String cjMinLogProbStr = getArg(args, "cjminlogprob");
    double cjMinLogProb = cjMinLogProbStr != null ? Double.parseDouble(cjMinLogProbStr) : 0;

    if (modelFile != null) {
      return CJClassifier.load(modelFile, cjMinLogProb);
    }
    // No --modelfile: use bundled default model
    return CJClassifier.load(cjMinLogProb);
  }

  /**
   * Checks for unrecognized --flags and exits with an error if any are found.
   *
   * @param args the full args array (args[0] is the command name)
   * @param validFlags set of valid flag names (without the "--" prefix)
   */
  private static void checkUnrecognizedArgs(String[] args, Set<String> validFlags) {
    // Flags whose values are separate args (--flag value); skip the value too.
    Set<String> valuedFlags =
        Set.of(
            "modelfile",
            "infile",
            "language",
            "duration",
            "byparagraph",
            "misses",
            "justideographs",
            "infiles",
            "cjminlogprob",
            "phrase");
    for (int i = 1; i < args.length; i++) {
      if (args[i].startsWith("--")) {
        String name = args[i].substring(2);
        if (!validFlags.contains(name)) {
          System.err.println("Error: unrecognized option: " + args[i]);
          printUsage();
          System.exit(1);
        }
        // Skip the value argument for valued flags
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
        "Usage: java -cp target/cjclassifier-tools-1.0.jar com.jlpka.cjclassifier.tools.EvalTool"
            + " <command> [options]");
    System.err.println();
    System.err.println("Commands:");
    System.err.println("  adhoc   - Detect language of a single phrase");
    System.err.println("            --phrase <text>       The phrase to detect (required)");
    System.err.println(
        "            --modelfile <file>    Combined logprob model file (default: bundled)");
    System.err.println(
        "            --cjminlogprob <val>  CJ classifier log-prob floor (0=use file value)");
    System.err.println();
    System.err.println("  phraseeval - Evaluate over a file of test phrases");
    System.err.println(
        "            --infile <file>      File with phrases in the expected language, one per"
            + " line");
    System.err.println("            --language <lang>    Expected language");
    System.err.println(
        "            --modelfile <file>    Combined logprob model file (default: bundled)");
    System.err.println(
        "            --cjminlogprob <val>  CJ classifier log-prob floor (0=use file value)");
    System.err.println();
    System.err.println("  wikieval  - Evaluate CJClassifier accuracy against Wikipedia dumps");
    System.err.println("            --infiles <list>   Comma-delimited file:lang pairs (required)");
    System.err.println("                                 e.g. wiki-zh.bz2:zh,wiki-ja.bz2:ja");
    System.err.println(
        "            --justideographs   Only look at ideographs, don't include Kana. This makes is"
            + " harder.");
    System.err.println(
        "            --minrun <n>       Minimum ideograph run length to evaluate (default 4)");
    System.err.println(
        "            --fulldoc          Eval over full docs, rather than on short segments");
    System.err.println(
        "            --modelfile <file> Combined logprob model file (default: bundled)");
    System.err.println(
        "            --cjminlogprob <val>  CJ classifier log-prob floor (0=use file value)");
    System.err.println();
    System.err.println("  loadmodelspeed - Benchmark model loading time (load + clear cache loop)");
    System.err.println("            --duration <sec>   Benchmark duration in seconds (default 30)");
    System.err.println(
        "            --modelfile <file> Combined logprob model file (default: bundled)");
    System.err.println(
        "            --cjminlogprob <val>  CJ classifier log-prob floor (0=use file value)");
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
    if (command.equalsIgnoreCase("adhoc")) {
      checkUnrecognizedArgs(args, Set.of("modelfile", "cjminlogprob", "phrase"));
      String phrase = getArg(args, "phrase");
      if (phrase == null) {
        System.err.println("Error: adhoc requires --phrase");
        System.exit(1);
      }
      CJClassifier detector = buildDetector(args);
      adhocCommand(detector, phrase);
    } else if (command.equalsIgnoreCase("phraseeval")) {
      checkUnrecognizedArgs(
          args, Set.of("modelfile", "cjminlogprob", "infile", "language", "misses"));
      String infile = getArg(args, "infile");
      CJLanguage language = CJLanguage.fromString(getArg(args, "language"));
      if (infile == null || language == CJLanguage.UNKNOWN) {
        System.err.println("Error: adhoc requires --infile and --language");
        System.exit(1);
      }
      CJClassifier detector = buildDetector(args);
      phraseEvalCommand(detector, infile, language, hasFlag(args, "misses"));
    } else if (command.equalsIgnoreCase("wikieval")) {
      checkUnrecognizedArgs(
          args, Set.of("modelfile", "infiles", "minrun", "fulldoc", "justideographs"));
      String modelFile = getArg(args, "modelfile");
      String infilesArg = getArg(args, "infiles");
      if (modelFile == null || infilesArg == null) {
        System.err.println("Error: wikieval requires --modelfile and --infiles");
        System.err.println(
            "  --infiles format: file1:lang1,file2:lang2  (e.g. wiki-zh.bz2:zh,wiki-ja.bz2:ja)");
        System.exit(1);
      }
      String minRunStr = getArg(args, "minrun");
      int minRun = minRunStr != null ? Integer.parseInt(minRunStr) : 4;
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
      wikiEvalCommand(
          langFiles, modelFile, minRun, hasFlag(args, "fulldoc"), hasFlag(args, "justideographs"));
    } else if (command.equalsIgnoreCase("loadmodelspeed")) {
      checkUnrecognizedArgs(args, Set.of("modelfile", "cjminlogprob", "duration"));
      String durationStr = getArg(args, "duration");
      int duration = durationStr != null ? Integer.parseInt(durationStr) : 30;
      modelLoadingBenchmarkCommand(args, duration);
    } else {
      System.err.println("Unknown command: " + command);
      printUsage();
      System.exit(1);
    }
  }
}
