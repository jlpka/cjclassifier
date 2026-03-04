package com.jlpka.cjclassifier;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests for CJClassifier using synthetic model files.
 *
 * <p>We pick CJK Unified Ideograph characters in the main range (0x4E00–0x9FFF) and assign
 * artificial log-probabilities to create clear language signals. The model file format is: header:
 * Languages: zh-hans,zh-hant,ja UnigramTotals: ... BigramTotals: ... MinLogProb: -10.0 unigram
 * lines: <char> <zhHansProb> <zhHantProb> <jaProb> bigram lines: <char1><char2> <zhHansProb>
 * <zhHantProb> <jaProb>
 */
class CJClassifierBaseTest {

  @TempDir static Path tempDir;

  static CJClassifier detector;

  // Pick chars in CJK Unified Ideographs block (0x4E00+), well within CJ_RANGE
  // Simplified-leaning chars
  static final char ZH_HANS_A = '\u4e00'; // 一
  static final char ZH_HANS_B = '\u4e8c'; // 二
  static final char ZH_HANS_C = '\u4e09'; // 三

  // Traditional-leaning chars
  static final char ZH_HANT_A = '\u5b78'; // 學
  static final char ZH_HANT_B = '\u6a23'; // 樣
  static final char ZH_HANT_C = '\u9a57'; // 驗

  // Japanese-leaning chars
  static final char JA_A = '\u8fbc'; // 込
  static final char JA_B = '\u7573'; // 畳
  static final char JA_C = '\u586b'; // 填

  @BeforeAll
  static void createModelAndDetector() throws IOException {
    Path modelFile = tempDir.resolve("test-model.txt");
    try (PrintWriter out =
        new PrintWriter(Files.newBufferedWriter(modelFile, StandardCharsets.UTF_8))) {
      // Header: columns are zh-hans, zh-hant, ja
      out.println(
          "Languages: zh-hans,zh-hant,ja UnigramTotals: 1000,1000,1000 BigramTotals: 1000,1000,1000"
              + " MinLogProb: -10.0");

      // Simplified-leaning unigrams: high zh-hans, low others
      out.printf("%c -2.0 -8.0 -8.0%n", ZH_HANS_A);
      out.printf("%c -2.0 -8.0 -8.0%n", ZH_HANS_B);
      out.printf("%c -2.0 -8.0 -8.0%n", ZH_HANS_C);

      // Traditional-leaning unigrams: high zh-hant, low others
      out.printf("%c -8.0 -2.0 -8.0%n", ZH_HANT_A);
      out.printf("%c -8.0 -2.0 -8.0%n", ZH_HANT_B);
      out.printf("%c -8.0 -2.0 -8.0%n", ZH_HANT_C);

      // Japanese-leaning unigrams: high ja, low others
      out.printf("%c -8.0 -8.0 -2.0%n", JA_A);
      out.printf("%c -8.0 -8.0 -2.0%n", JA_B);
      out.printf("%c -8.0 -8.0 -2.0%n", JA_C);

      // Bigrams: reinforce the same signals
      out.printf("%c%c -2.0 -8.0 -8.0%n", ZH_HANS_A, ZH_HANS_B);
      out.printf("%c%c -2.0 -8.0 -8.0%n", ZH_HANS_B, ZH_HANS_C);
      out.printf("%c%c -8.0 -2.0 -8.0%n", ZH_HANT_A, ZH_HANT_B);
      out.printf("%c%c -8.0 -2.0 -8.0%n", ZH_HANT_B, ZH_HANT_C);
      out.printf("%c%c -8.0 -8.0 -2.0%n", JA_A, JA_B);
      out.printf("%c%c -8.0 -8.0 -2.0%n", JA_B, JA_C);
    }

    detector = new CJClassifier(modelFile.toString());
  }

  private CJClassifier.Results stats;

  @BeforeEach
  void resetStats() {
    stats = new CJClassifier.Results();
  }

  // ========================================================================
  // Basic detection
  // ========================================================================

  @Test
  void detectSimplifiedChinese() {
    String text = "" + ZH_HANS_A + ZH_HANS_B + ZH_HANS_C;
    CJLanguage result = detector.detect(text, stats);
    assertEquals(CJLanguage.CHINESE_SIMPLIFIED, result);
    assertEquals(0, stats.scores.kanaCount);
  }

  @Test
  void detectTraditionalChinese() {
    String text = "" + ZH_HANT_A + ZH_HANT_B + ZH_HANT_C;
    CJLanguage result = detector.detect(text, stats);
    assertEquals(CJLanguage.CHINESE_TRADITIONAL, result);
    assertEquals(0, stats.scores.kanaCount);
  }

  @Test
  void detectJapaneseByIdeographs() {
    String text = "" + JA_A + JA_B + JA_C;
    CJLanguage result = detector.detect(text, stats);
    assertEquals(CJLanguage.JAPANESE, result);
    assertEquals(0, stats.scores.kanaCount);
  }

  // ========================================================================
  // Kana shortcut
  // ========================================================================

  @Test
  void detectJapaneseByKana() {
    // Hiragana chars
    CJLanguage result = detector.detect("ひらがな", stats);
    assertEquals(CJLanguage.JAPANESE, result);
    assertTrue(stats.scores.kanaCount > 0);
    // katakana chars
    result = detector.detect("カタカナ", stats);
    assertEquals(CJLanguage.JAPANESE, result);
    assertTrue(stats.scores.kanaCount > 0);
  }

  @Test
  void kanaOverridesIdeographSignal() {
    // Start with simplified-leaning ideographs, then hit a hiragana
    String text = "" + ZH_HANS_A + ZH_HANS_B + '\u3053';
    CJLanguage result = detector.detect(text, stats);
    assertEquals(CJLanguage.JAPANESE, result);
    assertTrue(stats.scores.kanaCount > 0);
  }

  @Test
  void smallKanaFractionStillChinese() {
    // 100+ simplified ideographs with 1 kana char — kana fraction < 1% threshold.
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < 101; i++) {
      sb.append(ZH_HANS_A);
    }
    sb.append('\u3053'); // one hiragana
    CJLanguage result = detector.detect(sb.toString(), stats);
    assertEquals(CJLanguage.CHINESE_SIMPLIFIED, result);
    assertEquals(1, stats.scores.kanaCount);
    assertEquals(101, stats.scores.cjCharCount);
  }

  @Test
  void largeKanaFractionIsJapanese() {
    // 2 ideographs + 1 kana = 33% kana, well above 1% threshold.
    String text = "" + ZH_HANS_A + ZH_HANS_B + '\u3053';
    CJLanguage result = detector.detect(text, stats);
    assertEquals(CJLanguage.JAPANESE, result);
    assertTrue(stats.scores.kanaCount > 0);
  }

  // ========================================================================
  // Edge cases
  // ========================================================================

  @Test
  void emptyTextReturnsUnknown() {
    CJLanguage result = detector.detect("", stats);
    assertEquals(CJLanguage.UNKNOWN, result);
  }

  @Test
  void nonCJTextReturnsUnknown() {
    CJLanguage result = detector.detect("Hello world 123", stats);
    assertEquals(CJLanguage.UNKNOWN, result);
  }

  @Test
  void singleCharDetection() {
    // Even a single char should produce a result (not UNKNOWN)
    CJLanguage result = detector.detect("" + JA_A, stats);
    assertEquals(CJLanguage.JAPANESE, result);
  }

  // ========================================================================
  // Stats
  // ========================================================================

  @Test
  void statsPopulatedCorrectly() {
    String text = "" + ZH_HANS_A + ZH_HANS_B + ZH_HANS_C;
    detector.detect(text, stats);

    // 3 unigrams, 2 bigrams (AB, BC)
    assertTrue(stats.scores.unigramScores[0] != 0, "zh-hans unigram score should be nonzero");
    assertTrue(stats.totalScores[0] != 0, "zh-hans total score should be nonzero");
  }

  @Test
  void clearResetsAllFields() {
    // First detection
    detector.detect("" + ZH_HANS_A + ZH_HANS_B, stats);
    assertNotNull(stats.result);

    // Clear
    stats.clear();
    assertNull(stats.result);
    assertEquals(0, stats.scores.kanaCount);
    for (int i = 0; i < 3; i++) {
      assertEquals(0.0, stats.scores.unigramScores[i]);
      assertEquals(0.0, stats.scores.bigramScores[i]);
      assertEquals(0.0, stats.totalScores[i]);
      assertEquals(0, stats.scores.unigramHitsPerLang[i]);
      assertEquals(0, stats.scores.bigramHitsPerLang[i]);
    }
  }

  @Test
  void clearThenReuseGivesCleanResult() {
    // Detect simplified Chinese
    detector.detect("" + ZH_HANS_A + ZH_HANS_B + ZH_HANS_C, stats);
    assertEquals(CJLanguage.CHINESE_SIMPLIFIED, stats.result);

    // Clear and detect Japanese
    stats.clear();
    detector.detect("" + JA_A + JA_B + JA_C, stats);
    assertEquals(CJLanguage.JAPANESE, stats.result);
    assertEquals(0, stats.scores.kanaCount);
  }

  // ========================================================================
  // toString / toShortString
  // ========================================================================

  @Test
  void toStringContainsResult() {
    detector.detect("" + ZH_HANS_A + ZH_HANS_B, stats);
    String s = stats.toString();
    assertTrue(s.contains("zh-hans"), "toString should contain language code");
    assertTrue(s.contains("Results{"), "toString should start with class name");
  }

  @Test
  void toShortStringFormat() {
    detector.detect("" + ZH_HANS_A + ZH_HANS_B + ZH_HANS_C, stats);
    String s = stats.toShortString();
    assertFalse(s.isEmpty());
    // Should contain all three language codes
    assertTrue(s.contains("zh-hans"));
    assertTrue(s.contains("zh-hant"));
    assertTrue(s.contains("ja"));
    // Format is lang:ratio,lang:ratio,lang:ratio
    assertEquals(2, s.chars().filter(c -> c == ',').count(), "Should have 2 commas");
  }

  @Test
  void toShortStringKana() {
    detector.detect("\u3053\u3093", stats);
    assertEquals("ja:1.0,zh-hans:0,zh-hant:0", stats.toShortString());
  }

  @Test
  void toShortStringUnknownIsEmpty() {
    detector.detect("Hello", stats);
    assertEquals("", stats.toShortString());
  }

  // ========================================================================
  // MinProb parsing
  // ========================================================================

  @Test
  void minProbParsedFromHeader() throws IOException {
    Path modelFile = tempDir.resolve("minprob-model.txt");
    try (PrintWriter out =
        new PrintWriter(Files.newBufferedWriter(modelFile, StandardCharsets.UTF_8))) {
      out.println(
          "Languages: zh-hans,zh-hant,ja UnigramTotals: 100,100,100 BigramTotals: 100,100,100"
              + " MinLogProb: -5.0");
      out.printf("%c -2.0 -4.0 -4.0%n", ZH_HANS_A);
    }
    // If MinProb is parsed correctly, defaultLogProb should be -5.0
    // We can verify indirectly: a char not in the model should use defaultLogProb
    CJClassifier d = new CJClassifier(modelFile.toString());
    CJClassifier.Results s = new CJClassifier.Results();
    // Use a char that IS in range but NOT in the model — should get defaultLogProb for all langs
    // char 0x4E10 is in range but we only defined 0x4E00
    d.detect("\u4e10", s);
    // All unigram scores should be equal (all got defaultLogProb = -5.0)
    assertEquals(s.scores.unigramScores[0], s.scores.unigramScores[1], 0.001);
    assertEquals(s.scores.unigramScores[1], s.scores.unigramScores[2], 0.001);
  }

  @Test
  void explicitFloorAppliedWhenMinProbAbsent() throws IOException {
    // No MinLogProb in header — with an explicit logProbFloor of -12.0, that floor is used.
    Path modelFile = tempDir.resolve("no-minprob-model.txt");
    try (PrintWriter out =
        new PrintWriter(Files.newBufferedWriter(modelFile, StandardCharsets.UTF_8))) {
      out.println(
          "Languages: zh-hans,zh-hant,ja UnigramTotals: 100,100,100 BigramTotals: 100,100,100");
      out.printf("%c -20.0 -4.0 -4.0%n", ZH_HANS_A);
    }
    CJClassifier d = new CJClassifier(modelFile.toString(), -12.0);
    CJClassifier.Results s = new CJClassifier.Results();
    d.detect("" + ZH_HANS_A, s);
    // zh-hans column had -20.0, floored to -12.0
    int zhHansIdx = 0; // cjLangIndex: zh-hans=0
    assertEquals(
        -12.0,
        s.scores.unigramScores[zhHansIdx],
        0.001,
        "Value below logProbFloor should be floored");
    // zh-hant had -4.0, above -12.0, stays as -4.0
    int zhHantIdx = 1;
    assertEquals(-4.0, s.scores.unigramScores[zhHantIdx], 0.001);
  }

  @Test
  void noArgConstructorThrowsWhenMinProbAbsent() throws IOException {
    // No MinLogProb in header and no explicit floor => IllegalArgumentException.
    Path modelFile = tempDir.resolve("no-minprob-noarg-model.txt");
    try (PrintWriter out =
        new PrintWriter(Files.newBufferedWriter(modelFile, StandardCharsets.UTF_8))) {
      out.println(
          "Languages: zh-hans,zh-hant,ja UnigramTotals: 100,100,100 BigramTotals: 100,100,100");
      out.printf("%c -2.0 -4.0 -4.0%n", ZH_HANS_A);
    }
    assertThrows(IllegalArgumentException.class, () -> new CJClassifier(modelFile.toString()));
  }

  // ========================================================================
  // Convenience detect(CharSequence) overload
  // ========================================================================

  @Test
  void detectWithoutStatsWorks() {
    CJLanguage result = detector.detect("" + ZH_HANS_A + ZH_HANS_B + ZH_HANS_C);
    assertEquals(CJLanguage.CHINESE_SIMPLIFIED, result);
  }

  // ========================================================================
  // Column order independence
  // ========================================================================

  @Test
  void differentColumnOrderProducesSameResult() throws IOException {
    // Create a model with reversed column order: ja, zh-hant, zh-hans
    Path modelFile = tempDir.resolve("reversed-model.txt");
    try (PrintWriter out =
        new PrintWriter(Files.newBufferedWriter(modelFile, StandardCharsets.UTF_8))) {
      out.println(
          "Languages: ja,zh-hant,zh-hans UnigramTotals: 1000,1000,1000 BigramTotals: 1000,1000,1000"
              + " MinLogProb: -10.0");
      // Same data but columns in ja, zh-hant, zh-hans order
      out.printf("%c -8.0 -8.0 -2.0%n", ZH_HANS_A);
      out.printf("%c -8.0 -8.0 -2.0%n", ZH_HANS_B);
      out.printf("%c -8.0 -8.0 -2.0%n", ZH_HANS_C);
      out.printf("%c -2.0 -8.0 -8.0%n", JA_A);
      out.printf("%c -2.0 -8.0 -8.0%n", JA_B);
      out.printf("%c -2.0 -8.0 -8.0%n", JA_C);
    }
    CJClassifier d = new CJClassifier(modelFile.toString());
    assertEquals(CJLanguage.CHINESE_SIMPLIFIED, d.detect("" + ZH_HANS_A + ZH_HANS_B + ZH_HANS_C));
    assertEquals(CJLanguage.JAPANESE, d.detect("" + JA_A + JA_B + JA_C));
  }
}
