package com.jlpka.cjclassifier;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** These tests cases run CJClassifier using real model files. */
class CJClassifierReadModelTest {
  private static CJClassifier cjc;

  @BeforeAll
  static void loadModel() throws IOException {
    cjc = CJClassifier.load();
  }

  @Test
  void detectCJ() {
    assertEquals(CJLanguage.CHINESE_SIMPLIFIED, cjc.detect("今天天气很好，我们去公园散步"));
    assertEquals(CJLanguage.CHINESE_TRADITIONAL, cjc.detect("今天天氣很好，我們去公園散步"));

    // Kanji only
    assertEquals(CJLanguage.JAPANESE, cjc.detect("事務所"));
    // All Kana.
    assertEquals(CJLanguage.JAPANESE, cjc.detect("ひらがなとカタカナと"));
    // Mixed Kanji/kana
    assertEquals(CJLanguage.JAPANESE, cjc.detect("日本語は日本で使われている言語です。ひらがなとカタカナと漢字を使います"));
    assertEquals(CJLanguage.UNKNOWN, cjc.detect("hello"));
  }

  @Test
  void chineseAliases() {
    // CHINESE_SIMPLIFIED: primary "zh-hans", iso3 "zho-hans",
    // aliases "chinese,zh,zh-cn,zh-hans-cn,zh-hans-sg"
    for (String alias :
        List.of(
            "zh-hans",
            "zho-hans",
            "chinese",
            "zh",
            "zh-cn",
            "zh-hans-cn",
            "zh-hans-sg",
            "chinese_simplified")) {
      assertEquals(
          CJLanguage.CHINESE_SIMPLIFIED,
          CJLanguage.fromString(alias),
          "Expected CHINESE_SIMPLIFIED for \"" + alias + "\"");
    }
    // CHINESE_TRADITIONAL: primary "zh-hant", iso3 "zho-hant",
    // aliases "zh-hant-hk,zh-hk,zh-hant-tw"
    for (String alias :
        List.of(
            "zh-hant", "zho-hant", "zh-hant-hk", "zh-hk", "zh-hant-tw", "chinese_traditional")) {
      assertEquals(
          CJLanguage.CHINESE_TRADITIONAL,
          CJLanguage.fromString(alias),
          "Expected CHINESE_TRADITIONAL for \"" + alias + "\"");
    }
  }

  // ========================================================================
  // CJ totalScores populated
  // ========================================================================

  @Test
  void cjTotalScoresPopulated() {
    // CJ classifier path: simplified Chinese text
    CJClassifier.Results r = new CJClassifier.Results();
    cjc.detect("今天天气很好，我们去公园散步", r);
    assertEquals(CJLanguage.CHINESE_SIMPLIFIED, r.result);
    assertTrue(r.gap > 0, "Expected positive gap");
    // totalScores should have non-zero entries for CJ languages
    assertTrue(
        r.totalScores[CJClassifier.cjLangIndex(CJLanguage.CHINESE_SIMPLIFIED)] != 0,
        "Expected non-zero totalScore for zh-hans");
    // The winning language should have the highest (least negative) score
    assertTrue(
        r.totalScores[CJClassifier.cjLangIndex(CJLanguage.CHINESE_SIMPLIFIED)]
            >= r.totalScores[CJClassifier.cjLangIndex(CJLanguage.CHINESE_TRADITIONAL)]);
    assertTrue(
        r.totalScores[CJClassifier.cjLangIndex(CJLanguage.CHINESE_SIMPLIFIED)]
            >= r.totalScores[CJClassifier.cjLangIndex(CJLanguage.JAPANESE)]);
  }

  @Test
  void kanaTotalScoresPopulated() {
    // Kana early return: all kana text
    CJClassifier.Results r = new CJClassifier.Results();
    cjc.detect("ひらがなとカタカナと", r);
    assertEquals(CJLanguage.JAPANESE, r.result);
    assertEquals(1.0, r.gap, 1e-9);
    // totalScores should be 1.0 for Japanese, 0 for everything else
    assertEquals(1.0, r.totalScores[CJClassifier.cjLangIndex(CJLanguage.JAPANESE)], 1e-9);
    for (int i = 0; i < r.totalScores.length; i++) {
      if (i != CJClassifier.cjLangIndex(CJLanguage.JAPANESE)) {
        assertEquals(0.0, r.totalScores[i], "Expected 0 for non-Japanese lang idx " + i);
      }
    }
  }

  // ========================================================================
  // Boosts
  // ========================================================================

  @Test
  void boostFlipsAmbiguousChineseResult() {
    // "中国人民共和" — characters shared across simplified/traditional; model picks zh-hans.
    // Use the lower-level API because detect() calls clear() which zeros boosts.
    String text = "中国人民共和";

    // Baseline: without boosts, should be zh-hans
    CJClassifier.Results baseline = new CJClassifier.Results();
    cjc.addText(text, baseline.scores);
    cjc.computeResult(baseline);
    assertEquals(CJLanguage.CHINESE_SIMPLIFIED, baseline.result);

    // Now boost zh-hant enough to flip the result
    int zhHantIdx = CJClassifier.cjLangIndex(CJLanguage.CHINESE_TRADITIONAL);
    CJClassifier.Results boosted = new CJClassifier.Results();
    cjc.addText(text, boosted.scores);
    boosted.boosts[zhHantIdx] = 0.5; // large positive boost favors zh-hant
    cjc.computeResult(boosted);
    assertEquals(CJLanguage.CHINESE_TRADITIONAL, boosted.result);

    // The boosted zh-hant total should be higher (less negative) than baseline
    assertTrue(
        boosted.totalScores[zhHantIdx] > baseline.totalScores[zhHantIdx],
        "Boost should improve zh-hant totalScore");
  }

  @Test
  void mixedKanaHanTotalScoresPopulated() {
    // Mixed kanji + kana: kana presence should still give Japanese with totalScores set
    CJClassifier.Results r = new CJClassifier.Results();
    cjc.detect("日本語は日本で使われている言語です。ひらがなとカタカナと漢字を使います", r);
    assertEquals(CJLanguage.JAPANESE, r.result);
    assertEquals(1.0, r.gap, 1e-9);
    assertEquals(0.0, r.totalScores[CJClassifier.cjLangIndex(CJLanguage.CHINESE_SIMPLIFIED)], 1e-9);
    assertEquals(
        0.0, r.totalScores[CJClassifier.cjLangIndex(CJLanguage.CHINESE_TRADITIONAL)], 1e-9);
    assertEquals(1.0, r.totalScores[CJClassifier.cjLangIndex(CJLanguage.JAPANESE)], 1e-9);
  }
}
