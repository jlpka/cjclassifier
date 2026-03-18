# Copyright 2026 Jeremy Lilley (jeremy@jlilley.net)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for CJClassifier using synthetic model files.

Port of CJClassifierBaseTest.java.
"""

import os
import tempfile
import unittest

from cjclassifier import CJClassifier, CJLanguage
from cjclassifier.classifier import Results

# Pick chars in CJK Unified Ideographs block (0x4E00+), well within CJ_RANGE
# Simplified-leaning chars
ZH_HANS_A = "\u4e00"  # 一
ZH_HANS_B = "\u4e8c"  # 二
ZH_HANS_C = "\u4e09"  # 三

# Traditional-leaning chars
ZH_HANT_A = "\u5b78"  # 學
ZH_HANT_B = "\u6a23"  # 樣
ZH_HANT_C = "\u9a57"  # 驗

# Japanese-leaning chars
JA_A = "\u8fbc"  # 込
JA_B = "\u7573"  # 畳
JA_C = "\u586b"  # 填


def _create_test_model(path):
    """Create a synthetic model file for testing."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "Languages: zh-hans,zh-hant,ja "
            "UnigramTotals: 1000,1000,1000 "
            "BigramTotals: 1000,1000,1000 "
            "MinLogProb: -10.0\n"
        )
        # Simplified-leaning unigrams
        for c in [ZH_HANS_A, ZH_HANS_B, ZH_HANS_C]:
            f.write(f"{c} -2.0 -8.0 -8.0\n")
        # Traditional-leaning unigrams
        for c in [ZH_HANT_A, ZH_HANT_B, ZH_HANT_C]:
            f.write(f"{c} -8.0 -2.0 -8.0\n")
        # Japanese-leaning unigrams
        for c in [JA_A, JA_B, JA_C]:
            f.write(f"{c} -8.0 -8.0 -2.0\n")
        # Bigrams
        f.write(f"{ZH_HANS_A}{ZH_HANS_B} -2.0 -8.0 -8.0\n")
        f.write(f"{ZH_HANS_B}{ZH_HANS_C} -2.0 -8.0 -8.0\n")
        f.write(f"{ZH_HANT_A}{ZH_HANT_B} -8.0 -2.0 -8.0\n")
        f.write(f"{ZH_HANT_B}{ZH_HANT_C} -8.0 -2.0 -8.0\n")
        f.write(f"{JA_A}{JA_B} -8.0 -8.0 -2.0\n")
        f.write(f"{JA_B}{JA_C} -8.0 -8.0 -2.0\n")


class TestCJClassifierBase(unittest.TestCase):

    _detector = None
    _tmpdir = None

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp()
        model_path = os.path.join(cls._tmpdir, "test-model.txt")
        _create_test_model(model_path)
        CJClassifier.clear_cached_models()
        cls._detector = CJClassifier.load(path=model_path)

    def setUp(self):
        self.results = Results()

    # ========================================================================
    # Basic detection
    # ========================================================================

    def test_detect_simplified_chinese(self):
        text = ZH_HANS_A + ZH_HANS_B + ZH_HANS_C
        result = self._detector.detect(text, self.results)
        self.assertEqual(CJLanguage.CHINESE_SIMPLIFIED, result)
        self.assertEqual(0, self.results.scores.kana_count)

    def test_detect_traditional_chinese(self):
        text = ZH_HANT_A + ZH_HANT_B + ZH_HANT_C
        result = self._detector.detect(text, self.results)
        self.assertEqual(CJLanguage.CHINESE_TRADITIONAL, result)
        self.assertEqual(0, self.results.scores.kana_count)

    def test_detect_japanese_by_ideographs(self):
        text = JA_A + JA_B + JA_C
        result = self._detector.detect(text, self.results)
        self.assertEqual(CJLanguage.JAPANESE, result)
        self.assertEqual(0, self.results.scores.kana_count)

    # ========================================================================
    # Kana shortcut
    # ========================================================================

    def test_detect_japanese_by_kana(self):
        result = self._detector.detect("ひらがな", self.results)
        self.assertEqual(CJLanguage.JAPANESE, result)
        self.assertGreater(self.results.scores.kana_count, 0)

        result = self._detector.detect("カタカナ", self.results)
        self.assertEqual(CJLanguage.JAPANESE, result)
        self.assertGreater(self.results.scores.kana_count, 0)

    def test_kana_overrides_ideograph_signal(self):
        text = ZH_HANS_A + ZH_HANS_B + "\u3053"
        result = self._detector.detect(text, self.results)
        self.assertEqual(CJLanguage.JAPANESE, result)
        self.assertGreater(self.results.scores.kana_count, 0)

    def test_small_kana_fraction_still_chinese(self):
        sb = ZH_HANS_A * 101 + "\u3053"  # one hiragana
        result = self._detector.detect(sb, self.results)
        self.assertEqual(CJLanguage.CHINESE_SIMPLIFIED, result)
        self.assertEqual(1, self.results.scores.kana_count)
        self.assertEqual(101, self.results.scores.cj_char_count)

    def test_large_kana_fraction_is_japanese(self):
        text = ZH_HANS_A + ZH_HANS_B + "\u3053"
        result = self._detector.detect(text, self.results)
        self.assertEqual(CJLanguage.JAPANESE, result)
        self.assertGreater(self.results.scores.kana_count, 0)

    # ========================================================================
    # Edge cases
    # ========================================================================

    def test_empty_text_returns_unknown(self):
        result = self._detector.detect("", self.results)
        self.assertEqual(CJLanguage.UNKNOWN, result)

    def test_non_cj_text_returns_unknown(self):
        result = self._detector.detect("Hello world 123", self.results)
        self.assertEqual(CJLanguage.UNKNOWN, result)

    def test_single_char_detection(self):
        result = self._detector.detect(JA_A, self.results)
        self.assertEqual(CJLanguage.JAPANESE, result)

    # ========================================================================
    # Stats
    # ========================================================================

    def test_stats_populated_correctly(self):
        text = ZH_HANS_A + ZH_HANS_B + ZH_HANS_C
        self._detector.detect(text, self.results)
        self.assertNotEqual(0, self.results.scores.unigram_scores[0])
        self.assertNotEqual(0, self.results.total_scores[0])

    def test_clear_resets_all_fields(self):
        self._detector.detect(ZH_HANS_A + ZH_HANS_B, self.results)
        self.assertIsNotNone(self.results.result)
        self.results.clear()
        self.assertIsNone(self.results.result)
        self.assertEqual(0, self.results.scores.kana_count)
        for i in range(3):
            self.assertEqual(0.0, self.results.scores.unigram_scores[i])
            self.assertEqual(0.0, self.results.scores.bigram_scores[i])
            self.assertEqual(0.0, self.results.total_scores[i])
            self.assertEqual(0, self.results.scores.unigram_hits_per_lang[i])
            self.assertEqual(0, self.results.scores.bigram_hits_per_lang[i])

    def test_clear_then_reuse_gives_clean_result(self):
        self._detector.detect(ZH_HANS_A + ZH_HANS_B + ZH_HANS_C, self.results)
        self.assertEqual(CJLanguage.CHINESE_SIMPLIFIED, self.results.result)
        self.results.clear()
        self._detector.detect(JA_A + JA_B + JA_C, self.results)
        self.assertEqual(CJLanguage.JAPANESE, self.results.result)
        self.assertEqual(0, self.results.scores.kana_count)

    # ========================================================================
    # to_short_string
    # ========================================================================

    def test_to_short_string_format(self):
        self._detector.detect(ZH_HANS_A + ZH_HANS_B + ZH_HANS_C, self.results)
        s = self.results.to_short_string()
        self.assertTrue(len(s) > 0)
        self.assertIn("zh-hans", s)
        self.assertIn("zh-hant", s)
        self.assertIn("ja", s)
        self.assertEqual(2, s.count(","))

    def test_to_short_string_kana(self):
        self._detector.detect("\u3053\u3093", self.results)
        self.assertEqual("ja:1.0,zh-hans:0,zh-hant:0", self.results.to_short_string())

    def test_to_short_string_unknown_is_empty(self):
        self._detector.detect("Hello", self.results)
        self.assertEqual("", self.results.to_short_string())

    # ========================================================================
    # MinProb parsing
    # ========================================================================

    def test_min_prob_parsed_from_header(self):
        path = os.path.join(self._tmpdir, "minprob-model.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "Languages: zh-hans,zh-hant,ja "
                "UnigramTotals: 100,100,100 "
                "BigramTotals: 100,100,100 "
                "MinLogProb: -5.0\n"
            )
            f.write(f"{ZH_HANS_A} -2.0 -4.0 -4.0\n")

        CJClassifier.clear_cached_models()
        d = CJClassifier.load(path=path)
        r = Results()
        d.detect("\u4e10", r)
        # All unigram scores should be equal (all got defaultLogProb = -5.0)
        self.assertAlmostEqual(r.scores.unigram_scores[0], r.scores.unigram_scores[1], places=3)
        self.assertAlmostEqual(r.scores.unigram_scores[1], r.scores.unigram_scores[2], places=3)

    def test_explicit_floor_applied_when_min_prob_absent(self):
        path = os.path.join(self._tmpdir, "no-minprob-model.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "Languages: zh-hans,zh-hant,ja "
                "UnigramTotals: 100,100,100 "
                "BigramTotals: 100,100,100\n"
            )
            f.write(f"{ZH_HANS_A} -20.0 -4.0 -4.0\n")

        CJClassifier.clear_cached_models()
        d = CJClassifier.load(path=path, log_prob_floor=-12.0)
        r = Results()
        d.detect(ZH_HANS_A, r)
        self.assertAlmostEqual(-12.0, r.scores.unigram_scores[0], places=3)
        self.assertAlmostEqual(-4.0, r.scores.unigram_scores[1], places=3)

    def test_no_arg_raises_when_min_prob_absent(self):
        path = os.path.join(self._tmpdir, "no-minprob-noarg-model.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "Languages: zh-hans,zh-hant,ja "
                "UnigramTotals: 100,100,100 "
                "BigramTotals: 100,100,100\n"
            )
            f.write(f"{ZH_HANS_A} -2.0 -4.0 -4.0\n")

        CJClassifier.clear_cached_models()
        with self.assertRaises(ValueError):
            CJClassifier.load(path=path)

    # ========================================================================
    # Convenience detect() overload
    # ========================================================================

    def test_detect_without_results_works(self):
        result = self._detector.detect(ZH_HANS_A + ZH_HANS_B + ZH_HANS_C)
        self.assertEqual(CJLanguage.CHINESE_SIMPLIFIED, result)

    # ========================================================================
    # Column order independence
    # ========================================================================

    def test_different_column_order_produces_same_result(self):
        path = os.path.join(self._tmpdir, "reversed-model.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "Languages: ja,zh-hant,zh-hans "
                "UnigramTotals: 1000,1000,1000 "
                "BigramTotals: 1000,1000,1000 "
                "MinLogProb: -10.0\n"
            )
            for c in [ZH_HANS_A, ZH_HANS_B, ZH_HANS_C]:
                f.write(f"{c} -8.0 -8.0 -2.0\n")
            for c in [JA_A, JA_B, JA_C]:
                f.write(f"{c} -2.0 -8.0 -8.0\n")

        CJClassifier.clear_cached_models()
        d = CJClassifier.load(path=path)
        self.assertEqual(
            CJLanguage.CHINESE_SIMPLIFIED,
            d.detect(ZH_HANS_A + ZH_HANS_B + ZH_HANS_C),
        )
        self.assertEqual(
            CJLanguage.JAPANESE,
            d.detect(JA_A + JA_B + JA_C),
        )


if __name__ == "__main__":
    unittest.main()
