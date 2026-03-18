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

"""Tests for CJClassifier using the real bundled model.

Port of CJClassifierReadModelTest.java.
"""

import unittest

from cjclassifier import CJClassifier, CJLanguage
from cjclassifier.classifier import Results


class TestCJClassifierRealModel(unittest.TestCase):

    _cjc = None

    @classmethod
    def setUpClass(cls):
        CJClassifier.clear_cached_models()
        cls._cjc = CJClassifier.load()

    def test_detect_cj(self):
        self.assertEqual(CJLanguage.CHINESE_SIMPLIFIED,
                         self._cjc.detect("今天天气很好，我们去公园散步"))
        self.assertEqual(CJLanguage.CHINESE_TRADITIONAL,
                         self._cjc.detect("今天天氣很好，我們去公園散步"))
        # Kanji only
        self.assertEqual(CJLanguage.JAPANESE, self._cjc.detect("事務所"))
        # All kana
        self.assertEqual(CJLanguage.JAPANESE, self._cjc.detect("ひらがなとカタカナと"))
        # Mixed kanji/kana
        self.assertEqual(
            CJLanguage.JAPANESE,
            self._cjc.detect("日本語は日本で使われている言語です。ひらがなとカタカナと漢字を使います"),
        )
        self.assertEqual(CJLanguage.UNKNOWN, self._cjc.detect("hello"))

    def test_chinese_aliases(self):
        for alias in [
            "zh-hans", "zho-hans", "chinese", "zh", "zh-cn",
            "zh-hans-cn", "zh-hans-sg", "chinese_simplified",
        ]:
            self.assertEqual(
                CJLanguage.CHINESE_SIMPLIFIED,
                CJLanguage.from_string(alias),
                f'Expected CHINESE_SIMPLIFIED for "{alias}"',
            )
        for alias in [
            "zh-hant", "zho-hant", "zh-hant-hk", "zh-hk",
            "zh-hant-tw", "chinese_traditional",
        ]:
            self.assertEqual(
                CJLanguage.CHINESE_TRADITIONAL,
                CJLanguage.from_string(alias),
                f'Expected CHINESE_TRADITIONAL for "{alias}"',
            )

    def test_cj_total_scores_populated(self):
        r = Results()
        self._cjc.detect("今天天气很好，我们去公园散步", r)
        self.assertEqual(CJLanguage.CHINESE_SIMPLIFIED, r.result)
        self.assertGreater(r.gap, 0)
        self.assertNotEqual(0, r.total_scores[CJLanguage.CHINESE_SIMPLIFIED])
        self.assertGreaterEqual(
            r.total_scores[CJLanguage.CHINESE_SIMPLIFIED],
            r.total_scores[CJLanguage.CHINESE_TRADITIONAL],
        )
        self.assertGreaterEqual(
            r.total_scores[CJLanguage.CHINESE_SIMPLIFIED],
            r.total_scores[CJLanguage.JAPANESE],
        )

    def test_kana_total_scores_populated(self):
        r = Results()
        self._cjc.detect("ひらがなとカタカナと", r)
        self.assertEqual(CJLanguage.JAPANESE, r.result)
        self.assertAlmostEqual(1.0, r.gap, places=9)
        self.assertAlmostEqual(1.0, r.total_scores[CJLanguage.JAPANESE], places=9)
        for i in range(len(r.total_scores)):
            if i != CJLanguage.JAPANESE:
                self.assertAlmostEqual(0.0, r.total_scores[i])

    def test_boost_flips_ambiguous_chinese_result(self):
        text = "中国人民共和"

        # Baseline
        baseline = Results()
        self._cjc.add_text(text, baseline.scores)
        self._cjc.compute_result(baseline)
        self.assertEqual(CJLanguage.CHINESE_SIMPLIFIED, baseline.result)

        # Boosted
        boosted = Results()
        self._cjc.add_text(text, boosted.scores)
        boosted.boosts[CJLanguage.CHINESE_TRADITIONAL] = 0.5
        self._cjc.compute_result(boosted)
        self.assertEqual(CJLanguage.CHINESE_TRADITIONAL, boosted.result)
        self.assertGreater(
            boosted.total_scores[CJLanguage.CHINESE_TRADITIONAL],
            baseline.total_scores[CJLanguage.CHINESE_TRADITIONAL],
        )

    def test_mixed_kana_han_total_scores_populated(self):
        r = Results()
        self._cjc.detect("日本語は日本で使われている言語です。ひらがなとカタカナと漢字を使います", r)
        self.assertEqual(CJLanguage.JAPANESE, r.result)
        self.assertAlmostEqual(1.0, r.gap, places=9)
        self.assertAlmostEqual(0.0, r.total_scores[CJLanguage.CHINESE_SIMPLIFIED], places=9)
        self.assertAlmostEqual(0.0, r.total_scores[CJLanguage.CHINESE_TRADITIONAL], places=9)
        self.assertAlmostEqual(1.0, r.total_scores[CJLanguage.JAPANESE], places=9)


if __name__ == "__main__":
    unittest.main()
