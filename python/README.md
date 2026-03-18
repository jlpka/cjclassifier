# CJClassifier

A focused, pure-Python library for distinguishing between **Japanese**,
**Chinese Simplified**, and **Chinese Traditional** text using a statistical
model of ideograph frequencies built from Japanese and Chinese language Wikipedia
corpora.

No external dependencies. The bundled model is ~7 MB on disk and ~29 MB in
memory (loaded once and cached).

## Install

```bash
pip install cjclassifier
```

## Usage

```python
from cjclassifier import CJClassifier, CJLanguage

cjc = CJClassifier.load()

cjc.detect("今天天气很好，我们去公园散步")   # => CJLanguage.CHINESE_SIMPLIFIED
cjc.detect("今天天氣很好，我們去公園散步")   # => CJLanguage.CHINESE_TRADITIONAL
cjc.detect("事務所")                         # => CJLanguage.JAPANESE  (all Kanji)
cjc.detect("ひらがなとカタカナと")           # => CJLanguage.JAPANESE  (all kana)
cjc.detect("hello")                          # => CJLanguage.UNKNOWN
```

## Detailed results

```python
from cjclassifier.classifier import Results

results = Results()
cjc.detect("今天天气很好", results)

results.result             # CJLanguage.CHINESE_SIMPLIFIED
results.gap                # confidence gap: 0 = dead heat, 1 = no contest
results.total_scores       # per-language log-probability totals
results.to_short_string()  # e.g. "zh-hans:1.00,zh-hant:0.97,ja:0.85"
```

## How it works

CJClassifier uses a **unigram + bigram statistical model** trained on the
Chinese and Japanese Wikipedia corpora. For every character and character-pair
in the CJ range, the model stores per-language log-probabilities. At
classification time the library sums these log-probabilities across the input
and picks the language with the highest score.

A Java implementation and the model-building tools are also available in the
same repository: [github.com/jlpka/cjclassifier](https://github.com/jlpka/cjclassifier)

## License

Apache License 2.0
