# CJClassifier

A focused, self-contained Java library for distinguishing between **Japanese**,
**Chinese Simplified**, and **Chinese Traditional** text.

## The problem

Chinese characters (Hanzi) and Japanese Kanji share the same Unicode CJ Unified
Ideograph code space. While Japanese also uses Hiragana and Katakana, short
texts — or all-Kanji Japanese phrases like 事務所 (office) — give no script-level
signal to work with. Generic language-detection libraries exist, but they spread
their models across dozens of languages and tend to underperform on this specific
three-way classification.

## How it works

CJClassifier uses a **unigram + bigram statistical model** trained on the Chinese
and Japanese Wikipedia corpora. For every character and character-pair in the CJ
range, the model stores per-language log-probabilities. At classification time
the library sums these log-probabilities across the input and picks the language
with the highest score.

Hiragana and Katakana are treated as strong Japanese signals: if the kana
fraction of the input exceeds a configurable threshold (default 1%), the text is
classified as Japanese immediately, without consulting the model.

## Features

- **Three-way CJK classification**: Chinese Simplified, Chinese Traditional, and
  Japanese — including all-Kanji Japanese text
- **No external runtime dependencies**: the core library is a single JAR with
  zero transitive dependencies (Java 11+)
- **Bundled model**: a pre-trained model (~7 MB compressed, ~36 MB in memory) is
  included in the JAR — no separate download required
- **Thread-safe**: the classifier is immutable; concurrent threads can classify
  simultaneously using their own `Results` instances
- **Fast**: classification is a single pass over the input with array lookups — no
  regex, no tokenization, no heap allocation in the hot path
- **Incremental API**: feed text in chunks via `addText()` and compute the result
  once at the end
- **Boosts**: optionally bias toward a language when you have a regional hint

## Quick start

### Maven

```xml
<dependency>
    <groupId>com.jlpka</groupId>
    <artifactId>cjclassifier</artifactId>
    <version>1.0</version>
</dependency>
```

### Usage

```java
import com.jlpka.cjclassifier.CJClassifier;
import com.jlpka.cjclassifier.CJLanguage;

// Load the bundled model (cached singleton — first call ~0.3 s, subsequent calls free)
CJClassifier cjc = CJClassifier.load();

cjc.detect("今天天气很好，我们去公园散步");   // → CHINESE_SIMPLIFIED
cjc.detect("今天天氣很好，我們去公園散步");   // → CHINESE_TRADITIONAL
cjc.detect("事務所");                         // → JAPANESE  (all Kanji)
cjc.detect("ひらがなとカタカナと");           // → JAPANESE  (all kana)
cjc.detect("hello");                          // → UNKNOWN
```

### Detailed results

```java
CJClassifier.Results results = new CJClassifier.Results();
cjc.detect("今天天气很好", results);

results.result;       // CJLanguage.CHINESE_SIMPLIFIED
results.gap;          // confidence gap: 0 = dead heat, 1 = no contest
results.totalScores;  // per-language log-probability totals
results.toShortString();  // e.g. "zh-hans:1.00,zh-hant:0.97,ja:0.85"
```

### Incremental classification

If text arrives in chunks, accumulate scores incrementally and compute the result
at the end:

```java
CJClassifier.Results results = new CJClassifier.Results();
cjc.addText(chunk1, results.scores);
cjc.addText(chunk2, results.scores);
cjc.computeResult(results);
// results.result is now available
```

### Boosts

When you have a hint about the source region (e.g. content is from a
Hong Kong website), you can slightly favor one language. Boosts range from 0
(no effect) to 1.0 (heavy). A value around 0.05 is reasonable for nudging
traditional vs simplified Chinese:

```java
CJClassifier.Results results = new CJClassifier.Results();
cjc.addText(text, results.scores);

int zhHantIdx = CJClassifier.cjLangIndex(CJLanguage.CHINESE_TRADITIONAL);
results.boosts[zhHantIdx] = 0.05;

cjc.computeResult(results);
```

Note: `detect()` calls `clear()` internally, so boosts must be set when using
the lower-level `addText()` / `computeResult()` API.

## Project structure

```
cjclassifier/
├── core/           cjclassifier – the library JAR (no external dependencies)
└── tools/          cjclassifier-tools – offline model-building utilities
```

The **core** module is all you need as a consumer. It contains the classifier and
a bundled pre-trained model.

The **tools** module contains the utilities (`ModelBuilder`, `EvalTool`,
`ContentUtils`) that were used to generate the statistical model from Wikipedia
dumps. You do not need this module to use the library. The tools module has
additional dependencies (commons-compress, opencc4j) and produces a shaded
uber-JAR for CLI usage.

## Building from source

```bash
mvn clean package
```

This produces:

- `core/target/cjclassifier-1.0.jar` — the library (≈ 8 MB, includes bundled model)
- `tools/target/cjclassifier-tools-1.0.jar` — uber-JAR for offline tooling

To run tests:

```bash
mvn test
```

Requires Java 11+.

## Contributing

Contributions are welcome! Please open an issue or pull request at
[github.com/jlpka/cjclassifier](https://github.com/jlpka/cjclassifier).

Before submitting a PR, make sure all tests pass:

```bash
mvn test
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

The bundled model contains statistical parameters derived from Wikipedia text.
The model does not contain or reproduce Wikipedia text. See [NOTICE](NOTICE).
