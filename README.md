# CJClassifier

A focused, self-contained library for distinguishing between **Japanese**,
**Chinese Simplified**, and **Chinese Traditional** text. The primary
implementation and model-building tools are in **Java**, with **Python** and
**Rust** implementations also included. The Rust library also provides a
**C/C++ FFI** interface.

## The problem

Chinese characters and Japanese Kanji share the same Unicode CJ Unified
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

The focus here is a model that distinguishes between Japanese Kanji and Traditional/Simplified
Chinese ideographs, particularly in shorter contexts like headings/titles.

That said, it we see any Japanese Hiragana or Katakana characters, we will treat them as
strong Japanese signals; after all, it would be reasonable to treat such text as automatically
Japanese. The one caveat is that we've seen Chinese text with a short embedded Japanese quote -
and for that reason, we will only treat Hiragana/Katakana as definitively Japanese if it's at least 1% of
the text (in typical long-form Japanese text, Hiragana is closer to 50% of the characters).

## Features

- **Three-way CJK classification**: Chinese Simplified, Chinese Traditional, and
  Japanese — including all-Kanji Japanese text
- **No external runtime dependencies**: the Java library is a single JAR (Java 11+);
  the Python package is pure Python (3.8+) with no third-party dependencies;
  the Rust crate depends only on `flate2` for gzip decompression
- **Bundled model**: a pre-trained model (~7 MB compressed) is included — no
  separate download required
- **Fast**: classification is a single pass over the input with array lookups — no
  regex, no tokenization
- **Incremental API**: feed text in chunks and compute the result once at the end
- **Boosts**: optionally bias toward a language when you have a regional hint

## Quick start — Java

### Maven

```xml
<dependency>
    <groupId>com.jlpka</groupId>
    <artifactId>cjclassifier</artifactId>
    <version>1.0.2</version>
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

```java
CJClassifier.Results results = new CJClassifier.Results();
cjc.addText(chunk1, results.scores);
cjc.addText(chunk2, results.scores);
cjc.computeResult(results);
// results.result is now available
```

### Boosts

```java
CJClassifier.Results results = new CJClassifier.Results();
cjc.addText(text, results.scores);

int zhHantIdx = CJClassifier.cjLangIndex(CJLanguage.CHINESE_TRADITIONAL);
results.boosts[zhHantIdx] = 0.05;

cjc.computeResult(results);
```

Note: `detect()` calls `clear()` internally, so boosts must be set when using
the lower-level `addText()` / `computeResult()` API.

## Quick start — Python

### Install

```bash
pip install cjclassifier
```

### Usage

```python
from cjclassifier import CJClassifier, CJLanguage

cjc = CJClassifier.load()

cjc.detect("今天天气很好，我们去公园散步")   # => CJLanguage.CHINESE_SIMPLIFIED
cjc.detect("今天天氣很好，我們去公園散步")   # => CJLanguage.CHINESE_TRADITIONAL
cjc.detect("事務所")                         # => CJLanguage.JAPANESE  (all Kanji)
cjc.detect("ひらがなとカタカナと")           # => CJLanguage.JAPANESE  (all kana)
cjc.detect("hello")                          # => CJLanguage.UNKNOWN
```

### Detailed results

```python
from cjclassifier.classifier import Results

results = Results()
cjc.detect("今天天气很好", results)

results.result             # CJLanguage.CHINESE_SIMPLIFIED
results.gap                # confidence gap: 0 = dead heat, 1 = no contest
results.total_scores       # per-language log-probability totals
results.to_short_string()  # e.g. "zh-hans:1.00,zh-hant:0.97,ja:0.85"
```

## Quick start — Rust / C / C++

A Rust crate is available with the same functionality, and also builds as a
C-compatible shared or static library for use from C/C++.

See [`rust/cjclassifier/README.md`](rust/cjclassifier/README.md) for Rust API
usage, detailed results, and C/C++ FFI documentation.

## Project structure

```
cjclassifier/
├── core/                  Java library JAR (no external dependencies)
├── python/                Python package (pure Python, no external dependencies)
├── rust/cjclassifier/     Rust crate (also builds as C/C++ shared/static library)
├── rust/eval/             Rust evaluation and benchmarking tools
└── tools/                 Java model-building utilities
```

The **core** module (Java), the **python** directory, and/or the **rust/cjclassifier**
crate are all you need as a consumer. Each contains the classifier and a bundled
pre-trained model. The Rust crate also builds as a C-compatible shared or static
library — see `rust/eval/src/usingffi.cpp` for a working C++ example.

The **tools** module contains Java utilities (`ModelBuilder`, `EvalTool`,
`ContentUtils`) that were used to generate the statistical model from Wikipedia
dumps. You do not need this module to use the library. The tools module has
additional dependencies (commons-compress, opencc4j) and produces a shaded
uber-JAR for CLI usage.

## Building from source

### Java

```bash
mvn clean package
```

This produces:

- `core/target/cjclassifier-1.0.2.jar` — the library (≈ 8 MB, includes bundled model)
- `tools/target/cjclassifier-tools-1.0.2.jar` — uber-JAR for offline tooling

Requires Java 11+.

### Python

```bash
cd python
make install-dev   # copies model from core/ and installs in editable mode
make test          # run tests
```

Requires Python 3.8+.

### Rust

```bash
cd rust/cjclassifier
cargo build --release
cargo test
```

See `rust/cjclassifier/README.md` for API documentation and FFI usage.

## Contributing

Contributions are welcome! Please open an issue or pull request at
[github.com/jlpka/cjclassifier](https://github.com/jlpka/cjclassifier).

Before submitting a PR, make sure all tests pass:

```bash
mvn test                              # Java
cd python && make test                # Python
cd rust/cjclassifier && cargo test    # Rust
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

The bundled model contains statistical parameters derived from Wikipedia text.
The model does not contain or reproduce Wikipedia text. See [NOTICE](NOTICE).
