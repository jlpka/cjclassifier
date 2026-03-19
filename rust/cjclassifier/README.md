# CJClassifier

A focused Rust library for distinguishing between **Japanese**,
**Chinese Simplified**, and **Chinese Traditional** text using a statistical
model of ideograph frequencies built from Japanese and Chinese language Wikipedia
corpora.

No external runtime dependencies beyond `flate2` (for gzip decompression of the
bundled model). The model is ~7 MB on disk and ~36 MB in memory (loaded once).

## Install

```toml
[dependencies]
cjclassifier = "0.1"
```

## Usage

```rust
use cjclassifier::{CJClassifier, CJLanguage};

let classifier = CJClassifier::load_bundled().unwrap();

assert_eq!(classifier.detect("今天天气很好，我们去公园散步"), CJLanguage::ChineseSimplified);
assert_eq!(classifier.detect("今天天氣很好，我們去公園散步"), CJLanguage::ChineseTraditional);
assert_eq!(classifier.detect("事務所"), CJLanguage::Japanese);       // all Kanji
assert_eq!(classifier.detect("ひらがなとカタカナと"), CJLanguage::Japanese); // all kana
assert_eq!(classifier.detect("hello"), CJLanguage::Unknown);
```

## Detailed results

```rust
use cjclassifier::{CJClassifier, CJLanguage, Results};

let classifier = CJClassifier::load_bundled().unwrap();
let mut results = Results::new();
classifier.detect_with_results("今天天气很好", &mut results);

assert_eq!(results.result, Some(CJLanguage::ChineseSimplified));
// results.gap         — confidence: 0 = dead heat, 1 = no contest
// results.total_scores — per-language log-probability totals
// results.to_short_string() — e.g. "zh-hans:1.00,zh-hant:0.97,ja:0.85"
```

## C/C++ FFI

The crate builds as a C-compatible shared or static library (`cdylib` / `staticlib`).
See `rust/eval/src/usingffi.cpp` for a complete working example with build instructions.

```c
#include <string.h>

// Opaque handle
typedef struct CJClassifierHandle CJClassifierHandle;

CJClassifierHandle *model = cj_load_bundled();
if (model) {
    const char *text = "今天天气很好";
    int lang = cj_detect(model, text, strlen(text));
    // lang: 0=Unknown, 1=ChineseSimplified, 2=ChineseTraditional, 3=Japanese
    cj_free(model);
}
```

## How it works

CJClassifier uses a **unigram + bigram statistical model** trained on the
Chinese and Japanese Wikipedia corpora (licensed under CC BY-SA 3.0). For every
character and character-pair in the CJK ideograph range, the model stores
per-language log-probabilities. At classification time the library sums these
log-probabilities across the input and picks the language with the highest score.

## Evaluation tools

The `rust/eval/` directory in the Git repository contains tools for evaluating
classifier accuracy and performance:

- `adhoc` — detect language of a single phrase with detailed scoring
- `phrase_eval` — evaluate accuracy against a file of test phrases
- `load_speed` — benchmark model loading speed

Run with e.g. `cargo run --bin adhoc -- --phrase "今天天气很好"`.

Java, Python, and Rust implementations and the model-building tools are all
available in the same repository:
[github.com/jlpka/cjclassifier](https://github.com/jlpka/cjclassifier)

## License

Apache License 2.0
