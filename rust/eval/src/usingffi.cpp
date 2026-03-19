// Copyright 2026 Jeremy Lilley (jeremy@jlilley.net)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Example of using the cjclassifier Rust library from C++ via FFI.
//
// Build (from rust/eval/):
//   cargo build --release --manifest-path ../cjclassifier/Cargo.toml
//   c++ -std=c++11 -O2 -o usingffi src/usingffi.cpp \
//       -I ../cjclassifier/src \
//       -L ../cjclassifier/target/release \
//       -lcjclassifier
//
// Run (macOS):
//   DYLD_LIBRARY_PATH=../cjclassifier/target/release ./usingffi "今天天气很好"
//
// Run (Linux):
//   LD_LIBRARY_PATH=../cjclassifier/target/release ./usingffi "今天天气很好"

#include "cjclassifier.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *lang_name(CJLanguageCode code) {
    switch (code) {
        case CJ_CHINESE_SIMPLIFIED:  return "zh-hans (ChineseSimplified)";
        case CJ_CHINESE_TRADITIONAL: return "zh-hant (ChineseTraditional)";
        case CJ_JAPANESE:            return "ja (Japanese)";
        default:                     return "UNKNOWN";
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <phrase>\n", argv[0]);
        return 1;
    }

    CJClassifierHandle *model = cj_load_bundled();
    if (!model) {
        fprintf(stderr, "Failed to load bundled model\n");
        return 1;
    }

    const char *phrase = argv[1];
    size_t len = strlen(phrase);

    /* Simple detection */
    CJLanguageCode lang = cj_detect(model, phrase, len);
    printf("Result: %s\n", lang_name(lang));

    /* Detailed detection */
    CJResult result = {0};
    cj_detect_detailed(model, phrase, len, &result);
    printf("Gap: %.4f\n", result.gap);
    printf("Scores: zh-hans=%.2f  zh-hant=%.2f  ja=%.2f\n",
           result.total_scores[0],
           result.total_scores[1],
           result.total_scores[2]);

    cj_free(model);
    return 0;
}
