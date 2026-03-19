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

//! Phrase-level evaluation tool for CJ language detection.
//!
//! Evaluates classifier accuracy against a file of test phrases with an expected
//! language. Treats simplified and traditional Chinese as equivalent since they're
//! mixed in corpora.
//!
//! # Examples
//!
//! ```sh
//! cargo run --bin phrase_eval -- --infile phrases_zh.txt --language zh
//! cargo run --bin phrase_eval -- --infile phrases_ja.txt --language ja --misses
//! cargo run --bin phrase_eval -- --modelfile model.gz --infile phrases.txt --language zh-hant
//! ```

use cjclassifier::{CJClassifier, CJLanguage, Results};
use clap::Parser;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;
use std::process;

#[derive(Parser)]
#[command(about = "Evaluate CJ classifier accuracy against a phrase file")]
struct Args {
    /// File with phrases in the expected language, one per line.
    #[arg(long)]
    infile: String,

    /// Expected language code (e.g. ja, zh, zh-hans, zh-hant).
    #[arg(long)]
    language: String,

    /// Print details for each misclassified phrase.
    #[arg(long, default_value_t = false)]
    misses: bool,

    /// Path to a combined logprob model file (default: bundled model).
    #[arg(long)]
    modelfile: Option<String>,

    /// Log-probability floor (0 = use model file value).
    #[arg(long, default_value_t = 0.0)]
    logprobfloor: f64,
}

/// In Wikipedia and other corpora, simplified and traditional Chinese are mixed,
/// so treat both Chinese variants as equivalent.
fn is_equivalent(a: CJLanguage, b: CJLanguage) -> bool {
    a == b || (a.is_chinese() && b.is_chinese())
}

fn main() {
    let args = Args::parse();

    let expected = CJLanguage::from_string(&args.language);
    if expected == CJLanguage::Unknown {
        eprintln!("Error: unknown language code: {}", args.language);
        process::exit(1);
    }

    let detector = load_model(&args.modelfile, args.logprobfloor);

    let file = File::open(&args.infile)
        .unwrap_or_else(|e| panic!("Failed to open {}: {}", args.infile, e));
    let reader = BufReader::new(file);

    let mut correct = 0u64;
    let mut total = 0u64;
    let mut results = Results::new();

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Warning: read error: {}", e);
                continue;
            }
        };
        if line.is_empty() || line.chars().all(char::is_whitespace) {
            continue;
        }
        total += 1;
        detector.detect_with_results(&line, &mut results);

        let detected = results.result.unwrap_or(CJLanguage::Unknown);
        if is_equivalent(detected, expected) {
            correct += 1;
        } else if args.misses {
            let phrase = if line.len() > 80 {
                format!("{}...", &line[..line.floor_char_boundary(80)])
            } else {
                line.clone()
            };
            let detected_code = detected.iso_code();
            let detected_str = if detected_code.is_empty() {
                "unknown"
            } else {
                detected_code
            };
            eprintln!(
                "  MISS [expected={} detected={}] {{{}}} \"{}\"",
                expected.iso_code(),
                detected_str,
                results,
                phrase
            );
        }
    }

    let pct = if total > 0 {
        100.0 * correct as f64 / total as f64
    } else {
        0.0
    };
    eprintln!("Overall: {}/{} correct ({:.1}%)", correct, total, pct);
}

fn load_model(modelfile: &Option<String>, logprobfloor: f64) -> Arc<CJClassifier> {
    match modelfile {
        Some(path) => CJClassifier::load_file_with_floor(path, logprobfloor)
            .unwrap_or_else(|e| panic!("Failed to load model from {}: {}", path, e)),
        None => {
            if logprobfloor != 0.0 {
                CJClassifier::load_with_floor(logprobfloor)
            } else {
                CJClassifier::load()
            }
            .unwrap_or_else(|e| panic!("Failed to load bundled model: {}", e))
        }
    }
}
