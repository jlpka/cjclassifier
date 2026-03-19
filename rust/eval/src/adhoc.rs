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

//! Ad-hoc CJ language detection tool.
//!
//! Detects whether a phrase is Japanese, Chinese Simplified, or Chinese Traditional,
//! and prints detailed scoring results.
//!
//! # Examples
//!
//! ```sh
//! cargo run --bin adhoc -- --phrase "羊驼是一种很好的动物。"
//! cargo run --bin adhoc -- --phrase "事務所へ行きます"
//! cargo run --bin adhoc -- --modelfile path/to/cjlogprobs.gz --phrase "東京皇居"
//! ```

use cjclassifier::{CJClassifier, Results};
use clap::Parser;
use std::sync::Arc;

#[derive(Parser)]
#[command(about = "Ad-hoc CJ language detection")]
struct Args {
    /// The phrase to detect.
    #[arg(long)]
    phrase: String,

    /// Path to a combined logprob model file (default: bundled model).
    #[arg(long)]
    modelfile: Option<String>,

    /// Log-probability floor (0 = use model file value).
    #[arg(long, default_value_t = 0.0)]
    logprobfloor: f64,
}

fn main() {
    let args = Args::parse();

    let detector = load_model(&args.modelfile, args.logprobfloor);

    let mut results = Results::new();
    let result = detector.detect_with_results(&args.phrase, &mut results);

    let code = result.iso_code();
    if code.is_empty() {
        println!("Result: UNKNOWN");
    } else {
        println!("Result: {} ({:?})", code, result);
    }
    println!("{}", results);
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
