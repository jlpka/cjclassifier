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

//! Benchmark model loading speed and memory consumption.
//!
//! # Examples
//!
//! ```sh
//! cargo run --bin load_speed
//! cargo run --bin load_speed -- --modelfile path/to/cjlogprobs.gz
//! cargo run --bin load_speed -- --duration 60
//! ```

use cjclassifier::CJClassifier;
use clap::Parser;
use std::time::Instant;

#[derive(Parser)]
#[command(about = "Benchmark CJ classifier model loading speed")]
struct Args {
    /// Path to a combined logprob model file (default: bundled model).
    #[arg(long)]
    modelfile: Option<String>,

    /// Log-probability floor (0 = use model file value).
    #[arg(long, default_value_t = 0.0)]
    logprobfloor: f64,

    /// Benchmark duration in seconds.
    #[arg(long, default_value_t = 10)]
    duration: u64,
}

fn load_model(modelfile: &Option<String>, logprobfloor: f64) -> CJClassifier {
    match modelfile {
        Some(path) => CJClassifier::load_file_uncached_with_floor(path, logprobfloor)
            .unwrap_or_else(|e| panic!("Failed to load model from {}: {}", path, e)),
        None => {
            if logprobfloor != 0.0 {
                CJClassifier::load_bundled_with_floor(logprobfloor)
            } else {
                CJClassifier::load_bundled()
            }
            .unwrap_or_else(|e| panic!("Failed to load bundled model: {}", e))
        }
    }
}

fn main() {
    let args = Args::parse();

    // Warmup run with memory measurement.
    // Rust doesn't have tracemalloc, but we can measure RSS change on supported platforms.
    eprintln!("Warmup...");
    let rss_before = get_rss_bytes();
    let _model = load_model(&args.modelfile, args.logprobfloor);
    let rss_after = get_rss_bytes();

    if let (Some(before), Some(after)) = (rss_before, rss_after) {
        let diff_mb = (after as f64 - before as f64) / (1024.0 * 1024.0);
        eprintln!("** MODEL MEMORY (RSS delta): {:.1}MB", diff_mb);
    }
    drop(_model);

    // Benchmark loop
    let deadline = Instant::now() + std::time::Duration::from_secs(args.duration);
    let mut iteration = 0u64;
    let mut total_sec = 0.0f64;
    let mut min_sec = f64::INFINITY;
    let mut max_sec = 0.0f64;

    while Instant::now() < deadline {
        iteration += 1;
        let t0 = Instant::now();
        let _model = load_model(&args.modelfile, args.logprobfloor);
        let sec = t0.elapsed().as_secs_f64();
        drop(_model);

        total_sec += sec;
        min_sec = min_sec.min(sec);
        max_sec = max_sec.max(sec);
        eprintln!("  iteration {}: {:.3}s", iteration, sec);
    }

    if iteration > 0 {
        let avg_sec = total_sec / iteration as f64;
        eprintln!();
        eprintln!(
            "Benchmark results: {} iterations in {:.1}s",
            iteration, total_sec
        );
        eprintln!(
            "  avg={:.3}s  min={:.3}s  max={:.3}s",
            avg_sec, min_sec, max_sec
        );
    }
}

/// Get current RSS in bytes (macOS and Linux).
fn get_rss_bytes() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        use std::mem;
        unsafe {
            let mut info: libc::rusage = mem::zeroed();
            if libc::getrusage(libc::RUSAGE_SELF, &mut info) == 0 {
                // macOS reports maxrss in bytes
                return Some(info.ru_maxrss as usize);
            }
        }
        None
    }
    #[cfg(target_os = "linux")]
    {
        use std::mem;
        unsafe {
            let mut info: libc::rusage = mem::zeroed();
            if libc::getrusage(libc::RUSAGE_SELF, &mut info) == 0 {
                // Linux reports maxrss in kilobytes
                return Some(info.ru_maxrss as usize * 1024);
            }
        }
        None
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}
