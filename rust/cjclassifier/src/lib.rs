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

//! Detects whether CJK text is Japanese, Chinese Simplified, or Chinese Traditional
//! based on unigram and bigram log-probability models.
//!
//! # Quick start
//!
//! ```no_run
//! use cjclassifier::{CJClassifier, CJLanguage};
//!
//! let classifier = CJClassifier::load_bundled().unwrap();
//! assert_eq!(classifier.detect("今天天气很好"), CJLanguage::ChineseSimplified);
//! assert_eq!(classifier.detect("ひらがな"), CJLanguage::Japanese);
//! ```

mod bigram_map;
pub mod ffi;
mod language;

pub use language::CJLanguage;

use bigram_map::{BigramMap, BigramMapBuilder};
use flate2::read::GzDecoder;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::sync::{Arc, Mutex};

const CJ_RANGE_START: u32 = 0x3400;
const CJ_RANGE_END: u32 = 0x9FFF;
const CJ_RANGE_SIZE: usize = (CJ_RANGE_END - CJ_RANGE_START + 1) as usize;
const LANG_COUNT: usize = 3;

/// The three CJ languages in index order.
pub const CJ_LANGUAGES: [CJLanguage; LANG_COUNT] = [
    CJLanguage::ChineseSimplified,
    CJLanguage::ChineseTraditional,
    CJLanguage::Japanese,
];

/// The bundled model file, compiled into the binary.
const BUNDLED_MODEL: &[u8] = include_bytes!("../data/cjlogprobs.gz");

/// Returns the internal array index for a CJ language, or None for Unknown.
pub fn cj_lang_index(lang: CJLanguage) -> Option<usize> {
    match lang {
        CJLanguage::ChineseSimplified => Some(0),
        CJLanguage::ChineseTraditional => Some(1),
        CJLanguage::Japanese => Some(2),
        CJLanguage::Unknown => None,
    }
}

/// Accumulates per-language scoring data during text processing.
#[derive(Debug, Clone)]
pub struct Scores {
    /// Accumulated unigram log-probability sums, indexed by language.
    pub unigram_scores: [f64; LANG_COUNT],
    /// Accumulated bigram log-probability sums, indexed by language.
    pub bigram_scores: [f64; LANG_COUNT],
    /// Number of unigram model hits per language.
    pub unigram_hits_per_lang: [i32; LANG_COUNT],
    /// Number of bigram model hits per language.
    pub bigram_hits_per_lang: [i32; LANG_COUNT],
    /// Number of kana characters seen.
    pub kana_count: i32,
    /// Number of characters in the main CJ range (0x3400-0x9FFF).
    pub cj_char_count: i32,
}

impl Scores {
    pub fn new() -> Self {
        Scores {
            unigram_scores: [0.0; LANG_COUNT],
            bigram_scores: [0.0; LANG_COUNT],
            unigram_hits_per_lang: [0; LANG_COUNT],
            bigram_hits_per_lang: [0; LANG_COUNT],
            kana_count: 0,
            cj_char_count: 0,
        }
    }

    pub fn clear(&mut self) {
        *self = Scores::new();
    }

    /// Returns true if at least one unigram was matched in any language.
    pub fn any_hits(&self) -> bool {
        self.unigram_hits_per_lang.iter().any(|&h| h > 0)
    }
}

impl Default for Scores {
    fn default() -> Self {
        Self::new()
    }
}

/// Detection results, including accumulated scores and the computed result.
#[derive(Debug, Clone)]
pub struct Results {
    /// The underlying per-language scoring data.
    pub scores: Scores,
    /// Combined (unigram + bigram + boost-adjusted) scores, indexed by language.
    pub total_scores: [f64; LANG_COUNT],
    /// Per-language boosts (0..1.0). A boost makes the language more likely to win.
    pub boosts: [f64; LANG_COUNT],
    /// The winning language, or None if not yet computed.
    pub result: Option<CJLanguage>,
    /// Gap between best and runner-up (0 = dead heat, 1 = no contest).
    pub gap: f64,
}

impl Results {
    pub fn new() -> Self {
        Results {
            scores: Scores::new(),
            total_scores: [0.0; LANG_COUNT],
            boosts: [0.0; LANG_COUNT],
            result: None,
            gap: 0.0,
        }
    }

    pub fn clear(&mut self) {
        self.scores.clear();
        self.total_scores = [0.0; LANG_COUNT];
        self.boosts = [0.0; LANG_COUNT];
        self.result = None;
        self.gap = 0.0;
    }

    fn compute_totals(&mut self, placeholder_score: f64) {
        let max_unigram = *self.scores.unigram_hits_per_lang.iter().max().unwrap_or(&0);
        let max_bigram = *self.scores.bigram_hits_per_lang.iter().max().unwrap_or(&0);
        for i in 0..LANG_COUNT {
            self.total_scores[i] = self.scores.unigram_scores[i]
                + (max_unigram - self.scores.unigram_hits_per_lang[i]) as f64 * placeholder_score
                + self.scores.bigram_scores[i]
                + (max_bigram - self.scores.bigram_hits_per_lang[i]) as f64 * placeholder_score;
            // Implement boosts: since logprob values are negative, a favorable boost is negative.
            self.total_scores[i] -= self.boosts[i] * self.total_scores[i];
        }
    }

    /// Returns a compact, comma-separated representation of per-language relative scores,
    /// e.g. "zh-hans:1.00,zh-hant:0.97,ja:0.85", ordered from best to worst.
    pub fn to_short_string(&self) -> String {
        match self.result {
            None | Some(CJLanguage::Unknown) => return String::new(),
            Some(CJLanguage::Japanese) if self.scores.kana_count > 0 => {
                return "ja:1.0,zh-hans:0,zh-hant:0".to_string();
            }
            _ => {}
        }

        let mut order: Vec<usize> = vec![0, 1, 2];
        order.sort_by(|&a, &b| {
            self.total_scores[b]
                .partial_cmp(&self.total_scores[a])
                .unwrap()
        });

        let best = self.total_scores[order[0]];
        if best == 0.0 {
            return String::new();
        }

        let mut parts = Vec::with_capacity(LANG_COUNT);
        for &li in &order {
            let ratio = if self.total_scores[li] != 0.0 {
                best / self.total_scores[li]
            } else {
                0.0
            };
            parts.push(format!("{}:{:.2}", CJ_LANGUAGES[li].iso_code(), ratio));
        }
        parts.join(",")
    }
}

impl Default for Results {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for Results {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Results{{result={}",
            self.result.map_or("null", |r| r.iso_code())
        )?;
        if self.scores.kana_count > 0 {
            write!(
                f,
                " kana={}/{}",
                self.scores.kana_count,
                self.scores.kana_count + self.scores.cj_char_count
            )?;
        }
        for li in 0..LANG_COUNT {
            write!(
                f,
                " | {}: uni={:.2} bi={:.2} total={:.2} biHits={}",
                CJ_LANGUAGES[li].iso_code(),
                self.scores.unigram_scores[li],
                self.scores.bigram_scores[li],
                self.total_scores[li],
                self.scores.bigram_hits_per_lang[li]
            )?;
        }
        write!(f, "}}")
    }
}

/// Singleton cache for loaded models, keyed by cache key string.
static CACHE: Mutex<Option<HashMap<String, Arc<CJClassifier>>>> = Mutex::new(None);

/// The CJK language classifier.
pub struct CJClassifier {
    /// unigramLogProbs[(charIndex*LANG_COUNT) + langIndex]
    unigram_log_probs: Vec<f32>,
    /// Bigram probability map.
    bigram_log_probs: BigramMap,
    /// Default log-probability used when a bigram is not in the model.
    default_log_prob: f64,
    /// Tolerated kana fraction threshold for Chinese.
    tolerated_kana_threshold: f64,
}

impl CJClassifier {
    // ========================================================================
    // Cached loaders (return Arc, singleton per cache key)
    // ========================================================================

    /// Returns a cached classifier loaded from the bundled model.
    /// Subsequent calls return the same instance. Thread-safe.
    pub fn load() -> io::Result<Arc<Self>> {
        Self::load_cached("classpath:default:0", |_| {
            Self::load_from_reader(
                GzDecoder::new(std::io::Cursor::new(BUNDLED_MODEL)),
                "bundled:cjlogprobs.gz",
                0.0,
            )
        })
    }

    /// Returns a cached classifier loaded from the bundled model with a custom
    /// log-probability floor. Thread-safe.
    pub fn load_with_floor(log_prob_floor: f64) -> io::Result<Arc<Self>> {
        let key = format!("classpath:default:{}", log_prob_floor);
        Self::load_cached(&key, |_| {
            Self::load_from_reader(
                GzDecoder::new(std::io::Cursor::new(BUNDLED_MODEL)),
                "bundled:cjlogprobs.gz",
                log_prob_floor,
            )
        })
    }

    /// Returns a cached classifier loaded from a filesystem path. Thread-safe.
    pub fn load_file(path: &str) -> io::Result<Arc<Self>> {
        Self::load_file_with_floor(path, 0.0)
    }

    /// Returns a cached classifier loaded from a filesystem path with a custom
    /// log-probability floor. Thread-safe.
    pub fn load_file_with_floor(path: &str, log_prob_floor: f64) -> io::Result<Arc<Self>> {
        let key = format!("{}:{}", path, log_prob_floor);
        Self::load_cached(&key, |_| Self::load_from_reader_file(path, log_prob_floor))
    }

    /// Clears the singleton cache, primarily for testing and benchmarking.
    pub fn clear_cached_models() {
        let mut guard = CACHE.lock().unwrap();
        if let Some(map) = guard.as_mut() {
            map.clear();
        }
    }

    fn load_cached<F>(key: &str, loader: F) -> io::Result<Arc<Self>>
    where
        F: FnOnce(&str) -> io::Result<Self>,
    {
        let mut guard = CACHE.lock().unwrap();
        let map = guard.get_or_insert_with(HashMap::new);
        if let Some(cached) = map.get(key) {
            return Ok(Arc::clone(cached));
        }
        let instance = Arc::new(loader(key)?);
        map.insert(key.to_string(), Arc::clone(&instance));
        Ok(instance)
    }

    // ========================================================================
    // Uncached loaders (return owned CJClassifier, for benchmarking etc.)
    // ========================================================================

    /// Load from the bundled model without caching.
    pub fn load_bundled() -> io::Result<Self> {
        Self::load_from_reader(
            GzDecoder::new(std::io::Cursor::new(BUNDLED_MODEL)),
            "bundled:cjlogprobs.gz",
            0.0,
        )
    }

    /// Load from the bundled model with a custom log-probability floor, without caching.
    pub fn load_bundled_with_floor(log_prob_floor: f64) -> io::Result<Self> {
        Self::load_from_reader(
            GzDecoder::new(std::io::Cursor::new(BUNDLED_MODEL)),
            "bundled:cjlogprobs.gz",
            log_prob_floor,
        )
    }

    /// Load from a filesystem path without caching.
    pub fn load_file_uncached(path: &str) -> io::Result<Self> {
        Self::load_from_reader_file(path, 0.0)
    }

    /// Load from a filesystem path with a custom log-probability floor, without caching.
    pub fn load_file_uncached_with_floor(path: &str, log_prob_floor: f64) -> io::Result<Self> {
        Self::load_from_reader_file(path, log_prob_floor)
    }

    fn load_from_reader_file(path: &str, log_prob_floor: f64) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::with_capacity(1024 * 1024, file);
        if path.ends_with(".gz") {
            Self::load_from_reader(GzDecoder::new(reader), path, log_prob_floor)
        } else {
            Self::load_from_reader(reader, path, log_prob_floor)
        }
    }

    /// Load from any reader.
    pub fn load_from_reader<R: Read>(
        reader: R,
        label: &str,
        log_prob_floor: f64,
    ) -> io::Result<Self> {
        let buf_reader = BufReader::with_capacity(1024 * 1024, reader);
        let mut unigram_log_probs = vec![0.0f32; CJ_RANGE_SIZE * LANG_COUNT];
        let mut bigram_builder = BigramMapBuilder::new(16 << 10);

        let mut lines = buf_reader.lines();

        // Parse header
        let header = lines
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Empty model file"))??;

        if !header.starts_with("Languages: ") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid model file (bad header): {}", label),
            ));
        }

        let header_parts: Vec<&str> = header.split(' ').collect();
        let lang_codes: Vec<&str> = header_parts[1].split(',').collect();
        let mut lang_map = Vec::with_capacity(lang_codes.len());
        for code in &lang_codes {
            let lang = CJLanguage::from_string(code);
            match cj_lang_index(lang) {
                Some(li) => lang_map.push(li),
                None => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Unknown CJ language in header: {} in {}", code, label),
                    ));
                }
            }
        }

        // Parse MinLogProb from header
        let mut parsed_min_prob: Option<f64> = None;
        for i in 0..header_parts.len().saturating_sub(1) {
            if header_parts[i] == "MinLogProb:" {
                parsed_min_prob = header_parts[i + 1].parse().ok();
                break;
            }
        }

        let default_log_prob = if log_prob_floor == 0.0 {
            parsed_min_prob.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "No MinLogProb in header and no explicit log_prob_floor: {}",
                        label
                    ),
                )
            })?
        } else {
            match parsed_min_prob {
                Some(p) => log_prob_floor.max(p),
                None => log_prob_floor,
            }
        };

        // Read unigram and bigram lines
        let mut probs;
        let mut unigram_count = 0usize;
        let mut bigram_count = 0usize;

        for line_result in lines {
            let line = line_result?;
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split(' ').collect();
            let key = parts[0];
            if parts.len() != lang_map.len() + 1 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Column count mismatch on line: {} in {}", line, label),
                ));
            }

            let chars: Vec<char> = key.chars().collect();
            if chars.len() == 1 {
                // Unigram
                let c = chars[0] as u32;
                if c >= CJ_RANGE_START && c <= CJ_RANGE_END {
                    let idx = (c - CJ_RANGE_START) as usize;
                    for (col, &li) in lang_map.iter().enumerate() {
                        let val: f32 = parts[col + 1].parse().map_err(|e| {
                            io::Error::new(io::ErrorKind::InvalidData, format!("Parse error: {}", e))
                        })?;
                        unigram_log_probs[idx * LANG_COUNT + li] =
                            val.max(default_log_prob as f32);
                    }
                    unigram_count += 1;
                }
            } else if chars.len() == 2 {
                // Bigram
                probs = [0.0; LANG_COUNT];
                let mut any_higher_than_default = false;
                for (col, &li) in lang_map.iter().enumerate() {
                    let val: f32 = parts[col + 1].parse().map_err(|e| {
                        io::Error::new(io::ErrorKind::InvalidData, format!("Parse error: {}", e))
                    })?;
                    if val < default_log_prob as f32 || val == 0.0 {
                        probs[li] = default_log_prob as f32;
                    } else {
                        any_higher_than_default = true;
                        probs[li] = val;
                    }
                }
                if any_higher_than_default {
                    bigram_builder.put(chars[0] as u32, chars[1] as u32, &probs);
                    bigram_count += 1;
                }
            }
        }

        eprintln!(
            "Loaded {} unigrams and {} bigrams, minLogProb: {:.2}, from {}",
            unigram_count, bigram_count, default_log_prob, label
        );

        Ok(CJClassifier {
            unigram_log_probs,
            bigram_log_probs: bigram_builder.build(),
            default_log_prob,
            tolerated_kana_threshold: 0.01,
        })
    }

    /// Set the kana fraction threshold above which text is classified as Japanese.
    pub fn set_tolerated_kana_threshold(&mut self, threshold: f64) {
        self.tolerated_kana_threshold = threshold;
    }

    /// Returns the current kana fraction threshold.
    pub fn tolerated_kana_threshold(&self) -> f64 {
        self.tolerated_kana_threshold
    }

    /// Detect language, returning only the language enum.
    pub fn detect(&self, text: &str) -> CJLanguage {
        let mut results = Results::new();
        self.add_text(text, &mut results.scores);
        self.compute_result(&mut results)
    }

    /// Detect language, populating the provided Results with scoring info.
    pub fn detect_with_results(&self, text: &str, results: &mut Results) -> CJLanguage {
        results.clear();
        self.add_text(text, &mut results.scores);
        self.compute_result(results)
    }

    /// Add text to the scoring accumulator. For incremental processing.
    pub fn add_text(&self, text: &str, scores: &mut Scores) {
        let mut prev: u32 = 0;
        let mut prev_in_range = false;

        for c in text.chars() {
            let cp = c as u32;
            if is_kana(cp) {
                scores.kana_count += 1;
                continue;
            }
            let in_range = in_main_cj_range(cp);
            if in_range {
                scores.cj_char_count += 1;
                self.calc_with_prev(cp, prev, prev_in_range, scores);
            }
            prev = cp;
            prev_in_range = in_range;
        }
    }

    /// Compute the final result from accumulated scores.
    pub fn compute_result(&self, results: &mut Results) -> CJLanguage {
        if results.scores.kana_count > 0 {
            let kana_ratio = results.scores.kana_count as f64
                / (results.scores.kana_count + results.scores.cj_char_count) as f64;
            if kana_ratio > self.tolerated_kana_threshold {
                results.result = Some(CJLanguage::Japanese);
                results.gap = 1.0;
                results.total_scores[cj_lang_index(CJLanguage::Japanese).unwrap()] = 1.0;
                return CJLanguage::Japanese;
            }
        }
        if !results.scores.any_hits() {
            results.result = Some(CJLanguage::Unknown);
            results.gap = 0.0;
            return CJLanguage::Unknown;
        }
        results.compute_totals(self.default_log_prob);

        // Find best and second-best language
        let mut best_idx = 0usize;
        let mut second_idx: Option<usize> = None;
        for li in 1..LANG_COUNT {
            if results.total_scores[li] > results.total_scores[best_idx] {
                second_idx = Some(best_idx);
                best_idx = li;
            } else if second_idx.is_none()
                || results.total_scores[li] > results.total_scores[second_idx.unwrap()]
            {
                second_idx = Some(li);
            }
        }
        results.result = Some(CJ_LANGUAGES[best_idx]);

        // Compute gap
        let best = results.total_scores[best_idx];
        let second = second_idx.map_or(best, |si| results.total_scores[si]);
        results.gap = if second != 0.0 {
            1.0 - (best / second)
        } else {
            0.0
        };

        CJ_LANGUAGES[best_idx]
    }

    #[inline]
    fn calc_with_prev(
        &self,
        c: u32,
        prev: u32,
        prev_in_range: bool,
        scores: &mut Scores,
    ) {
        let idx = (c - CJ_RANGE_START) as usize;
        for li in 0..LANG_COUNT {
            let u_prob = self.unigram_log_probs[idx * LANG_COUNT + li];
            scores.unigram_scores[li] += (u_prob as f64).max(self.default_log_prob);
            if u_prob != 0.0 {
                scores.unigram_hits_per_lang[li] += 1;
            }
        }
        if prev_in_range {
            if let Some(probs) = self.bigram_log_probs.get(prev, c) {
                for li in 0..LANG_COUNT {
                    let bp = probs[li];
                    scores.bigram_scores[li] += (bp as f64).max(self.default_log_prob);
                    if bp != 0.0 {
                        scores.bigram_hits_per_lang[li] += 1;
                    }
                }
            }
        }
    }
}

#[inline]
fn is_kana(c: u32) -> bool {
    (0x3040..=0x30FF).contains(&c)
        || (0x31F0..=0x31FF).contains(&c)
        || (0xFF65..=0xFF9F).contains(&c)
}

#[inline]
fn in_main_cj_range(c: u32) -> bool {
    c >= CJ_RANGE_START && c <= CJ_RANGE_END
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_test_model() -> CJClassifier {
        // Simplified-leaning chars
        let zh_hans_a = '\u{4e00}'; // 一
        let zh_hans_b = '\u{4e8c}'; // 二
        let zh_hans_c = '\u{4e09}'; // 三
        // Traditional-leaning chars
        let zh_hant_a = '\u{5b78}'; // 學
        let zh_hant_b = '\u{6a23}'; // 樣
        let zh_hant_c = '\u{9a57}'; // 驗
        // Japanese-leaning chars
        let ja_a = '\u{8fbc}'; // 込
        let ja_b = '\u{7573}'; // 畳
        let ja_c = '\u{586b}'; // 填

        let model_text = format!(
            "Languages: zh-hans,zh-hant,ja UnigramTotals: 1000,1000,1000 BigramTotals: 1000,1000,1000 MinLogProb: -10.0\n\
             {} -2.0 -8.0 -8.0\n\
             {} -2.0 -8.0 -8.0\n\
             {} -2.0 -8.0 -8.0\n\
             {} -8.0 -2.0 -8.0\n\
             {} -8.0 -2.0 -8.0\n\
             {} -8.0 -2.0 -8.0\n\
             {} -8.0 -8.0 -2.0\n\
             {} -8.0 -8.0 -2.0\n\
             {} -8.0 -8.0 -2.0\n\
             {}{} -2.0 -8.0 -8.0\n\
             {}{} -2.0 -8.0 -8.0\n\
             {}{} -8.0 -2.0 -8.0\n\
             {}{} -8.0 -2.0 -8.0\n\
             {}{} -8.0 -8.0 -2.0\n\
             {}{} -8.0 -8.0 -2.0\n",
            zh_hans_a, zh_hans_b, zh_hans_c,
            zh_hant_a, zh_hant_b, zh_hant_c,
            ja_a, ja_b, ja_c,
            zh_hans_a, zh_hans_b,
            zh_hans_b, zh_hans_c,
            zh_hant_a, zh_hant_b,
            zh_hant_b, zh_hant_c,
            ja_a, ja_b,
            ja_b, ja_c,
        );

        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("test-model.txt");
        let mut file = File::create(&model_path).unwrap();
        file.write_all(model_text.as_bytes()).unwrap();
        drop(file);

        // We need to keep dir alive, but for tests we'll just load immediately
        let classifier = CJClassifier::load_file_uncached(model_path.to_str().unwrap()).unwrap();
        // dir drops here, that's fine - model is already loaded
        classifier
    }

    #[test]
    fn detect_simplified_chinese() {
        let detector = create_test_model();
        let text = "\u{4e00}\u{4e8c}\u{4e09}";
        assert_eq!(detector.detect(text), CJLanguage::ChineseSimplified);
    }

    #[test]
    fn detect_traditional_chinese() {
        let detector = create_test_model();
        let text = "\u{5b78}\u{6a23}\u{9a57}";
        assert_eq!(detector.detect(text), CJLanguage::ChineseTraditional);
    }

    #[test]
    fn detect_japanese_by_ideographs() {
        let detector = create_test_model();
        let text = "\u{8fbc}\u{7573}\u{586b}";
        assert_eq!(detector.detect(text), CJLanguage::Japanese);
    }

    #[test]
    fn detect_japanese_by_kana() {
        let detector = create_test_model();
        assert_eq!(detector.detect("ひらがな"), CJLanguage::Japanese);
        assert_eq!(detector.detect("カタカナ"), CJLanguage::Japanese);
    }

    #[test]
    fn kana_overrides_ideograph_signal() {
        let detector = create_test_model();
        let text = "\u{4e00}\u{4e8c}\u{3053}";
        assert_eq!(detector.detect(text), CJLanguage::Japanese);
    }

    #[test]
    fn small_kana_fraction_still_chinese() {
        let detector = create_test_model();
        let mut text = "\u{4e00}".repeat(101);
        text.push('\u{3053}'); // one hiragana
        assert_eq!(detector.detect(&text), CJLanguage::ChineseSimplified);
    }

    #[test]
    fn empty_text_returns_unknown() {
        let detector = create_test_model();
        assert_eq!(detector.detect(""), CJLanguage::Unknown);
    }

    #[test]
    fn non_cj_text_returns_unknown() {
        let detector = create_test_model();
        assert_eq!(detector.detect("Hello world 123"), CJLanguage::Unknown);
    }

    #[test]
    fn single_char_detection() {
        let detector = create_test_model();
        assert_eq!(detector.detect("\u{8fbc}"), CJLanguage::Japanese);
    }

    #[test]
    fn stats_populated_correctly() {
        let detector = create_test_model();
        let mut results = Results::new();
        detector.detect_with_results("\u{4e00}\u{4e8c}\u{4e09}", &mut results);
        assert!(results.scores.unigram_scores[0] != 0.0);
        assert!(results.total_scores[0] != 0.0);
    }

    #[test]
    fn clear_resets_all_fields() {
        let detector = create_test_model();
        let mut results = Results::new();
        detector.detect_with_results("\u{4e00}\u{4e8c}", &mut results);
        assert!(results.result.is_some());
        results.clear();
        assert!(results.result.is_none());
        assert_eq!(results.scores.kana_count, 0);
        for i in 0..3 {
            assert_eq!(results.scores.unigram_scores[i], 0.0);
            assert_eq!(results.scores.bigram_scores[i], 0.0);
            assert_eq!(results.total_scores[i], 0.0);
        }
    }

    #[test]
    fn to_short_string_format() {
        let detector = create_test_model();
        let mut results = Results::new();
        detector.detect_with_results("\u{4e00}\u{4e8c}\u{4e09}", &mut results);
        let s = results.to_short_string();
        assert!(!s.is_empty());
        assert!(s.contains("zh-hans"));
        assert!(s.contains("zh-hant"));
        assert!(s.contains("ja"));
        assert_eq!(s.matches(',').count(), 2);
    }

    #[test]
    fn to_short_string_kana() {
        let detector = create_test_model();
        let mut results = Results::new();
        detector.detect_with_results("\u{3053}\u{3093}", &mut results);
        assert_eq!(results.to_short_string(), "ja:1.0,zh-hans:0,zh-hant:0");
    }

    #[test]
    fn to_short_string_unknown_is_empty() {
        let detector = create_test_model();
        let mut results = Results::new();
        detector.detect_with_results("Hello", &mut results);
        assert_eq!(results.to_short_string(), "");
    }

    #[test]
    fn different_column_order_produces_same_result() {
        let model_text = format!(
            "Languages: ja,zh-hant,zh-hans UnigramTotals: 1000,1000,1000 BigramTotals: 1000,1000,1000 MinLogProb: -10.0\n\
             \u{4e00} -8.0 -8.0 -2.0\n\
             \u{4e8c} -8.0 -8.0 -2.0\n\
             \u{4e09} -8.0 -8.0 -2.0\n\
             \u{8fbc} -2.0 -8.0 -8.0\n\
             \u{7573} -2.0 -8.0 -8.0\n\
             \u{586b} -2.0 -8.0 -8.0\n"
        );
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("reversed-model.txt");
        let mut file = File::create(&model_path).unwrap();
        file.write_all(model_text.as_bytes()).unwrap();
        drop(file);

        let d = CJClassifier::load_file_uncached(model_path.to_str().unwrap()).unwrap();
        assert_eq!(
            d.detect("\u{4e00}\u{4e8c}\u{4e09}"),
            CJLanguage::ChineseSimplified
        );
        assert_eq!(
            d.detect("\u{8fbc}\u{7573}\u{586b}"),
            CJLanguage::Japanese
        );
    }

    #[test]
    fn load_bundled_model() {
        let classifier = CJClassifier::load().unwrap();
        assert_eq!(
            classifier.detect("今天天气很好，我们去公园散步"),
            CJLanguage::ChineseSimplified
        );
        assert_eq!(
            classifier.detect("今天天氣很好，我們去公園散步"),
            CJLanguage::ChineseTraditional
        );
        assert_eq!(classifier.detect("事務所"), CJLanguage::Japanese);
        assert_eq!(
            classifier.detect("ひらがなとカタカナと"),
            CJLanguage::Japanese
        );
        assert_eq!(
            classifier.detect("日本語は日本で使われている言語です。ひらがなとカタカナと漢字を使います"),
            CJLanguage::Japanese
        );
        assert_eq!(classifier.detect("hello"), CJLanguage::Unknown);
    }
}
