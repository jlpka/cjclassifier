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

//! C FFI for using cjclassifier from C/C++.
//!
//! # Usage from C/C++
//!
//! ```c
//! #include "cjclassifier.h"
//!
//! CJClassifierHandle *model = cj_load_bundled();
//! if (model) {
//!     const char *text = "今天天气很好";
//!     CJLanguageCode lang = cj_detect(model, text, strlen(text));
//!     // lang == CJ_CHINESE_SIMPLIFIED (1)
//!     cj_free(model);
//! }
//! ```

use crate::{CJClassifier, CJLanguage, Results};
use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::Arc;

/// Opaque handle to a classifier instance (an Arc behind the scenes).
pub type CJClassifierHandle = Arc<CJClassifier>;

/// Integer language codes for C interop.
/// 0 = Unknown, 1 = Chinese Simplified, 2 = Chinese Traditional, 3 = Japanese
pub type CJLanguageCode = i32;

const CJ_UNKNOWN: CJLanguageCode = 0;
const CJ_CHINESE_SIMPLIFIED: CJLanguageCode = 1;
const CJ_CHINESE_TRADITIONAL: CJLanguageCode = 2;
const CJ_JAPANESE: CJLanguageCode = 3;

fn lang_to_code(lang: CJLanguage) -> CJLanguageCode {
    match lang {
        CJLanguage::Unknown => CJ_UNKNOWN,
        CJLanguage::ChineseSimplified => CJ_CHINESE_SIMPLIFIED,
        CJLanguage::ChineseTraditional => CJ_CHINESE_TRADITIONAL,
        CJLanguage::Japanese => CJ_JAPANESE,
    }
}

/// Convert a (pointer, length) pair to a &str, returning None on null or invalid UTF-8.
unsafe fn ptr_len_to_str<'a>(text: *const c_char, len: usize) -> Option<&'a str> {
    if text.is_null() {
        return None;
    }
    let bytes = unsafe { std::slice::from_raw_parts(text as *const u8, len) };
    std::str::from_utf8(bytes).ok()
}

/// C result struct for detailed detection.
#[repr(C)]
pub struct CJResult {
    pub language: CJLanguageCode,
    pub gap: f64,
    pub total_scores: [f64; 3],
}

/// Load the bundled model. Returns null on failure.
///
/// # Safety
/// The returned pointer must be freed with `cj_free`.
#[no_mangle]
pub extern "C" fn cj_load_bundled() -> *mut CJClassifierHandle {
    match CJClassifier::load() {
        Ok(arc) => Box::into_raw(Box::new(arc)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Load a model from a file path. Returns null on failure.
///
/// # Safety
/// `path` must be a valid null-terminated UTF-8 string.
/// The returned pointer must be freed with `cj_free`.
#[no_mangle]
pub unsafe extern "C" fn cj_load_file(path: *const c_char) -> *mut CJClassifierHandle {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    match CJClassifier::load_file(path_str) {
        Ok(arc) => Box::into_raw(Box::new(arc)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Detect the language of a UTF-8 text buffer with explicit length.
///
/// # Safety
/// `model` must be a valid pointer from `cj_load_bundled` or `cj_load_file`.
/// `text` must point to at least `len` bytes of valid UTF-8.
#[no_mangle]
pub unsafe extern "C" fn cj_detect(
    model: *const CJClassifierHandle,
    text: *const c_char,
    len: usize,
) -> CJLanguageCode {
    if model.is_null() {
        return CJ_UNKNOWN;
    }
    let model = unsafe { &*model };
    match unsafe { ptr_len_to_str(text, len) } {
        Some(s) => lang_to_code(model.detect(s)),
        None => CJ_UNKNOWN,
    }
}

/// Detect with detailed results, using a UTF-8 text buffer with explicit length.
///
/// # Safety
/// `model` must be a valid pointer from `cj_load_bundled` or `cj_load_file`.
/// `text` must point to at least `len` bytes of valid UTF-8.
/// `result` must be a valid pointer to a `CJResult` struct.
#[no_mangle]
pub unsafe extern "C" fn cj_detect_detailed(
    model: *const CJClassifierHandle,
    text: *const c_char,
    len: usize,
    result: *mut CJResult,
) -> CJLanguageCode {
    if model.is_null() || result.is_null() {
        return CJ_UNKNOWN;
    }
    let model = unsafe { &*model };
    let text_str = match unsafe { ptr_len_to_str(text, len) } {
        Some(s) => s,
        None => return CJ_UNKNOWN,
    };

    let mut results = Results::new();
    let lang = model.detect_with_results(text_str, &mut results);
    let code = lang_to_code(lang);

    let out = unsafe { &mut *result };
    out.language = code;
    out.gap = results.gap;
    out.total_scores = results.total_scores;

    code
}

/// Free a classifier handle.
///
/// # Safety
/// `model` must be a valid pointer from `cj_load_bundled` or `cj_load_file`,
/// or null (in which case this is a no-op).
#[no_mangle]
pub unsafe extern "C" fn cj_free(model: *mut CJClassifierHandle) {
    if !model.is_null() {
        drop(unsafe { Box::from_raw(model) });
    }
}
