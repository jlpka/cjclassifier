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

use std::fmt;

/// The CJK languages detected by the classifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub enum CJLanguage {
    Unknown,
    ChineseSimplified,
    ChineseTraditional,
    Japanese,
}

impl CJLanguage {
    /// Returns the ISO 639-1-style language code (e.g. "zh-hans", "ja").
    pub fn iso_code(self) -> &'static str {
        match self {
            CJLanguage::Unknown => "",
            CJLanguage::ChineseSimplified => "zh-hans",
            CJLanguage::ChineseTraditional => "zh-hant",
            CJLanguage::Japanese => "ja",
        }
    }

    /// Returns the ISO 639-3-style language code (e.g. "zho-hans", "jpn").
    pub fn iso_code3(self) -> &'static str {
        match self {
            CJLanguage::Unknown => "",
            CJLanguage::ChineseSimplified => "zho-hans",
            CJLanguage::ChineseTraditional => "zho-hant",
            CJLanguage::Japanese => "jpn",
        }
    }

    /// Returns true if this is Chinese Simplified or Chinese Traditional.
    pub fn is_chinese(self) -> bool {
        matches!(
            self,
            CJLanguage::ChineseSimplified | CJLanguage::ChineseTraditional
        )
    }

    /// Returns true if this is Japanese.
    pub fn is_japanese(self) -> bool {
        self == CJLanguage::Japanese
    }

    /// Look up a language by name, ISO code, or alternate name (case-insensitive).
    pub fn from_string(s: &str) -> CJLanguage {
        match s.to_lowercase().as_str() {
            "zh-hans" | "zho-hans" | "chinese" | "zh" | "zh-cn" | "zh-hans-cn" | "zh-hans-sg"
            | "chinese_simplified" | "chinesesimplified" => CJLanguage::ChineseSimplified,
            "zh-hant" | "zho-hant" | "zh-hant-hk" | "zh-hk" | "zh-hant-tw"
            | "chinese_traditional" | "chinesetraditional" => CJLanguage::ChineseTraditional,
            "ja" | "jpn" | "jp" | "japanese" => CJLanguage::Japanese,
            _ => CJLanguage::Unknown,
        }
    }
}

impl fmt::Display for CJLanguage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.iso_code())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_string_aliases() {
        assert_eq!(CJLanguage::from_string("zh-hans"), CJLanguage::ChineseSimplified);
        assert_eq!(CJLanguage::from_string("zho-hans"), CJLanguage::ChineseSimplified);
        assert_eq!(CJLanguage::from_string("chinese"), CJLanguage::ChineseSimplified);
        assert_eq!(CJLanguage::from_string("zh"), CJLanguage::ChineseSimplified);
        assert_eq!(CJLanguage::from_string("zh-cn"), CJLanguage::ChineseSimplified);
        assert_eq!(CJLanguage::from_string("ZH-HANS"), CJLanguage::ChineseSimplified);

        assert_eq!(CJLanguage::from_string("zh-hant"), CJLanguage::ChineseTraditional);
        assert_eq!(CJLanguage::from_string("zho-hant"), CJLanguage::ChineseTraditional);
        assert_eq!(CJLanguage::from_string("zh-hk"), CJLanguage::ChineseTraditional);

        assert_eq!(CJLanguage::from_string("ja"), CJLanguage::Japanese);
        assert_eq!(CJLanguage::from_string("jpn"), CJLanguage::Japanese);
        assert_eq!(CJLanguage::from_string("jp"), CJLanguage::Japanese);
        assert_eq!(CJLanguage::from_string("JA"), CJLanguage::Japanese);

        assert_eq!(CJLanguage::from_string("xyz"), CJLanguage::Unknown);
        assert_eq!(CJLanguage::from_string(""), CJLanguage::Unknown);
    }

    #[test]
    fn iso_codes() {
        assert_eq!(CJLanguage::ChineseSimplified.iso_code(), "zh-hans");
        assert_eq!(CJLanguage::ChineseTraditional.iso_code3(), "zho-hant");
        assert_eq!(CJLanguage::Japanese.iso_code(), "ja");
        assert_eq!(CJLanguage::Unknown.iso_code(), "");
    }

    #[test]
    fn predicates() {
        assert!(CJLanguage::ChineseSimplified.is_chinese());
        assert!(CJLanguage::ChineseTraditional.is_chinese());
        assert!(!CJLanguage::Japanese.is_chinese());
        assert!(CJLanguage::Japanese.is_japanese());
        assert!(!CJLanguage::ChineseSimplified.is_japanese());
    }
}
