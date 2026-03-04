package com.jlpka.cjclassifier;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * An enum for Chinese vs Japanese for {@link CJClassifier}. Chinese Simplified vs Traditional are
 * also separate entries.
 *
 * <p>Each constant carries ISO 639 codes and alternate name strings that are recognized by {@link
 * #fromString(String)}.
 */
public enum CJLanguage {
  UNKNOWN("", "", ""),
  CHINESE_SIMPLIFIED("zh-hans", "zho-hans", "chinese,zh,zh-cn,zh-hans-cn,zh-hans-sg"),
  CHINESE_TRADITIONAL("zh-hant", "zho-hant", "zh-hant-hk,zh-hk,zh-hant-tw"),
  JAPANESE("ja", "jpn", "jp");

  private final String isoCode;
  private final String isoCode3;
  private final List<String> altNames;

  CJLanguage(String isoCode, String isoCode3, String altNames) {
    this.isoCode = isoCode;
    this.isoCode3 = isoCode3;
    this.altNames =
        altNames.isEmpty()
            ? Collections.emptyList()
            : Collections.unmodifiableList(Arrays.asList(altNames.split(",")));
  }

  /**
   * Returns the ISO 639-1-style language code (e.g. {@code "zh-hans"}, {@code "ja"}).
   *
   * @return the short ISO code, or {@code ""} for {@link #UNKNOWN}
   */
  public String isoCode() {
    return isoCode;
  }

  /**
   * Returns the ISO 639-3-style language code (e.g. {@code "zho-hans"}, {@code "jpn"}).
   *
   * @return the three-letter ISO code, or {@code ""} for {@link #UNKNOWN}
   */
  public String isoCode3() {
    return isoCode3;
  }

  /**
   * Returns alternate name strings recognized by {@link #fromString(String)}.
   *
   * @return an unmodifiable list of alternate names (may be empty)
   */
  public List<String> altNames() {
    return altNames;
  }

  /**
   * Looks up a {@link CJLanguage} by name, ISO code, or alternate name (case-insensitive).
   *
   * @return the matching language, or {@link #UNKNOWN} if not recognized
   */
  public static CJLanguage fromString(String s) {
    if (s == null) return UNKNOWN;
    return BY_NAME.getOrDefault(s.toLowerCase(), UNKNOWN);
  }

  /**
   * Returns {@code true} if this is {@link #CHINESE_SIMPLIFIED} or {@link #CHINESE_TRADITIONAL}.
   *
   * @return whether this language is a Chinese variant
   */
  public boolean isChinese() {
    return this == CHINESE_SIMPLIFIED || this == CHINESE_TRADITIONAL;
  }

  /**
   * Returns {@code true} if this is {@link #JAPANESE}.
   *
   * @return whether this language is Japanese
   */
  public boolean isJapanese() {
    return this == JAPANESE;
  }

  private static final Map<String, CJLanguage> BY_NAME;

  static {
    Map<String, CJLanguage> map = new HashMap<>();
    for (CJLanguage lang : values()) {
      map.put(lang.name().toLowerCase(), lang);
      if (!lang.isoCode.isEmpty()) {
        map.put(lang.isoCode, lang);
      }
      if (!lang.isoCode3.isEmpty()) {
        map.put(lang.isoCode3, lang);
      }
      for (String name : lang.altNames()) {
        map.put(name, lang);
      }
    }
    BY_NAME = map;
  }
}
