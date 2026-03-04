/*
 * Copyright 2026 Jeremy Lilley (jeremy@jlilley.net)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.jlpka.cjclassifier;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;
import java.util.zip.GZIPInputStream;

/**
 * Detects whether a CJ text is Japanese, Chinese Simplified, or Chinese Traditional based on
 * unigram and bigram log-probability models.
 *
 * <p>Instantiate with {@link #load()}. The contents will be cached as a singleton, so subsequent
 * calls will be free. The model takes 36.5MB in memory, but is only loaded once.
 *
 * <p>Call {@link #detect(CharSequence)} to classify text.
 *
 * <p>For more detail on results, use the {@link #detect(CharSequence, Results)} API.
 *
 * <p>Similarly, if text comes in chunks, it's possible to call {@link #addText(CharSequence,
 * Scores)} on incremental text, and then {@link #computeResult(Results)} at the end.
 *
 * <p>It is possible to use slightly less memory (e.g. load with logprobfloor=-12 for 25MB model
 * size; the default is -16 for 36.5MB) with less accuracy.
 *
 * <p>CJClassifier itself is thread-safe and keeps all mutable state in the {@link Results} and
 * {@link Scores} classes. Multiple threads may classify concurrently with the same classifier as
 * long as each uses its own {@link Results} instance.
 */
public class CJClassifier {

  private static final Logger logger = Logger.getLogger(CJClassifier.class.getName());

  // Singleton cache to avoid loading the same model file twice.
  private static final Map<String, CJClassifier> cache = new HashMap<>();

  /** Default classpath resource name for the bundled model. */
  private static final String DEFAULT_MODEL_RESOURCE = "cjlogprobs.gz";

  /**
   * Returns a cached CJClassifier loaded from the bundled classpath resource. Uses the MinLogProb
   * value from the file header. Thread-safe: callers may invoke from any thread.
   *
   * @return the cached (or newly-loaded) classifier instance
   */
  public static synchronized CJClassifier load() throws IOException {
    return load(0.0);
  }

  /**
   * Returns a cached CJClassifier loaded from the bundled classpath resource with a custom
   * logProbFloor. If logProbFloor is 0, uses the file's MinLogProb header value. Thread-safe.
   *
   * @return the cached (or newly-loaded) classifier instance
   */
  public static synchronized CJClassifier load(double logProbFloor) throws IOException {
    String cacheKey = "classpath:default:" + logProbFloor;
    CJClassifier cached = cache.get(cacheKey);
    if (cached != null) {
      return cached;
    }
    InputStream in = CJClassifier.class.getResourceAsStream(DEFAULT_MODEL_RESOURCE);
    if (in == null) {
      throw new IOException("Bundled model not found on classpath: " + DEFAULT_MODEL_RESOURCE);
    }
    CJClassifier instance =
        new CJClassifier(in, "classpath:" + DEFAULT_MODEL_RESOURCE, logProbFloor);
    cache.put(cacheKey, instance);
    return instance;
  }

  /**
   * Returns a cached CJClassifier for the given path, loading it if necessary. Uses the MinLogProb
   * value from the file header. Thread-safe: callers may invoke from any thread.
   *
   * @return the cached (or newly-loaded) classifier instance
   */
  public static synchronized CJClassifier load(String path) throws IOException {
    return load(path, 0.0);
  }

  /**
   * Returns a cached CJClassifier for the given path and logProbFloor, loading it if necessary. If
   * logProbFloor is 0, uses the file's MinLogProb header value (IllegalArgumentException if
   * absent). Otherwise the logProbFloor allows restricting to a higher floor than what's in the
   * file.
   *
   * @return the cached (or newly-loaded) classifier instance
   */
  public static synchronized CJClassifier load(String path, double logProbFloor)
      throws IOException {
    String cacheKey = path + ":" + logProbFloor;
    CJClassifier cached = cache.get(cacheKey);
    if (cached != null) {
      return cached;
    }
    CJClassifier instance = new CJClassifier(path, logProbFloor);
    cache.put(cacheKey, instance);
    return instance;
  }

  /** Clears the loaded model(s), primarily for testing. */
  public static synchronized void clearCachedModels() {
    cache.clear();
  }

  // ========================================================================
  // Model data
  // ========================================================================

  private static final int CJ_RANGE_START = 0x3400;
  private static final int CJ_RANGE_END = 0x9FFF;
  private static final int CJ_RANGE_SIZE = CJ_RANGE_END - CJ_RANGE_START + 1;
  private static final int LANG_COUNT = 3; // zh-hans=0, zh-hant=1, ja=2

  static final CJLanguage[] CJ_LANGUAGES = {
    CJLanguage.CHINESE_SIMPLIFIED, CJLanguage.CHINESE_TRADITIONAL, CJLanguage.JAPANESE
  };

  /** The three CJ languages in index order: zh-hans, zh-hant, ja. */
  public static final List<CJLanguage> CJ_LANGUAGES_LIST =
      Collections.unmodifiableList(Arrays.asList(CJ_LANGUAGES));

  /**
   * unigramLogProbs[(charIndex*LANG_COUNT) + langIndex] — log-probabilities per language for each
   * unigram
   */
  private final float[] unigramLogProbs;

  /** bigramLogProbs: bigramKey -> offset into flat probData array (0 = absent) */
  private final BigramToFloatArrayMap bigramLogProbs;

  /** Cached reference to bigramLogProbs.probData() for fast inner-loop access. */
  private final float[] bigramProbData;

  /** Default log-probability used when a bigram is not in the model */
  private final double defaultLogProb;

  /** Tolerated kana fraction threshold for Chinese, see setToleratedKanaThreshold() */
  private double toleratedKanaThreshold = 0.01;

  /**
   * Typically any Hiragana or Katakana should be classified as Japanese, since these are characters
   * in Japanese but not Chinese.
   *
   * <p>But if e.g. we see a few Hiragana characters vs 1000 Ideographs (e.g. a Chinese article with
   * an embedded Japanese quote), we shouldn't automatically tag the article as Japanese strictly on
   * this basis, but actually analyze the ideographs.
   *
   * <p>For context, Hiragana is often ~50% of the characters in a typical Japanese text.
   *
   * <p>This value should rightly be set quite low, we default to 1%.
   */
  public void setToleratedKanaThreshold(double threshold) {
    this.toleratedKanaThreshold = threshold;
  }

  /**
   * @return the current kana fraction threshold
   */
  public double getToleratedKanaThreshold() {
    return toleratedKanaThreshold;
  }

  // ========================================================================
  // Constructor
  // ========================================================================

  /**
   * Load a combined log-probability model file (possibly gzipped). Uses the MinLogProb value from
   * the file header (IllegalArgumentException if absent). Prefer {@link #load(String)}.
   *
   * @param filename path to the combine output file (with --aslogprob)
   */
  CJClassifier(String filename) throws IOException {
    this(filename, 0.0);
  }

  /**
   * Load from a filesystem path.
   *
   * @param logProbFloor if 0, uses the file's MinLogProb header value; otherwise uses the given
   *     floor (max of this and the file's value).
   */
  CJClassifier(String filename, double logProbFloor) throws IOException {
    this(
        new BufferedInputStream(new FileInputStream(filename), 1024 * 1024),
        filename,
        logProbFloor);
  }

  /**
   * Load from an already-opened InputStream (e.g. from classpath or other source).
   *
   * @param in the input stream to read from (will be closed when done)
   * @param label a human-readable label for error messages and logging (e.g. filename or classpath
   *     location)
   * @param logProbFloor if 0, uses the file's MinLogProb header value; otherwise uses the given
   *     floor (max of this and the file's value).
   */
  CJClassifier(InputStream in, String label, double logProbFloor) throws IOException {
    long startTime = System.nanoTime();
    this.unigramLogProbs = new float[CJ_RANGE_SIZE * LANG_COUNT];
    BigramToFloatArrayMap.Builder bigramBuilder = new BigramToFloatArrayMap.Builder(16 << 10);

    try (BufferedReader reader =
        new BufferedReader(
            new InputStreamReader(maybeDecompress(in, label), StandardCharsets.UTF_8),
            1024 * 1024)) {

      // Parse header: Languages: ja,zh-hans,zh-hant UnigramTotals: ... BigramTotals: ...
      // MinLogProb: -16.0
      String header = reader.readLine();
      if (header == null || !header.startsWith("Languages: ")) {
        throw new IllegalArgumentException("Invalid model file (bad header): " + label);
      }
      String[] headerParts = header.split(" ");
      String[] langCodes = headerParts[1].split(",");
      int[] langMap = new int[langCodes.length];
      for (int i = 0; i < langCodes.length; i++) {
        CJLanguage lang = CJLanguage.fromString(langCodes[i]);
        int li = cjLangIndex(lang);
        if (li < 0) {
          throw new IllegalArgumentException(
              "Unknown CJ language in header: " + langCodes[i] + " in " + label);
        }
        langMap[i] = li;
      }

      // Parse MinLogProb from header if present.
      double parsedMinProb = Double.NaN;
      for (int i = 0; i < headerParts.length - 1; i++) {
        if ("MinLogProb:".equals(headerParts[i])) {
          parsedMinProb = Double.parseDouble(headerParts[i + 1]);
          break;
        }
      }
      if (logProbFloor == 0.0) {
        // Use file's MinLogProb; require it to be present.
        if (Double.isNaN(parsedMinProb)) {
          throw new IllegalArgumentException(
              "No MinLogProb in header and no explicit logProbFloor: " + label);
        }
        this.defaultLogProb = parsedMinProb;
      } else {
        // Explicit floor: use max of floor and file value (if present).
        this.defaultLogProb =
            Double.isNaN(parsedMinProb) ? logProbFloor : Math.max(logProbFloor, parsedMinProb);
      }

      // Read unigram and bigram lines
      String line;
      int unigramCount = 0, bigramCount = 0;
      float[] probs = new float[LANG_COUNT]; // reused across bigram lines (builder copies)
      while ((line = reader.readLine()) != null) {
        if (line.isEmpty()) continue;
        String[] parts = line.split(" ");
        String key = parts[0];
        if (parts.length != langMap.length + 1) {
          throw new IllegalArgumentException(
              "Column count mismatch on line: " + line + " in " + label);
        }
        if (key.length() == 1) {
          // Unigram
          char c = key.charAt(0);
          int idx = c - CJ_RANGE_START;
          if (idx >= 0 && idx < CJ_RANGE_SIZE) {
            for (int col = 0; col < langMap.length; col++) {
              unigramLogProbs[(idx * LANG_COUNT) + langMap[col]] =
                  (float) Math.max(Float.parseFloat(parts[col + 1]), this.defaultLogProb);
            }
            unigramCount++;
          }
        } else if (key.length() == 2) {
          // Bigram
          Arrays.fill(probs, 0);
          boolean anyHigherThanDefault = false;
          for (int col = 0; col < langMap.length; col++) {
            float val = Float.parseFloat(parts[col + 1]);
            if (val < this.defaultLogProb) {
              val = (float) this.defaultLogProb;
            } else {
              anyHigherThanDefault = true;
            }
            probs[langMap[col]] = val;
          }
          if (anyHigherThanDefault) {
            bigramBuilder.put(key.charAt(0), key.charAt(1), probs);
            bigramCount++;
          }
        }
      }

      double elapsedSec = (System.nanoTime() - startTime) / 1_000_000_000.0;
      logger.info(
          String.format(
              "Loaded %d unigrams and %d bigrams, minLogProb: %.2f, from %s [%.1fs]",
              unigramCount, bigramCount, this.defaultLogProb, label, elapsedSec));
    }

    this.bigramLogProbs = bigramBuilder.build();
    this.bigramProbData = this.bigramLogProbs.probData();
  }

  // ========================================================================
  // Detection
  // ========================================================================

  /**
   * Detect language, only returning the language, rather than the Results details.
   *
   * @return the detected {@link CJLanguage}, or {@link CJLanguage#UNKNOWN} if undetermined
   */
  public CJLanguage detect(CharSequence text) {
    Results results = new Results();
    addText(text, results.scores);
    return computeResult(results);
  }

  /**
   * Detect language, populating the provided Results object with scoring info.
   *
   * @return the detected {@link CJLanguage}, or {@link CJLanguage#UNKNOWN} if undetermined
   */
  public CJLanguage detect(CharSequence text, Results results) {
    results.clear();
    addText(text, results.scores);
    return computeResult(results);
  }

  /**
   * A lower-level api, where one can add multiple sequences to be detected to the provided scores
   * object from a Results class.
   *
   * <p>When we want the result, we should call computeResult(). The main advantage is allowing
   * incremental computation.
   */
  public void addText(CharSequence text, Scores scores) {
    char prev = 0;
    boolean prevInRange = false;

    final int len = text.length();
    for (int i = 0; i < len; i++) {
      char c = text.charAt(i);
      if (isKana(c)) {
        scores.kanaCount++;
        continue;
      }
      boolean inRange = inMainCJRange(c);
      if (inRange) {
        scores.cjCharCount++;
        calcWithPrev(c, prev, prevInRange, scores);
      }
      prev = c;
      prevInRange = inRange;
    }
  }

  /** {@code char[]}-based variant of {@link #addText(CharSequence, Scores)}. */
  public void addText(char[] text, int ofs, int len, Scores scores) {
    char prev = 0;
    boolean prevInRange = false;

    for (int i = 0; i < len; i++) {
      char c = text[i + ofs];
      if (isKana(c)) {
        scores.kanaCount++;
        continue;
      }
      boolean inRange = inMainCJRange(c);
      if (inRange) {
        scores.cjCharCount++;
        calcWithPrev(c, prev, prevInRange, scores);
      }
      prev = c;
      prevInRange = inRange;
    }
  }

  /**
   * Sums the underlying scores and computes the language result.
   *
   * @return the detected {@link CJLanguage}, or {@link CJLanguage#UNKNOWN} if undetermined
   */
  public CJLanguage computeResult(Results results) {
    if (results.scores.kanaCount > 0) {
      double kanaRatio =
          (double) results.scores.kanaCount
              / (results.scores.kanaCount + results.scores.cjCharCount);
      if (kanaRatio > toleratedKanaThreshold) {
        results.result = CJLanguage.JAPANESE;
        results.gap = 1.0;
        results.totalScores[cjLangIndex(CJLanguage.JAPANESE)] = 1.0;
        return results.result;
      }
    }
    if (!results.scores.anyHits()) {
      results.result = CJLanguage.UNKNOWN;
      results.gap = 0;
      return results.result;
    }
    results.computeTotals(defaultLogProb);

    // Find best and second-best language
    int bestIdx = 0;
    int secondIdx = -1;
    for (int li = 1; li < LANG_COUNT; li++) {
      if (results.totalScores[li] > results.totalScores[bestIdx]) {
        secondIdx = bestIdx;
        bestIdx = li;
      } else if (secondIdx < 0 || results.totalScores[li] > results.totalScores[secondIdx]) {
        secondIdx = li;
      }
    }
    results.result = CJ_LANGUAGES[bestIdx];

    // Compute gap: 1 - (best / second). Scores are negative logprobs, so best is least negative.
    double best = results.totalScores[bestIdx];
    double second = secondIdx >= 0 ? results.totalScores[secondIdx] : best;
    results.gap = (second != 0.0) ? 1.0 - (best / second) : 0.0;

    return results.result;
  }

  // ========================================================================
  // Scores: accumulated during addText
  // ========================================================================

  /** Accumulates per-language scoring data during CJ text processing. */
  public static class Scores {
    public final double[] unigramScores = new double[LANG_COUNT];
    public final double[] bigramScores = new double[LANG_COUNT];
    public final int[] unigramHitsPerLang = new int[LANG_COUNT];
    public final int[] bigramHitsPerLang = new int[LANG_COUNT];

    /** Number of kana characters seen. */
    public int kanaCount;

    /** Number of characters in the main CJ range (0x3400-0x9FFF). */
    public int cjCharCount;

    public void clear() {
      Arrays.fill(unigramScores, 0);
      Arrays.fill(bigramScores, 0);
      Arrays.fill(unigramHitsPerLang, 0);
      Arrays.fill(bigramHitsPerLang, 0);
      kanaCount = 0;
      cjCharCount = 0;
    }

    public boolean anyHits() {
      return maxVal(unigramHitsPerLang) > 0;
    }

    static int maxVal(int[] a) {
      int maxVal = a[0];
      for (int i = 1; i < a.length; ++i) {
        if (a[i] > maxVal) {
          maxVal = a[i];
        }
      }
      return maxVal;
    }
  }

  // ========================================================================
  // Results: computed after scoring
  // ========================================================================

  /** Detection results, including accumulated {@link Scores} and the computed result. */
  public static class Results {
    public final Scores scores;
    public final double[] totalScores = new double[LANG_COUNT];

    /**
     * Per-language boosts, in the range 0..1.0 where 0 is no boost and 1.0 is a heavy boost. A
     * boost makes the corresponding language more likely to win. For example, 0.05 might be a
     * reasonable boost to help favor traditional vs simplified Chinese.
     *
     * <p>Boosts are applied during {@link CJClassifier#computeResult(Results)}. Note that {@link
     * CJClassifier#detect(CharSequence)} calls {@link #clear()}, so boosts must be set when using
     * the lower-level {@link CJClassifier#addText(CharSequence, Scores)} / {@link
     * CJClassifier#computeResult(Results)} API.
     */
    public final double[] boosts = new double[LANG_COUNT];

    public CJLanguage result;

    /** Gap between best and runner-up (0 = dead heat, 1 = no contest). */
    public double gap;

    /** Creates a Results with its own internal Scores. */
    public Results() {
      this.scores = new Scores();
    }

    /** Creates a Results that wraps an externally-owned Scores instance. */
    public Results(Scores scores) {
      this.scores = scores;
    }

    public void clear() {
      scores.clear();
      Arrays.fill(totalScores, 0);
      Arrays.fill(boosts, 0);
      result = null;
      gap = 0;
    }

    void computeTotals(double placeholderScore) {
      int maxUnigram = Scores.maxVal(scores.unigramHitsPerLang);
      int maxBigram = Scores.maxVal(scores.bigramHitsPerLang);
      for (int i = 0; i < LANG_COUNT; i++) {
        // Add bigram + unigram scores. Plus, normalize for missing entries.
        totalScores[i] =
            scores.unigramScores[i]
                + (maxUnigram - scores.unigramHitsPerLang[i]) * placeholderScore
                + scores.bigramScores[i]
                + (maxBigram - scores.bigramHitsPerLang[i]) * placeholderScore;
        // Implement "boosts." Since logprob values are negative, a favorable boost is negative.
        totalScores[i] -= boosts[i] * totalScores[i];
      }
    }

    @Override
    public String toString() {
      StringBuilder sb = new StringBuilder();
      sb.append("Results{result=").append(result != null ? result.isoCode() : "null");
      if (scores.kanaCount > 0) {
        sb.append(
            String.format(" kana=%d/%d", scores.kanaCount, scores.kanaCount + scores.cjCharCount));
      }
      for (int li = 0; li < LANG_COUNT; li++) {
        sb.append(
            String.format(
                " | %s: uni=%.2f bi=%.2f total=%.2f biHits=%d",
                CJ_LANGUAGES[li].isoCode(),
                scores.unigramScores[li],
                scores.bigramScores[li],
                totalScores[li],
                scores.bigramHitsPerLang[li]));
      }
      sb.append('}');
      return sb.toString();
    }

    /**
     * Returns a compact, comma-separated representation of the per-language relative scores, e.g.
     * {@code "zh-hans:1.00,zh-hant:0.97,ja:0.85"}, ordered from best to worst. Scores are expressed
     * as ratios of the best language's totalScore. Returns an empty string if no result has been
     * computed or the result is unknown.
     *
     * @return the short score string, or {@code ""} if unavailable
     */
    public String toShortString() {
      if (result == null || result == CJLanguage.UNKNOWN) return "";
      if (scores.kanaCount > 0 && result == CJLanguage.JAPANESE) {
        return "ja:1.0,zh-hans:0,zh-hant:0";
      }

      // Sort indices by totalScores descending
      Integer[] order = {0, 1, 2};
      Arrays.sort(order, (a, b) -> Double.compare(totalScores[b], totalScores[a]));

      double best = totalScores[order[0]];
      if (best == 0) return "";

      StringBuilder sb = new StringBuilder();
      for (int i = 0; i < LANG_COUNT; i++) {
        if (i > 0) sb.append(',');
        int li = order[i];
        sb.append(CJ_LANGUAGES[li].isoCode()).append(':');
        sb.append(String.format("%.2f", totalScores[li] != 0 ? best / totalScores[li] : 0));
      }
      return sb.toString();
    }
  }

  private void calcWithPrev(char c, char prev, boolean prevInRange, Scores scores) {
    int idx = c - CJ_RANGE_START;
    for (int li = 0; li < LANG_COUNT; li++) {
      float uProb = unigramLogProbs[(idx * LANG_COUNT) + li];
      scores.unigramScores[li] += Math.max(uProb, defaultLogProb);
      if (uProb != 0) scores.unigramHitsPerLang[li]++;
    }
    if (prevInRange) {
      int bOff = bigramLogProbs.getOffset(prev, c);
      if (bOff != 0) {
        for (int li = 0; li < LANG_COUNT; li++) {
          float bp = bigramProbData[bOff + li];
          scores.bigramScores[li] += Math.max(bp, defaultLogProb);
          if (bp != 0) scores.bigramHitsPerLang[li]++;
        }
      }
    }
  }

  // ========================================================================
  // Character helpers
  // ========================================================================

  private static boolean isKana(char c) {
    if (c >= 0x3040 && c <= 0x30FF) return true;
    if (c >= 0x31F0 && c <= 0x31FF) return true;
    if (c >= 0xFF65 && c <= 0xFF9F) return true;
    return false;
  }

  private static boolean inMainCJRange(char c) {
    return c >= CJ_RANGE_START && c <= CJ_RANGE_END;
  }

  /** Wraps an InputStream with GZIPInputStream if the label ends with .gz. */
  private static InputStream maybeDecompress(InputStream in, String label) throws IOException {
    InputStream buffered = in.markSupported() ? in : new BufferedInputStream(in, 1024 * 1024);
    if (label.endsWith(".gz")) {
      return new GZIPInputStream(buffered, 1024 * 1024);
    }
    return buffered;
  }

  /**
   * Returns the internal array index for the given CJ language.
   *
   * @return 0 for zh-hans, 1 for zh-hant, 2 for ja, or -1 for unknown/unsupported
   */
  public static int cjLangIndex(CJLanguage lang) {
    switch (lang) {
      case CHINESE_SIMPLIFIED:
        return 0;
      case CHINESE_TRADITIONAL:
        return 1;
      case JAPANESE:
        return 2;
      default:
        return -1;
    }
  }
}
