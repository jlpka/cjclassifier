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

/**
 * Functional equivalent of {@code Map<Pair<Character, Character>, float[]>} except that we avoid
 * Object boxing and do open addressing. This is both faster and avoids Integer allocs on a
 * per-lookup basis.
 *
 * <p>Probabilities are stored in a single flat {@code float[]} ({@link #probData()}) to avoid one
 * small {@code float[LANG_COUNT]} object per entry. Each hash-table slot stores an int index into
 * probData (0&nbsp;=&nbsp;empty/absent; real entries start at offset&nbsp;1).
 */
class BigramToFloatArrayMap {
  private static final int EMPTY = 0;
  private static final int LANG_COUNT = 3;
  private static final float MAX_LOAD_FACTOR = 0.75f;

  private final int[] keys;
  private final int[] valueIndices; // index into probData, 0 = empty
  private final float[] probData; // flat: entries at offsets 1, 1+LANG_COUNT, ...
  private final int size;
  private final int mask;

  private BigramToFloatArrayMap(
      int[] keys, int[] valueIndices, float[] probData, int size, int mask) {
    this.keys = keys;
    this.valueIndices = valueIndices;
    this.probData = probData;
    this.size = size;
    this.mask = mask;
  }

  /**
   * Look up the offset in {@link #probData()} for a bigram key.
   *
   * @return the offset into {@link #probData()}, or 0 if not present. When non-zero, the {@code
   *     LANG_COUNT} floats starting at that offset are the per-language log-probabilities.
   */
  public int getOffset(char c1, char c2) {
    int key = bigramKey(c1, c2);
    int idx = mix(key) & mask;
    while (true) {
      int k = keys[idx];
      if (k == key) {
        return valueIndices[idx];
      }
      if (k == EMPTY) {
        return 0;
      }
      idx = (idx + 1) & mask;
    }
  }

  /**
   * Returns the flat probability data array.
   *
   * @return the backing {@code float[]} containing all per-language log-probabilities
   */
  public float[] probData() {
    return probData;
  }

  /**
   * Returns the number of entries in this map.
   *
   * @return the entry count
   */
  public int size() {
    return size;
  }

  private static int bigramKey(char c1, char c2) {
    return ((int) c1 << 16) | (int) c2;
  }

  // Mix/hash function to spread bits of the key.
  private static int mix(int k) {
    k ^= (k >>> 16);
    k *= 0x85ebca6b;
    k ^= (k >>> 13);
    return k;
  }

  /** Return the smallest power of 2 that can hold expectedSize at &lt;=75% load. */
  private static int tableSizeFor(int expectedSize) {
    int minCapacity = (int) (expectedSize / MAX_LOAD_FACTOR) + 1;
    int capacity = Integer.highestOneBit(minCapacity);
    if (capacity < minCapacity) {
      capacity <<= 1;
    }
    return Math.max(capacity, 16);
  }

  // ========================================================================
  // Builder
  // ========================================================================

  /** Mutable builder for constructing a {@link BigramToFloatArrayMap}. */
  public static class Builder {
    private int[] keys;
    private int[] valueIndices;
    private float[] probData;
    private int probDataSize; // next free offset in probData (starts at 1)
    private int size;
    private int mask;
    private int threshold;

    public Builder(int expectedSize) {
      int capacity = tableSizeFor(expectedSize);
      keys = new int[capacity];
      valueIndices = new int[capacity];
      mask = capacity - 1;
      threshold = (int) (capacity * MAX_LOAD_FACTOR);
      probData = new float[expectedSize * LANG_COUNT + 1];
      probDataSize = 1; // offset 0 is reserved (means "empty")
    }

    /**
     * Store a value for the given key bigram. The contents of {@code value} are copied into the
     * flat probData array; the caller may reuse the array afterwards.
     */
    public void put(char c1, char c2, float[] value) {
      put(bigramKey(c1, c2), value);
    }

    private void put(int key, float[] value) {
      if (key == EMPTY) {
        throw new IllegalArgumentException("Key 0 is reserved as the empty sentinel");
      }
      if (size >= threshold) {
        resize();
      }
      int idx = mix(key) & mask;
      while (true) {
        int k = keys[idx];
        if (k == EMPTY) {
          keys[idx] = key;
          valueIndices[idx] = appendProbData(value);
          size++;
          return;
        }
        if (k == key) {
          // Update existing entry in-place
          System.arraycopy(value, 0, probData, valueIndices[idx], LANG_COUNT);
          return;
        }
        idx = (idx + 1) & mask;
      }
    }

    private int appendProbData(float[] value) {
      if (probDataSize + LANG_COUNT > probData.length) {
        float[] grown = new float[Math.max(probData.length * 2, probDataSize + LANG_COUNT)];
        System.arraycopy(probData, 0, grown, 0, probDataSize);
        probData = grown;
      }
      int off = probDataSize;
      System.arraycopy(value, 0, probData, off, LANG_COUNT);
      probDataSize += LANG_COUNT;
      return off;
    }

    /**
     * Returns the number of entries added so far.
     *
     * @return the current entry count
     */
    public int size() {
      return size;
    }

    /**
     * Build an immutable {@link BigramToFloatArrayMap}. The builder should not be used after this
     * call.
     *
     * @return a new immutable map containing all entries added via {@link #put}
     */
    public BigramToFloatArrayMap build() {
      // Trim probData to exact size
      if (probData.length != probDataSize) {
        float[] trimmed = new float[probDataSize];
        System.arraycopy(probData, 0, trimmed, 0, probDataSize);
        probData = trimmed;
      }
      BigramToFloatArrayMap map =
          new BigramToFloatArrayMap(keys, valueIndices, probData, size, mask);
      keys = null;
      valueIndices = null;
      probData = null;
      return map;
    }

    private void resize() {
      int newCapacity = keys.length << 1;
      int[] oldKeys = keys;
      int[] oldValueIndices = valueIndices;

      keys = new int[newCapacity];
      valueIndices = new int[newCapacity];
      mask = newCapacity - 1;
      threshold = (int) (newCapacity * MAX_LOAD_FACTOR);
      size = 0;

      for (int i = 0; i < oldKeys.length; i++) {
        if (oldKeys[i] != EMPTY) {
          rehashPut(oldKeys[i], oldValueIndices[i]);
        }
      }
    }

    /** Insert during rehash — probData offsets are unchanged. */
    private void rehashPut(int key, int valueIndex) {
      int idx = mix(key) & mask;
      while (true) {
        if (keys[idx] == EMPTY) {
          keys[idx] = key;
          valueIndices[idx] = valueIndex;
          size++;
          return;
        }
        idx = (idx + 1) & mask;
      }
    }
  }
}
