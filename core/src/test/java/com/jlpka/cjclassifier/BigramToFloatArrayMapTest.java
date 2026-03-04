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

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class BigramToFloatArrayMapTest {

  // ========================================================================
  // Basic put / get
  // ========================================================================

  @Test
  void putAndGetSingleEntry() {
    BigramToFloatArrayMap.Builder b = new BigramToFloatArrayMap.Builder(16);
    float[] val = {1.0f, 2.0f, 3.0f};
    b.put('a', 'c', val);
    assertEquals(1, b.size());
    BigramToFloatArrayMap map = b.build();
    int off = map.getOffset('a', 'c');
    assertNotEquals(0, off);
    float[] pd = map.probData();
    assertEquals(1.0f, pd[off]);
    assertEquals(2.0f, pd[off + 1]);
    assertEquals(3.0f, pd[off + 2]);
    assertEquals(1, map.size());
  }

  @Test
  void getMissingKeyReturnsZero() {
    BigramToFloatArrayMap.Builder b = new BigramToFloatArrayMap.Builder(16);
    BigramToFloatArrayMap map = b.build();
    assertEquals(0, map.getOffset('a', 'a'));
  }

  @Test
  void putOverwritesExistingKey() {
    BigramToFloatArrayMap.Builder b = new BigramToFloatArrayMap.Builder(16);
    float[] v1 = {1.0f, 0.0f, 0.0f};
    float[] v2 = {2.0f, 0.0f, 0.0f};
    b.put('a', 'b', v1);
    b.put('a', 'b', v2);
    assertEquals(1, b.size());
    BigramToFloatArrayMap map = b.build();
    int off = map.getOffset('a', 'b');
    assertNotEquals(0, off);
    assertEquals(2.0f, map.probData()[off]);
    assertEquals(1, map.size());
  }

  // ========================================================================
  // Key 0 is sentinel
  // ========================================================================

  @Test
  void putKeyZeroThrows() {
    BigramToFloatArrayMap.Builder b = new BigramToFloatArrayMap.Builder(16);
    assertThrows(
        IllegalArgumentException.class, () -> b.put('\0', '\0', new float[] {1.0f, 0.0f, 0.0f}));
  }

  @Test
  void getKeyZeroReturnsZero() {
    // Key 0 is the EMPTY sentinel — getOffset(0) will hit an EMPTY slot and return 0
    BigramToFloatArrayMap.Builder b = new BigramToFloatArrayMap.Builder(16);
    b.put('a', 'b', new float[] {1.0f, 0.0f, 0.0f});
    BigramToFloatArrayMap map = b.build();
    assertEquals(0, map.getOffset('\0', '\0'));
  }

  // ========================================================================
  // Resizing
  // ========================================================================

  @Test
  void resizePreservesEntries() {
    // Start small, force multiple resizes
    BigramToFloatArrayMap.Builder b = new BigramToFloatArrayMap.Builder(4);
    int count = 1000;
    for (int i = 1; i <= count; i++) {
      b.put('a', (char) i, new float[] {(float) i, 0.0f, 0.0f});
    }
    assertEquals(count, b.size());
    BigramToFloatArrayMap map = b.build();
    assertEquals(count, map.size());
    float[] pd = map.probData();
    // Verify all entries survived resizing
    for (int i = 1; i <= count; i++) {
      int off = map.getOffset('a', (char) i);
      assertNotEquals(0, off, "Missing key: " + i);
      assertEquals((float) i, pd[off]);
    }
  }

  // ========================================================================
  // Bigram-style keys (the actual use case)
  // ========================================================================

  @Test
  void bigramKeyPattern() {
    // Simulate bigram keys: (c1 << 16) | c2 with c1,c2 in CJ range (0x4E00-0x9FFF)
    BigramToFloatArrayMap.Builder b = new BigramToFloatArrayMap.Builder(256);
    int base = 0x4E00;
    int count = 0;
    for (int c1 = base; c1 < base + 20; c1++) {
      for (int c2 = base; c2 < base + 20; c2++) {
        b.put((char) c1, (char) c2, new float[] {(float) c1, (float) c2, 0.0f});
        count++;
      }
    }
    assertEquals(count, b.size());
    BigramToFloatArrayMap map = b.build();
    assertEquals(count, map.size());
    float[] pd = map.probData();

    // Verify all retrievable
    for (int c1 = base; c1 < base + 20; c1++) {
      for (int c2 = base; c2 < base + 20; c2++) {
        int off = map.getOffset((char) c1, (char) c2);
        assertNotEquals(0, off);
        assertEquals((float) c1, pd[off]);
        assertEquals((float) c2, pd[off + 1]);
      }
    }

    // Non-existent bigram key should return 0
    assertEquals(0, map.getOffset((char) 0x9FFF, (char) 0x9FFF));
  }

  // ========================================================================
  // Edge cases
  // ========================================================================

  @Test
  void emptyMapSizeIsZero() {
    BigramToFloatArrayMap.Builder b = new BigramToFloatArrayMap.Builder(16);
    BigramToFloatArrayMap map = b.build();
    assertEquals(0, map.size());
  }

  @Test
  void expectedSizeZeroStillWorks() {
    BigramToFloatArrayMap.Builder b = new BigramToFloatArrayMap.Builder(0);
    b.put('a', 'b', new float[] {1.0f, 2.0f, 3.0f});
    BigramToFloatArrayMap map = b.build();
    int off = map.getOffset('a', 'b');
    assertNotEquals(0, off);
    assertEquals(1.0f, map.probData()[off]);
  }
}
