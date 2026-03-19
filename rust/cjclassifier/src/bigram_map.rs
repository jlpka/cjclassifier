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

//! Memory-efficient open-addressing hash map for bigram -> float[LANG_COUNT] lookups.
//!
//! Each slot stores the key and probabilities together (16 bytes), so lookups
//! require only a single probe chain with no pointer chasing.

const LANG_COUNT: usize = 3;
const EMPTY: u32 = 0;
const MAX_LOAD_FACTOR: f32 = 0.75;

/// A single slot in the hash table: key + inlined probabilities.
#[derive(Clone, Copy)]
#[repr(C)]
struct Entry {
    key: u32,                  // 0 = empty
    probs: [f32; LANG_COUNT],  // per-language log-probabilities
}

impl Entry {
    const EMPTY: Entry = Entry {
        key: EMPTY,
        probs: [0.0; LANG_COUNT],
    };
}

/// Immutable bigram-to-probability map using open addressing with inlined values.
pub struct BigramMap {
    entries: Vec<Entry>,
    mask: usize,
    size: usize,
}

#[allow(dead_code)]
impl BigramMap {
    /// Look up the probabilities for a bigram key.
    /// Returns None if not present, or a reference to the LANG_COUNT floats.
    #[inline]
    pub fn get(&self, c1: u32, c2: u32) -> Option<&[f32; LANG_COUNT]> {
        let key = bigram_key(c1, c2);
        let mut idx = (mix(key) as usize) & self.mask;
        loop {
            let entry = &self.entries[idx];
            if entry.key == key {
                return Some(&entry.probs);
            }
            if entry.key == EMPTY {
                return None;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    /// Returns the number of entries.
    pub fn size(&self) -> usize {
        self.size
    }
}

/// Mutable builder for constructing a `BigramMap`.
pub struct BigramMapBuilder {
    entries: Vec<Entry>,
    size: usize,
    mask: usize,
    threshold: usize,
}

#[allow(dead_code)]
impl BigramMapBuilder {
    pub fn new(expected_size: usize) -> Self {
        let capacity = table_size_for(expected_size);
        BigramMapBuilder {
            entries: vec![Entry::EMPTY; capacity],
            size: 0,
            mask: capacity - 1,
            threshold: (capacity as f32 * MAX_LOAD_FACTOR) as usize,
        }
    }

    /// Store probabilities for a bigram key. The value is copied.
    pub fn put(&mut self, c1: u32, c2: u32, value: &[f32; LANG_COUNT]) {
        let key = bigram_key(c1, c2);
        assert!(key != EMPTY, "Key 0 is reserved as the empty sentinel");
        if self.size >= self.threshold {
            self.resize();
        }
        let mut idx = (mix(key) as usize) & self.mask;
        loop {
            let entry = &mut self.entries[idx];
            if entry.key == EMPTY {
                entry.key = key;
                entry.probs = *value;
                self.size += 1;
                return;
            }
            if entry.key == key {
                entry.probs = *value;
                return;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    /// Build an immutable `BigramMap`. The builder should not be used after this.
    pub fn build(self) -> BigramMap {
        BigramMap {
            entries: self.entries,
            mask: self.mask,
            size: self.size,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    fn resize(&mut self) {
        let new_capacity = self.entries.len() * 2;
        let old_entries = std::mem::replace(&mut self.entries, vec![Entry::EMPTY; new_capacity]);

        self.mask = new_capacity - 1;
        self.threshold = (new_capacity as f32 * MAX_LOAD_FACTOR) as usize;
        self.size = 0;

        for entry in &old_entries {
            if entry.key != EMPTY {
                self.rehash_put(entry.key, &entry.probs);
            }
        }
    }

    fn rehash_put(&mut self, key: u32, probs: &[f32; LANG_COUNT]) {
        let mut idx = (mix(key) as usize) & self.mask;
        loop {
            if self.entries[idx].key == EMPTY {
                self.entries[idx].key = key;
                self.entries[idx].probs = *probs;
                self.size += 1;
                return;
            }
            idx = (idx + 1) & self.mask;
        }
    }
}

#[inline]
fn bigram_key(c1: u32, c2: u32) -> u32 {
    (c1 << 16) | c2
}

#[inline]
fn mix(mut k: u32) -> u32 {
    k ^= k >> 16;
    k = k.wrapping_mul(0x85ebca6b);
    k ^= k >> 13;
    k
}

/// Return the smallest power of 2 that can hold expected_size at <=75% load.
fn table_size_for(expected_size: usize) -> usize {
    let min_capacity = (expected_size as f32 / MAX_LOAD_FACTOR) as usize + 1;
    let mut capacity = min_capacity.next_power_of_two();
    if capacity < min_capacity {
        capacity *= 2;
    }
    capacity.max(16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_put_and_get() {
        let mut builder = BigramMapBuilder::new(10);
        builder.put(0x4E00, 0x4E8C, &[-2.0, -8.0, -8.0]);
        builder.put(0x4E8C, 0x4E09, &[-8.0, -2.0, -8.0]);
        let map = builder.build();

        assert_eq!(map.size(), 2);

        let probs1 = map.get(0x4E00, 0x4E8C).unwrap();
        assert_eq!(probs1[0], -2.0);
        assert_eq!(probs1[1], -8.0);
        assert_eq!(probs1[2], -8.0);

        let probs2 = map.get(0x4E8C, 0x4E09).unwrap();
        assert_eq!(probs2[0], -8.0);
        assert_eq!(probs2[1], -2.0);

        // Missing key
        assert!(map.get(0x4E00, 0x4E09).is_none());
    }

    #[test]
    fn stress_many_entries() {
        let n = 10_000;
        let mut builder = BigramMapBuilder::new(n);
        for i in 0..n as u32 {
            let c1 = 0x4E00 + (i / 100);
            let c2 = 0x4E00 + (i % 100);
            builder.put(c1, c2, &[-(i as f32), -1.0, -1.0]);
        }
        let map = builder.build();
        assert_eq!(map.size(), n);

        for i in 0..n as u32 {
            let c1 = 0x4E00 + (i / 100);
            let c2 = 0x4E00 + (i % 100);
            let probs = map.get(c1, c2).unwrap();
            assert_eq!(probs[0], -(i as f32));
        }
    }
}
