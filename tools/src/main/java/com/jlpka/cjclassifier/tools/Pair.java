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

package com.jlpka.cjclassifier.tools;

import java.util.Objects;

/** A generic pair of two values. */
public class Pair<A, B> {
  private final A first;
  private final B second;

  public Pair(A first, B second) {
    this.first = first;
    this.second = second;
  }

  public static <A, B> Pair<A, B> of(A first, B second) {
    return new Pair<>(first, second);
  }

  public A first() {
    return first;
  }

  public B second() {
    return second;
  }

  @Override
  public int hashCode() {
    return Objects.hash(first, second);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) return true;
    if (!(obj instanceof Pair)) return false;
    Pair<?, ?> other = (Pair<?, ?>) obj;
    return Objects.equals(first, other.first) && Objects.equals(second, other.second);
  }
}
