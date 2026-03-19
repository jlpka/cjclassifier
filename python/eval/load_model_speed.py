#!/usr/bin/env python3
# Copyright 2026 Jeremy Lilley (jeremy@jlilley.net)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark model loading speed and memory consumption.

Analogous to the 'loadmodelspeed' command in EvalTool.java.

Usage:
  python3 load_model_speed.py
  python3 load_model_speed.py --modelfile ../path/to/cjlogprobs.gz
  python3 load_model_speed.py --duration 60
"""

import argparse
import gc
import sys
import time
import tracemalloc

from cjclassifier.classifier import CJClassifier


def load_model(modelfile, logprobfloor):
  """Load the model, bypassing the cache."""
  return CJClassifier.load(path=modelfile, log_prob_floor=logprobfloor)


def main():
  parser = argparse.ArgumentParser(
      description="Benchmark CJ classifier model loading speed and memory")
  parser.add_argument(
      "--modelfile", default=None,
      help="Path to a combined logprob model file (default: bundled model)")
  parser.add_argument(
      "--logprobfloor", type=float, default=0.0,
      help="Log-probability floor (0 = use model file value)")
  parser.add_argument(
      "--duration", type=int, default=10,
      help="Benchmark duration in seconds (default: 10)")
  args = parser.parse_args()

  # Warmup run with memory measurement
  print("Warmup...", file=sys.stderr)
  tracemalloc.start()
  gc.collect()
  before = tracemalloc.take_snapshot()

  load_model(args.modelfile, args.logprobfloor)

  gc.collect()
  after = tracemalloc.take_snapshot()
  stats = after.compare_to(before, "filename")
  model_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)
  print(f"** MODEL MEMORY: {model_bytes / (1024 * 1024):.1f}MB", file=sys.stderr)

  tracemalloc.stop()
  CJClassifier.clear_cached_models()

  deadline = time.monotonic() + args.duration
  iteration = 0
  total_sec = 0.0
  min_sec = float("inf")
  max_sec = 0.0

  while time.monotonic() < deadline:
    iteration += 1
    t0 = time.perf_counter()
    load_model(args.modelfile, args.logprobfloor)
    sec = time.perf_counter() - t0
    CJClassifier.clear_cached_models()
    total_sec += sec
    min_sec = min(min_sec, sec)
    max_sec = max(max_sec, sec)
    print(f"  iteration {iteration}: {sec:.3f}s", file=sys.stderr)

  if iteration > 0:
    avg_sec = total_sec / iteration
    print(file=sys.stderr)
    print(
        f"Benchmark results: {iteration} iterations in {total_sec:.1f}s",
        file=sys.stderr)
    print(
        f"  avg={avg_sec:.3f}s  min={min_sec:.3f}s  max={max_sec:.3f}s",
        file=sys.stderr)


if __name__ == "__main__":
  main()
