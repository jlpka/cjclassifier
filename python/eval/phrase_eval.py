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

"""Phrase-level evaluation tool for CJ language detection.

Evaluates classifier accuracy against a file of test phrases with an expected
language. Analogous to the 'phraseeval' command in EvalTool.java.

Usage:
  python3 phrase_eval.py --infile phrases_zh.txt --language zh
  python3 phrase_eval.py --infile phrases_ja.txt --language ja --misses
  python3 phrase_eval.py --modelfile model.gz --infile phrases.txt --language zh-hant
"""

import argparse
import sys

from cjclassifier.classifier import CJClassifier, Results
from cjclassifier.language import CJLanguage


def is_equivalent(a: CJLanguage, b: CJLanguage) -> bool:
  """Check if two languages are equivalent.

  In Wikipedia and other corpora, simplified and traditional Chinese are mixed
  together, so treat both Chinese variants as equivalent.
  """
  return a == b or (a.is_chinese() and b.is_chinese())


def phrase_eval(detector: CJClassifier, infile: str,
                expected_language: CJLanguage, show_misses: bool):
  correct = 0
  total = 0

  with open(infile, "r", encoding="utf-8") as f:
    for line in f:
      line = line.rstrip("\n")
      if not line or line.isspace():
        continue
      total += 1
      results = Results()
      detector.detect(line, results)
      if is_equivalent(results.result, expected_language):
        correct += 1
      elif show_misses:
        phrase = line[:80] + "..." if len(line) > 80 else line
        detected = results.result.iso_code if results.result is not None else "null"
        print(
            f'  MISS [expected={expected_language.iso_code}'
            f' detected={detected}]'
            f' scores={{{repr(results)}}} "{phrase}"',
            file=sys.stderr)

  pct = 100.0 * correct / total if total > 0 else 0.0
  print(f"Overall: {correct}/{total} correct ({pct:.1f}%)", file=sys.stderr)


def main():
  parser = argparse.ArgumentParser(
      description="Evaluate CJ classifier accuracy against a phrase file")
  parser.add_argument(
      "--infile", required=True,
      help="File with phrases in the expected language, one per line")
  parser.add_argument(
      "--language", required=True,
      help="Expected language code (e.g. ja, zh, zh-hans, zh-hant)")
  parser.add_argument(
      "--misses", action="store_true",
      help="Print details for each misclassified phrase")
  parser.add_argument(
      "--modelfile", default=None,
      help="Path to a combined logprob model file (default: bundled model)")
  parser.add_argument(
      "--logprobfloor", type=float, default=0.0,
      help="Log-probability floor (0 = use model file value)")
  args = parser.parse_args()

  expected = CJLanguage.from_string(args.language)
  if expected is CJLanguage.UNKNOWN:
    print(f"Error: unknown language code: {args.language}", file=sys.stderr)
    sys.exit(1)

  detector = CJClassifier.load(path=args.modelfile, log_prob_floor=args.logprobfloor)
  phrase_eval(detector, args.infile, expected, args.misses)


if __name__ == "__main__":
  main()
