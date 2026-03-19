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

"""Ad-hoc CJ language detection tool.

Detects whether a phrase is Japanese, Chinese Simplified, or Chinese Traditional,
and prints detailed scoring results. Analogous to the 'adhoc' command in EvalTool.java.

Usage:
  python3 adhoc.py --phrase "羊驼是一种很好的动物。"
  python3 adhoc.py --modelfile ../path/to/cjlogprobs.gz --phrase "東京皇居"
  python3 adhoc.py --phrase "事務所へ行きます"
"""

import argparse
import sys

from cjclassifier.classifier import CJClassifier, Results


def main():
  parser = argparse.ArgumentParser(
      description="Ad-hoc CJ language detection")
  parser.add_argument(
      "--phrase", required=True,
      help="The phrase to detect")
  parser.add_argument(
      "--modelfile", default=None,
      help="Path to a combined logprob model file (default: bundled model)")
  parser.add_argument(
      "--logprobfloor", type=float, default=0.0,
      help="Log-probability floor (0 = use model file value)")
  args = parser.parse_args()

  detector = CJClassifier.load(path=args.modelfile, log_prob_floor=args.logprobfloor)

  results = Results()
  result = detector.detect(args.phrase, results)

  print(f"Result: {result.iso_code} ({result.name})" if result.iso_code else "Result: UNKNOWN")
  print(repr(results))


if __name__ == "__main__":
  main()
