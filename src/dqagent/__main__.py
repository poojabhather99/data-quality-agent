"""CLI: python -m dqagent data.csv [--llm]"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from .proposer import groq_complete, propose
from .rules import check


def main() -> None:
    ap = argparse.ArgumentParser(prog="dqagent")
    ap.add_argument("csv")
    ap.add_argument("--llm", action="store_true", help="use Groq (needs GROQ_API_KEY)")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    rules, rejected = propose(df, groq_complete if a.llm else None)
    print(json.dumps({"rules": check(df, rules), "rejected": rejected}, indent=2))


main()
