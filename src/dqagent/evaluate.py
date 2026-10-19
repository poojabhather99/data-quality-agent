"""Evaluate a proposer: rules are fit on CLEAN data, then run on the corrupted copy."""

from __future__ import annotations

from typing import Any, Callable

from .proposer import propose
from .rules import violations
from .synth import DEFECTS, clean_table, inject_defects


def evaluate(llm: Callable[[str], str] | None = None) -> dict[str, Any]:
    clean = clean_table()
    bad, groups = inject_defects(clean)
    rules, rejected = propose(clean, llm)
    caught: dict[str, float] = {}
    for name, rows in groups.items():
        col, kind = DEFECTS[name]
        hit: set[int] = set()
        for r in rules:
            if r.column == col and r.kind == kind:
                hit |= set(bad.index[violations(bad, r)])
        caught[name] = len(hit & set(rows)) / len(rows)
    false_pos = sum(int(violations(clean, r).sum()) for r in rules)
    return {
        "rules": len(rules),
        "rejected": len(rejected),
        "defect_recall": caught,
        "mean_recall": round(sum(caught.values()) / len(caught), 3),
        "false_positive_rows_on_clean": false_pos,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(evaluate(), indent=2))
