"""Synthetic table with known injected defects, used for demos and evaluation."""

from __future__ import annotations

import random

import pandas as pd

DEFECTS = {
    "null_worker": ("worker_id", "not_null"),
    "bad_age": ("age", "range"),
    "bad_status": ("status", "allowed_values"),
}


def clean_table(n: int = 200, seed: int = 5) -> pd.DataFrame:
    rng = random.Random(seed)
    return pd.DataFrame(
        {
            "record_id": list(range(1, n + 1)),
            "worker_id": [rng.randint(1, 30) for _ in range(n)],
            "age": [rng.randint(18, 60) for _ in range(n)],
            "status": [rng.choice(["active", "inactive", "leave"]) for _ in range(n)],
        }
    )


def inject_defects(df: pd.DataFrame, seed: int = 5) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """Return a corrupted copy plus {defect_name: row indexes}."""
    rng = random.Random(seed + 1)
    bad = df.copy()
    idx = list(range(len(bad)))
    rng.shuffle(idx)
    groups = {"null_worker": idx[:4], "bad_age": idx[4:8], "bad_status": idx[8:12]}
    bad["worker_id"] = bad["worker_id"].astype("float")
    for i in groups["null_worker"]:
        bad.loc[i, "worker_id"] = None
    for i in groups["bad_age"]:
        bad.loc[i, "age"] = rng.choice([-5, 250, 999])
    for i in groups["bad_status"]:
        bad.loc[i, "status"] = rng.choice(["ACTIVE", "unknown", "??"])
    return bad, groups
