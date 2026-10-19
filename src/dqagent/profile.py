"""Column profiling: the evidence the rule proposer sees."""

from __future__ import annotations

from typing import Any

import pandas as pd


def profile_column(s: pd.Series) -> dict[str, Any]:
    n = len(s)
    nn = s.dropna()
    p: dict[str, Any] = {
        "name": s.name,
        "dtype": str(s.dtype),
        "rows": n,
        "null_rate": round(float(s.isna().mean()), 4) if n else 0.0,
        "unique": int(nn.nunique()),
        "unique_rate": round(nn.nunique() / len(nn), 4) if len(nn) else 0.0,
    }
    if pd.api.types.is_numeric_dtype(s) and len(nn):
        p.update(min=float(nn.min()), max=float(nn.max()), mean=round(float(nn.mean()), 3))
    if (s.dtype == object or pd.api.types.is_string_dtype(s)) and len(nn):
        if nn.nunique() <= 10:
            p["values"] = sorted(map(str, nn.unique()))
        p["sample"] = [str(v) for v in nn.head(3)]
    return p


def profile_table(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [profile_column(df[c]) for c in df.columns]
