"""Declarative rules. The LLM proposes JSON specs; it never supplies executable code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

KINDS = {"not_null", "unique", "range", "allowed_values", "regex"}


class InvalidRule(ValueError):
    pass


@dataclass(frozen=True)
class Rule:
    kind: str
    column: str
    params: tuple[tuple[str, Any], ...] = ()

    def p(self, key: str, default: Any = None) -> Any:
        return dict(self.params).get(key, default)

    def label(self) -> str:
        return f"{self.kind}({self.column})"


def parse_rule(spec: dict[str, Any], columns: list[str]) -> Rule:
    """Validate an untrusted dict into a Rule or raise InvalidRule."""
    if not isinstance(spec, dict):
        raise InvalidRule("rule must be an object")
    kind, col = spec.get("kind"), spec.get("column")
    if kind not in KINDS:
        raise InvalidRule(f"unknown kind: {kind!r}")
    if col not in columns:
        raise InvalidRule(f"unknown column: {col!r}")
    params: dict[str, Any] = {}
    if kind == "range":
        lo, hi = spec.get("min"), spec.get("max")
        if lo is None and hi is None:
            raise InvalidRule("range needs min and/or max")
        for k, v in (("min", lo), ("max", hi)):
            if v is not None:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise InvalidRule(f"{k} must be a number")
                params[k] = v
        if lo is not None and hi is not None and lo > hi:
            raise InvalidRule("min greater than max")
    elif kind == "allowed_values":
        vals = spec.get("values")
        if not isinstance(vals, list) or not vals:
            raise InvalidRule("allowed_values needs a non-empty list")
        params["values"] = tuple(str(v) for v in vals)
    elif kind == "regex":
        import re

        pat = spec.get("pattern")
        if not isinstance(pat, str) or len(pat) > 200:
            raise InvalidRule("regex needs a pattern string (<=200 chars)")
        try:
            re.compile(pat)
        except re.error as e:
            raise InvalidRule(f"bad regex: {e}") from e
        params["pattern"] = pat
    return Rule(kind, col, tuple(sorted(params.items())))


def violations(df: pd.DataFrame, rule: Rule) -> pd.Series:
    """Boolean mask of rows violating the rule."""
    s = df[rule.column]
    if rule.kind == "not_null":
        return s.isna()
    if rule.kind == "unique":
        return s.notna() & s.duplicated(keep=False)
    if rule.kind == "range":
        num = pd.to_numeric(s, errors="coerce")
        bad = s.notna() & num.isna()
        lo, hi = rule.p("min"), rule.p("max")
        if lo is not None:
            bad |= num < lo
        if hi is not None:
            bad |= num > hi
        return bad
    if rule.kind == "allowed_values":
        return s.notna() & ~s.astype(str).isin(rule.p("values"))
    if rule.kind == "regex":
        return s.notna() & ~s.astype(str).str.fullmatch(rule.p("pattern"))
    raise InvalidRule(rule.kind)


def check(df: pd.DataFrame, rules: list[Rule]) -> list[dict[str, Any]]:
    out = []
    for r in rules:
        mask = violations(df, r)
        rate = round(float(mask.mean()), 4) if len(df) else 0.0
        out.append(
            {"rule": r.label(), "kind": r.kind, "column": r.column,
             "violations": int(mask.sum()), "rate": rate}
        )
    return out
