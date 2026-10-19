"""Rule proposers: an offline heuristic baseline and an optional LLM (Groq) backend."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

import pandas as pd

from .profile import profile_table
from .rules import InvalidRule, Rule, parse_rule

ID_HINT = re.compile(r"(^id$|_id$|^id_)", re.IGNORECASE)


def heuristic_specs(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Conservative rules inferred from a profile of (assumed mostly clean) data."""
    specs: list[dict[str, Any]] = []
    for p in profiles:
        col = p["name"]
        if p["null_rate"] == 0:
            specs.append({"kind": "not_null", "column": col})
        if ID_HINT.search(str(col)) and p["unique_rate"] == 1.0 and p["rows"] > 1:
            specs.append({"kind": "unique", "column": col})
        if "min" in p and not ID_HINT.search(str(col)):
            pad = (p["max"] - p["min"]) * 0.1
            specs.append({"kind": "range", "column": col,
                          "min": round(p["min"] - pad, 3), "max": round(p["max"] + pad, 3)})
        if "values" in p and p["unique"] <= 10:
            specs.append({"kind": "allowed_values", "column": col, "values": p["values"]})
    return specs


def build_prompt(profiles: list[dict[str, Any]]) -> str:
    return (
        "You are a data quality engineer. Given this column profile, propose validation rules "
        "as a JSON array. Allowed kinds: not_null, unique, range (min/max), allowed_values "
        "(values), regex (pattern). Each rule: {\"kind\":..., \"column\":..., ...params}. "
        "Only reference listed columns. Output JSON only.\n\n" + json.dumps(profiles)
    )


def extract_json_array(text: str) -> list[Any]:
    """Pull the first JSON array out of a model reply (handles code fences/prose)."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON array in model output")
    data = json.loads(m.group(0))
    if not isinstance(data, list):
        raise ValueError("expected a JSON array")
    return data


def groq_complete(prompt: str) -> str:  # pragma: no cover - needs network and a key
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return r.choices[0].message.content


def propose(
    df: pd.DataFrame,
    llm: Callable[[str], str] | None = None,
) -> tuple[list[Rule], list[str]]:
    """Return (valid rules, rejection messages). Falls back to heuristics without an LLM."""
    profiles = profile_table(df)
    cols = list(df.columns)
    rejected: list[str] = []
    if llm is None:
        raw = heuristic_specs(profiles)
    else:
        try:
            raw = extract_json_array(llm(build_prompt(profiles)))
        except (ValueError, json.JSONDecodeError) as e:
            return [], [f"unparseable model output: {e}"]
    rules: list[Rule] = []
    seen: set[Rule] = set()
    for spec in raw:
        try:
            r = parse_rule(spec, cols)
        except InvalidRule as e:
            rejected.append(f"{spec!r}: {e}")
            continue
        if r not in seen:
            seen.add(r)
            rules.append(r)
    return rules, rejected
