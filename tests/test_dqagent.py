import json

import pandas as pd
import pytest

from dqagent.evaluate import evaluate
from dqagent.profile import profile_column, profile_table
from dqagent.proposer import extract_json_array, propose
from dqagent.rules import InvalidRule, check, parse_rule, violations
from dqagent.synth import clean_table, inject_defects

COLS = ["a", "b"]


def df():
    return pd.DataFrame({"a": [1, 2, 2, None], "b": ["x", "y", "z", "bad"]})


def test_profile_numeric_and_categorical():
    p = profile_table(clean_table())
    by = {c["name"]: c for c in p}
    assert by["age"]["min"] >= 18 and by["age"]["max"] <= 60
    assert set(by["status"]["values"]) == {"active", "inactive", "leave"}
    assert by["record_id"]["unique_rate"] == 1.0


def test_profile_empty_column():
    assert profile_column(pd.Series([], name="x", dtype=float))["null_rate"] == 0.0


def test_not_null_and_unique():
    d = df()
    assert violations(d, parse_rule({"kind": "not_null", "column": "a"}, COLS)).sum() == 1
    assert violations(d, parse_rule({"kind": "unique", "column": "a"}, COLS)).sum() == 2


def test_range_catches_non_numeric():
    d = pd.DataFrame({"a": ["1", "oops", "5"]})
    r = parse_rule({"kind": "range", "column": "a", "min": 0, "max": 10}, ["a"])
    assert violations(d, r).tolist() == [False, True, False]


def test_allowed_values_and_regex():
    d = df()
    av = parse_rule({"kind": "allowed_values", "column": "b", "values": ["x", "y", "z"]}, COLS)
    assert violations(d, av).sum() == 1
    rx = parse_rule({"kind": "regex", "column": "b", "pattern": "[xyz]"}, COLS)
    assert violations(d, rx).sum() == 1


@pytest.mark.parametrize(
    "spec",
    [
        {"kind": "drop_table", "column": "a"},
        {"kind": "not_null", "column": "missing"},
        {"kind": "range", "column": "a"},
        {"kind": "range", "column": "a", "min": 9, "max": 1},
        {"kind": "range", "column": "a", "min": "1"},
        {"kind": "allowed_values", "column": "b", "values": []},
        {"kind": "regex", "column": "b", "pattern": "("},
        {"kind": "regex", "column": "b", "pattern": "a" * 300},
        "not a dict",
    ],
)
def test_parse_rejects_bad_specs(spec):
    with pytest.raises(InvalidRule):
        parse_rule(spec, COLS)


def test_extract_json_array_from_fenced_reply():
    txt = 'Sure!\n```json\n[{"kind":"not_null","column":"a"}]\n```'
    assert extract_json_array(txt)[0]["kind"] == "not_null"
    with pytest.raises(ValueError):
        extract_json_array("no json here")


def test_llm_output_is_validated_and_deduped():
    reply = json.dumps(
        [
            {"kind": "not_null", "column": "a"},
            {"kind": "not_null", "column": "a"},
            {"kind": "not_null", "column": "ghost"},
            {"kind": "exec", "column": "a"},
        ]
    )
    rules, rejected = propose(df(), lambda _p: reply)
    assert len(rules) == 1 and len(rejected) == 2


def test_llm_garbage_does_not_crash():
    rules, rejected = propose(df(), lambda _p: "I cannot help")
    assert rules == [] and "unparseable" in rejected[0]


def test_check_reports_counts():
    d = df()
    out = check(d, [parse_rule({"kind": "not_null", "column": "a"}, COLS)])
    assert out[0]["violations"] == 1 and out[0]["rate"] == 0.25


def test_injection_is_deterministic_and_disjoint():
    _, g1 = inject_defects(clean_table())
    _, g2 = inject_defects(clean_table())
    assert g1 == g2
    rows = [i for v in g1.values() for i in v]
    assert len(rows) == len(set(rows)) == 12


def test_heuristic_baseline_quality():
    res = evaluate()
    assert res["false_positive_rows_on_clean"] == 0
    assert res["defect_recall"]["bad_age"] == 1.0
    assert res["defect_recall"]["bad_status"] == 1.0
    assert res["defect_recall"]["null_worker"] == 1.0
