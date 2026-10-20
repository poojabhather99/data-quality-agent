# data-quality-agent

[![CI](https://github.com/poojabhather99/data-quality-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/poojabhather99/data-quality-agent/actions/workflows/ci.yml)

**Live demo (runs in your browser):** https://poojabhather99.github.io/data-quality-agent/

Profile a table, have an LLM (or an offline heuristic) propose validation rules, and run them safely.

The main design point: **the model never produces executable code.** It proposes rules as JSON specs (`not_null`, `unique`, `range`, `allowed_values`, `regex`). Every spec is validated against a fixed schema and the table's real column names before it can run. Unknown kinds, unknown columns, bad regexes, inverted ranges and non-numeric bounds are rejected and reported instead of executed.

## Pipeline

1. `profile.py` summarises each column (null rate, uniqueness, min/max, small value sets).
2. `proposer.py` builds a prompt from the profile and parses the reply, or uses the heuristic baseline when no LLM is given. It tolerates code fences and prose around the JSON and returns rejections rather than raising.
3. `rules.py` validates specs into `Rule` objects and computes a violation mask per rule.
4. `evaluate.py` fits rules on a clean synthetic table, injects known defects, and measures recall per defect type plus false positives on the clean data.

## Run

```bash
pip install -e ".[dev]"
pytest -q                                   # 20 tests
python -m dqagent.evaluate                  # offline evaluation of the heuristic baseline
python -m dqagent data.csv                  # heuristic rules checked against your CSV
pip install -e ".[llm]" && GROQ_API_KEY=... python -m dqagent data.csv --llm
```

## Evaluation (synthetic)

| Proposer | Mean defect recall | False-positive rows on clean data |
|---|---|---|
| Heuristic baseline | 1.00 | 0 |

Read this with care. The data is synthetic, the injected defects (nulls, out-of-range ages, invalid status values) are the kinds the heuristics are built to catch, and the rules are fitted on the same clean table the defects were injected into. A recall of 1.0 shows the plumbing works; it is not evidence of accuracy on real data. The LLM backend is implemented and its output path is unit tested with stubbed replies, but I have not run it against a live model, so there are no LLM numbers here.

## Limitations

- Rules fit on already-clean data. Fitting on dirty data would bake the defects into the rules.
- Single-column rules only; no cross-column or referential checks.
- The heuristic range padding (10%) and the "10 distinct values means categorical" cutoff are arbitrary defaults.
