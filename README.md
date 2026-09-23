# actuation-runner

[![tests](https://github.com/dpemmanouilidis/actuation-runner/actions/workflows/tests.yml/badge.svg)](https://github.com/dpemmanouilidis/actuation-runner/actions/workflows/tests.yml)

This project measures how often LLM-generated vehicle-actuation payloads
fail before reaching an actuator, and decomposes those failures by
category. A Profiler model reads vehicle telemetry and produces a
sentence-level assessment; a Planner model consumes that assessment and
is required to emit a JSON payload matching a fixed schema. Each trial's
Planner output is classified into one of eight categories describing
whether, and how, it would have failed before actuation.

**Result.** Across two runs of 100 trials each, 181/200 Planner payloads
passed validation. All 19/200 failures were `empty_response`: the Planner
call stopped at its length limit (`done_reason: length`) and returned an
empty string. No payload was malformed, missing a key, out of enum, out of
bounds or wrongly typed (0/200 each). Failures fell almost entirely on
trials whose telemetry reported clear traffic (18/102) rather than blocked
traffic (1/98).

## Categories

- `pass` -- the Planner's raw response parses as JSON and validates
  against the schema.
- `empty_response` -- the Planner's raw response was the empty string.
- `malformed_json` -- the raw response is non-empty but does not parse
  as JSON.
- `missing_required_key` -- the parsed payload is missing `drive_mode`
  or `max_acceleration_m_s2`.
- `out_of_enum_mode` -- `drive_mode` is present but is not one of the
  four schema-permitted values (`Eco`, `Comfort`, `Sport`, `Emergency`).
- `out_of_bounds_value` -- a payload that is well-formed, correctly
  typed and schema-shaped, and is a failure only against a physical
  limit (`max_acceleration_m_s2` outside `[0.0, 5.0]`).
- `wrong_type` -- a field has the wrong JSON type, or is the classifier's
  default fallthrough (see Limitations).
- `infrastructure_error` -- the chain call itself raised an exception
  (e.g. an Ollama call failure) before any response was produced.

Each category except `infrastructure_error` has a classifier test in
[`tests/test_classify.py`](tests/test_classify.py), so the five classifier categories that
never fired in the runs below are detectable, not merely defined.
`infrastructure_error` is assigned by the runner, not the classifier. Of the
two acceleration bounds, only the upper one is covered by a test.

## Results

Two independent runs of 100 trials each, same configuration
(`config_hash` identical across both:
`89e6c0f44dfd342dfdcbb1ec4ed6e8f609feba6761956e22967016b70402df14`),
model `qwen3.5:9b`, `ollama_client_version` `0.6.2`. The two runs are
reported separately below and are not pooled.

**Run `20260831T162121370079Z-d52279db`**

| category | count |
|---|---|
| pass | 89/100 |
| empty_response | 11/100 |
| malformed_json | 0/100 |
| missing_required_key | 0/100 |
| out_of_enum_mode | 0/100 |
| out_of_bounds_value | 0/100 |
| wrong_type | 0/100 |
| infrastructure_error | 0/100 |

Blocked/Clear telemetry split: 49/51.

**Run `20260831T172152727117Z-85941c1e`**

| category | count |
|---|---|
| pass | 92/100 |
| empty_response | 8/100 |
| malformed_json | 0/100 |
| missing_required_key | 0/100 |
| out_of_enum_mode | 0/100 |
| out_of_bounds_value | 0/100 |
| wrong_type | 0/100 |
| infrastructure_error | 0/100 |

Blocked/Clear telemetry split: 49/51.

**Cross-tab, both runs combined (200 trials, denominator 200):**

Across the two runs, planner `done_reason` corresponded to category
without exception: `stop` -> `pass` in 181/181 of the trials where it
occurred, and `length` -> `empty_response` in 19/19 of the trials where
it occurred. Profiler truncation (`done_reason: length`) occurred in
7/200 trials and in every one of those 7 the trial still classified as
`pass` -- profiler truncation did not cause a failure in either run.

Raw records for both runs, as written by the runner, are in
[`results/`](results/): `run.json` and all 100 per-trial records in
`trials.json` for each.

## Latency

Seconds per trial, median (IQR), rounded to two decimals. Quartiles are
computed with Python's `statistics.quantiles(..., method="inclusive")`, the
method `runner.py` uses.

| | run `d52279db` | run `85941c1e` |
|---|---|---|
| trial wall clock (both calls) | 34.42 (21.05) | 32.85 (25.55) |
| Profiler call (`total_duration`) | 18.21 (11.47) | 16.34 (14.97) |
| Planner call (`total_duration`) | 11.27 (20.22) | 9.74 (19.92) |

Each run of 100 trials took about an hour (60 m 23 s and 59 m 11 s).

## Setup

Beelink GTi13 Ultra with an NVIDIA GeForce RTX 5070 (12 GB) in a Beelink EX
dock. CARLA was not running. Hardware and the Ollama server version are not
recorded in the run artefacts; the hardware is stated by the author, and
the server version at the time of the runs is unknown.

## Reproduce

Requires Python 3.12. Live runs also require [Ollama](https://ollama.com)
running locally with the model pulled.

```
python -m venv .venv
.venv\Scripts\activate          # Windows; Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest -q                       # offline; Ollama not needed
ollama pull qwen3.5:9b          # live runs only
python runner.py --trials <N>
```

Add `--dry-run` to exercise the pipeline without calling Ollama (used
by the test suite). Output is written to `runs/<run_id>/run.json` and
`runs/<run_id>/trials.json` under the repo-local `runs/` directory by
default, or under the directory passed to `--runs-root`.

## Limitations

- Inputs are synthetic and deterministic. The runner has never fed the
  pipeline CARLA-produced telemetry. This makes runs reproducible
  without a simulator.
- `wrong_type` absorbs any jsonschema validator not present in
  `_VALIDATOR_TO_CATEGORY`, via a default fallthrough. It is a category
  and a catch-all, and the two are not distinguished at the return
  boundary.
- One model, one prompt pair, one input distribution, one machine, two
  runs of 100. Six categories were never observed to fire.
- The `run.json` `status` field conflates process completion with
  trial success; `utc_end` and `trials_completed` disambiguate it.
- `carla_present` is recorded metadata and branches no logic.

## Origin

This instrument exists because prior work of mine reported a failure
decomposition that its logging could not actually have produced --
categories were claimed without the instrumentation needed to
distinguish them. This repository is that instrumentation, built to
actually produce the decomposition it reports.

## Licence

Apache License 2.0. See [LICENSE](LICENSE).
