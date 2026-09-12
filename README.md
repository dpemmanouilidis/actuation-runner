actuation-runner

This project measures how often LLM-generated vehicle-actuation payloads
fail before reaching an actuator, and decomposes those failures by
category. A Profiler model reads vehicle telemetry and produces a
sentence-level assessment; a Planner model consumes that assessment and
is required to emit a JSON payload matching a fixed schema. Each trial's
Planner output is classified into one of eight categories describing
whether, and how, it would have failed before actuation.

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

## Reproduce

```
pip install -r requirements.txt
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
