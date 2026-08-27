import json
import re
import statistics

import runner
import subject


VALID_PAYLOAD = '{"drive_mode": "Sport", "max_acceleration_m_s2": 3.0}'


class FakeResponse(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            return None


def _canned_response(content, done_reason="stop", total_duration=123456, load_duration=111,
                      prompt_eval_count=12, prompt_eval_duration=222, eval_count=34, eval_duration=333):
    return FakeResponse({
        "message": {"content": content},
        "done_reason": done_reason,
        "total_duration": total_duration,
        "load_duration": load_duration,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration": prompt_eval_duration,
        "eval_count": eval_count,
        "eval_duration": eval_duration,
    })


# --- 1. Trial-record field test ------------------------------------------

def test_written_trial_record_has_all_required_fields(tmp_path):
    content = VALID_PAYLOAD

    def chain_fn(telemetry):
        return content, _canned_response(content), _canned_response("profile sentence")

    runner.run_trials(chain_fn, telemetries=["t0"], run_dir=tmp_path)

    with open(tmp_path / "trials.json", encoding="utf-8") as f:
        trial_records = json.load(f)

    assert len(trial_records) == 1
    record = trial_records[0]

    required_keys = {
        "profiler_total_duration",
        "profiler_load_duration",
        "profiler_prompt_eval_count",
        "profiler_prompt_eval_duration",
        "profiler_eval_count",
        "profiler_eval_duration",
        "planner_total_duration",
        "planner_load_duration",
        "planner_prompt_eval_count",
        "planner_prompt_eval_duration",
        "planner_eval_count",
        "planner_eval_duration",
        "profiler_done_reason",
        "planner_done_reason",
        "planner_response_content_len",
        "planner_raw_response_repr",
        "trial_index",
        "category",
        "all_violations",
        "wall_clock_s",
        "telemetry",
        "profiler_response_content",
    }
    missing = required_keys - set(record.keys())
    assert not missing, f"missing keys in written trial record: {missing}"
    assert len(required_keys) == 22
    assert set(record.keys()) == required_keys

    assert record["planner_total_duration"] == 123456
    assert record["planner_load_duration"] == 111
    assert record["planner_prompt_eval_count"] == 12
    assert record["planner_prompt_eval_duration"] == 222
    assert record["planner_eval_count"] == 34
    assert record["planner_eval_duration"] == 333
    assert record["planner_done_reason"] == "stop"
    assert record["planner_response_content_len"] == len(content)
    assert record["planner_raw_response_repr"] == repr(content)
    assert record["trial_index"] == 0
    assert record["category"] == "pass"
    assert record["all_violations"] == []


# --- 1b. Anti-aliasing test -------------------------------------------------

def test_profiler_and_planner_timing_fields_do_not_alias(tmp_path):
    content = VALID_PAYLOAD

    def chain_fn(telemetry):
        planner_response = _canned_response(
            content, total_duration=100, load_duration=200,
            prompt_eval_count=300, prompt_eval_duration=400,
            eval_count=500, eval_duration=600, done_reason="stop",
        )
        profiler_response = _canned_response(
            "profile sentence", total_duration=1, load_duration=2,
            prompt_eval_count=3, prompt_eval_duration=4,
            eval_count=5, eval_duration=6, done_reason="length",
        )
        return content, planner_response, profiler_response

    runner.run_trials(chain_fn, telemetries=["t0"], run_dir=tmp_path)

    with open(tmp_path / "trials.json", encoding="utf-8") as f:
        trial_records = json.load(f)

    record = trial_records[0]

    for field in ["total_duration", "load_duration", "prompt_eval_count",
                  "prompt_eval_duration", "eval_count", "eval_duration"]:
        assert record[f"profiler_{field}"] != record[f"planner_{field}"], (
            f"profiler_{field} and planner_{field} must not alias"
        )

    assert record["profiler_done_reason"] != record["planner_done_reason"]


# --- 2. Subject call-site test --------------------------------------------

def test_profiler_and_planner_call_sites(monkeypatch):
    recorded_calls = []

    def fake_chat(**kwargs):
        recorded_calls.append(kwargs)
        raise RuntimeError("no network allowed")

    monkeypatch.setattr(subject.ollama, "chat", fake_chat)

    # Profiler path
    try:
        subject.ollama.chat(**subject.profiler_call_args("Vehicle Speed: 60 km/h."))
    except RuntimeError:
        pass

    # Planner path
    try:
        subject.ollama.chat(**subject.planner_call_args("The driver is travelling normally."))
    except RuntimeError:
        pass

    assert len(recorded_calls) == 2
    profiler_kwargs, planner_kwargs = recorded_calls

    assert profiler_kwargs["options"] == {"temperature": 0.1}
    assert "format" not in profiler_kwargs

    assert planner_kwargs["format"] == "json"
    assert "options" not in planner_kwargs

    assert profiler_kwargs["model"] == "qwen3.5:9b"
    assert planner_kwargs["model"] == "qwen3.5:9b"


def test_run_chain_makes_no_network_call_and_uses_correct_call_sites(monkeypatch):
    recorded_calls = []

    def fake_chat(**kwargs):
        recorded_calls.append(kwargs)
        raise RuntimeError("no network allowed")

    monkeypatch.setattr(subject.ollama, "chat", fake_chat)

    try:
        subject.run_chain("Vehicle Speed: 60 km/h.")
    except RuntimeError:
        pass

    assert len(recorded_calls) == 1
    profiler_kwargs = recorded_calls[0]
    assert profiler_kwargs["options"] == {"temperature": 0.1}
    assert "format" not in profiler_kwargs
    assert profiler_kwargs["model"] == "qwen3.5:9b"


# --- 3. Statistics test ----------------------------------------------------

def test_median_and_iqr_even_count():
    # Even count, values chosen so mean and median differ substantially.
    # data: [1, 2, 3, 4, 100] is odd; use even-count dataset here.
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
    # sorted: 1,2,3,4,5,100 -> median = (3+4)/2 = 3.5 ; mean = 115/6 ≈ 19.1667
    expected_median = 3.5
    expected_mean = sum(data) / len(data)
    assert abs(expected_mean - 19.16666666666667) < 1e-9
    assert abs(expected_median - expected_mean) > 10  # mean/median differ substantially

    trial_records = [{"wall_clock_s": v, "category": "pass"} for v in data]
    summary = runner.summarize(trial_records)

    assert summary["latency"]["median_s"] == expected_median
    assert summary["latency"]["median_s"] != expected_mean

    q1_expected = statistics.quantiles(sorted(data), n=100, method="inclusive")[24]
    q3_expected = statistics.quantiles(sorted(data), n=100, method="inclusive")[74]
    expected_iqr = q3_expected - q1_expected

    assert summary["latency"]["iqr_s"] == expected_iqr


def test_median_and_iqr_odd_count():
    # Odd count, mean/median differ substantially.
    data = [1.0, 2.0, 3.0, 4.0, 200.0]
    # sorted: 1,2,3,4,200 -> median = 3 ; mean = 210/5 = 42.0
    expected_median = 3.0
    expected_mean = 42.0
    assert sum(data) / len(data) == expected_mean
    assert abs(expected_median - expected_mean) > 10

    trial_records = [{"wall_clock_s": v, "category": "pass"} for v in data]
    summary = runner.summarize(trial_records)

    assert summary["latency"]["median_s"] == expected_median
    assert summary["latency"]["median_s"] != expected_mean

    q1_expected = statistics.quantiles(sorted(data), n=100, method="inclusive")[24]
    q3_expected = statistics.quantiles(sorted(data), n=100, method="inclusive")[74]
    expected_iqr = q3_expected - q1_expected

    assert summary["latency"]["iqr_s"] == expected_iqr


# --- 4. Denominator test ----------------------------------------------------

def test_summary_denominator_renders_both_numbers():
    categories = ["out_of_enum_mode"] * 3 + ["pass"] * 17
    assert len(categories) == 20

    trial_records = [{"category": c, "wall_clock_s": 0.1} for c in categories]
    summary = runner.summarize(trial_records)

    count_str = summary["counts"]["out_of_enum_mode"]
    assert count_str == "3/20"

    match = re.search(r"(\d+)\s*/\s*(\d+)", count_str)
    assert match is not None
    assert match.group(1) == "3"
    assert match.group(2) == "20"

    assert "20" in count_str
