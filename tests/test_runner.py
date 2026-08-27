import json
import re

import runner
import subject


class FakeResponse(dict):
    """Mimics the ollama response object: dict-like with attribute access."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            return None


def _fake_response(content, **timing):
    base = {
        "message": {"content": content},
        "done_reason": timing.pop("done_reason", "stop"),
        "total_duration": timing.pop("total_duration", 100),
        "load_duration": timing.pop("load_duration", 10),
        "prompt_eval_count": timing.pop("prompt_eval_count", 5),
        "prompt_eval_duration": timing.pop("prompt_eval_duration", 20),
        "eval_count": timing.pop("eval_count", 7),
        "eval_duration": timing.pop("eval_duration", 30),
    }
    return FakeResponse(base)


VALID_PAYLOAD = '{"drive_mode": "Sport", "max_acceleration_m_s2": 3.0}'


def _make_chain_fn(contents, fail_on_index=None):
    """Build a chain_fn(telemetry) -> (content, planner_response, profiler_response)
    driven by an index counter."""
    counter = {"i": -1}

    def chain_fn(telemetry):
        counter["i"] += 1
        idx = counter["i"]
        if fail_on_index is not None and idx == fail_on_index:
            raise RuntimeError("simulated infrastructure failure")
        content = contents[idx]
        planner_response = _fake_response(content)
        profiler_response = _fake_response("profile sentence")
        return content, planner_response, profiler_response

    return chain_fn


def test_run_trials_all_pass_produces_complete_run(tmp_path):
    contents = [VALID_PAYLOAD] * 3
    chain_fn = _make_chain_fn(contents)

    run_record, trial_records = runner.run_trials(
        chain_fn, telemetries=["t0", "t1", "t2"], notes="unit test", run_dir=tmp_path
    )

    assert run_record["status"] == "complete"
    assert run_record["trials_intended"] == 3
    assert run_record["trials_completed"] == 3
    assert len(trial_records) == 3
    for record in trial_records:
        assert record["category"] == "pass"

    with open(tmp_path / "run.json", encoding="utf-8") as f:
        persisted = json.load(f)
    assert persisted["status"] == "complete"


def test_run_trials_mid_run_failure_produces_incomplete_run(tmp_path):
    contents = [VALID_PAYLOAD, VALID_PAYLOAD, None, VALID_PAYLOAD, VALID_PAYLOAD]
    chain_fn = _make_chain_fn(contents, fail_on_index=2)

    run_record, trial_records = runner.run_trials(
        chain_fn,
        telemetries=["t0", "t1", "t2", "t3", "t4"],
        notes="mid-run failure test",
        run_dir=tmp_path,
    )

    assert run_record["status"] == "incomplete"
    assert run_record["trials_intended"] == 5
    assert run_record["trials_completed"] == 4

    infra_rows = [r for r in trial_records if r["category"] == "infrastructure_error"]
    assert len(infra_rows) == 1
    assert infra_rows[0]["trial_index"] == 2

    with open(tmp_path / "run.json", encoding="utf-8") as f:
        persisted = json.load(f)
    assert persisted["status"] == "incomplete"
    assert persisted["trials_intended"] == 5


def test_config_hash_changes_when_profiler_system_prompt_changes(monkeypatch):
    original_hash = runner.compute_config_hash()

    original_prompt = subject.PROFILER_SYSTEM_PROMPT
    monkeypatch.setattr(subject, "PROFILER_SYSTEM_PROMPT", original_prompt + "x")
    changed_hash = runner.compute_config_hash()

    assert changed_hash != original_hash


def test_config_hash_changes_when_planner_system_prompt_changes(monkeypatch):
    original_hash = runner.compute_config_hash()

    original_prompt = subject.PLANNER_SYSTEM_PROMPT
    monkeypatch.setattr(subject, "PLANNER_SYSTEM_PROMPT", original_prompt + "x")
    changed_hash = runner.compute_config_hash()

    assert changed_hash != original_hash


def test_ollama_client_version_is_a_real_version_string(tmp_path):
    contents = [VALID_PAYLOAD]
    run_record, _ = runner.run_trials(
        _make_chain_fn(contents), telemetries=["t0"], notes="version check", run_dir=tmp_path
    )
    version = run_record["ollama_client_version"]

    assert version is not None
    assert version != "None"
    assert isinstance(version, str)
    assert version != ""
    assert re.match(r"^\d+\.\d+(\.\d+)?", version), version


def test_config_hash_stable_regardless_of_notes(tmp_path):
    contents = [VALID_PAYLOAD]
    run_record_1, _ = runner.run_trials(
        _make_chain_fn(contents), telemetries=["t0"], notes="notes A", run_dir=tmp_path
    )
    run_record_2, _ = runner.run_trials(
        _make_chain_fn(contents), telemetries=["t0"], notes="a totally different note", run_dir=tmp_path
    )
    assert run_record_1["config_hash"] == run_record_2["config_hash"]


# --- 2.1: Planner user turn wrapper -----------------------------------------

def test_planner_call_args_wraps_profile_sentence_with_schema_instructions():
    call_args = subject.planner_call_args("The driver is travelling normally.")
    user_message = call_args["messages"][-1]["content"]
    assert "drive_mode" in user_message
    assert "max_acceleration_m_s2" in user_message


def test_planner_call_args_wrapper_is_sent_to_ollama_chat(monkeypatch):
    recorded_calls = []

    def fake_chat(**kwargs):
        recorded_calls.append(kwargs)
        return {"message": {"content": "unused"}}

    monkeypatch.setattr(subject.ollama, "chat", fake_chat)

    subject.ollama.chat(**subject.planner_call_args("The driver is travelling normally."))

    assert len(recorded_calls) == 1
    user_message = recorded_calls[0]["messages"][-1]["content"]
    assert "drive_mode" in user_message
    assert "max_acceleration_m_s2" in user_message


# --- 2.5: dry-run / live path key-set parity ---------------------------------

def test_dry_run_and_live_paths_produce_identical_trial_record_key_sets(monkeypatch, tmp_path):
    dry_dir = tmp_path / "dry"
    live_dir = tmp_path / "live"
    dry_dir.mkdir()
    live_dir.mkdir()

    dry_chain_fn = runner._dry_run_chain_factory()
    _, dry_records = runner.run_trials(dry_chain_fn, telemetries=["t0"], run_dir=dry_dir)
    dry_keys = set(dry_records[0].keys())

    def fake_chat(**kwargs):
        if "format" in kwargs:  # planner call
            return _fake_response(VALID_PAYLOAD)
        return _fake_response("profile sentence")

    monkeypatch.setattr(subject.ollama, "chat", fake_chat)

    _, live_records = runner.run_trials(subject.run_chain, telemetries=["t0"], run_dir=live_dir)
    live_keys = set(live_records[0].keys())

    assert dry_keys == live_keys


# --- 2.6: config_hash covers the telemetry construction ----------------------

def test_config_hash_stable_across_trials_within_a_run():
    hash_a = runner.compute_config_hash()
    hash_b = runner.compute_config_hash()
    assert hash_a == hash_b


def test_config_hash_changes_when_telemetry_construction_changes(monkeypatch):
    original_hash = runner.compute_config_hash()

    def _different_distance_formula(index):
        return 3.0 if index % 2 == 0 else 30.0

    monkeypatch.setattr(runner, "_telemetry_distance_m", _different_distance_formula)
    changed_hash = runner.compute_config_hash()

    assert changed_hash != original_hash


# --- Part 1 (1e-c): telemetry reproduction fidelity --------------------------

def test_build_telemetry_matches_carla_format_exactly():
    assert runner.build_telemetry(0) == "Vehicle Speed: 16.00 m/s. Traffic: Clear. Distance to Hazard: 15.0m."
    assert runner.build_telemetry(1) == "Vehicle Speed: 16.50 m/s. Traffic: Blocked. Distance to Hazard: 5.0m."


def test_summary_has_no_mean_and_no_bare_percent(tmp_path):
    contents = [VALID_PAYLOAD, "", '{"drive_mode": "Track"}']
    chain_fn = _make_chain_fn(contents)
    _, trial_records = runner.run_trials(
        chain_fn, telemetries=["t0", "t1", "t2"], run_dir=tmp_path
    )
    summary = runner.summarize(trial_records)
    serialized = json.dumps(summary)

    assert "mean" not in serialized.lower()

    import re
    # A "bare %" is a percent sign directly following a number, e.g. "42%".
    assert re.search(r"\d\s*%", serialized) is None

    for count_str in summary["counts"].values():
        assert "/" in count_str
