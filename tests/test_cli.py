import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PY = REPO_ROOT / "runner.py"


def _run_cli(args, cwd, runs_root):
    return subprocess.run(
        [sys.executable, str(RUNNER_PY), *args, "--runs-root", str(runs_root)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=60,
    )


def _find_run_dirs(runs_root):
    if not runs_root.exists():
        return []
    return sorted(p for p in runs_root.iterdir() if p.is_dir())


def test_cli_end_to_end_creates_run_dir_with_both_files(tmp_path):
    runs_root = tmp_path / "runs"

    result = _run_cli(
        ["--dry-run", "--trials", "3", "--carla-present", "false", "--notes", "cli test complete"],
        cwd=REPO_ROOT,
        runs_root=runs_root,
    )
    assert result.returncode == 0, result.stderr

    run_dirs = _find_run_dirs(runs_root)
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    run_json_path = run_dir / "run.json"
    trials_json_path = run_dir / "trials.json"
    assert run_json_path.exists()
    assert trials_json_path.exists()

    with open(run_json_path, encoding="utf-8") as f:
        run_record = json.load(f)
    with open(trials_json_path, encoding="utf-8") as f:
        trial_records = json.load(f)

    assert isinstance(run_record, dict)
    assert isinstance(trial_records, list)

    assert run_record["trials_intended"] == 3
    assert "config_hash" in run_record and run_record["config_hash"]
    assert run_record["carla_present"] == "false"
    assert run_record["carla_map"] == ""
    assert run_record["carla_ticking"] == "false"
    assert run_record["notes"] == "cli test complete"
    assert run_record["status"] == "complete"
    assert run_record["dry_run"] is True
    assert run_record["dry_run_fail_at"] is None


def test_cli_mid_run_failure_leaves_incomplete_status_on_disk(tmp_path):
    runs_root = tmp_path / "runs"

    result = _run_cli(
        [
            "--dry-run",
            "--dry-run-fail-at",
            "2",
            "--trials",
            "5",
            "--carla-present",
            "false",
            "--notes",
            "cli test incomplete",
        ],
        cwd=REPO_ROOT,
        runs_root=runs_root,
    )
    # The CLI process itself does not crash: run_trials catches the chain_fn
    # exception internally and continues to the next trial.
    assert result.returncode == 0, result.stderr

    run_dirs = _find_run_dirs(runs_root)
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    with open(run_dir / "run.json", encoding="utf-8") as f:
        run_record = json.load(f)

    assert run_record["status"] == "incomplete"
    assert run_record["trials_intended"] == 5
    assert run_record["dry_run"] is True
    assert run_record["dry_run_fail_at"] == 2

    with open(run_dir / "trials.json", encoding="utf-8") as f:
        trial_records = json.load(f)
    infra_rows = [r for r in trial_records if r["category"] == "infrastructure_error"]
    assert len(infra_rows) == 1
    assert infra_rows[0]["trial_index"] == 2
    assert infra_rows[0]["telemetry"]
    assert infra_rows[0]["profiler_response_content"] is None


def test_cli_trial_records_have_exact_twenty_two_key_set(tmp_path):
    runs_root = tmp_path / "runs"

    result = _run_cli(
        ["--dry-run", "--trials", "2", "--carla-present", "false", "--notes", "field set check"],
        cwd=REPO_ROOT,
        runs_root=runs_root,
    )
    assert result.returncode == 0, result.stderr

    run_dirs = _find_run_dirs(runs_root)
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    with open(run_dir / "trials.json", encoding="utf-8") as f:
        trial_records = json.load(f)

    assert len(trial_records) == 2

    expected_keys = {
        "trial_index",
        "category",
        "all_violations",
        "telemetry",
        "profiler_response_content",
        "profiler_done_reason",
        "planner_done_reason",
        "planner_raw_response_repr",
        "planner_response_content_len",
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
        "wall_clock_s",
    }
    assert len(expected_keys) == 22

    for record in trial_records:
        assert set(record.keys()) == expected_keys


def test_cli_output_files_are_utf8_without_bom(tmp_path):
    runs_root = tmp_path / "runs"

    result = _run_cli(
        ["--dry-run", "--trials", "1", "--carla-present", "false", "--notes", "bom check"],
        cwd=REPO_ROOT,
        runs_root=runs_root,
    )
    assert result.returncode == 0, result.stderr

    run_dirs = _find_run_dirs(runs_root)
    run_dir = run_dirs[0]

    for filename in ("run.json", "trials.json"):
        raw = (run_dir / filename).read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"{filename} has a UTF-8 BOM"
        raw.decode("utf-8")  # must not raise


def test_two_cli_runs_produce_distinct_run_dirs(tmp_path):
    runs_root = tmp_path / "runs"

    result1 = _run_cli(
        ["--dry-run", "--trials", "1", "--carla-present", "false", "--notes", "first"],
        cwd=REPO_ROOT,
        runs_root=runs_root,
    )
    assert result1.returncode == 0, result1.stderr

    result2 = _run_cli(
        ["--dry-run", "--trials", "1", "--carla-present", "false", "--notes", "second"],
        cwd=REPO_ROOT,
        runs_root=runs_root,
    )
    assert result2.returncode == 0, result2.stderr

    run_dirs = _find_run_dirs(runs_root)
    assert len(run_dirs) == 2

    dir1, dir2 = run_dirs
    assert dir1 != dir2
    with open(dir1 / "run.json", encoding="utf-8") as f:
        record1 = json.load(f)
    with open(dir2 / "run.json", encoding="utf-8") as f:
        record2 = json.load(f)
    assert record1["run_id"] != record2["run_id"]
