import argparse
import hashlib
import importlib.metadata
import inspect
import json
import secrets
import statistics
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone

import ollama

import classify
import subject


def _get_ollama_client_version():
    """Look up the installed ollama package version.

    ollama exposes no top-level __version__, so importlib.metadata is used
    instead. If the lookup itself fails, the exception type name is recorded
    as the field's value so a failed lookup is visible in the record rather
    than indistinguishable from an absent one.
    """
    try:
        return importlib.metadata.version("ollama")
    except Exception as exc:
        return type(exc).__name__


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


_TELEMETRY_SEED = "actuation-runner-telemetry-v1"


def _telemetry_pseudo_unit(stream, index):
    """Deterministic pseudo-random value in [0, 1) for a (stream, index) pair.

    Uses sha256 rather than the `random` module so the sequence is stable
    across processes and machines without relying on Mersenne Twister
    implementation details, and uses a distinct `stream` tag per field so
    speed and distance are independently derived from the same index.
    """
    digest = hashlib.sha256(f"{_TELEMETRY_SEED}:{stream}:{index}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _telemetry_distance_m(index):
    """Deterministic distance-to-hazard sequence (meters), 0.0-20.0 m.

    Spans the 10.0 m Blocked/Clear threshold on both sides across trials,
    independently of `_telemetry_speed_m_s` (distinct hash stream).
    """
    return round(_telemetry_pseudo_unit("distance", index) * 20.0, 1)


def _telemetry_speed_m_s(index):
    """Deterministic vehicle speed sequence (m/s), 0.0-30.0 m/s inclusive."""
    return round(_telemetry_pseudo_unit("speed", index) * 30.0, 2)


def build_telemetry(index):
    """Build the telemetry string for trial `index`.

    Reproduces the CARLA telemetry construction (three fields -- Vehicle
    Speed, Traffic, Distance to Hazard; speed in m/s; a Blocked/Clear
    traffic status derived from a 10.0 m hazard-distance threshold).
    """
    speed = _telemetry_speed_m_s(index)
    distance = _telemetry_distance_m(index)
    traffic_status = "Blocked" if distance < 10.0 else "Clear"
    return (
        f"Vehicle Speed: {speed:.2f} m/s. "
        f"Traffic: {traffic_status}. "
        f"Distance to Hazard: {distance:.1f}m."
    )


def compute_config_hash():
    """Hash the per-call-site configuration of both Profiler and Planner calls,
    plus the telemetry construction.

    Only fixed configuration is hashed: model, system prompt, few-shot
    messages, options dict, and format argument for the two call sites, and
    the source text of the telemetry-construction functions -- not per-trial
    telemetry strings, profile-sentence text, or rendered values. The hash is
    stable across trials within a run (it never depends on a trial's actual
    inputs) and changes when the call-site configuration or the telemetry
    construction itself changes.
    """
    profiler_config = {
        "model": subject.MODEL,
        "system_prompt": subject.PROFILER_SYSTEM_PROMPT,
        "fewshot_messages": [
            {"role": "user", "content": subject.PROFILER_FEWSHOT_USER},
            {"role": "assistant", "content": subject.PROFILER_FEWSHOT_ASSISTANT},
        ],
        "options": {"temperature": 0.1},
        "format": None,
    }
    planner_config = {
        "model": subject.MODEL,
        "system_prompt": subject.PLANNER_SYSTEM_PROMPT,
        "fewshot_messages": [],
        "options": None,
        "format": "json",
    }
    telemetry_config = {
        "distance_formula": inspect.getsource(_telemetry_distance_m),
        "speed_formula": inspect.getsource(_telemetry_speed_m_s),
        "template_formula": inspect.getsource(build_telemetry),
    }
    combined = {"profiler": profiler_config, "planner": planner_config, "telemetry": telemetry_config}
    canonical = _canonical(combined)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


TIMING_FIELDS = [
    "total_duration",
    "load_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
]


def _extract_timing_fields(response):
    result = {}
    for field in TIMING_FIELDS:
        if hasattr(response, field):
            result[field] = getattr(response, field)
        elif isinstance(response, dict):
            result[field] = response.get(field)
        else:
            result[field] = None
    return result


def _extract_done_reason(response):
    if hasattr(response, "done_reason"):
        return response.done_reason
    if isinstance(response, dict):
        return response.get("done_reason")
    return None


def _extract_message_content(response):
    if hasattr(response, "message"):
        message = response.message
    elif isinstance(response, dict):
        message = response.get("message")
    else:
        message = None

    if message is None:
        return None
    if hasattr(message, "content"):
        return message.content
    if isinstance(message, dict):
        return message.get("content")
    return None


def run_trials(chain_fn, telemetries, run_id=None, envelope=None, notes="", run_dir=None,
                dry_run=False, dry_run_fail_at=None):
    """Run `len(telemetries)` trials through chain_fn.

    chain_fn(telemetry) -> (content_str, planner_response_obj, profiler_response_obj),
    matching subject.run_chain. Writes a run record and per-trial records to
    run_dir (if given) as JSON. Returns (run_record, trial_records).

    dry_run / dry_run_fail_at are recorded verbatim into run_record for
    provenance; they do not select chain_fn themselves -- the caller already
    decided that. Defaults (False / None) describe a live run.
    """
    if run_id is None:
        run_id = str(uuid.uuid4())
    if envelope is None:
        envelope = {}

    trials_intended = len(telemetries)
    start_time = _utc_now_iso()

    run_record = {
        "run_id": run_id,
        "utc_start": start_time,
        "utc_end": None,
        "status": "incomplete",
        "trials_intended": trials_intended,
        "trials_completed": 0,
        "config_hash": compute_config_hash(),
        "carla_present": envelope.get("carla_present"),
        "carla_map": envelope.get("carla_map"),
        "carla_ticking": envelope.get("carla_ticking"),
        "notes": notes,
        "model": subject.MODEL,
        "ollama_client_version": _get_ollama_client_version(),
        "dry_run": dry_run,
        "dry_run_fail_at": dry_run_fail_at,
    }

    def _persist():
        if run_dir is not None:
            _write_run_record(run_dir, run_record)
            _write_trial_records(run_dir, trial_records)

    trial_records = []
    _persist()

    trials_completed = 0
    for index, telemetry in enumerate(telemetries):
        start = time.perf_counter()
        try:
            content, planner_response, profiler_response = chain_fn(telemetry)
        except Exception as exc:
            wall_clock = time.perf_counter() - start
            trial_records.append({
                "trial_index": index,
                "category": "infrastructure_error",
                "all_violations": [],
                "telemetry": telemetry,
                "profiler_response_content": None,
                "profiler_done_reason": None,
                "planner_done_reason": None,
                "planner_raw_response_repr": repr(exc),
                "planner_response_content_len": 0,
                **{f"profiler_{field}": None for field in TIMING_FIELDS},
                **{f"planner_{field}": None for field in TIMING_FIELDS},
                "wall_clock_s": wall_clock,
            })
            run_record["trials_completed"] = trials_completed
            _persist()
            continue

        wall_clock = time.perf_counter() - start
        classification = classify.classify(content)
        profiler_timing = {f"profiler_{k}": v for k, v in _extract_timing_fields(profiler_response).items()}
        planner_timing = {f"planner_{k}": v for k, v in _extract_timing_fields(planner_response).items()}
        profiler_done_reason = _extract_done_reason(profiler_response)
        planner_done_reason = _extract_done_reason(planner_response)
        profiler_response_content = _extract_message_content(profiler_response)

        trial_records.append({
            "trial_index": index,
            "category": classification["category"],
            "all_violations": classification["all_violations"],
            "telemetry": telemetry,
            "profiler_response_content": profiler_response_content,
            "profiler_done_reason": profiler_done_reason,
            "planner_done_reason": planner_done_reason,
            "planner_raw_response_repr": repr(content),
            "planner_response_content_len": len(content),
            **profiler_timing,
            **planner_timing,
            "wall_clock_s": wall_clock,
        })

        trials_completed += 1
        run_record["trials_completed"] = trials_completed
        _persist()

    run_record["utc_end"] = _utc_now_iso()
    run_record["status"] = "complete" if trials_completed == trials_intended else "incomplete"
    _persist()

    return run_record, trial_records


def _write_run_record(run_dir, run_record):
    path = run_dir / "run.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2)


def _write_trial_records(run_dir, trial_records):
    path = run_dir / "trials.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trial_records, f, indent=2)


def summarize(trial_records):
    """Build a summary: category counts as n/N, latency median/IQR. No mean, no bare percentages."""
    n_total = len(trial_records)
    categories = {}
    for record in trial_records:
        cat = record["category"]
        categories[cat] = categories.get(cat, 0) + 1

    counts = {cat: f"{count}/{n_total}" for cat, count in categories.items()}

    wall_clocks = sorted(r["wall_clock_s"] for r in trial_records if r.get("wall_clock_s") is not None)

    def _quantile(data, q):
        if not data:
            return None
        return statistics.quantiles(data, n=100, method="inclusive")[q - 1] if len(data) > 1 else data[0]

    latency_summary = None
    if wall_clocks:
        median = statistics.median(wall_clocks)
        if len(wall_clocks) > 1:
            q1 = _quantile(wall_clocks, 25)
            q3 = _quantile(wall_clocks, 75)
            iqr = q3 - q1
        else:
            q1 = q3 = wall_clocks[0]
            iqr = 0.0
        latency_summary = {
            "median_s": median,
            "iqr_s": iqr,
            "q1_s": q1,
            "q3_s": q3,
            "n": len(wall_clocks),
        }

    return {
        "counts": counts,
        "n_total": n_total,
        "latency": latency_summary,
        "raw_wall_clocks_s": wall_clocks,
    }


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Run offline/online actuation trials.")
    parser.add_argument("--carla-present", type=str, default="false")
    parser.add_argument("--carla-map", type=str, default="")
    parser.add_argument("--carla-ticking", type=str, default="false")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a canned in-process chain instead of calling Ollama. No network call is made.",
    )
    parser.add_argument(
        "--dry-run-fail-at",
        type=int,
        default=None,
        help="With --dry-run, raise on this trial index (0-based) to simulate an infrastructure failure.",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default=None,
        help="Directory under which the run's output directory is created. "
        "Defaults to the repo-local runs/ directory next to this file.",
    )
    return parser


def make_run_id():
    """Filesystem-safe, sortable run id: UTC timestamp plus a short random suffix."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = secrets.token_hex(4)
    return f"{timestamp}-{suffix}"


def make_run_dir(runs_root, run_id):
    """Create runs_root/run_id, failing rather than overwriting an existing directory."""
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


VALID_DRY_RUN_PAYLOAD = '{"drive_mode": "Sport", "max_acceleration_m_s2": 3.0}'


class _DryRunResponse(dict):
    """Mimics the ollama response object well enough for run_trials' extraction helpers."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            return None


def _dry_run_chain_factory(fail_at=None):
    """Build a chain_fn(telemetry) -> (content, planner_response, profiler_response)
    that never calls Ollama.

    Returns canned, schema-valid responses for every trial except, if fail_at is
    given, raises on that trial index to simulate an infrastructure failure.
    Profiler and Planner timing values are deliberately distinct so downstream
    consumers can verify the two are not aliased.
    """
    counter = {"i": -1}

    def chain_fn(telemetry):
        counter["i"] += 1
        index = counter["i"]
        if fail_at is not None and index == fail_at:
            raise RuntimeError("simulated infrastructure failure (--dry-run-fail-at)")
        content = VALID_DRY_RUN_PAYLOAD
        profiler_response = _DryRunResponse({
            "message": {"content": "dry-run profile sentence"},
            "done_reason": "stop",
            "total_duration": 1,
            "load_duration": 2,
            "prompt_eval_count": 3,
            "prompt_eval_duration": 4,
            "eval_count": 5,
            "eval_duration": 6,
        })
        planner_response = _DryRunResponse({
            "message": {"content": content},
            "done_reason": "stop",
            "total_duration": 100,
            "load_duration": 200,
            "prompt_eval_count": 300,
            "prompt_eval_duration": 400,
            "eval_count": 500,
            "eval_duration": 600,
        })
        return content, planner_response, profiler_response

    return chain_fn


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    envelope = {
        "carla_present": args.carla_present,
        "carla_map": args.carla_map,
        "carla_ticking": args.carla_ticking,
    }

    telemetries = [build_telemetry(i) for i in range(args.trials)]

    if args.dry_run:
        chain_fn = _dry_run_chain_factory(fail_at=args.dry_run_fail_at)
    else:
        chain_fn = subject.run_chain

    run_id = make_run_id()
    runs_root = Path(args.runs_root) if args.runs_root else Path(__file__).resolve().parent / "runs"
    run_dir = make_run_dir(runs_root, run_id)

    run_record, trial_records = run_trials(
        chain_fn, telemetries, run_id=run_id, envelope=envelope, notes=args.notes, run_dir=run_dir,
        dry_run=args.dry_run, dry_run_fail_at=args.dry_run_fail_at,
    )

    summary = summarize(trial_records)
    print(_canonical(run_record))
    print(_canonical(summary))


if __name__ == "__main__":
    main()
