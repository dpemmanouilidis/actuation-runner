import ollama

MODEL = "qwen3.5:9b"

PROFILER_SYSTEM_PROMPT = (
    "You are a strict data parser. You output EXACTLY one short sentence. "
    "No introductions, no explanations, no lists."
)

PROFILER_FEWSHOT_USER = (
    "Analyze this vehicle telemetry: Vehicle Speed: 120 km/h. Traffic: Clear. "
    "Driver Heart Rate: 110 bpm."
)

PROFILER_FEWSHOT_ASSISTANT = (
    "The driver is traveling at high speed with an elevated heart rate in clear traffic."
)

PLANNER_SYSTEM_PROMPT = (
    "You are a vehicle planner agent. You only output raw JSON dictionaries without markdown blocks."
)


def profiler_call_args(telemetry):
    """Build the exact keyword arguments passed to ollama.chat() for the Profiler call."""
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": PROFILER_SYSTEM_PROMPT},
            {"role": "user", "content": PROFILER_FEWSHOT_USER},
            {"role": "assistant", "content": PROFILER_FEWSHOT_ASSISTANT},
            {"role": "user", "content": f"Analyze this vehicle telemetry: {telemetry}"},
        ],
        "options": {"temperature": 0.1},
    }


def planner_call_args(profile_sentence):
    """Build the exact keyword arguments passed to ollama.chat() for the Planner call."""
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": profile_sentence},
        ],
        "format": "json",
    }


def run_chain(telemetry):
    """Run the Profiler->Planner chain against Ollama.

    Returns (planner_content, planner_response, profiler_response) where
    planner_content is the raw string content of the Planner's response, and
    planner_response / profiler_response are the full response objects
    returned by ollama.chat() for each call site.
    """
    profiler_response = ollama.chat(**profiler_call_args(telemetry))
    profile_sentence = profiler_response["message"]["content"]

    planner_response = ollama.chat(**planner_call_args(profile_sentence))
    planner_content = planner_response["message"]["content"]

    return planner_content, planner_response, profiler_response
