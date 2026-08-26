import json
import jsonschema

SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "VehicleSafetyGate",
    "type": "object",
    "properties": {
        "drive_mode": {"type": "string", "enum": ["Eco", "Comfort", "Sport", "Emergency"]},
        "max_acceleration_m_s2": {"type": "number", "minimum": 0.0, "maximum": 5.0},
    },
    "required": ["drive_mode", "max_acceleration_m_s2"],
}

_VALIDATOR = jsonschema.Draft7Validator(SCHEMA)

_CATEGORY_ORDER = [
    "missing_required_key",
    "out_of_enum_mode",
    "out_of_bounds_value",
    "wrong_type",
]

_VALIDATOR_TO_CATEGORY = {
    "required": "missing_required_key",
    "enum": "out_of_enum_mode",
    "maximum": "out_of_bounds_value",
    "minimum": "out_of_bounds_value",
    "type": "wrong_type",
}


def classify(response_content):
    """Classify a raw Planner response string.

    Returns a dict with keys: category, all_violations, payload.
    Consumes strings only; knows nothing about how the string was produced.
    """
    if response_content == "":
        return {"category": "empty_response", "all_violations": [], "payload": None}

    try:
        payload = json.loads(response_content)
    except (json.JSONDecodeError, ValueError):
        return {"category": "malformed_json", "all_violations": [], "payload": None}

    errors = list(_VALIDATOR.iter_errors(payload))
    if not errors:
        return {"category": "pass", "all_violations": [], "payload": payload}

    violated_validators = [e.validator for e in errors]
    categories_present = {_VALIDATOR_TO_CATEGORY[v] for v in violated_validators if v in _VALIDATOR_TO_CATEGORY}

    primary = None
    for cat in _CATEGORY_ORDER:
        if cat in categories_present:
            primary = cat
            break
    if primary is None:
        primary = "wrong_type"

    return {
        "category": primary,
        "all_violations": violated_validators,
        "payload": payload,
    }
