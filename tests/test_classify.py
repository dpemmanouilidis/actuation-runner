import ast
import json
from pathlib import Path

import classify


VALID_PAYLOAD = '{"drive_mode": "Sport", "max_acceleration_m_s2": 3.0}'
EMPTY = ""
TRUNCATED = '{"drive_mode": "Sport"'
MISSING_KEY = '{"drive_mode": "Sport"}'
BAD_ENUM = '{"drive_mode": "Track", "max_acceleration_m_s2": 3.0}'
BAD_BOUNDS = '{"drive_mode": "Sport", "max_acceleration_m_s2": 6.0}'
BAD_TYPE = '{"drive_mode": "Sport", "max_acceleration_m_s2": "fast"}'
BAD_ENUM_AND_BOUNDS = '{"drive_mode": "Track", "max_acceleration_m_s2": 6.0}'


def test_valid_payload_passes():
    result = classify.classify(VALID_PAYLOAD)
    assert result["category"] == "pass"
    assert result["all_violations"] == []


def test_empty_response():
    result = classify.classify(EMPTY)
    assert result["category"] == "empty_response"


def test_truncated_is_malformed_json_not_empty():
    result = classify.classify(TRUNCATED)
    assert result["category"] == "malformed_json"


def test_missing_required_key():
    result = classify.classify(MISSING_KEY)
    assert result["category"] == "missing_required_key"
    assert "required" in result["all_violations"]


def test_out_of_enum_mode():
    result = classify.classify(BAD_ENUM)
    assert result["category"] == "out_of_enum_mode"
    assert "enum" in result["all_violations"]


def test_out_of_bounds_value():
    result = classify.classify(BAD_BOUNDS)
    assert result["category"] == "out_of_bounds_value"
    assert "maximum" in result["all_violations"]


def test_wrong_type():
    result = classify.classify(BAD_TYPE)
    assert result["category"] == "wrong_type"
    assert "type" in result["all_violations"]


def test_simultaneous_enum_and_bounds_violation_records_both():
    result = classify.classify(BAD_ENUM_AND_BOUNDS)
    assert "enum" in result["all_violations"]
    assert "maximum" in result["all_violations"]
    assert len(result["all_violations"]) >= 2


def test_empty_checked_before_malformed_json():
    # '' also fails json.loads; must not be absorbed into malformed_json.
    assert classify.classify("") == {"category": "empty_response", "all_violations": [], "payload": None}


def test_no_ollama_or_carla_import_in_classify_module():
    source = Path(classify.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_names.add(node.module.split(".")[0])
    assert "ollama" not in imported_names
    assert "carla" not in imported_names


def test_finding_missing_required_key_validator_attribute():
    """Explicit finding: for {"drive_mode": "Sport"}, iter_errors yields an error
    whose .validator attribute is exactly 'required'."""
    import jsonschema

    payload = json.loads(MISSING_KEY)
    validator = jsonschema.Draft7Validator(classify.SCHEMA)
    errors = list(validator.iter_errors(payload))
    assert len(errors) == 1
    assert errors[0].validator == "required"
