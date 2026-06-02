"""Validate ledger / result rows against the JSON schemas in schemas/.

Uses `jsonschema` when available; falls back to a minimal required-field + enum
check otherwise so the harness still runs on a bare box.
"""

from __future__ import annotations

import json
import os

_SCHEMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schemas")
_CACHE: dict[str, dict] = {}


def _load(name: str) -> dict:
    if name not in _CACHE:
        with open(os.path.join(_SCHEMA_DIR, name)) as f:
            _CACHE[name] = json.load(f)
    return _CACHE[name]


def _minimal_check(row: dict, schema: dict) -> list[str]:
    errs = []
    for req in schema.get("required", []):
        if req not in row:
            errs.append(f"missing required field '{req}'")
    for key, spec in schema.get("properties", {}).items():
        if key in row and isinstance(spec, dict) and "enum" in spec and row[key] not in spec["enum"]:
            errs.append(f"'{key}'={row[key]!r} not in enum {spec['enum']}")
    return errs


def validate_rows(rows: list[dict], schema_name: str) -> list[tuple[int, str]]:
    """Return list of (row_index, error). Empty == all valid."""
    schema = _load(schema_name)
    out: list[tuple[int, str]] = []
    try:
        import jsonschema
        validator = jsonschema.Draft7Validator(schema)
        for i, row in enumerate(rows):
            for e in validator.iter_errors(row):
                loc = "/".join(str(p) for p in e.path) or "<root>"
                out.append((i, f"{loc}: {e.message}"))
    except ImportError:
        for i, row in enumerate(rows):
            for msg in _minimal_check(row, schema):
                out.append((i, msg))
    return out


def validate_ledger(rows: list[dict]) -> list[tuple[int, str]]:
    return validate_rows(rows, "shape_ledger.schema.json")


def validate_results(rows: list[dict]) -> list[tuple[int, str]]:
    return validate_rows(rows, "benchmark_result.schema.json")
