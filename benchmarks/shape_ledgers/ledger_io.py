"""Idempotent ledger merge shared by all importers.

Each importer owns one or more source.kind values. upsert_ledger() drops prior
rows from those kinds and adds the new ones, dedups by shape_id, and rewrites a
stably-sorted file -- so re-running any importer never clobbers rows another
importer (synthetic / sglang_trace / diagnostic) contributed.
"""

from __future__ import annotations

import os

from benchmarks import common, validate


def _sort_key(r: dict):
    a = r.get("args", {})
    return (r.get("op_type", ""), r.get("stage", ""), r.get("dtype", ""),
            a.get("N", 0), a.get("M", 0), r.get("shape_id", ""))


def upsert_ledger(path: str, new_rows: list[dict], replace_kinds: set[str],
                  *, validate_rows: bool = True) -> dict:
    existing = common.read_jsonl(path) if os.path.exists(path) else []
    kept = [r for r in existing if r.get("source", {}).get("kind") not in replace_kinds]
    by_id: dict[str, dict] = {}
    for r in kept + new_rows:
        by_id[r["shape_id"]] = r            # last writer wins
    rows = sorted(by_id.values(), key=_sort_key)
    if validate_rows:
        errs = validate.validate_ledger(rows)
        if errs:
            raise ValueError(f"{path}: {len(errs)} schema error(s); first: {errs[0]}")
    common.write_jsonl(path, rows)
    return {
        "path": path,
        "total": len(rows),
        "added_kinds": sorted(replace_kinds),
        "kept_other": len(kept),
        "new": len(new_rows),
    }
