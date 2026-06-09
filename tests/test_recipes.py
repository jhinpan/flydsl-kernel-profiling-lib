"""Validate tools/data/recipes.json.

A recipe tells the profiling harness how to run a kernel under rocprofv3. A stale
recipe — e.g. a ``test_file`` that was renamed/removed upstream — silently breaks
capture (the kernel never launches). This regression test guards that class:

  * structural checks run everywhere (required keys, types, the invocation actually
    references the test file it names);
  * the existence check resolves each ``test_file`` against the FlyDSL worktree and,
    when that worktree is present, asserts every recipe still points at a real file.
    It skips gracefully when no worktree is available (portable / CI-friendly).

Run:  pytest tests/test_recipes.py
      FLYPROF_WORKTREE=/path/to/FlyDSL pytest tests/test_recipes.py   # force existence check
"""
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RECIPES = REPO / "tools" / "data" / "recipes.json"
WORKTREE = Path(os.environ.get("FLYPROF_WORKTREE", "/sgl-workspace/FlyDSL-lab"))

REQUIRED = ("test_file", "kernels", "arch", "op_category", "invocation")


def _recipes():
    data = json.loads(RECIPES.read_text())
    assert isinstance(data, list) and data, "recipes.json must be a non-empty list"
    return data


def _resolve(test_file: str) -> Path:
    p = Path(test_file)
    return p if p.is_absolute() else WORKTREE / test_file


def test_recipes_parse():
    _recipes()


@pytest.mark.parametrize("rec", _recipes(), ids=lambda r: Path(r.get("test_file", "?")).stem)
def test_recipe_structure(rec):
    for k in REQUIRED:
        assert k in rec, f"missing required key {k!r}"
    tf = rec["test_file"]
    assert tf.endswith(".py"), f"test_file must be a .py: {tf}"
    base = os.path.basename(tf)
    assert base.startswith(("test_", "bench_")), f"unexpected test file name: {base}"
    assert isinstance(rec["kernels"], list) and rec["kernels"], "kernels must be a non-empty list"
    assert isinstance(rec["invocation"], str) and rec["invocation"].strip(), "invocation must be a non-empty string"
    # the invocation must run the file the recipe names (catches test_file/invocation drift)
    assert base in rec["invocation"], f"invocation does not reference {base}: {rec['invocation']!r}"


@pytest.mark.parametrize("rec", _recipes(), ids=lambda r: Path(r.get("test_file", "?")).stem)
def test_recipe_test_file_exists(rec):
    """Every recipe must point at a real test file (skipped when no worktree is present)."""
    resolved = _resolve(rec["test_file"])
    # Only enforce when the worktree these recipes target is actually checked out.
    if not WORKTREE.exists() and not Path(rec["test_file"]).is_absolute():
        pytest.skip(f"no FlyDSL worktree at {WORKTREE}; cannot resolve relative test_file")
    if Path(rec["test_file"]).is_absolute() and not WORKTREE.exists():
        # absolute paths point into a specific machine's worktree; skip off-box
        pytest.skip("FlyDSL worktree absent; absolute test_file not resolvable here")
    assert resolved.exists(), f"recipe test_file does not exist: {resolved} (renamed/removed upstream?)"
