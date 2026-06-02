#!/usr/bin/env bash
# Source this before running any GPU benchmark runner. Verified recipe for this
# MI350X node: the FlyDSL build tree provides flydsl._mlir, which ALSO unblocks
# `import aiter` (aiter/__init__ imports flydsl.expr transitively).
export FLYDSL_LAB="${FLYDSL_LAB:-/sgl-workspace/FlyDSL-lab}"
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${FLYDSL_LAB}/build-fly/python_packages:${FLYDSL_LAB}:${_REPO_ROOT}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${FLYDSL_LAB}/build-fly/python_packages/flydsl/_mlir/_mlir_libs:${LD_LIBRARY_PATH}"
export SGLANG_USE_AITER="${SGLANG_USE_AITER:-0}"   # let standalone triton import without forcing aiter
