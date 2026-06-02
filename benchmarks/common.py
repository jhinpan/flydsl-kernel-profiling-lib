"""Shared spine for the FlyDSL multi-shape benchmark harness.

Everything that more than one module needs lives here so the importers,
providers, runners, and report generators agree on:

  * stable shape hashing (canonical JSON -> sha1)
  * the on-disk dtype / tolerance vocabulary
  * GPU kernel timing (event timing + L2 flush + launch-overhead amortization)
  * environment bootstrap for the FlyDSL build tree (which ALSO unblocks `import aiter`
    on this node -- see notes below)
  * provenance helpers (rocm version, gpu name, arch, component git commits)

Design rule: nothing in this module imports torch at import time, so the
pure-data tools (importers, report generators) run on a CPU-only box. Functions
that need torch import it lazily.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from typing import Any, Iterable

import numpy as np

# --------------------------------------------------------------------------- #
# Environment bootstrap
# --------------------------------------------------------------------------- #
# On this node `flydsl` is pip-installed editable from FlyDSL-lab/python, but
# that tree has NO compiled _mlir extension. The built extension lives in the
# build tree. Putting the build tree first on sys.path makes `import flydsl.expr`
# resolve to the built copy -- which (verified) also unblocks `import aiter`,
# whose __init__ imports flydsl.expr transitively. The _mlir .so additionally
# needs its _mlir_libs dir on LD_LIBRARY_PATH; LD_LIBRARY_PATH must be set BEFORE
# the process starts (the loader reads it at exec), so GPU runners are launched
# via benchmarks/env.sh (or the `benchmarks/bench` wrapper). bootstrap_env()
# handles the sys.path (PYTHONPATH) half so imports resolve; flydsl_runtime_ok()
# reports whether the native half is actually loadable.
FLYDSL_LAB = os.environ.get("FLYDSL_LAB", "/sgl-workspace/FlyDSL-lab")
FLYDSL_BUILD_PKGS = os.path.join(FLYDSL_LAB, "build-fly", "python_packages")
FLYDSL_MLIR_LIBS = os.path.join(FLYDSL_BUILD_PKGS, "flydsl", "_mlir", "_mlir_libs")


def bootstrap_env() -> None:
    """Prepend the FlyDSL build tree (and repo root) to sys.path. Idempotent.

    Covers the PYTHONPATH half of the recipe so `import flydsl.*` / `import aiter`
    resolve to the built tree. Does NOT touch LD_LIBRARY_PATH (must be preset)."""
    for p in (FLYDSL_BUILD_PKGS, FLYDSL_LAB):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    # so `import benchmarks.*` works when launched as a plain script
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def flydsl_runtime_ok() -> tuple[bool, str | None]:
    """Return (ok, reason). ok == native flydsl._mlir is importable in THIS process."""
    bootstrap_env()
    if not os.path.isdir(FLYDSL_MLIR_LIBS):
        return False, f"missing build tree {FLYDSL_MLIR_LIBS} (FlyDSL not built)"
    if FLYDSL_MLIR_LIBS not in os.environ.get("LD_LIBRARY_PATH", ""):
        # native libs likely won't load; let the caller know to use env.sh
        try:
            import flydsl._mlir  # noqa: F401
        except Exception as e:  # pragma: no cover - depends on loader state
            return False, (
                "flydsl._mlir not loadable; launch via benchmarks/env.sh so "
                f"LD_LIBRARY_PATH includes {FLYDSL_MLIR_LIBS} ({type(e).__name__})"
            )
    try:
        import flydsl._mlir  # noqa: F401
        return True, None
    except Exception as e:  # pragma: no cover
        return False, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
# dtype + tolerance vocabulary
# --------------------------------------------------------------------------- #
# Canonical dtype strings used in the shape ledger ("bf16" etc). FlyDSL builders
# use a different spelling ("f32"/"f16"/"bf16"); FLYDSL_DTYPE maps to that.
def torch_dtype(name: str):
    import torch

    return {
        "fp32": torch.float32, "f32": torch.float32, "float32": torch.float32,
        "fp16": torch.float16, "f16": torch.float16, "float16": torch.float16,
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "fp8": torch.float8_e4m3fnuz, "fp8_e4m3": torch.float8_e4m3fnuz,
        "fp8_e5m2": torch.float8_e5m2fnuz,
    }[name]


FLYDSL_DTYPE = {
    "fp32": "f32", "f32": "f32", "float32": "f32",
    "fp16": "f16", "f16": "f16", "float16": "f16",
    "bf16": "bf16", "bfloat16": "bf16",
}

# (rtol, atol) for comparing a kernel output (upcast to fp32) against the fp32
# reference. From the discovery probe; tuned to each dtype's mantissa width.
TOL = {
    "fp32": (1e-5, 1e-6),
    "f32": (1e-5, 1e-6),
    "fp16": (2e-3, 1e-3),
    "f16": (2e-3, 1e-3),
    "bf16": (1.6e-2, 1e-2),
    "bfloat16": (1.6e-2, 1e-2),
    "fp8": (1.5e-1, 1.5e-1),
    "fp8_e4m3": (1.5e-1, 1.5e-1),
    "fp8_e5m2": (1.5e-1, 1.5e-1),
}


def tol_for(dtype: str) -> tuple[float, float]:
    return TOL.get(dtype, (1.6e-2, 1e-2))


# --------------------------------------------------------------------------- #
# Stable shape id
# --------------------------------------------------------------------------- #
def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, no NaN. Never repr()."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalize(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _normalize(v[k]) for k in sorted(v)}
    if isinstance(v, (list, tuple)):
        return [_normalize(x) for x in v]
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def stable_shape_id(*, op_type: str, model: str, stage: str, dtype: str,
                    layout: dict, args: dict, extra: dict | None = None) -> str:
    """sha1 over canonical JSON of the identity-defining fields. Stable across runs."""
    payload = _normalize({
        "op_type": op_type, "model": model, "stage": stage,
        "dtype": dtype, "layout": layout or {}, "args": args or {},
        **({"extra": extra} if extra else {}),
    })
    h = hashlib.sha1(canonical_json(payload).encode("utf-8")).hexdigest()
    return "sha1:" + h[:16]


# --------------------------------------------------------------------------- #
# JSONL helpers
# --------------------------------------------------------------------------- #
def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln}: bad JSON: {e}") from e
    return rows


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    n = 0
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(canonical_json(r) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.strip() or None
    except Exception:
        return None


def git_short(repo: str) -> str | None:
    return _run(["git", "-C", repo, "rev-parse", "--short", "HEAD"])


def rocm_version() -> str | None:
    for p in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
        try:
            with open(p) as f:
                return f.read().strip()
        except OSError:
            continue
    return None


def gpu_name() -> str | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def arch() -> str | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        pass
    return None


def provenance() -> dict:
    """One call -> the scope fields every benchmark_result row carries."""
    return {
        "gpu": gpu_name(),
        "arch": arch(),
        "rocm_version": rocm_version(),
        "driver_version": _run(["bash", "-lc", "cat /sys/module/amdgpu/version 2>/dev/null"]),
        "flydsl_commit": git_short(FLYDSL_LAB),
        "aiter_commit": git_short("/sgl-workspace/aiter"),
        "sglang_commit": git_short("/sgl-workspace/sglang"),
        "torch_version": _torch_version(),
        "triton_version": _triton_version(),
    }


def _torch_version() -> str | None:
    try:
        import torch
        return torch.__version__
    except Exception:
        return None


def _triton_version() -> str | None:
    try:
        import triton
        return triton.__version__
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
_L2_FLUSH_BYTES = 256 * 1024 * 1024  # Triton's AMD default L2-flush buffer size


def _stats(times_us: list[float], *, loops_per_measure: int = 1) -> dict:
    a = np.asarray(times_us, dtype=np.float64)
    if a.size == 0:
        return {k: None for k in
                ("median_us", "mean_us", "std_us", "min_us", "p10_us", "p90_us")} | {
                    "n_iters": 0, "loops_per_measure": loops_per_measure, "stable": False}
    p10, p90 = np.percentile(a, [10, 90])  # numpy linear interp, matches triton
    med = float(np.median(a))
    return {
        "median_us": med,
        "mean_us": float(a.mean()),
        "std_us": float(a.std()),
        "min_us": float(a.min()),
        "p10_us": float(p10),
        "p90_us": float(p90),
        "n_iters": int(a.size),
        "loops_per_measure": int(loops_per_measure),
        # a measurement is "unstable" if the p90/p10 spread exceeds 20% (the
        # spec's profiler-trigger threshold); guards against bad short-kernel timing
        "stable": bool(p10 > 0 and (p90 / p10) <= 1.20),
    }


def benchmark(fn, *, warmup_iters: int = 20, repeat_iters: int = 100,
              flush_l2: bool = True, auto_loops: bool = True,
              target_interval_us: float = 50.0, max_loops: int = 500) -> dict:
    """Time `fn` (a no-arg closure that launches exactly the kernel under test).

    Inputs MUST be allocated outside `fn` so allocation is excluded. JIT/autotune
    is excluded by the warmup. For very short kernels we run `loops` launches per
    measured interval and divide, which amortizes host launch overhead and the
    coarse ROCm event-timer quantization without needing CUDA-graph capture
    (graph capture is fragile across providers that allocate their own output).

    Returns the stats dict consumed by the benchmark_result schema.
    """
    import torch

    torch.cuda.synchronize()
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    # estimate single-launch cost to pick loops-per-measurement
    loops = 1
    if auto_loops:
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(5):
            fn()
        e1.record()
        torch.cuda.synchronize()
        est_us = e0.elapsed_time(e1) * 1e3 / 5.0
        if est_us > 0:
            loops = int(np.clip(round(target_interval_us / est_us), 1, max_loops))

    cache = (torch.empty(_L2_FLUSH_BYTES // 4, dtype=torch.int32, device="cuda")
             if flush_l2 else None)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]
    for i in range(repeat_iters):
        if cache is not None:
            cache.zero_()          # evict L2 so memory-bound kernels read cold
        starts[i].record()
        for _ in range(loops):
            fn()
        ends[i].record()
    torch.cuda.synchronize()       # REQUIRED before elapsed_time on ROCm
    times_us = [starts[i].elapsed_time(ends[i]) * 1e3 / loops for i in range(repeat_iters)]
    return _stats(times_us, loops_per_measure=loops)


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def benchmark_cudagraph(fn, *, warmup_iters: int = 15, rep_ms: float = 15.0,
                        n_retries: int = 15, max_capture: int = 2000) -> dict:
    """Kernel-only timing via CUDA/HIP graph capture.

    Pays host launch overhead + JIT/autotune + allocation ONCE (at warmup/capture),
    then replays N unrolled launches and divides -- so the result is pure device
    time, the fair metric for comparing kernel implementations on short shapes
    where Python launch overhead would otherwise dominate. Raises on capture
    failure (caller falls back to eager event timing).

    IMPORTANT: closures that launch on a specific stream MUST fetch
    torch.cuda.current_stream() at CALL time so they land on the capture stream.
    """
    import torch

    # warm on the default stream first: triggers JIT + Triton autotune (which
    # itself benchmarks and so cannot run inside capture) and caches them
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    # estimate single-launch device cost to size the capture
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(10):
        fn()
    e1.record()
    torch.cuda.synchronize()
    est_us = e0.elapsed_time(e1) * 1e3 / 10.0
    n = int(np.clip(round(rep_ms * 1000.0 / est_us) if est_us > 0 else max_capture, 1, max_capture))

    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    torch.cuda.synchronize()

    times_us = []
    for _ in range(n_retries):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        times_us.append(a.elapsed_time(b) * 1e3 / n)
    stats = _stats(times_us, loops_per_measure=n)
    stats["timing_method"] = "cudagraph"
    return stats


def measure_both(fn, *, warmup_iters: int = 20, repeat_iters: int = 100) -> dict:
    """Return {graph, eager, primary_us, eager_us, host_overhead_us, timing_method}.

    primary = kernel-only graph median when capture succeeds (the fair metric),
    else eager event median. host_overhead_us = eager - graph (per-call host
    launch cost), surfaced as a first-class signal."""
    eager = benchmark(fn, warmup_iters=warmup_iters, repeat_iters=repeat_iters)
    eager["timing_method"] = "eager_event"
    graph = None
    graph_err = None
    try:
        graph = benchmark_cudagraph(fn, warmup_iters=max(10, warmup_iters // 2))
    except Exception as e:  # capture not supported for this fn/shape
        graph_err = f"{type(e).__name__}: {e}"

    if graph and graph.get("median_us") is not None:
        primary = dict(graph)
        primary["timing_method"] = "cudagraph"
        host_ovh = (eager["median_us"] - graph["median_us"]
                    if eager.get("median_us") is not None else None)
    else:
        primary = dict(eager)
        host_ovh = None
    primary["eager_median_us"] = eager.get("median_us")
    primary["graph_median_us"] = graph.get("median_us") if graph else None
    primary["host_overhead_us"] = host_ovh
    primary["graph_capture_error"] = graph_err
    return primary


def make_generator(seed: int):
    import torch
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return g


def speedup(provider_us: float | None, flydsl_us: float | None) -> float | None:
    """provider_median / flydsl_median. >1 => FlyDSL faster. None if unmeasurable."""
    if not provider_us or not flydsl_us or flydsl_us <= 0:
        return None
    return provider_us / flydsl_us


def geomean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and v > 0]
    if not vals:
        return None
    return float(np.exp(np.mean(np.log(vals))))


def weighted_geomean(pairs: list[tuple[float, float]]) -> float | None:
    """pairs of (value, weight); value>0, weight>=0."""
    num = 0.0
    den = 0.0
    for v, w in pairs:
        if v is not None and v > 0 and w is not None and w > 0:
            num += w * np.log(v)
            den += w
    if den <= 0:
        return None
    return float(np.exp(num / den))
