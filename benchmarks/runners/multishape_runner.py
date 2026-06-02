"""Multi-shape benchmark orchestrator.

For one kernel: load the shape ledger + baseline matrix, build deterministic
inputs ONCE per shape, run the fp32 reference, then every enabled provider
(correctness + timing). Failed / unsupported / incorrect / disabled providers
are KEPT in the output with an explicit status -- never silently dropped.

Emits benchmark_results.jsonl + benchmark_results.csv. Speedups / summary /
coverage are computed by the report generators from these rows.

  benchmarks/bench -m benchmarks.runners.multishape_runner \
    --op rmsnorm \
    --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
    --baseline-matrix benchmarks/examples/rmsnorm/baseline_matrix.yaml \
    --out benchmarks/examples/rmsnorm --warmup-iters 25 --repeat-iters 100
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import traceback

import yaml

from benchmarks import common, ops, validate
from benchmarks.providers import base

_CSV_COLS = ["run_id", "shape_id", "model", "stage", "dtype", "op_type", "args",
             "provider", "provider_detail", "correct", "benchmark_status",
             "median_us", "timing_method", "eager_median_us", "host_overhead_us",
             "graph_median_us", "cache_state", "graph_replay_count",
             "p10_us", "p90_us", "mean_us", "std_us", "min_us",
             "stable", "effective_gbps", "effective_tflops",
             "includes_allocation", "includes_layout_conversion", "includes_jit",
             "skip_reason", "candidate_commit", "baseline_commit", "gpu", "arch",
             "rocm_version", "driver_version", "torch_version", "triton_version",
             "sglang_commit"]


def _now_id(op: str, prov: dict) -> str:
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{op}-{prov.get('flydsl_commit') or 'nocommit'}-{ts}"


def run(op_type: str, ledger_path: str, matrix_path: str, out_dir: str, *,
        warmup_iters: int, repeat_iters: int, seed: int, limit: int | None,
        stages: set[str] | None) -> dict:
    import torch

    rows = [r for r in common.read_jsonl(ledger_path) if r.get("op_type") == op_type]
    if stages:
        rows = [r for r in rows if r.get("stage") in stages]
    errs = validate.validate_ledger(rows)
    if errs:
        raise ValueError(f"ledger invalid: {len(errs)} error(s); first {errs[0]}")
    if limit:
        rows = rows[:limit]

    matrix = yaml.safe_load(open(matrix_path))
    prov_specs = matrix["providers"]
    ref_name = matrix.get("reference_provider", "pytorch")

    op = ops.get_op(op_type)
    if op is None:
        raise ValueError(f"no Op registered for {op_type}")
    prov = common.provenance()
    run_id = _now_id(op_type, prov)

    # instantiate adapters once (launcher/JIT caches persist across shapes)
    adapters: dict[str, base.ProviderAdapter] = {}
    for pname, spec in prov_specs.items():
        if not spec.get("enabled", False):
            continue
        try:
            adapters[pname] = base.load_entrypoint(spec["entrypoint"], op_type)
        except Exception as e:  # bad entrypoint -> keep as failed below
            adapters[pname] = e  # type: ignore

    results: list[dict] = []
    n = len(rows)
    for i, shape in enumerate(rows, 1):
        sid = shape["shape_id"]
        print(f"[{i}/{n}] {sid} {op.args_summary(shape)} {shape['dtype']} ({shape['stage']})", flush=True)
        try:
            inputs = op.make_inputs(shape, seed)
            reference = op.reference(shape, inputs)
        except Exception as e:
            for pname, spec in prov_specs.items():
                results.append(_row(run_id, shape, spec.get("provider", pname), prov,
                                    status="failed", skip_reason=f"input/reference build: {e}",
                                    warmup_iters=warmup_iters, repeat_iters=repeat_iters))
            continue

        for pname, spec in prov_specs.items():
            provider = spec.get("provider", pname)
            if not spec.get("enabled", False):
                results.append(_row(run_id, shape, provider, prov, status="not_configured",
                                    skip_reason=spec.get("skip_reason", "disabled in baseline_matrix"),
                                    warmup_iters=warmup_iters, repeat_iters=repeat_iters))
                continue
            adapter = adapters.get(pname)
            if isinstance(adapter, Exception):
                results.append(_row(run_id, shape, provider, prov, status="failed",
                                    skip_reason=f"adapter load: {adapter}",
                                    warmup_iters=warmup_iters, repeat_iters=repeat_iters))
                continue
            try:
                supported, reason = adapter.supports(shape)
            except Exception as e:
                supported, reason = False, f"supports() raised: {e}"
            if not supported:
                results.append(_row(run_id, shape, provider, prov, status="unsupported",
                                    skip_reason=reason, detail=getattr(adapter, "provider_detail", ""),
                                    warmup_iters=warmup_iters, repeat_iters=repeat_iters,
                                    adapter=adapter))
                continue
            # correctness first
            try:
                correct, cerr = adapter.check_correctness(shape, inputs, reference)
            except Exception as e:
                correct, cerr = False, f"{type(e).__name__}: {e}"
            # timing (kept even if incorrect, but labeled)
            stats, status, berr = {}, "ok", None
            try:
                stats = common.measure_both(lambda: adapter.run(shape, inputs),
                                            warmup_iters=warmup_iters, repeat_iters=repeat_iters)
            except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
                status, berr = "oom", str(e)
                torch.cuda.empty_cache()
            except Exception as e:
                status, berr = "failed", f"{type(e).__name__}: {e}"
                traceback.print_exc()
            if status == "ok" and not correct:
                status = "incorrect"
            eff = op.effective(shape, stats.get("median_us")) if stats else {"effective_gbps": None, "effective_tflops": None}
            results.append(_row(run_id, shape, provider, prov, status=status,
                                correct=correct, cerr=cerr or berr, stats=stats, eff=eff,
                                detail=getattr(adapter, "provider_detail", ""),
                                warmup_iters=warmup_iters, repeat_iters=repeat_iters, adapter=adapter))
        del inputs, reference
        if i % 8 == 0:
            torch.cuda.empty_cache()

    # validate + write
    rerrs = validate.validate_results(results)
    if rerrs:
        print(f"WARNING: {len(rerrs)} result rows failed schema; first: {rerrs[0]}")
    jsonl = os.path.join(out_dir, "benchmark_results.jsonl")
    common.write_jsonl(jsonl, results)
    _write_csv(os.path.join(out_dir, "benchmark_results.csv"), results)
    print(f"\nWrote {len(results)} result rows for {n} shapes -> {jsonl}")
    return {"run_id": run_id, "results": jsonl, "n_shapes": n, "n_rows": len(results)}


def _row(run_id, shape, provider, prov, *, status, correct=None, cerr=None, stats=None,
         eff=None, detail="", skip_reason=None, warmup_iters=0, repeat_iters=0, adapter=None) -> dict:
    stats = stats or {}
    eff = eff or {}
    return {
        "run_id": run_id,
        "shape_id": shape["shape_id"],
        "model": shape.get("model"),
        "stage": shape.get("stage"),
        "dtype": shape.get("dtype"),
        "op_type": shape.get("op_type"),
        "args": shape.get("args"),
        "provider": provider,
        "provider_detail": detail or "",
        "candidate_commit": prov.get("flydsl_commit") if provider == "flydsl" else (
            prov.get("aiter_commit") if provider.startswith("aiter") else None),
        "baseline_commit": prov.get("aiter_commit") if provider.startswith("aiter") else None,
        "gpu": prov.get("gpu"), "arch": prov.get("arch"), "rocm_version": prov.get("rocm_version"),
        "driver_version": prov.get("driver_version"),
        "torch_version": prov.get("torch_version"),
        "triton_version": prov.get("triton_version"),
        "sglang_commit": prov.get("sglang_commit"),
        "correct": correct, "correctness_error": cerr,
        "warmup_iters": warmup_iters, "repeat_iters": repeat_iters,
        "loops_per_measure": stats.get("loops_per_measure"),
        "median_us": stats.get("median_us"), "mean_us": stats.get("mean_us"),
        "std_us": stats.get("std_us"), "min_us": stats.get("min_us"),
        "p10_us": stats.get("p10_us"), "p90_us": stats.get("p90_us"),
        "stable": stats.get("stable"),
        "timing_method": stats.get("timing_method"),
        "eager_median_us": stats.get("eager_median_us"),
        "graph_median_us": stats.get("graph_median_us"),
        "cache_state": stats.get("cache_state"),
        "graph_replay_count": stats.get("graph_replay_count"),
        "host_overhead_us": stats.get("host_overhead_us"),
        "graph_capture_error": stats.get("graph_capture_error"),
        "effective_tflops": eff.get("effective_tflops"), "effective_gbps": eff.get("effective_gbps"),
        "workspace_bytes": None,
        "includes_allocation": bool(getattr(adapter, "includes_allocation", False)),
        "includes_layout_conversion": bool(getattr(adapter, "includes_layout_conversion", False)),
        "includes_jit": bool(getattr(adapter, "includes_jit", False)),
        "benchmark_status": status, "skip_reason": skip_reason, "profile_artifact": None,
    }


def _write_csv(path, results):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(_CSV_COLS)
        for r in results:
            w.writerow([common.canonical_json(r["args"]) if c == "args" else r.get(c) for c in _CSV_COLS])


def main(argv=None):
    ap = argparse.ArgumentParser(description="FlyDSL multi-shape benchmark runner")
    ap.add_argument("--op", "--kernel", dest="op", required=True)
    ap.add_argument("--shape-ledger", required=True)
    ap.add_argument("--baseline-matrix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup-iters", type=int, default=25)
    ap.add_argument("--repeat-iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stages", default=None, help="comma list to restrict stages")
    args = ap.parse_args(argv)
    common.bootstrap_env()
    stages = {s.strip() for s in args.stages.split(",")} if args.stages else None
    run(args.op, args.shape_ledger, args.baseline_matrix, args.out,
        warmup_iters=args.warmup_iters, repeat_iters=args.repeat_iters,
        seed=args.seed, limit=args.limit, stages=stages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
