"""Generate benchmark_summary.md from a shape ledger + benchmark results.

Headline metric = kernel-only (CUDA-graph) speedup vs the best available
baseline. The eager / host-overhead view is reported separately so a launcher
problem is never confused with a kernel problem. Hard failures (crashes,
incorrect) are surfaced, never hidden.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict

from benchmarks import common, ops
from benchmarks.reports import analysis
from benchmarks.reports.classify_bottleneck import classify, classify_eager


def _fmt(x, nd=2, suf=""):
    return f"{x:.{nd}f}{suf}" if isinstance(x, (int, float)) else "-"


def _args_summary(kernel: str, shape: dict) -> str:
    op = ops.get_op(kernel)
    return op.args_summary(shape) if op else common.canonical_json(shape.get("args", {}))


def _coverage(joined):
    c = Counter()
    for s in joined["shapes"]:
        provs = s["providers"]
        fly = provs.get("flydsl", {})
        c["total"] += 1
        c["flydsl_ok"] += 1 if fly.get("benchmark_status") == "ok" and fly.get("correct") else 0
        c["flydsl_failed"] += 1 if fly.get("benchmark_status") in ("failed", "oom") else 0
        c["flydsl_unsupported"] += 1 if fly.get("benchmark_status") == "unsupported" else 0
        c["flydsl_incorrect"] += 1 if fly.get("benchmark_status") == "incorrect" else 0
        if (s["best_baseline_us"] is not None and fly.get("median_us")
                and fly.get("benchmark_status") == "ok" and fly.get("correct")):
            c["measured_pairs"] += 1
    return c


def _failures(joined):
    out = []
    for s in joined["shapes"]:
        fly = s["providers"].get("flydsl", {})
        if fly.get("benchmark_status") in ("failed", "oom", "incorrect"):
            out.append((s, fly))
    return out


def decide(agg, failures):
    geo = agg["unweighted_geomean_vs_best"] or 0
    nfail = len(failures)
    if agg["n_measured"] == 0:
        return "blocked", "no correct+timed FlyDSL-vs-baseline pairs; fix coverage/failures before a speedup verdict"
    if nfail:
        if geo >= 1.0:
            return "tune_needed", (f"correct measured rows are parity-or-better (geomean {geo:.2f}x), "
                                   "but hard failures/incorrect rows block promotion")
        if geo >= 0.7:
            return "tune_needed", (f"sub-parity correct measured rows (geomean {geo:.2f}x) + hard failures on a shape class; "
                                   "needs per-shape tuning + bug fixes before broad promotion")
        return "rewrite_needed", (f"well below parity (geomean {geo:.2f}x) + hard failures on a shape class; "
                                  "structural rework needed")
    if nfail == 0 and geo >= 1.0:
        return "promote", "overall kernel-only geomean is parity-or-better, no FlyDSL hard failures"
    if nfail == 0 and geo >= 0.95:
        return "promote_with_guardrails", "near parity; guardrail the few sub-parity shapes"
    if geo < 0.9:
        if geo >= 0.7:
            return "tune_needed", (f"sub-parity overall (geomean {geo:.2f}x); "
                                   "wins on its target regime but needs per-shape tuning + bug fixes before broad promotion")
        return "rewrite_needed", f"well below parity (geomean {geo:.2f}x); structural rework needed"
    return "tune_needed", f"geomean {geo:.2f}x"


def _first(rows: list[dict], key: str):
    for r in rows:
        v = r.get(key)
        if v is not None:
            return v
    return None


def _commit_for(rows: list[dict], provider: str):
    vals = [r.get("candidate_commit") for r in rows
            if r.get("provider") == provider and r.get("candidate_commit")]
    if vals:
        return Counter(vals).most_common(1)[0][0]
    vals = [r.get("baseline_commit") for r in rows
            if r.get("provider") == provider and r.get("baseline_commit")]
    return Counter(vals).most_common(1)[0][0] if vals else None


def _result_provenance(joined: dict) -> dict:
    rows = joined["results"]
    measured = [r for r in rows if r.get("benchmark_status") != "not_configured"] or rows
    return {
        "gpu": _first(measured, "gpu"),
        "arch": _first(measured, "arch"),
        "rocm_version": _first(measured, "rocm_version"),
        "driver_version": _first(measured, "driver_version"),
        "flydsl_commit": _commit_for(rows, "flydsl"),
        "aiter_commit": _commit_for(rows, "aiter") or _commit_for(rows, "aiter_triton"),
        "sglang_commit": _first(measured, "sglang_commit"),
        "torch_version": _first(measured, "torch_version"),
        "triton_version": _first(measured, "triton_version"),
    }


def _cache_state_note(joined: dict) -> str:
    states = Counter((r.get("cache_state") or "not_recorded") for r in joined["results"]
                     if r.get("benchmark_status") == "ok" and r.get("timing_method") == "cudagraph")
    if not states:
        return "No CUDA-graph timing rows found."
    parts = ", ".join(f"{k}={v}" for k, v in sorted(states.items()))
    if "not_recorded" in states:
        return (f"Graph cache state: {parts}. Rows without `cache_state` were produced before "
                "the L2-flushed graph metric was recorded; regenerate latency results before "
                "treating effective bandwidth/roofline numbers as cold-cache evidence.")
    return f"Graph cache state: {parts}."


def _failure_reason(row: dict) -> str:
    return row.get("skip_reason") or row.get("correctness_error") or row.get("graph_capture_error") or ""


def _failure_diag(kernel: str, row: dict) -> dict:
    status = row.get("benchmark_status")
    reason = _failure_reason(row)
    if status == "incorrect":
        return {"classification": "baseline_unfair_or_unmatched",
                "likely_fix": "fix correctness before trusting timing for this shape"}
    if status == "oom":
        return {"classification": "measurement_issue",
                "likely_fix": "reduce shape size or isolate a larger-memory device before profiling"}
    if "max_flat_workgroup_size" in reason or "known_block_size" in reason:
        return {"classification": "flydsl_codegen_gap",
                "likely_fix": "annotate known_block_size or lower the generated workgroup size for this path"}
    if "CShuffle epilogue" in reason:
        return {"classification": "implementation_gap",
                "likely_fix": "enable/validate the stage-2 CShuffle epilogue path, or mark this MoE path unsupported"}
    if "input/reference build" in reason:
        return {"classification": "measurement_issue",
                "likely_fix": "exclude unsupported dtype rows from this op run, or implement input/reference support"}
    return {"classification": "measurement_issue",
            "likely_fix": "inspect the recorded failure reason and add an op-specific adapter/kernel fix"}


def _classify_shape(kernel: str, s: dict) -> dict:
    sp = s.get("speedup_vs_best")
    if s.get("flydsl_correct") is False:
        return {"classification": "baseline_unfair_or_unmatched",
                "evidence": "FlyDSL output failed correctness vs fp32 reference",
                "likely_fix": "fix correctness before trusting timing"}
    if sp is None:
        return {"classification": "measurement_issue",
                "evidence": "no comparable correct baseline was measured",
                "likely_fix": "ensure at least one correct baseline runs for this shape"}
    if kernel in ("rmsnorm", "layernorm"):
        return classify(s)
    if sp >= 0.95:
        return {"classification": "ok",
                "evidence": f"kernel-only vs-best {sp:.2f}x (near parity or better)",
                "likely_fix": "none"}
    args = s["shape"].get("args", {})
    if kernel == "softmax":
        return {"classification": "implementation_gap",
                "evidence": (f"FlyDSL softmax currently reports the generic scalar path; "
                             f"kernel-only vs-best {sp:.2f}x for args={args}"),
                "likely_fix": "profile and re-enable/fix the vectorized softmax path before promotion"}
    if kernel == "gemm":
        return {"classification": "algorithm_gap",
                "evidence": f"hgemm_splitk is slower than the best GEMM baseline ({sp:.2f}x) for args={args}",
                "likely_fix": "retune tile/split-K selection or route this shape to a stronger GEMM backend"}
    if kernel == "fused_rope_cache":
        return {"classification": "tuning_gap",
                "evidence": f"single fused RoPE/cache launch is below the best baseline ({sp:.2f}x) for args={args}",
                "likely_fix": "capture rocprofv3 and inspect occupancy, vectorization, and memory-store behavior"}
    if kernel == "moe_gemm":
        return {"classification": "implementation_gap",
                "evidence": f"full MoE path is below the best available baseline ({sp:.2f}x) for args={args}",
                "likely_fix": "separate adapter composition overhead from stage kernel tuning before filing a kernel verdict"}
    return {"classification": "tuning_gap",
            "evidence": f"kernel-only vs-best {sp:.2f}x for args={args}",
            "likely_fix": "profile the hot shape and add an op-specific diagnosis"}


def _promotion_notes(kernel: str, agg: dict, fails: list) -> list[str]:
    notes = [f"Correct+timed FlyDSL-vs-baseline pairs: {agg['n_measured']}/{agg['n_shapes']}."]
    if fails:
        notes.append("Hard FlyDSL failures must be fixed or explicitly scoped out before broad promotion.")
    if kernel == "moe_gemm" and agg["n_measured"] == 0:
        notes.append("No FlyDSL speedup verdict is available until the full-op adapter produces correct timed rows.")
    if kernel == "gemm":
        notes.append("Only supported bf16/f16 rows should drive the hgemm_splitk verdict; quantized rows need separate adapters.")
    if kernel == "fused_rope_cache":
        notes.append("Incorrect baselines are excluded from vs-best aggregates; check the coverage matrix before promoting.")
    if kernel in ("rmsnorm", "layernorm"):
        notes.append("Norm conclusions are shape-regime dependent; keep the fast-path/generic-path split visible.")
    return notes


def render(ledger_path, results_path, kernel="rmsnorm"):
    joined = analysis.join(ledger_path, results_path)
    agg = analysis.aggregates(joined)
    cov = _coverage(joined)
    fails = _failures(joined)
    prov = _result_provenance(joined)
    L = []
    P = L.append

    P(f"# Benchmark Summary: {kernel}\n")
    P("## Scope\n")
    P(f"- GPU: {prov.get('gpu')}  |  Arch: {prov.get('arch')}  |  ROCm: {prov.get('rocm_version')}")
    P(f"- torch: {prov.get('torch_version') or 'not recorded'}  |  triton: {prov.get('triton_version') or 'not recorded'}")
    P(f"- FlyDSL commit: {prov.get('flydsl_commit') or 'not recorded'}  |  AITER commit: {prov.get('aiter_commit') or 'not recorded'}  |  SGLang commit: {prov.get('sglang_commit') or 'not recorded'}")
    P(f"- Shapes: {agg['n_shapes']} (sources: " + ", ".join(f"{k}={v}" for k, v in
      Counter(s['shape']['source']['kind'] for s in joined['shapes']).items()) + ")")
    P("- Headline metric: **kernel-only** (CUDA-graph) median speedup vs best available baseline. "
      "Eager/host-overhead reported separately.")
    P(f"- {_cache_state_note(joined)}\n")

    P("## Coverage\n")
    P("| Category | Count |")
    P("|---|---:|")
    P(f"| total shapes | {cov['total']} |")
    P(f"| FlyDSL correct + timed | {cov['flydsl_ok']} |")
    P(f"| FlyDSL failed/oom | {cov['flydsl_failed']} |")
    P(f"| FlyDSL incorrect | {cov['flydsl_incorrect']} |")
    P(f"| FlyDSL unsupported | {cov['flydsl_unsupported']} |")
    P(f"| measured FlyDSL-vs-baseline pairs | {cov['measured_pairs']} |\n")

    P("## Overall Speedup (kernel-only, vs best available)\n")
    P("| Aggregate | value |")
    P("|---|---:|")
    P(f"| unweighted geomean vs best | {_fmt(agg['unweighted_geomean_vs_best'])}x  (n={agg['n_measured']}) |")
    wg = agg["weighted_geomean_vs_best"]
    P(f"| production-weighted geomean vs best | {(_fmt(wg)+'x') if wg else 'n/a (no weights yet — add a serving trace)'} |")
    for p, d in agg["per_baseline"].items():
        P(f"| vs {p} | {_fmt(d['geomean'])}x  (n={d['n']}) |")
    w = agg["worst_hot"]
    if w:
        P(f"| worst hot shape | {_fmt(w['speedup_vs_best'])}x  ({_args_summary(kernel, w['shape'])} vs {w['best_baseline']}) |")
    P("")

    P("## Stage Split (kernel-only vs best)\n")
    P("| Stage | Shapes | Geomean vs best |")
    P("|---|---:|---:|")
    for st, d in agg["stages"].items():
        if d["n"]:
            P(f"| {st} | {d['n']} | {_fmt(d['geomean'])}x |")
    P("")

    # model split (split merged model names)
    P("## Model Split (kernel-only vs best)\n")
    bym = defaultdict(list)
    for s in joined["shapes"]:
        if s["speedup_vs_best"] is not None and s["flydsl_correct"]:
            for m in str(s["shape"].get("model", "")).split("|"):
                bym[m].append(s["speedup_vs_best"])
    P("| Model | Shapes | Geomean vs best |")
    P("|---|---:|---:|")
    for m, vals in sorted(bym.items()):
        P(f"| {m} | {len(vals)} | {_fmt(common.geomean(vals))}x |")
    P("")

    P("## Top Wins (kernel-only)\n")
    P("| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |")
    P("|---|---|---|---:|---|---:|---:|")
    for s in agg["top_wins"]:
        P(f"| {_args_summary(kernel, s['shape'])} | {s['stage']} | {s['shape']['dtype']} | "
          f"{_fmt(s['flydsl_us'])} | {s['best_baseline']} | {_fmt(s['best_baseline_us'])} | {_fmt(s['speedup_vs_best'])}x |")
    P("")

    P("## Top Regressions (kernel-only) + diagnosis\n")
    P("| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |")
    P("|---|---|---|---:|---|---:|---:|---|")
    for s in agg["top_regressions"]:
        c = _classify_shape(kernel, s)
        P(f"| {_args_summary(kernel, s['shape'])} | {s['stage']} | {s['shape']['dtype']} | "
          f"{_fmt(s['flydsl_us'])} | {s['best_baseline']} | {_fmt(s['best_baseline_us'])} | "
          f"{_fmt(s['speedup_vs_best'])}x | {c['classification']} |")
    P("")

    if fails:
        P("## FlyDSL hard failures (crash / incorrect)\n")
        P("| shape | model | stage | dtype | status | reason |")
        P("|---|---|---|---|---|---|")
        for s, fly in fails:
            reason = (fly.get("skip_reason") or fly.get("correctness_error") or "")[:120].replace("|", "\\|")
            model = str(s["shape"].get("model", ""))[:20].replace("|", "\\|")
            P(f"| {_args_summary(kernel, s['shape'])} | {model} | {s['shape']['stage']} | "
              f"{s['shape']['dtype']} | {fly['benchmark_status']} | {reason} |")
        P("")

    # eager / launcher-overhead view
    P("## Eager vs kernel-only (host launch overhead)\n")
    P("FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every call; on short shapes this host "
      "overhead dwarfs the kernel. This is a launcher (host-side) issue, distinct from kernel speed.\n")
    P("| shape | FlyDSL kernel us | FlyDSL eager us | host overhead us |")
    P("|---|---:|---:|---:|")
    short = sorted([s for s in joined["shapes"] if s.get("flydsl_host_overhead_us")],
                   key=lambda s: -(s["flydsl_host_overhead_us"] or 0))[:6]
    for s in short:
        P(f"| {_args_summary(kernel, s['shape'])} | {_fmt(s['flydsl_us'])} | "
          f"{_fmt(s['flydsl_eager_us'])} | {_fmt(s['flydsl_host_overhead_us'])} |")
    if short:
        ce = classify_eager(short[0])
        if ce:
            P(f"\n**Eager verdict:** {ce['classification']} — {ce['evidence']}\n  - likely fix: {ce['likely_fix']}")
    P("")

    P("## Diagnosis\n")
    seen = set()
    for s in agg["top_regressions"][:6] + [f[0] for f in fails]:
        if s["shape_id"] in seen:
            continue
        seen.add(s["shape_id"])
        fly = s["providers"].get("flydsl", {})
        if fly.get("benchmark_status") in ("failed", "oom"):
            diag = _failure_diag(kernel, fly)
            reason = _failure_reason(fly)[:160]
            P(f"- `{_args_summary(kernel, s['shape'])}` ({s['shape']['dtype']}): **{fly['benchmark_status']}** — "
              f"{reason}\n  - classification: **{diag['classification']}**\n"
              f"  - likely fix: {diag['likely_fix']}")
            continue
        c = _classify_shape(kernel, s)
        P(f"- `{_args_summary(kernel, s['shape'])}` ({s['shape']['dtype']}, vs-best {_fmt(s['speedup_vs_best'])}x): "
          f"**{c['classification']}**\n  - evidence: {c['evidence']}\n  - likely fix: {c['likely_fix']}")
    P("")

    decision, why = decide(agg, fails)
    P("## Promotion Decision\n")
    P(f"**{decision}** — {why}\n")
    P("Reading:")
    for note in _promotion_notes(kernel, agg, fails):
        P(f"- {note}")
    P("")

    P("## Reproduction\n")
    P("```bash")
    P("# 1. use the checked-in shape ledger, or refresh model_config rows when this op is importer-backed")
    if kernel != "fused_rope_cache":
        P("python -m benchmarks.shape_ledgers.aiter_model_shapes_importer \\")
        P("  --aiter-model-shapes /sgl-workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json \\")
        P(f"  --out benchmarks/examples --tp 8 --gpu MI350X --arch gfx950 --ops {kernel}")
    else:
        P("# fused_rope_cache rows are derived from rope model_config rows; use the checked-in ledger here.")
    if kernel == "rmsnorm":
        P("python -m benchmarks.shape_ledgers.manual_shape_importer --op rmsnorm --out benchmarks/examples \\")
        P("  --synthetic-boundary --diagnostic 32768,8192,bf16")
    P("# 2. run (env.sh sets the FlyDSL build-tree PYTHONPATH/LD that also unblocks aiter)")
    P("HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.multishape_runner \\")
    P(f"  --op {kernel} --shape-ledger benchmarks/examples/{kernel}/shape_ledger.jsonl \\")
    P(f"  --baseline-matrix benchmarks/examples/{kernel}/baseline_matrix.yaml \\")
    P(f"  --out benchmarks/examples/{kernel} --warmup-iters 20 --repeat-iters 60")
    P("# 3. reports")
    P(f"python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/{kernel}/shape_ledger.jsonl \\")
    P(f"  --results benchmarks/examples/{kernel}/benchmark_results.jsonl --out benchmarks/examples/{kernel}/benchmark_summary.md \\")
    P(f"  --kernel {kernel}")
    P("```")
    P("\nRaw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`")
    return "\n".join(L) + "\n", {"decision": decision, "agg": agg, "failures": len(fails)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate benchmark_summary.md")
    ap.add_argument("--shape-ledger", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kernel", default="rmsnorm")
    args = ap.parse_args(argv)
    md, meta = render(args.shape_ledger, args.results, kernel=args.kernel)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(md)
    print(f"wrote {args.out}  (decision={meta['decision']}, "
          f"geomean_vs_best={_fmt(meta['agg']['unweighted_geomean_vs_best'])}x, failures={meta['failures']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
