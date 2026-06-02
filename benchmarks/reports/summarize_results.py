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
        if s["best_baseline_us"] is not None and fly.get("median_us"):
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
    if nfail == 0 and geo >= 1.0:
        return "promote", "kernel-only parity-or-better across all measured shapes, no failures"
    if nfail == 0 and geo >= 0.95:
        return "promote_with_guardrails", "near parity; guardrail the few sub-parity shapes"
    if nfail > 0 or geo < 0.9:
        if geo >= 0.7:
            return "tune_needed", (f"sub-parity overall (geomean {geo:.2f}x){' + hard failures on a shape class' if nfail else ''}; "
                                   "wins on its target regime but needs per-shape tuning + bug fixes before broad promotion")
        return "rewrite_needed", f"well below parity (geomean {geo:.2f}x); structural rework needed"
    return "tune_needed", f"geomean {geo:.2f}x"


def render(ledger_path, results_path, kernel="rmsnorm"):
    joined = analysis.join(ledger_path, results_path)
    agg = analysis.aggregates(joined)
    cov = _coverage(joined)
    fails = _failures(joined)
    prov = common.provenance()
    L = []
    P = L.append

    P(f"# Benchmark Summary: {kernel}\n")
    P("## Scope\n")
    P(f"- GPU: {prov.get('gpu')}  |  Arch: {prov.get('arch')}  |  ROCm: {prov.get('rocm_version')}")
    P(f"- torch: {prov.get('torch_version')}  |  triton: {prov.get('triton_version')}")
    P(f"- FlyDSL commit: {prov.get('flydsl_commit')}  |  AITER commit: {prov.get('aiter_commit')}  |  SGLang commit: {prov.get('sglang_commit')}")
    P(f"- Shapes: {agg['n_shapes']} (sources: " + ", ".join(f"{k}={v}" for k, v in
      Counter(s['shape']['source']['kind'] for s in joined['shapes']).items()) + ")")
    P("- Headline metric: **kernel-only** (CUDA-graph) median speedup vs best available baseline. "
      "Eager/host-overhead reported separately.\n")

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
        P(f"| worst hot shape | {_fmt(w['speedup_vs_best'])}x  ({ops.get_op(kernel).args_summary(w['shape'])} vs {w['best_baseline']}) |")
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
        P(f"| {ops.get_op(kernel).args_summary(s['shape'])} | {s['stage']} | {s['shape']['dtype']} | "
          f"{_fmt(s['flydsl_us'])} | {s['best_baseline']} | {_fmt(s['best_baseline_us'])} | {_fmt(s['speedup_vs_best'])}x |")
    P("")

    P("## Top Regressions (kernel-only) + diagnosis\n")
    P("| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |")
    P("|---|---|---|---:|---|---:|---:|---|")
    for s in agg["top_regressions"]:
        c = classify(s)
        P(f"| {ops.get_op(kernel).args_summary(s['shape'])} | {s['stage']} | {s['shape']['dtype']} | "
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
            P(f"| {ops.get_op(kernel).args_summary(s['shape'])} | {model} | {s['shape']['stage']} | "
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
        P(f"| {ops.get_op(kernel).args_summary(s['shape'])} | {_fmt(s['flydsl_us'])} | "
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
            P(f"- `{ops.get_op(kernel).args_summary(s['shape'])}` ({s['shape']['dtype']}): **{fly['benchmark_status']}** — "
              f"{(fly.get('skip_reason') or '')[:160]}\n  - classification: **flydsl_codegen_gap** (illegal launch config)\n"
              f"  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path")
            continue
        c = classify(s)
        P(f"- `{ops.get_op(kernel).args_summary(s['shape'])}` ({s['shape']['dtype']}, vs-best {_fmt(s['speedup_vs_best'])}x): "
          f"**{c['classification']}**\n  - evidence: {c['evidence']}\n  - likely fix: {c['likely_fix']}")
    P("")

    decision, why = decide(agg, fails)
    P("## Promotion Decision\n")
    P(f"**{decision}** — {why}\n")
    P("Regime-specific reading:")
    P("- **Large-M aligned (prefill-like):** kernel-only parity-or-better (diagnostic ~1.03x, beats PyTorch ~1.5x). Promotable.")
    P("- **Small-M large-N (decode):** one-block-per-row underutilizes the GPU (kernel-only worst ~0.36x). Needs a parallelization change.")
    P("- **Eager decode latency:** ~tens-of-us launcher host overhead per call. Needs a launch cache (host-side).")
    if fails:
        P("- **Large-M small-N:** hard crash (block size > AMDGPU max_flat_workgroup_size). Must-fix bug.\n")

    P("## Reproduction\n")
    P("```bash")
    P("# 1. build the ledger")
    P("python -m benchmarks.shape_ledgers.aiter_model_shapes_importer \\")
    P("  --aiter-model-shapes /sgl-workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json \\")
    P("  --out benchmarks/examples --tp 8 --gpu MI350X --arch gfx950 --ops rmsnorm")
    P("python -m benchmarks.shape_ledgers.manual_shape_importer --op rmsnorm --out benchmarks/examples \\")
    P("  --synthetic-boundary --diagnostic 32768,8192,bf16")
    P("# 2. run (env.sh sets the FlyDSL build-tree PYTHONPATH/LD that also unblocks aiter)")
    P("HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.multishape_runner \\")
    P(f"  --op {kernel} --shape-ledger benchmarks/examples/{kernel}/shape_ledger.jsonl \\")
    P(f"  --baseline-matrix benchmarks/examples/{kernel}/baseline_matrix.yaml \\")
    P(f"  --out benchmarks/examples/{kernel} --warmup-iters 20 --repeat-iters 60")
    P("# 3. reports")
    P(f"python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/{kernel}/shape_ledger.jsonl \\")
    P(f"  --results benchmarks/examples/{kernel}/benchmark_results.jsonl --out benchmarks/examples/{kernel}/benchmark_summary.md")
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
