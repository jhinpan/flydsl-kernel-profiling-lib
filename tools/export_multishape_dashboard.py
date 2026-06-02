#!/usr/bin/env python3
"""Merge the multi-shape benchmark results into the dashboard dataset.

Post-processes docs/data/kernels.json (built by export_dashboard_data.py from the
single-shape rocprof ATT sweep): for every kernel with a multi-shape benchmark in
benchmarks/examples/<op>/, attach a `multishape` block (cold-cache geomean vs best
correct baseline, #shapes, #flydsl-correct, weighted geomean, models covered,
verdict) and a link to its benchmark_summary.md. Multishape-only kernels (no
rocprof record) are appended as new records. Rewrites kernels.json + data.js.
"""
import json, math, os, sys

REPO = "/sgl-workspace/flydsl-kernel-profiling"
DOCS = f"{REPO}/docs"
EX = f"{REPO}/benchmarks/examples"

# dashboard `example` name  ->  multishape op_type (renames where they differ)
MS_OPS = {
    "rmsnorm": "rmsnorm", "layernorm": "layernorm", "softmax": "softmax",
    "hgemm_splitk": "gemm", "fused_rope_cache": "fused_rope_cache",
    "moe_gemm": "moe_gemm", "flash_attn_func": "flash_attn", "pa": "pa_decode",
    "mla_decode": "mla_decode", "preshuffle_gemm": "preshuffle_gemm",
    "blockscale_preshuffle_gemm": "blockscale_preshuffle_gemm",
    "fp8_gemm_rowscale": "fp8_gemm_rowscale", "moe_blockscale": "moe_blockscale",
    "moe_reduce": "moe_reduce", "topk_gating_softmax": "topk_gating_softmax",
    "quant": "quant", "vec_add": "vec_add",
}
PRETTY = {
    "vec_add": "Vector Add", "quant": "Per-Token Quant",
    "topk_gating_softmax": "TopK Gating Softmax", "moe_reduce": "MoE Reduction",
    "preshuffle_gemm": "Preshuffle GEMM",
    "blockscale_preshuffle_gemm": "Block-Scale Preshuffle GEMM",
    "fp8_gemm_rowscale": "FP8 Row-Scale GEMM", "moe_blockscale": "MoE Block-Scale (2-stage)",
    "flash_attn": "Flash Attention", "pa_decode": "Paged-Attn Decode (PS)",
    "mla_decode": "MLA Decode (fp8)",
}
CAT = {
    "vec_add": "elementwise", "quant": "quant", "topk_gating_softmax": "moe",
    "moe_reduce": "moe", "preshuffle_gemm": "gemm",
    "blockscale_preshuffle_gemm": "gemm", "fp8_gemm_rowscale": "gemm",
    "moe_blockscale": "moe", "flash_attn": "attention", "pa_decode": "attention",
    "mla_decode": "attention",
}
GH = "https://github.com/jhinpan/flydsl-kernel-profiling/blob/main/benchmarks/examples"


def _t(r):
    return r.get("graph_median_us") or r.get("median_us")


def compute_ms(op):
    path = f"{EX}/{op}/benchmark_results.jsonl"
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path)]
    by = {}
    for r in rows:
        by.setdefault(r["shape_id"], []).append(r)
    ratios, wlog, wsum = [], 0.0, 0.0
    n_fly_ok = 0
    models = set()
    for sid, rs in by.items():
        for r in rs:
            if r.get("model") and r["model"] not in ("micro", "seed", "generic"):
                models.add(r["model"])
        fly = [r for r in rs if r["provider"] == "flydsl"
               and r["benchmark_status"] == "ok" and r.get("correct")]
        if not fly:
            continue
        n_fly_ok += 1
        ft = _t(fly[0])
        bases = [_t(r) for r in rs if r["provider"] != "flydsl"
                 and r["benchmark_status"] == "ok" and r.get("correct") and _t(r)]
        if not ft or not bases:
            continue
        ratio = min(bases) / ft  # <1 => flydsl slower than best correct baseline
        ratios.append(ratio)
        w = 0.0
        for r in rs:  # use the ledger traffic_weight if present (live-trace kernels)
            w = (r.get("weight") or {}).get("traffic_weight") or w
        if w:
            wlog += w * math.log(ratio)
            wsum += w
    if not ratios:
        return {"n_shapes": len(by), "n_flydsl_correct": n_fly_ok,
                "geomean_vs_best": None, "weighted_geomean": None, "vs_best_n": 0,
                "models": sorted(models), "verdict": "blocked",
                "summary_url": f"{GH}/{op}/benchmark_summary.md"}
    g = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
    wg = math.exp(wlog / wsum) if wsum > 0 else None
    # geomean >10x almost always means the only *correct* baseline left is a slow
    # eager reference (optimized baseline excluded as incorrect/failed), so the
    # vs-best number is not a fair FlyDSL-vs-optimized verdict -- flag it.
    if g > 10:
        verdict = "baseline_blocked"
    else:
        verdict = ("promote" if g >= 1.03 else "parity" if g >= 0.97
                   else "tune_needed" if g >= 0.6 else "rewrite_needed")
    return {"n_shapes": len(by), "n_flydsl_correct": n_fly_ok,
            "geomean_vs_best": round(g, 3),
            "weighted_geomean": round(wg, 3) if wg else None,
            "models": sorted(models), "verdict": verdict,
            "vs_best_n": len(ratios),
            "summary_url": f"{GH}/{op}/benchmark_summary.md"}


def main():
    kp = f"{DOCS}/data/kernels.json"
    data = json.load(open(kp))
    kernels = data["kernels"]
    by_example = {k.get("example"): k for k in kernels}
    attached, added = 0, 0
    seen_ops = set()

    for ex_name, op in MS_OPS.items():
        ms = compute_ms(op)
        if ms is None:
            continue
        seen_ops.add(op)
        k = by_example.get(ex_name)
        if k is not None:
            k["multishape"] = ms
            attached += 1
        else:
            kernels.append({
                "name": PRETTY.get(op, op), "test": f"test_{op}.py", "stem": f"test_{op}",
                "example": op, "op_category": CAT.get(op, "other"),
                "report_url": ms["summary_url"], "bundle_url": f"{GH}/{op}/",
                "flydsl_us": None, "baseline_us": None, "speedup_vs_baseline": None,
                "verdict": None, "has_bundle": False, "multishape": ms,
            })
            added += 1

    ms_kernels = [k for k in kernels if k.get("multishape")]
    data["multishape_summary"] = {
        "kernels": len(ms_kernels),
        "total_shapes": sum(k["multishape"]["n_shapes"] for k in ms_kernels),
        "promote": sorted(k["example"] for k in ms_kernels if k["multishape"]["verdict"] == "promote"),
        "parity": sorted(k["example"] for k in ms_kernels if k["multishape"]["verdict"] == "parity"),
        "tune_needed": sorted(k["example"] for k in ms_kernels if k["multishape"]["verdict"] == "tune_needed"),
        "rewrite_needed": sorted(k["example"] for k in ms_kernels if k["multishape"]["verdict"] == "rewrite_needed"),
        "baseline_blocked": sorted(k["example"] for k in ms_kernels if k["multishape"]["verdict"] == "baseline_blocked"),
        "blocked": sorted(k["example"] for k in ms_kernels if k["multishape"]["verdict"] == "blocked"),
        "models": sorted({m for k in ms_kernels for m in k["multishape"]["models"]}),
        "metric": "kernel-only cold-cache (CUDA-graph + L2 flush), geomean of best-correct-baseline / flydsl",
    }
    data["summary"]["total"] = len(kernels)

    json.dump(data, open(kp, "w"), indent=2)
    with open(f"{DOCS}/data/data.js", "w") as f:
        f.write("window.KERNEL_DATA = ")
        json.dump(data, f, indent=2)
        f.write(";\n")
    print(f"multishape merged: {attached} attached, {added} new kernels, "
          f"{len(ms_kernels)} with multishape data, "
          f"{data['multishape_summary']['total_shapes']} total shapes")
    for k in sorted(ms_kernels, key=lambda x: x["multishape"]["geomean_vs_best"] or 0):
        m = k["multishape"]
        print(f"  {k['example']:26s} geo={str(m['geomean_vs_best']):>7s} "
              f"shapes={m['n_shapes']:3d} fly_ok={m['n_flydsl_correct']:3d} "
              f"{m['verdict']:14s} models={m['models']}")


if __name__ == "__main__":
    sys.exit(main())
