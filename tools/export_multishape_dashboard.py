#!/usr/bin/env python3
"""Merge the multi-shape benchmark results into the dashboard dataset.

Post-processes docs/data/kernels.json (built by export_dashboard_data.py from the
single-shape rocprof ATT sweep): for every kernel with a multi-shape benchmark in
benchmarks/examples/<op>/, attach a `multishape` block (cold-cache geomean vs best
correct baseline, #shapes, #flydsl-correct, weighted geomean, models covered,
verdict) and a link to its benchmark_summary.md. Multishape-only kernels (no
rocprof record) are appended as new records. Rewrites kernels.json + data.js.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
import sys
from pathlib import Path

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


def _read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _shape_weight(shape):
    """baseline_time_weight wins over traffic_weight, matching reports.analysis."""
    w = shape.get("weight") or {}
    if w.get("baseline_time_weight") is not None:
        return w.get("baseline_time_weight")
    return w.get("traffic_weight")


def _model_family(model):
    m = (model or "unknown").lower()
    if "|" in m:
        return "Mixed"
    if "deepseek" in m and "kimi" in m:
        return "DeepSeek/Kimi"
    if "deepseek" in m:
        return "DeepSeek"
    if "kimi" in m:
        return "Kimi"
    if "qwen" in m:
        return "Qwen"
    if "llama" in m:
        return "Llama"
    if "gpt-oss" in m:
        return "GPT-OSS"
    if "mixtral" in m:
        return "Mixtral"
    if "sliding-window" in m or "test_" in m:
        return "Proxy/test"
    if any(x in m for x in ("synthetic", "diagnostic", "stress", "flydsl_test")):
        return "Synthetic/test"
    return "Other"


def _fmt_val(v):
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, bool):
        return str(v).lower()
    return str(v)


def _args_summary(shape):
    return ",".join(f"{k}={_fmt_val(v)}" for k, v in (shape.get("args") or {}).items())


def _geomean(vals):
    vals = [v for v in vals if v is not None and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def _weighted_geomean(pairs):
    num, den = 0.0, 0.0
    for v, w in pairs:
        if v is not None and v > 0 and w is not None and w > 0:
            num += w * math.log(v)
            den += w
    return math.exp(num / den) if den > 0 else None


def _round(x):
    return round(x, 3) if x is not None else None


def _public_shape_record(s):
    return {
        "shape_id": s.get("shape_id"),
        "stage": s.get("stage"),
        "dtype": s.get("dtype"),
        "args_summary": s.get("args_summary"),
        "weight": s.get("weight"),
        "occurrences": s.get("occurrences"),
        "speedup_vs_best": s.get("speedup_vs_best"),
        "best_baseline": s.get("best_baseline"),
    }


def compute_ms(op, examples_dir, github_base):
    op_dir = examples_dir / op
    result_path = op_dir / "benchmark_results.jsonl"
    if not result_path.exists():
        return None
    rows = _read_jsonl(result_path)
    ledger_path = op_dir / "shape_ledger.jsonl"
    ledger = {r["shape_id"]: r for r in _read_jsonl(ledger_path)} if ledger_path.exists() else {}
    by = {}
    for r in rows:
        by.setdefault(r["shape_id"], []).append(r)
    ratios, wlog, wsum = [], 0.0, 0.0
    n_fly_ok = 0
    models = set()
    shape_records = []
    for sid, rs in by.items():
        shape = ledger.get(sid, {})
        model = shape.get("model") or next((r.get("model") for r in rs if r.get("model")), "unknown")
        if model and model not in ("micro", "seed", "generic"):
            models.add(model)
        for r in rs:
            if r.get("model") and r["model"] not in ("micro", "seed", "generic"):
                models.add(r["model"])
        fly = [r for r in rs if r["provider"] == "flydsl"
               and r["benchmark_status"] == "ok" and r.get("correct")]
        fly_row = fly[0] if fly else next((r for r in rs if r["provider"] == "flydsl"), {})
        if fly:
            n_fly_ok += 1
        ft = _t(fly[0]) if fly else None
        bases = [_t(r) for r in rs if r["provider"] != "flydsl"
                 and r["benchmark_status"] == "ok" and r.get("correct") and _t(r)]
        w = _shape_weight(ledger.get(sid, {}))
        ratio = min(bases) / ft if ft and bases else None  # <1 => flydsl slower
        if ratio:
            ratios.append(ratio)
        if ratio and w:
            wlog += w * math.log(ratio)
            wsum += w
        best = None
        if ft and bases:
            best_us = min(bases)
            best = next((r for r in rs if r["provider"] != "flydsl"
                         and r["benchmark_status"] == "ok" and r.get("correct")
                         and _t(r) == best_us), None)
        weight = shape.get("weight") or {}
        shape_records.append({
            "shape_id": sid,
            "model": model,
            "family": _model_family(model),
            "stage": shape.get("stage") or fly_row.get("stage"),
            "dtype": shape.get("dtype") or fly_row.get("dtype"),
            "args_summary": _args_summary(shape or fly_row),
            "weight": w,
            "occurrences": weight.get("occurrences"),
            "flydsl_status": fly_row.get("benchmark_status"),
            "flydsl_correct": fly_row.get("correct"),
            "flydsl_us": _round(ft),
            "best_baseline": best.get("provider") if best else None,
            "best_us": _round(_t(best)) if best else None,
            "speedup_vs_best": _round(ratio),
        })
    model_records = []
    grouped = defaultdict(list)
    for rec in shape_records:
        grouped[rec["model"]].append(rec)
    for model, shapes in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        stage_counts = Counter(s.get("stage") or "unknown" for s in shapes)
        dtype_counts = Counter(s.get("dtype") or "unknown" for s in shapes)
        model_records.append({
            "model": model,
            "family": _model_family(model),
            "n_shapes": len(shapes),
            "n_flydsl_correct": sum(1 for s in shapes if s.get("flydsl_correct")),
            "stages": dict(sorted(stage_counts.items())),
            "dtypes": dict(sorted(dtype_counts.items())),
            "geomean_vs_best": _round(_geomean([s.get("speedup_vs_best") for s in shapes])),
            "weighted_geomean": _round(_weighted_geomean(
                [(s.get("speedup_vs_best"), s.get("weight")) for s in shapes])),
            "shapes": [_public_shape_record(s) for s in sorted(shapes, key=lambda s: (
                s.get("stage") or "", s.get("dtype") or "", s.get("args_summary") or "", s.get("shape_id") or ""))],
        })
    if not ratios:
        return {"n_shapes": len(by), "n_flydsl_correct": n_fly_ok,
                "geomean_vs_best": None, "weighted_geomean": None, "vs_best_n": 0,
                "models": sorted(models), "model_shapes": model_records, "verdict": "blocked",
                "summary_url": f"{github_base}/{op}/benchmark_summary.md"}
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
            "models": sorted(models), "model_shapes": model_records, "verdict": verdict,
            "vs_best_n": len(ratios),
            "summary_url": f"{github_base}/{op}/benchmark_summary.md"}


def parse_args(argv=None):
    default_repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Merge multi-shape benchmark results into the dashboard dataset")
    ap.add_argument("--repo-root", type=Path, default=default_repo,
                    help="repository root; defaults to this script's parent repo")
    ap.add_argument("--github-base", default=GH,
                    help="base URL for benchmark_summary.md links")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    repo = args.repo_root.resolve()
    docs_dir = repo / "docs"
    examples_dir = repo / "benchmarks" / "examples"

    kp = docs_dir / "data" / "kernels.json"
    with kp.open(encoding="utf-8") as f:
        data = json.load(f)
    kernels = data["kernels"]
    by_example = {k.get("example"): k for k in kernels}
    attached, added = 0, 0
    seen_ops = set()

    for ex_name, op in MS_OPS.items():
        ms = compute_ms(op, examples_dir, args.github_base)
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
                "report_url": ms["summary_url"], "bundle_url": f"{args.github_base}/{op}/",
                "flydsl_us": None, "baseline_us": None, "speedup_vs_baseline": None,
                "verdict": None, "has_bundle": False, "multishape": ms,
            })
            added += 1

    ms_kernels = [k for k in kernels if k.get("multishape")]
    family_stats = defaultdict(lambda: {"kernels": set(), "models": set(), "n_shapes": 0})
    for k in ms_kernels:
        for m in k["multishape"].get("model_shapes", []):
            fam = m.get("family") or _model_family(m.get("model"))
            family_stats[fam]["kernels"].add(k["example"])
            family_stats[fam]["models"].add(m["model"])
            family_stats[fam]["n_shapes"] += m["n_shapes"]
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
        "model_families": [
            {"family": fam, "n_kernels": len(v["kernels"]), "n_shapes": v["n_shapes"],
             "models": sorted(v["models"])}
            for fam, v in sorted(family_stats.items(), key=lambda kv: (-kv[1]["n_shapes"], kv[0]))
        ],
        "metric": "kernel-only cold-cache (CUDA-graph + L2 flush), geomean of best-correct-baseline / flydsl",
    }
    data["summary"]["total"] = len(kernels)

    with kp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    with (docs_dir / "data" / "data.js").open("w", encoding="utf-8") as f:
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
