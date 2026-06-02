"""Join shape ledger + benchmark results and compute the comparison.

One place computes per-shape speedups, best-available baseline, and the
aggregates everything else renders. Speedups use the PRIMARY median_us
(kernel-only / graph time when available) -- the fair kernel metric. The
eager / host-overhead view is carried alongside for the launcher story.
"""

from __future__ import annotations

from benchmarks import common

PROVIDER_ORDER = ["flydsl", "aiter", "aiter_triton", "aiter_ck", "aiter_asm",
                  "ck", "triton", "gluon", "hipblaslt", "pytorch"]
CANDIDATE = "flydsl"


def join(ledger_path: str, results_path: str) -> dict:
    ledger = {r["shape_id"]: r for r in common.read_jsonl(ledger_path)}
    results = common.read_jsonl(results_path)
    by_shape: dict[str, dict] = {}
    for r in results:
        by_shape.setdefault(r["shape_id"], {})[r["provider"]] = r

    shapes = []
    for sid, provs in by_shape.items():
        shape = ledger.get(sid)
        if shape is None:
            continue
        fly = provs.get(CANDIDATE, {})
        fly_us = fly.get("median_us") if fly.get("benchmark_status") in ("ok", "incorrect") else None
        # baselines = correct, timed, non-candidate providers
        baselines = {p: row for p, row in provs.items()
                     if p != CANDIDATE and row.get("benchmark_status") == "ok"
                     and row.get("correct") and row.get("median_us")}
        best_name, best_us = None, None
        for p, row in baselines.items():
            if best_us is None or row["median_us"] < best_us:
                best_name, best_us = p, row["median_us"]
        sp = {p: common.speedup(row["median_us"], fly_us) for p, row in baselines.items()}
        shapes.append({
            "shape_id": sid, "shape": shape, "providers": provs,
            "flydsl_us": fly_us, "flydsl_correct": fly.get("correct"),
            "flydsl_status": fly.get("benchmark_status"),
            "flydsl_eager_us": fly.get("eager_median_us"),
            "flydsl_host_overhead_us": fly.get("host_overhead_us"),
            "flydsl_stable": fly.get("stable"),
            "baselines": baselines,
            "best_baseline": best_name, "best_baseline_us": best_us,
            "speedup_vs_best": common.speedup(best_us, fly_us),
            "speedups": sp,
            "weight": shape.get("weight", {}),
            "stage": shape.get("stage"),
        })
    return {"shapes": shapes, "ledger": ledger, "results": results}


def _w(shape_row) -> float | None:
    w = shape_row.get("weight", {})
    return w.get("baseline_time_weight") if w.get("baseline_time_weight") is not None else w.get("traffic_weight")


def aggregates(joined: dict) -> dict:
    shapes = joined["shapes"]
    measured = [s for s in shapes if s["speedup_vs_best"] is not None and s["flydsl_correct"]]
    sp_best = [s["speedup_vs_best"] for s in measured]

    # per named-baseline geomean (over shapes where that baseline ran)
    per_baseline = {}
    for p in PROVIDER_ORDER:
        if p == CANDIDATE:
            continue
        vals = [s["speedups"][p] for s in measured if p in s["speedups"] and s["speedups"][p]]
        if vals:
            per_baseline[p] = {"geomean": common.geomean(vals), "n": len(vals)}

    def stage_geo(stage):
        v = [s["speedup_vs_best"] for s in measured if s["stage"] == stage]
        return {"geomean": common.geomean(v), "n": len(v)} if v else {"geomean": None, "n": 0}

    weighted = None
    wpairs = [(s["speedup_vs_best"], _w(s)) for s in measured if _w(s)]
    if wpairs:
        weighted = common.weighted_geomean(wpairs)

    worst = min(measured, key=lambda s: s["speedup_vs_best"], default=None)
    ranked = sorted(measured, key=lambda s: s["speedup_vs_best"])
    return {
        "n_shapes": len(shapes),
        "n_measured": len(measured),
        "unweighted_geomean_vs_best": common.geomean(sp_best),
        "weighted_geomean_vs_best": weighted,
        "per_baseline": per_baseline,
        "stages": {st: stage_geo(st) for st in
                   ("prefill", "decode", "mixed", "synthetic", "diagnostic", "model_config")},
        "worst_hot": worst,
        "top_regressions": ranked[:8],
        "top_wins": list(reversed(ranked))[:8],
    }
