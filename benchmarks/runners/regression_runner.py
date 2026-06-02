"""Regression runner (Tier-3): diff current vs previous benchmark results.

Two independent checks, both emitted to one report:

  1. Per-(shape_id, provider) latency regression. For every pair present in BOTH
     runs with a median_us, ratio = current / previous. Flag when current is
     slower: ratio > 1.05 (latency up >5%) OR speedup_vs_previous
     (= previous / current) < 0.95. Covers any provider, not just FlyDSL.

  2. Hot-shape guard on the CURRENT run only. Recompute FlyDSL's kernel-only
     speedup_vs_best via reports.analysis.join + aggregates (needs the shape
     ledger) and flag any shape where speedup_vs_best < 0.90 -- a shape where
     FlyDSL is meaningfully behind the best baseline right now, regardless of
     whether it moved since last run. These are the shapes a profiler pass
     (rocprofv3/ATT) should target.

Emits regression_summary.md (human) + regressions.jsonl (machine). regressions
rows: {shape_id, provider, current_us, previous_us, ratio, kind, hot} where
kind in {latency_regression, hot_below_0.90}.

  benchmarks/bench -m benchmarks.runners.regression_runner \
    --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
    --current  benchmarks/examples/rmsnorm/benchmark_results.jsonl \
    --previous benchmarks/baselines/rmsnorm/benchmark_results.jsonl \
    --out-dir  benchmarks/examples/rmsnorm
"""

from __future__ import annotations

import argparse
import os

from benchmarks import common
from benchmarks.reports import analysis

LATENCY_RATIO_THRESH = 1.05   # current/previous > this => latency regression
SPEEDUP_VS_PREV_THRESH = 0.95  # previous/current < this => latency regression
HOT_THRESH = 0.90              # speedup_vs_best < this => hot shape behind best


def _timed_by_pair(rows: list[dict]) -> dict[tuple[str, str], float]:
    """{(shape_id, provider): median_us} for rows that actually produced a time."""
    out: dict[tuple[str, str], float] = {}
    for r in rows:
        med = r.get("median_us")
        if med is None or r.get("benchmark_status") not in ("ok", "incorrect"):
            continue
        out[(r["shape_id"], r["provider"])] = float(med)
    return out


def find_latency_regressions(current_rows: list[dict],
                             previous_rows: list[dict]) -> list[dict]:
    cur = _timed_by_pair(current_rows)
    prev = _timed_by_pair(previous_rows)
    flags: list[dict] = []
    for key in sorted(cur.keys() & prev.keys()):
        sid, provider = key
        c, p = cur[key], prev[key]
        if c <= 0 or p <= 0:
            continue
        ratio = c / p
        speedup_vs_prev = p / c
        if ratio > LATENCY_RATIO_THRESH or speedup_vs_prev < SPEEDUP_VS_PREV_THRESH:
            flags.append({
                "shape_id": sid,
                "provider": provider,
                "current_us": c,
                "previous_us": p,
                "ratio": ratio,
                "speedup_vs_previous": speedup_vs_prev,
                "kind": "latency_regression",
                "hot": False,
            })
    return flags


def find_hot_shapes(ledger_path: str, current_path: str) -> tuple[list[dict], dict]:
    """Recompute FlyDSL speedup_vs_best on the current run; flag < HOT_THRESH."""
    joined = analysis.join(ledger_path, current_path)
    agg = analysis.aggregates(joined)
    flags: list[dict] = []
    for s in joined["shapes"]:
        svb = s.get("speedup_vs_best")
        if svb is None or not s.get("flydsl_correct"):
            continue
        if svb < HOT_THRESH:
            flags.append({
                "shape_id": s["shape_id"],
                "provider": analysis.CANDIDATE,
                "current_us": s.get("flydsl_us"),
                "previous_us": s.get("best_baseline_us"),
                "ratio": None,
                "speedup_vs_best": svb,
                "best_baseline": s.get("best_baseline"),
                "kind": "hot_below_0.90",
                "hot": True,
            })
    flags.sort(key=lambda f: f["speedup_vs_best"])
    return flags, agg


def _fmt(x, nd=2, suf=""):
    return f"{x:.{nd}f}{suf}" if isinstance(x, (int, float)) else "-"


def render_md(lat_flags: list[dict], hot_flags: list[dict], agg: dict, *,
              current_path: str, previous_path: str, ledger_path: str) -> str:
    prov = common.provenance()
    L: list[str] = []
    P = L.append
    P("# Regression Report\n")
    P("## Scope\n")
    P(f"- current : `{current_path}`")
    P(f"- previous: `{previous_path}`")
    P(f"- ledger  : `{ledger_path}`")
    P(f"- GPU: {prov.get('gpu')}  |  Arch: {prov.get('arch')}  |  ROCm: {prov.get('rocm_version')}")
    P(f"- FlyDSL commit: {prov.get('flydsl_commit')}  |  AITER commit: {prov.get('aiter_commit')}")
    P(f"- Thresholds: latency ratio > {LATENCY_RATIO_THRESH} OR speedup_vs_previous < "
      f"{SPEEDUP_VS_PREV_THRESH}; hot when speedup_vs_best < {HOT_THRESH}\n")

    verdict = "PASS" if not lat_flags and not hot_flags else "REGRESSED"
    P(f"## Verdict: **{verdict}**\n")
    P(f"- latency regressions: {len(lat_flags)}")
    P(f"- hot shapes (FlyDSL below {HOT_THRESH}x of best): {len(hot_flags)}")
    geo = agg.get("unweighted_geomean_vs_best")
    P(f"- current geomean vs best: {(_fmt(geo) + 'x') if geo else '-'}  (n={agg.get('n_measured', 0)})\n")

    P("## Latency regressions (current slower than previous)\n")
    if lat_flags:
        P("| shape_id | provider | current us | previous us | ratio | speedup_vs_prev |")
        P("|---|---|---:|---:|---:|---:|")
        for f in sorted(lat_flags, key=lambda f: -f["ratio"]):
            P(f"| {f['shape_id']} | {f['provider']} | {_fmt(f['current_us'])} | "
              f"{_fmt(f['previous_us'])} | {_fmt(f['ratio'])}x | {_fmt(f['speedup_vs_previous'])}x |")
    else:
        P("_None._")
    P("")

    P(f"## Hot shapes (FlyDSL kernel-only speedup_vs_best < {HOT_THRESH}, current run)\n")
    if hot_flags:
        P("| shape_id | FlyDSL us | best baseline | best us | speedup_vs_best |")
        P("|---|---:|---|---:|---:|")
        for f in hot_flags:
            P(f"| {f['shape_id']} | {_fmt(f['current_us'])} | {f.get('best_baseline') or '-'} | "
              f"{_fmt(f['previous_us'])} | {_fmt(f['speedup_vs_best'])}x |")
        P("\nThese are the shapes a profiler pass (rocprofv3/ATT) should target.")
    else:
        P("_None._")
    P("")
    return "\n".join(L) + "\n"


def run(ledger_path: str, current_path: str, previous_path: str, out_dir: str) -> dict:
    current_rows = common.read_jsonl(current_path)
    previous_rows = common.read_jsonl(previous_path)

    lat_flags = find_latency_regressions(current_rows, previous_rows)
    hot_flags, agg = find_hot_shapes(ledger_path, current_path)

    os.makedirs(os.path.abspath(out_dir), exist_ok=True)
    # regressions.jsonl carries only the documented columns
    cols = ("shape_id", "provider", "current_us", "previous_us", "ratio", "kind", "hot")
    jsonl_rows = [{k: f.get(k) for k in cols} for f in (lat_flags + hot_flags)]
    jsonl = os.path.join(out_dir, "regressions.jsonl")
    common.write_jsonl(jsonl, jsonl_rows)

    md = render_md(lat_flags, hot_flags, agg, current_path=current_path,
                   previous_path=previous_path, ledger_path=ledger_path)
    md_path = os.path.join(out_dir, "regression_summary.md")
    with open(md_path, "w") as f:
        f.write(md)

    verdict = "PASS" if not lat_flags and not hot_flags else "REGRESSED"
    print(f"{verdict}: {len(lat_flags)} latency regression(s), {len(hot_flags)} hot shape(s) "
          f"-> {md_path}, {jsonl}")
    return {"verdict": verdict, "n_latency": len(lat_flags), "n_hot": len(hot_flags),
            "summary": md_path, "regressions": jsonl}


def main(argv=None):
    ap = argparse.ArgumentParser(description="FlyDSL regression runner (current vs previous)")
    ap.add_argument("--shape-ledger", required=True)
    ap.add_argument("--current", required=True, help="current benchmark_results.jsonl")
    ap.add_argument("--previous", required=True, help="previous benchmark_results.jsonl")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    common.bootstrap_env()
    res = run(args.shape_ledger, args.current, args.previous, args.out_dir)
    # nonzero exit on regression so a CI gate can fail on it
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
