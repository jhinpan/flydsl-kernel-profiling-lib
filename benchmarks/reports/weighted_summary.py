"""Weighted vs unweighted aggregate speedups (kernel-only, vs best available).

Weighted geomean uses baseline_time_weight (preferred) or traffic_weight from
the ledger; if no shape carries a weight it reports unweighted only and says so.
"""

from __future__ import annotations

import argparse

from benchmarks import common
from benchmarks.reports import analysis


def render(ledger_path, results_path) -> str:
    agg = analysis.aggregates(analysis.join(ledger_path, results_path))
    L = ["# Weighted summary (kernel-only, vs best available)\n"]
    L.append(f"- shapes measured: {agg['n_measured']}/{agg['n_shapes']}")
    g = agg["unweighted_geomean_vs_best"]
    L.append(f"- unweighted geomean vs best: {g:.3f}x" if g else "- unweighted geomean: n/a")
    w = agg["weighted_geomean_vs_best"]
    L.append(f"- production-weighted geomean vs best: {w:.3f}x" if w else
             "- production-weighted geomean: **n/a — no shape carries a weight yet** "
             "(add a serving trace via sglang_trace_importer to populate weights)")
    L.append("\n## Per-baseline (unweighted geomean of speedup)")
    for p, d in agg["per_baseline"].items():
        L.append(f"- vs {p}: {d['geomean']:.3f}x (n={d['n']})")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Weighted/unweighted aggregate speedups")
    ap.add_argument("--shape-ledger", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    md = render(args.shape_ledger, args.results)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md)
        print(f"wrote {args.out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
