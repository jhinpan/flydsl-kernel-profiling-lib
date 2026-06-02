"""Render coverage_matrix.md: one row per shape, one column per provider.

Cell = the benchmark_status (ok/incorrect/unsupported/failed/oom/not_configured),
so nothing is hidden -- an unsupported or disabled provider shows its state
instead of silently vanishing.
"""

from __future__ import annotations

import argparse

from benchmarks import common, ops
from benchmarks.reports.analysis import PROVIDER_ORDER

_GLYPH = {"ok": "ok", "incorrect": "incorrect", "unsupported": "unsupported",
          "failed": "failed", "oom": "oom", "not_configured": "n/c", "skipped": "skipped"}


def render(ledger_path: str, results_path: str) -> str:
    ledger = {r["shape_id"]: r for r in common.read_jsonl(ledger_path)}
    results = common.read_jsonl(results_path)
    by_shape: dict[str, dict] = {}
    for r in results:
        by_shape.setdefault(r["shape_id"], {})[r["provider"]] = r
    provs = [p for p in PROVIDER_ORDER if any(p in v for v in by_shape.values())]

    hdr = ["shape_id", "model", "stage", "dtype", "args"] + provs + ["profile"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for sid, shape in sorted(ledger.items(), key=lambda kv: (kv[1].get("stage", ""), kv[1].get("args", {}).get("N", 0))):
        if sid not in by_shape:
            continue
        op = ops.get_op(shape["op_type"])
        argsum = op.args_summary(shape) if op else str(shape.get("args"))
        row = [sid.replace("sha1:", ""), shape.get("model", "")[:22], shape.get("stage", ""),
               shape.get("dtype", ""), argsum]
        prof = ""
        for p in provs:
            r = by_shape[sid].get(p)
            row.append(_GLYPH.get(r.get("benchmark_status"), "-") if r else "-")
            if r and r.get("profile_artifact"):
                prof = "yes"
        row.append(prof)
        lines.append("| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render coverage_matrix.md")
    ap.add_argument("--shape-ledger", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    md = render(args.shape_ledger, args.results)
    with open(args.out, "w") as f:
        f.write("# Coverage matrix\n\n" + md)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
