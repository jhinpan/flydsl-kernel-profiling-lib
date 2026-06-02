"""One-shot report renderer: writes coverage_matrix.md + benchmark_summary.md.

Convenience wrapper so a single command regenerates every per-kernel markdown
artifact from the raw ledger + results. Also exposes md_table() used elsewhere.
"""

from __future__ import annotations

import argparse
import os

from benchmarks.reports import coverage_matrix, summarize_results


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render_all(ledger_path: str, results_path: str, out_dir: str, kernel: str = "rmsnorm") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cov = coverage_matrix.render(ledger_path, results_path)
    with open(os.path.join(out_dir, "coverage_matrix.md"), "w") as f:
        f.write(f"# Coverage matrix: {kernel}\n\n" + cov)
    summ, meta = summarize_results.render(ledger_path, results_path, kernel=kernel)
    with open(os.path.join(out_dir, "benchmark_summary.md"), "w") as f:
        f.write(summ)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render coverage_matrix.md + benchmark_summary.md")
    ap.add_argument("--shape-ledger", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--kernel", default="rmsnorm")
    args = ap.parse_args(argv)
    meta = render_all(args.shape_ledger, args.results, args.out, kernel=args.kernel)
    print(f"wrote coverage_matrix.md + benchmark_summary.md to {args.out} "
          f"(decision={meta['decision']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
