"""Latency runner (Tier-2): thin wrapper over multishape_runner.run.

The multishape runner already measures correctness AND latency for every
enabled provider. This entry is the "latency only / single-provider" front door:
pass --provider to time exactly one implementation (e.g. just `flydsl`, or just
`aiter`) across the ledger -- useful for an isolated A/B, a profiler pre-pass, or
a CI latency gate that should not pay for the whole baseline matrix.

Implementation: load the baseline_matrix, and if --provider is given, write a
temp matrix with every provider but the requested one disabled (status
not_configured), then delegate to multishape_runner.run unchanged. With no
--provider it is just multishape_runner.run with latency flags. Output is the
standard benchmark_results.jsonl/.csv so all the existing reports work as-is.

  benchmarks/bench -m benchmarks.runners.latency_runner \
    --op rmsnorm --provider flydsl \
    --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
    --baseline-matrix benchmarks/examples/rmsnorm/baseline_matrix.yaml \
    --out benchmarks/examples/rmsnorm --warmup-iters 25 --repeat-iters 100
"""

from __future__ import annotations

import argparse
import os
import tempfile

import yaml

from benchmarks import common
from benchmarks.runners import multishape_runner


def _filter_matrix(matrix: dict, provider: str) -> dict:
    """Return a copy of `matrix` with only `provider` enabled.

    Matched by the spec's `provider` field (falls back to the spec key), so it
    works regardless of whether the matrix keys the provider by name or alias.
    Every other provider is disabled with an explicit skip_reason so it still
    appears in the output as not_configured (never silently dropped)."""
    out = dict(matrix)
    specs = {}
    matched = False
    for pname, spec in matrix["providers"].items():
        spec = dict(spec)
        if spec.get("provider", pname) == provider or pname == provider:
            spec["enabled"] = True
            matched = True
        else:
            spec["enabled"] = False
            spec["skip_reason"] = f"latency_runner --provider={provider}"
        specs[pname] = spec
    if not matched:
        raise ValueError(f"--provider {provider!r} not present in baseline matrix "
                         f"(have: {sorted(s.get('provider', k) for k, s in matrix['providers'].items())})")
    out["providers"] = specs
    return out


def run(op_type: str, ledger_path: str, matrix_path: str, out_dir: str, *,
        provider: str | None, warmup_iters: int, repeat_iters: int, seed: int,
        limit: int | None = None, stages: set[str] | None = None) -> dict:
    if not provider:
        return multishape_runner.run(
            op_type, ledger_path, matrix_path, out_dir,
            warmup_iters=warmup_iters, repeat_iters=repeat_iters, seed=seed,
            limit=limit, stages=stages)

    matrix = yaml.safe_load(open(matrix_path))
    filtered = _filter_matrix(matrix, provider)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                      prefix=f"baseline_matrix_{provider}_")
    try:
        yaml.safe_dump(filtered, tmp)
        tmp.flush()
        tmp.close()
        return multishape_runner.run(
            op_type, ledger_path, tmp.name, out_dir,
            warmup_iters=warmup_iters, repeat_iters=repeat_iters, seed=seed,
            limit=limit, stages=stages)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="FlyDSL latency runner (single-provider wrapper)")
    ap.add_argument("--op", "--kernel", dest="op", required=True)
    ap.add_argument("--provider", default=None,
                    help="restrict timing to this one provider; default = whole matrix")
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
        provider=args.provider, warmup_iters=args.warmup_iters,
        repeat_iters=args.repeat_iters, seed=args.seed, limit=args.limit, stages=stages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
