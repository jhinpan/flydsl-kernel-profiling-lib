"""Correctness-only runner (Tier-1): does NOT time anything.

Mirrors the control flow of multishape_runner.run -- load the ledger + baseline
matrix, build deterministic inputs ONCE per shape via the registered Op, run the
fp32 reference, then for every enabled provider call adapter.supports +
adapter.check_correctness -- but skips the benchmarking entirely. This is the
fast gate to answer "does FlyDSL (and every baseline) produce the right answer
on this op's whole ledger" without paying for warmup/timing.

Every provider stays in the output with an explicit benchmark_status:
  ok            -> supported + correct
  incorrect     -> supported but check_correctness said wrong
  unsupported   -> adapter.supports() returned False
  not_configured-> disabled in the baseline_matrix
  failed        -> adapter load / input build / supports() crashed

Emits correctness_results.jsonl (compact rows) so it can be diffed across runs
or fed to a correctness dashboard.

  benchmarks/bench -m benchmarks.runners.correctness_runner \
    --op rmsnorm \
    --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
    --baseline-matrix benchmarks/examples/rmsnorm/baseline_matrix.yaml \
    --out benchmarks/examples/rmsnorm --seed 1234
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os

import yaml

from benchmarks import common, ops, validate
from benchmarks.providers import base


def _row(run_id: str, shape: dict, provider: str, *, status: str,
         correct=None, correctness_error=None) -> dict:
    """Compact correctness row (the documented Tier-1 surface)."""
    return {
        "run_id": run_id,
        "shape_id": shape["shape_id"],
        "provider": provider,
        "correct": correct,
        "correctness_error": correctness_error,
        "benchmark_status": status,
    }


def _now_id(op: str, prov: dict) -> str:
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{op}-correctness-{prov.get('flydsl_commit') or 'nocommit'}-{ts}"


def run(op_type: str, ledger_path: str, matrix_path: str, out_dir: str, *,
        seed: int, limit: int | None = None, stages: set[str] | None = None) -> dict:
    import torch  # noqa: F401  (kept for parity / empty_cache below)

    rows = [r for r in common.read_jsonl(ledger_path) if r.get("op_type") == op_type]
    if stages:
        rows = [r for r in rows if r.get("stage") in stages]
    errs = validate.validate_ledger(rows)
    if errs:
        raise ValueError(f"ledger invalid: {len(errs)} error(s); first {errs[0]}")
    if limit:
        rows = rows[:limit]

    matrix = yaml.safe_load(open(matrix_path))
    prov_specs = matrix["providers"]

    op = ops.get_op(op_type)
    if op is None:
        raise ValueError(f"no Op registered for {op_type}")
    prov = common.provenance()
    run_id = _now_id(op_type, prov)

    # instantiate adapters once (launcher/JIT caches persist across shapes)
    adapters: dict[str, base.ProviderAdapter] = {}
    for pname, spec in prov_specs.items():
        if not spec.get("enabled", False):
            continue
        try:
            adapters[pname] = base.load_entrypoint(spec["entrypoint"], op_type)
        except Exception as e:  # bad entrypoint -> keep as failed below
            adapters[pname] = e  # type: ignore

    results: list[dict] = []
    n = len(rows)
    for i, shape in enumerate(rows, 1):
        sid = shape["shape_id"]
        print(f"[{i}/{n}] {sid} {op.args_summary(shape)} {shape['dtype']} ({shape['stage']})", flush=True)
        try:
            inputs = op.make_inputs(shape, seed)
            reference = op.reference(shape, inputs)
        except Exception as e:
            for pname, spec in prov_specs.items():
                results.append(_row(run_id, shape, spec.get("provider", pname),
                                    status="failed",
                                    correctness_error=f"input/reference build: {e}"))
            continue

        for pname, spec in prov_specs.items():
            provider = spec.get("provider", pname)
            if not spec.get("enabled", False):
                results.append(_row(run_id, shape, provider, status="not_configured",
                                    correctness_error=spec.get("skip_reason",
                                                               "disabled in baseline_matrix")))
                continue
            adapter = adapters.get(pname)
            if isinstance(adapter, Exception):
                results.append(_row(run_id, shape, provider, status="failed",
                                    correctness_error=f"adapter load: {adapter}"))
                continue
            try:
                supported, reason = adapter.supports(shape)
            except Exception as e:
                supported, reason = False, f"supports() raised: {e}"
            if not supported:
                results.append(_row(run_id, shape, provider, status="unsupported",
                                    correctness_error=reason))
                continue
            try:
                correct, cerr = adapter.check_correctness(shape, inputs, reference)
            except Exception as e:
                correct, cerr = False, f"{type(e).__name__}: {e}"
            status = "ok" if correct else "incorrect"
            results.append(_row(run_id, shape, provider, status=status,
                                correct=correct, correctness_error=cerr))
        del inputs, reference
        if i % 8 == 0:
            torch.cuda.empty_cache()

    jsonl = os.path.join(out_dir, "correctness_results.jsonl")
    common.write_jsonl(jsonl, results)
    n_incorrect = sum(1 for r in results if r["benchmark_status"] == "incorrect")
    n_ok = sum(1 for r in results if r["benchmark_status"] == "ok")
    print(f"\nWrote {len(results)} correctness rows for {n} shapes -> {jsonl}  "
          f"(ok={n_ok}, incorrect={n_incorrect})")
    return {"run_id": run_id, "results": jsonl, "n_shapes": n, "n_rows": len(results),
            "n_ok": n_ok, "n_incorrect": n_incorrect}


def main(argv=None):
    ap = argparse.ArgumentParser(description="FlyDSL correctness-only runner (no timing)")
    ap.add_argument("--op", "--kernel", dest="op", required=True)
    ap.add_argument("--shape-ledger", required=True)
    ap.add_argument("--baseline-matrix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stages", default=None, help="comma list to restrict stages")
    args = ap.parse_args(argv)
    common.bootstrap_env()
    stages = {s.strip() for s in args.stages.split(",")} if args.stages else None
    run(args.op, args.shape_ledger, args.baseline_matrix, args.out,
        seed=args.seed, limit=args.limit, stages=stages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
