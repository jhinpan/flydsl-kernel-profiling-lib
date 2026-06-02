"""Manual + synthetic-boundary + diagnostic shapes -> shape_ledger.jsonl.

Three sources, all upserted independently of the AITER/serving rows:

  --synthetic-boundary OP  generate canned boundary shapes (source.kind=synthetic)
  --diagnostic "M,N,dtype" add the existing ATT-capture shape (source.kind=diagnostic)
  --manual-file PATH       JSON/YAML list of explicit shapes (source.kind=manual,
                           or per-entry "source_kind")

Usage (rmsnorm):
  python -m benchmarks.shape_ledgers.manual_shape_importer --op rmsnorm \
    --out benchmarks/examples --synthetic-boundary --diagnostic 32768,8192,bf16
"""

from __future__ import annotations

import argparse
import os

from benchmarks import common
from benchmarks.shape_ledgers import ledger_io

_BASELINES = {"rmsnorm": ["aiter", "aiter_triton", "triton", "pytorch"]}


def _row(*, op, model, stage, dtype, args, kind, notes, gpu, arch):
    return {
        "op_type": op, "kernel_name": op, "model": model, "stage": stage,
        "gpu": gpu, "arch": arch, "dtype": dtype,
        "layout": {"row_major": True}, "args": args,
        "shape_id": common.stable_shape_id(op_type=op, model=model, stage=stage,
                                           dtype=dtype, layout={"row_major": True}, args=args),
        "source": {"kind": kind, "file": None, "input_len": None, "output_len": None,
                   "concurrency": None, "notes": notes},
        "weight": {"occurrences": None, "traffic_weight": None, "baseline_time_weight": None},
        "baselines_available": _BASELINES.get(op, []),
        "notes": "",
    }


def rmsnorm_boundary(gpu, arch) -> list[dict]:
    """Probe the FlyDSL fast-path cliff: fast iff N>=2048 & N%2048==0 & 16-bit."""
    rows = []
    # around the 2048-multiple alignment edges (fast vs generic scalar path)
    edge_Ns = [2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192, 8193]
    nonpow2 = [3000, 5333, 12288]
    for N in edge_Ns + nonpow2:
        aligned = (N >= 2048 and N % 2048 == 0)
        note = f"alignment edge: N%2048=={N % 2048} -> FlyDSL {'fast' if aligned else 'generic'} path"
        for M in (1, 4096):
            rows.append(_row(op="rmsnorm", model="synthetic", stage="synthetic", dtype="bf16",
                             args={"M": M, "N": N, "eps": 1e-5}, kind="synthetic",
                             notes=note, gpu=gpu, arch=arch))
    # extremes: tiny single-token decode + very large prefill row count
    for M, N, n in [(1, 8192, "tiny single-token decode"),
                    (131072, 8192, "very large prefill token count"),
                    (1, 2048, "tiny decode, fast-path width")]:
        rows.append(_row(op="rmsnorm", model="synthetic", stage="synthetic", dtype="bf16",
                         args={"M": M, "N": N, "eps": 1e-5}, kind="synthetic", notes=n,
                         gpu=gpu, arch=arch))
    # dtype variants on a fixed fast-path shape
    for dt in ("f16", "f32"):
        rows.append(_row(op="rmsnorm", model="synthetic", stage="synthetic", dtype=dt,
                         args={"M": 4096, "N": 8192, "eps": 1e-5}, kind="synthetic",
                         notes=f"dtype variant ({dt})", gpu=gpu, arch=arch))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Manual/synthetic/diagnostic shape importer")
    ap.add_argument("--op", default="rmsnorm")
    ap.add_argument("--out", default="benchmarks/examples")
    ap.add_argument("--gpu", default="MI350X")
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--synthetic-boundary", action="store_true")
    ap.add_argument("--diagnostic", default=None, help='"M,N,dtype" of the existing ATT-capture shape')
    ap.add_argument("--manual-file", default=None, help="JSON/YAML list of explicit shapes")
    args = ap.parse_args(argv)

    new_rows: list[dict] = []
    kinds: set[str] = set()

    if args.synthetic_boundary:
        if args.op != "rmsnorm":
            raise SystemExit(f"synthetic-boundary not implemented for op={args.op}")
        new_rows += rmsnorm_boundary(args.gpu, args.arch)
        kinds.add("synthetic")

    if args.diagnostic:
        M, N, dt = [x.strip() for x in args.diagnostic.split(",")]
        new_rows.append(_row(op=args.op, model="diagnostic", stage="diagnostic", dtype=dt,
                             args={"M": int(M), "N": int(N), "eps": 1e-5}, kind="diagnostic",
                             notes="shape used by the existing rocprofv3/ATT capture", gpu=args.gpu, arch=args.arch))
        kinds.add("diagnostic")

    if args.manual_file:
        import json
        if args.manual_file.endswith((".yaml", ".yml")):
            import yaml
            spec = yaml.safe_load(open(args.manual_file))
        else:
            spec = json.load(open(args.manual_file))
        for e in spec:
            op = e.get("op_type", args.op)
            kind = e.get("source_kind", "manual")
            kinds.add(kind)
            new_rows.append(_row(op=op, model=e.get("model", "manual"),
                                 stage=e.get("stage", "synthetic"), dtype=e["dtype"],
                                 args=e["args"], kind=kind, notes=e.get("notes", ""),
                                 gpu=args.gpu, arch=args.arch))

    if not new_rows:
        raise SystemExit("nothing to do: pass --synthetic-boundary, --diagnostic, or --manual-file")

    path = os.path.join(args.out, args.op, "shape_ledger.jsonl")
    info = ledger_io.upsert_ledger(path, new_rows, replace_kinds=kinds)
    print(f"Added {len(new_rows)} row(s) [{', '.join(sorted(kinds))}] -> {info['total']} total  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
