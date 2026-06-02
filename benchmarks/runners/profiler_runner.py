"""Tier-1 profiler gate: rocprofv3 kernel-trace for a (shape, provider).

Triggered on hot regressions (kernel-only speedup_vs_best < 0.90). Generates a
self-contained repro that launches the provider's kernel N times for one shape,
runs it under `rocprofv3 --kernel-trace`, and parses per-kernel dispatch
duration + grid/workgroup so the report can corroborate a diagnosis with ground
truth (e.g. "grid = 1 block -> GPU under-occupied" for small-M rmsnorm).

  benchmarks/bench -m benchmarks.runners.profiler_runner \
    --op rmsnorm --shape-id sha1:... --provider flydsl \
    --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
    --out benchmarks/examples/rmsnorm/profiles
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess

from benchmarks import common

_REPRO = '''\
import os, sys, json
sys.path[:0] = {paths!r}
import torch
from benchmarks import ops
from benchmarks.providers.base import load_entrypoint
shape = json.loads({shape_json!r})
op = ops.get_op({op!r})
inputs = op.make_inputs(shape, {seed})
ad = load_entrypoint({entry!r}, {op!r})
ok, why = ad.supports(shape)
assert ok, "provider does not support shape: " + str(why)
for _ in range(10):
    ad.run(shape, inputs)
torch.cuda.synchronize()
for _ in range({iters}):
    ad.run(shape, inputs)
torch.cuda.synchronize()
'''

_ENTRYPOINTS = {
    "flydsl": "benchmarks.providers.flydsl:RmsNormAdapter",
    "pytorch": "benchmarks.providers.pytorch:RmsNormAdapter",
    "aiter": "benchmarks.providers.aiter:RmsNormAdapter",
    "aiter_triton": "benchmarks.providers.aiter_triton:RmsNormAdapter",
    "triton": "benchmarks.providers.triton:RmsNormAdapter",
}


def _parse_kernel_trace(outdir: str) -> list[dict]:
    files = (glob.glob(os.path.join(outdir, "**", "*kernel_trace*.csv"), recursive=True)
             or glob.glob(os.path.join(outdir, "**", "*.csv"), recursive=True))
    agg: dict[str, dict] = {}
    for fp in files:
        with open(fp) as f:
            for row in csv.DictReader(f):
                name = row.get("Kernel_Name") or row.get("kernel_name") or row.get("Name")
                if not name:
                    continue
                dur = None
                for a, b in (("Start_Timestamp", "End_Timestamp"), ("start_timestamp", "end_timestamp")):
                    if row.get(a) and row.get(b):
                        try:
                            dur = (int(row[b]) - int(row[a])) / 1000.0  # ns -> us
                        except ValueError:
                            dur = None
                if dur is None and row.get("Duration"):
                    try:
                        dur = float(row["Duration"]) / 1000.0
                    except ValueError:
                        dur = None
                def _i(k):
                    try:
                        return int(row.get(k) or 0)
                    except ValueError:
                        return 0
                # rocprofv3 reports Grid_Size_* as TOTAL work-items; blocks = grid/workgroup
                gx, gy, gz = _i("Grid_Size_X"), _i("Grid_Size_Y"), _i("Grid_Size_Z")
                wx, wy, wz = _i("Workgroup_Size_X"), _i("Workgroup_Size_Y"), _i("Workgroup_Size_Z")
                e = agg.setdefault(name, {"count": 0, "durations_us": [],
                                          "grid": (gx, gy, gz), "workgroup": (wx, wy, wz),
                                          "vgpr": _i("VGPR_Count"), "sgpr": _i("SGPR_Count"),
                                          "lds": _i("LDS_Block_Size"), "scratch": _i("Scratch_Size")})
                e["count"] += 1
                if dur is not None:
                    e["durations_us"].append(dur)
    out = []
    for name, e in agg.items():
        ds = e["durations_us"]
        gx, gy, gz = e["grid"]
        wx, wy, wz = e["workgroup"]
        wg_threads = max(wx * max(wy, 1) * max(wz, 1), 1)
        total_wi = gx * max(gy, 1) * max(gz, 1)
        n_blocks = total_wi // wg_threads if wg_threads else None
        out.append({"kernel": name, "dispatches": e["count"],
                    "median_us": (sorted(ds)[len(ds) // 2] if ds else None),
                    "grid_workitems": [gx, gy, gz], "workgroup": [wx, wy, wz],
                    "workgroups_launched": n_blocks, "wg_threads": wg_threads,
                    "vgpr": e["vgpr"], "sgpr": e["sgpr"], "lds_bytes": e["lds"],
                    "scratch_bytes": e["scratch"]})
    return sorted(out, key=lambda d: -(d["median_us"] or 0))


def run(op: str, shape_id: str, provider: str, ledger_path: str, out_root: str,
        *, iters: int = 60, seed: int = 1234) -> dict:
    shapes = {r["shape_id"]: r for r in common.read_jsonl(ledger_path)}
    shape = shapes.get(shape_id)
    if shape is None:
        raise SystemExit(f"shape_id {shape_id} not in {ledger_path}")
    entry = _ENTRYPOINTS.get(provider)
    if entry is None:
        raise SystemExit(f"no entrypoint for provider {provider}")

    outdir = os.path.join(out_root, shape_id.replace("sha1:", ""), provider)
    os.makedirs(outdir, exist_ok=True)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    paths = [common.FLYDSL_BUILD_PKGS, common.FLYDSL_LAB, repo_root]
    repro = os.path.join(outdir, "repro.py")
    with open(repro, "w") as f:
        f.write(_REPRO.format(paths=paths, shape_json=json.dumps(shape), op=op,
                              seed=seed, entry=entry, iters=iters))

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(paths + [env.get("PYTHONPATH", "")])
    env["LD_LIBRARY_PATH"] = os.pathsep.join([common.FLYDSL_MLIR_LIBS, env.get("LD_LIBRARY_PATH", "")])
    env.setdefault("SGLANG_USE_AITER", "0")
    cmd = ["rocprofv3", "--kernel-trace", "--output-format", "csv", "-d", outdir,
           "--", env.get("PYTHON", "python"), repro]
    print("  $ " + " ".join(cmd))
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    (open(os.path.join(outdir, "rocprofv3.stdout.txt"), "w").write(proc.stdout))
    (open(os.path.join(outdir, "rocprofv3.stderr.txt"), "w").write(proc.stderr))

    kernels = _parse_kernel_trace(outdir)
    diag = {
        "op": op, "shape_id": shape_id, "provider": provider, "args": shape["args"],
        "dtype": shape["dtype"], "iters": iters, "rc": proc.returncode,
        "kernels": kernels[:10],
        "note": ("grid reports total work-items; a small grid (e.g. one workgroup for small M) "
                 "under-occupies the ~256-CU MI350X"),
    }
    with open(os.path.join(outdir, "diagnosis.json"), "w") as f:
        json.dump(diag, f, indent=2)
    print(f"  wrote {outdir}/diagnosis.json  (kernels: {len(kernels)}, rc={proc.returncode})")
    for k in kernels[:4]:
        print(f"    {k['kernel'][:46]:46s} disp={k['dispatches']:3d} med={k['median_us']}us "
              f"blocks={k.get('workgroups_launched')} wg={k.get('wg_threads')} vgpr={k.get('vgpr')}")
    return {"outdir": outdir, "diagnosis": diag}


def main(argv=None):
    ap = argparse.ArgumentParser(description="rocprofv3 kernel-trace gate for a (shape, provider)")
    ap.add_argument("--op", required=True)
    ap.add_argument("--shape-id", required=True)
    ap.add_argument("--provider", default="flydsl")
    ap.add_argument("--shape-ledger", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)
    common.bootstrap_env()
    run(args.op, args.shape_id, args.provider, args.shape_ledger, args.out,
        iters=args.iters, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
