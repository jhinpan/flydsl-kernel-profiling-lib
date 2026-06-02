"""Import ATOM-style serving anchors (ISL/OSL/concurrency) into the ledger.

HONESTY RULE: ISL / OSL / concurrency are *serving-workload* metadata, not
kernel shapes. An anchor like {isl:1024, osl:1024, concurrency:[...]} describes a
serving scenario (input/output sequence length, in-flight request count); it does
NOT, by itself, name a tensor shape any kernel sees. So this importer keeps the
two honest:

  1. It always emits one *serving-scenario* ledger row per (anchor, concurrency)
     with op_type="serving_scenario", stage="mixed". These carry the workload
     anchor verbatim and are NOT a kernel benchmark -- the multishape runner
     filters by op_type, so op_type="serving_scenario" never pollutes a kernel
     run. They live in examples/serving_scenarios/shape_ledger.jsonl.

  2. ONLY when a concrete model config is supplied (--model-config a HF
     config.json) AND a kernel is named (--op rmsnorm) does it *derive* real
     kernel shapes from the anchor, making the token-count assumption explicit:
        prefill: M = isl * concurrency   (all prompt tokens normalized at once)
        decode:  M = concurrency         (one new token per in-flight sequence)
        N      = hidden_size from the config
     Weights are left null: these are synthetic anchors, so the occurrence /
     traffic weight is genuinely unknown (a live serving trace fills those in).
     Derived rows go into examples/<op>/shape_ledger.jsonl.

This importer OWNS source.kind="atom_workload". upsert_ledger(replace_kinds=
{"atom_workload"}) makes re-runs idempotent and never clobbers rows another
importer (aiter_model_shapes / sglang_trace / synthetic / diagnostic) contributed.

Usage:
  # serving-scenario rows only (no kernel shapes -- no model config):
  python -m benchmarks.shape_ledgers.atom_workload_importer \
    --out benchmarks/examples --gpu MI350X --arch gfx950

  # also derive rmsnorm prefill/decode shapes from a model's hidden_size:
  python -m benchmarks.shape_ledgers.atom_workload_importer \
    --anchors anchors.json --model-config /path/to/config.json --op rmsnorm \
    --model llama-3.1-8b --out benchmarks/examples --gpu MI350X --arch gfx950
"""

from __future__ import annotations

import argparse
import json
import os

from benchmarks import common
from benchmarks.shape_ledgers import ledger_io

SOURCE_KIND = "atom_workload"

# The two canonical ATOM serving anchors. Used verbatim when --anchors is omitted.
DEFAULT_ANCHORS = [
    {"isl": 1024, "osl": 1024,
     "concurrency": [4, 8, 16, 32, 64, 128, 256, 512, 1024],
     "random_range_ratio": 0.8},
    {"isl": 8192, "osl": 1024,
     "concurrency": [4, 8, 16, 32, 64, 128, 256, 512, 1024],
     "random_range_ratio": 0.8},
]

# kernels we know how to derive a (M, N) tensor shape for from a serving anchor.
# Each maps stage -> a function (isl, osl, c) -> token-count M; N comes from the
# model config (hidden_size). Extend this dict to teach the importer a new op.
_DERIVERS = {
    "rmsnorm": {
        "prefill": lambda isl, osl, c: isl * c,   # all prompt tokens at once
        "decode": lambda isl, osl, c: c,          # one token per in-flight seq
    },
}

# HF config keys that name the model's hidden width, most-specific first.
_HIDDEN_KEYS = ("hidden_size", "n_embd", "d_model", "hidden_dim", "model_dim")

_BASELINES = {"rmsnorm": ["aiter", "aiter_triton", "triton", "pytorch"]}


# --------------------------------------------------------------------------- #
# serving-scenario rows (always emitted; NOT kernel shapes)
# --------------------------------------------------------------------------- #
def serving_rows(anchors: list[dict], *, model: str, gpu: str, arch: str,
                 src_file: str | None) -> list[dict]:
    """One row per (anchor, concurrency). op_type=serving_scenario, stage=mixed."""
    note = ("serving workload anchor (ISL/OSL/concurrency), NOT a kernel shape; "
            "derive kernel shapes by supplying --model-config + --op")
    rows: list[dict] = []
    for a in anchors:
        isl, osl = int(a["isl"]), int(a["osl"])
        rrr = a.get("random_range_ratio")
        for c in a["concurrency"]:
            c = int(c)
            args = {"isl": isl, "osl": osl, "concurrency": c}
            if rrr is not None:
                args["random_range_ratio"] = float(rrr)
            rows.append(_row(
                op="serving_scenario", kernel_name="serving_scenario", model=model,
                stage="mixed", dtype="n/a", args=args, gpu=gpu, arch=arch,
                input_len=isl, output_len=osl, concurrency=c, src_file=src_file,
                src_notes=note, baselines=[],
                notes="workload anchor; no kernel is benchmarked from this row"))
    return rows


# --------------------------------------------------------------------------- #
# derived kernel rows (only with a model config + a known op)
# --------------------------------------------------------------------------- #
def derived_rows(anchors: list[dict], *, op: str, hidden_size: int, dtype: str,
                 model: str, gpu: str, arch: str, eps: float,
                 src_file: str | None, config_file: str) -> list[dict]:
    """Derive (M, N) kernel shapes from each (anchor, concurrency, stage)."""
    derivers = _DERIVERS.get(op)
    if derivers is None:
        raise SystemExit(
            f"--op {op} has no anchor->shape deriver; known: {sorted(_DERIVERS)}. "
            "(serving_scenario rows are still emitted regardless.)")
    rows: list[dict] = []
    for a in anchors:
        isl, osl = int(a["isl"]), int(a["osl"])
        for c in a["concurrency"]:
            c = int(c)
            for stage, m_of in derivers.items():
                M = int(m_of(isl, osl, c))
                clamp_note = ""
                # real serving uses chunked prefill: a single rmsnorm forward never
                # sees isl*concurrency tokens, only up to one chunk. Clamp to keep
                # derived prefill shapes representative (and benchmarkable).
                if stage == "prefill" and M > _CHUNKED_PREFILL_MAX:
                    clamp_note = f"; clamped from {M} to chunked-prefill cap {_CHUNKED_PREFILL_MAX}"
                    M = _CHUNKED_PREFILL_MAX
                note = (f"derived from ATOM anchor isl={isl} osl={osl} concurrency={c} "
                        f"({stage}: M={_m_formula(op, stage)}{clamp_note}); "
                        f"N=hidden_size={hidden_size} from {os.path.basename(config_file)}; "
                        "synthetic anchor -> weights unknown")
                rows.append(_row(
                    op=op, kernel_name=op, model=model, stage=stage, dtype=dtype,
                    args={"M": M, "N": int(hidden_size), "eps": eps},
                    gpu=gpu, arch=arch, input_len=isl, output_len=osl, concurrency=c,
                    src_file=src_file, src_notes=note,
                    baselines=_BASELINES.get(op, []), notes=""))
    return rows


_CHUNKED_PREFILL_MAX = 131072  # SGLang chunked prefill caps tokens per forward


def _m_formula(op: str, stage: str) -> str:
    return {"prefill": "isl*concurrency", "decode": "concurrency"}.get(stage, "?")


# --------------------------------------------------------------------------- #
# row builder + config helpers
# --------------------------------------------------------------------------- #
def _row(*, op, kernel_name, model, stage, dtype, args, gpu, arch, input_len,
         output_len, concurrency, src_file, src_notes, baselines, notes) -> dict:
    layout = {"row_major": True}
    return {
        "op_type": op, "kernel_name": kernel_name, "model": model, "stage": stage,
        "gpu": gpu, "arch": arch, "dtype": dtype, "layout": layout, "args": args,
        "shape_id": common.stable_shape_id(op_type=op, model=model, stage=stage,
                                           dtype=dtype, layout=layout, args=args),
        "source": {"kind": SOURCE_KIND,
                   "file": os.path.basename(src_file) if src_file else None,
                   "input_len": input_len, "output_len": output_len,
                   "concurrency": concurrency, "notes": src_notes},
        "weight": {"occurrences": None, "traffic_weight": None,
                   "baseline_time_weight": None},
        "baselines_available": baselines,
        "notes": notes,
    }


def load_anchors(path: str | None) -> tuple[list[dict], str | None]:
    """Return (anchors, file). Built-in default when path is None."""
    if path is None:
        return [dict(a) for a in DEFAULT_ANCHORS], None
    with open(path) as f:
        anchors = json.load(f)
    if not isinstance(anchors, list):
        raise SystemExit(f"{path}: expected a JSON list of anchor objects")
    for a in anchors:
        for k in ("isl", "osl", "concurrency"):
            if k not in a:
                raise SystemExit(f"{path}: anchor missing required key '{k}': {a}")
    return anchors, path


def hidden_size_from_config(path: str) -> int:
    with open(path) as f:
        cfg = json.load(f)
    # HF text configs sometimes nest the language model under text_config.
    for scope in (cfg, cfg.get("text_config", {}) or {}):
        for k in _HIDDEN_KEYS:
            if isinstance(scope, dict) and isinstance(scope.get(k), int):
                return int(scope[k])
    raise SystemExit(f"{path}: no hidden-size key {(_HIDDEN_KEYS)} found in config.json")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Import ATOM serving anchors (ISL/OSL/concurrency) into the ledger")
    ap.add_argument("--anchors", default=None,
                    help="JSON list of anchors; built-in ATOM defaults if omitted")
    ap.add_argument("--model-config", default=None,
                    help="HF config.json; REQUIRED to derive kernel shapes (else only "
                         "serving-scenario rows are written)")
    ap.add_argument("--op", default=None,
                    help=f"kernel to derive shapes for, e.g. one of {sorted(_DERIVERS)}")
    ap.add_argument("--model", default="atom-anchor",
                    help="model label for the rows")
    ap.add_argument("--out", default="benchmarks/examples",
                    help="examples dir; serving rows -> serving_scenarios/, derived -> <op>/")
    ap.add_argument("--gpu", default="MI350X")
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--dtype", default="bf16",
                    help="dtype for derived kernel rows")
    ap.add_argument("--eps", type=float, default=1e-5,
                    help="rmsnorm epsilon for derived rows")
    args = ap.parse_args(argv)

    anchors, anchor_file = load_anchors(args.anchors)
    summary = []

    # 1. serving-scenario rows -- always, never a kernel benchmark
    srows = serving_rows(anchors, model=args.model, gpu=args.gpu, arch=args.arch,
                         src_file=anchor_file)
    spath = os.path.join(args.out, "serving_scenarios", "shape_ledger.jsonl")
    sinfo = ledger_io.upsert_ledger(spath, srows, replace_kinds={SOURCE_KIND})
    summary.append(("serving_scenario", len(srows), sinfo["total"], spath))

    # 2. derived kernel rows -- only with BOTH a model config and a known op
    if args.model_config and args.op:
        hidden = hidden_size_from_config(args.model_config)
        drows = derived_rows(anchors, op=args.op, hidden_size=hidden, dtype=args.dtype,
                             model=args.model, gpu=args.gpu, arch=args.arch, eps=args.eps,
                             src_file=anchor_file, config_file=args.model_config)
        dpath = os.path.join(args.out, args.op, "shape_ledger.jsonl")
        dinfo = ledger_io.upsert_ledger(dpath, drows, replace_kinds={SOURCE_KIND})
        summary.append((args.op, len(drows), dinfo["total"], dpath))
    elif args.model_config or args.op:
        print("note: deriving kernel shapes needs BOTH --model-config and --op; "
              "wrote serving-scenario rows only.")

    print(f"Imported {len(anchors)} ATOM anchor(s) [kind={SOURCE_KIND}]:")
    for label, n_new, total, path in summary:
        print(f"  {label:18s} +{n_new:3d} rows -> {total:3d} total  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
