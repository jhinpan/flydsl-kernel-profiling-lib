"""Import AITER model_shapes.json -> per-op shape_ledger.jsonl.

model_shapes.json is model -> kernel_key -> [shape_dict]. The shape_dict carries
only kernel-class fields (N for rmsnorm; N,K,TP_dim for gemm; hq,hkv,dqk,dv for
attention; ...). M / batch / seq are NOT in the file -- bench_models.py
synthesizes M = batch_size * seq_len at run time. We therefore cross the
model-fixed fields with a configurable iteration sweep and label those rows
stage=model_config (rmsnorm/gemm/moe/rope) or comment-derived (attention).
TP sharding mirrors bench_models.py exactly (shard = max(v//tp, 1)).

Verified rules (discovery probe, bench_models.py):
  * rmsnorm:    field N; NEVER sharded.
  * gemm:       N,K,TP_dim; shard the single dim named by TP_dim in {N,K,B}; null=unsharded.
  * batched_gemm: B,N,K,TP_dim; same TP_dim rule.
  * moe:        E,Dim1,Dim2,TopK; ALWAYS shard Dim2 (ignores TP_dim).
  * rope:       num_heads,num_kv_heads,head_dim,...; shard num_heads+num_kv_heads; seq_len=M.
  * mha/mla/unified_attention: hq,hkv,dqk,dv; shard hq+hkv; stage from optional 'comment'.

Usage:
  python -m benchmarks.shape_ledgers.aiter_model_shapes_importer \
    --aiter-model-shapes /sgl-workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json \
    --out benchmarks/examples --tp 8 --gpu MI350X --arch gfx950 --dtype bf16 --ops rmsnorm
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from benchmarks import common
from benchmarks.shape_ledgers import ledger_io

SOURCE_KIND = "aiter_model_shapes"

# aiter kernel key -> canonical op_type (verified proposal from discovery)
OPTYPE = {
    "gemm_a8w8_per_token_scale": "gemm", "gemm_a8w8_blockscale": "gemm",
    "gemm_a16w16": "gemm", "gemm_afp4wfp4": "gemm",
    "batched_gemm_a8w8": "batched_gemm", "batched_gemm_a16wfp4": "batched_gemm",
    "batched_gemm_afp4wfp4": "batched_gemm",
    "moe_op_gemm_a8w8": "moe_gemm", "moe_op_gemm_a8w8_blockscale": "moe_gemm",
    "moe_op_gemm_a8w4": "moe_gemm", "moe_op_gemm_a4w4": "moe_gemm",
    "rmsnorm": "rmsnorm", "rope": "rope",
    "mha": "attention_prefill", "mla": "mla_decode", "unified_attention": "paged_attention",
}

# dtype implied by the aiter kernel name (for provenance/ledger dtype of gemm/moe)
KERNEL_DTYPE = {
    "gemm_a16w16": "bf16", "gemm_a8w8_per_token_scale": "int8",
    "gemm_a8w8_blockscale": "fp8", "gemm_afp4wfp4": "fp4",
    "batched_gemm_a8w8": "int8", "batched_gemm_a16wfp4": "fp4", "batched_gemm_afp4wfp4": "fp4",
    "moe_op_gemm_a8w8": "int8", "moe_op_gemm_a8w8_blockscale": "fp8",
    "moe_op_gemm_a8w4": "mixed_a8w4", "moe_op_gemm_a4w4": "fp4",
}

BASELINES = {
    "rmsnorm": ["aiter", "aiter_triton", "triton", "pytorch"],
    "gemm": ["aiter", "ck", "hipblaslt", "triton", "pytorch"],
    "batched_gemm": ["aiter", "ck", "triton"],
    "moe_gemm": ["aiter", "aiter_triton", "triton"],
    "rope": ["aiter", "aiter_triton", "triton", "pytorch"],
    "attention_prefill": ["aiter", "aiter_triton", "ck", "triton"],
    "mla_decode": ["aiter", "aiter_triton", "ck", "triton"],
    "paged_attention": ["aiter", "aiter_triton", "ck", "triton"],
}

ATTN_STAGE_DEFAULT = {"mha": "prefill", "mla": "decode", "unified_attention": "mixed"}
COMMENT_STAGE = {"prefill": "prefill", "decode": "decode", "text": "mixed", "vision": "mixed"}


def shard(v: int, tp: int) -> int:
    return max(int(v) // tp, 1)


def _truthy(v) -> bool:
    return str(v).lower() in ("true", "1", "yes")


def build_rows(data: dict, *, tp: int, gpu: str, arch: str, dtype: str,
               m_values: list[int], attn_batch: list[int], attn_seq: list[int],
               eps: float, src_file: str, ops_filter: set[str] | None) -> dict[str, list[dict]]:
    """Return {op_type: [pre-dedup rows]}."""
    out: dict[str, list[dict]] = defaultdict(list)
    note_M = f"M synthesized from default token sweep {m_values} (not in model_shapes.json)"

    for model, kernels in data.items():
        for kernel, shapes in kernels.items():
            op = OPTYPE.get(kernel)
            if op is None:
                continue
            if ops_filter and op not in ops_filter and kernel not in ops_filter:
                continue
            for sh in shapes:
                cls = _kernel_class(kernel)
                if cls == "rmsnorm":
                    for M in m_values:
                        out[op].append(_mk(model, kernel, op, gpu, arch, dtype,
                                           {"M": M, "N": int(sh["N"]), "eps": eps},
                                           "model_config", src_file, note_M))
                elif cls in ("gemm", "batched_gemm"):
                    N, K = int(sh["N"]), int(sh["K"])
                    tpdim = sh.get("TP_dim")
                    if tpdim == "N":
                        N = shard(N, tp)
                    elif tpdim == "K":
                        K = shard(K, tp)
                    B = sh.get("B")
                    if tpdim == "B" and B is not None:
                        B = shard(B, tp)
                    dt = KERNEL_DTYPE.get(kernel, dtype)
                    for M in m_values:
                        args = {"M": M, "N": N, "K": K}
                        if B is not None:
                            args["B"] = int(B)
                        out[op].append(_mk(model, kernel, op, gpu, arch, dt, args,
                                           "model_config", src_file,
                                           f"{note_M}; TP={tp} on {tpdim}"))
                elif cls == "moe":
                    dt = KERNEL_DTYPE.get(kernel, dtype)
                    args0 = {"E": int(sh["E"]), "Dim1": int(sh["Dim1"]),
                             "Dim2": shard(int(sh["Dim2"]), tp), "TopK": int(sh["TopK"])}
                    for M in m_values:
                        out[op].append(_mk(model, kernel, op, gpu, arch, dt,
                                           {"M": M, **args0}, "model_config", src_file,
                                           f"{note_M}; TP={tp} shards Dim2"))
                elif cls == "rope":
                    base = {"num_heads": shard(int(sh["num_heads"]), tp),
                            "num_kv_heads": shard(int(sh["num_kv_heads"]), tp),
                            "head_dim": int(sh["head_dim"]),
                            "rotate_style": str(sh.get("rotate_style", "neox")),
                            "two_inputs": _truthy(sh.get("two_inputs", False)),
                            "positions": _truthy(sh.get("positions", False))}
                    for S in m_values:
                        out[op].append(_mk(model, kernel, op, gpu, arch, dtype,
                                           {"seq_len": S, **base}, "model_config", src_file,
                                           f"seq_len from sweep {m_values}; TP={tp} shards heads"))
                elif cls == "attention":
                    stage = COMMENT_STAGE.get(str(sh.get("comment", "")).lower(),
                                              ATTN_STAGE_DEFAULT[kernel])
                    base = {"hq": shard(int(sh["hq"]), tp), "hkv": shard(int(sh["hkv"]), tp),
                            "dqk": int(sh["dqk"]), "dv": int(sh["dv"])}
                    for k in ("sink", "sliding_window_left", "sliding_window"):
                        if k in sh:
                            base[k] = sh[k]
                    seqs = attn_seq if stage != "decode" else [1]
                    for b in attn_batch:
                        for s in seqs:
                            out[op].append(_mk(model, kernel, op, gpu, arch, dtype,
                                               {"batch": b, "seq": s, **base}, stage, src_file,
                                               f"batch/seq from anchors; TP={tp} shards heads"))
    return out


def _kernel_class(kernel: str) -> str:
    if kernel == "rmsnorm":
        return "rmsnorm"
    if kernel == "rope":
        return "rope"
    if kernel in ("mha", "mla", "unified_attention"):
        return "attention"
    if "moe" in kernel:
        return "moe"
    if "batched_gemm" in kernel:
        return "batched_gemm"
    if "gemm" in kernel:
        return "gemm"
    return "other"


def _mk(model, kernel, op, gpu, arch, dtype, args, stage, src_file, note) -> dict:
    return {
        "_model": model, "kernel_name": kernel, "op_type": op, "gpu": gpu, "arch": arch,
        "dtype": dtype, "layout": {"row_major": True}, "args": args, "stage": stage,
        "source": {"kind": SOURCE_KIND, "file": os.path.basename(src_file),
                   "input_len": None, "output_len": None, "concurrency": None, "notes": note},
        "weight": {"occurrences": None, "traffic_weight": None, "baseline_time_weight": None},
        "baselines_available": BASELINES.get(op, []),
        "notes": "",
    }


def dedup(rows: list[dict]) -> list[dict]:
    """Merge rows with identical (op_type,stage,dtype,args); join model names."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        key = common.canonical_json([r["op_type"], r["stage"], r["dtype"], r["args"]])
        groups[key].append(r)
    merged = []
    for key, grp in groups.items():
        models = sorted({g.pop("_model") for g in grp})
        r = dict(grp[0])
        r["model"] = "|".join(models)
        if len(models) > 1:
            r["source"] = dict(r["source"])
            r["source"]["notes"] = (r["source"]["notes"] + f"; models: {', '.join(models)}").strip("; ")
        r["shape_id"] = common.stable_shape_id(
            op_type=r["op_type"], model=r["model"], stage=r["stage"],
            dtype=r["dtype"], layout=r["layout"], args=r["args"])
        merged.append(r)
    return merged


def main(argv=None):
    ap = argparse.ArgumentParser(description="Import AITER model_shapes.json into per-op shape ledgers")
    ap.add_argument("--aiter-model-shapes", required=True)
    ap.add_argument("--out", default="benchmarks/examples", help="examples dir; writes <op_type>/shape_ledger.jsonl")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--gpu", default="MI350X")
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--dtype", default="bf16", help="dtype for ops whose kernel name does not imply one")
    ap.add_argument("--m-values", default="1,32,256,2048,16384", help="token-count sweep for M/seq_len")
    ap.add_argument("--attn-batch", default="1,16,128")
    ap.add_argument("--attn-seq", default="1024,8192")
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--ops", default="rmsnorm", help="comma list of op_types/kernels, or 'all'")
    args = ap.parse_args(argv)

    with open(args.aiter_model_shapes) as f:
        data = json.load(f)
    m_values = [int(x) for x in args.m_values.split(",") if x.strip()]
    attn_batch = [int(x) for x in args.attn_batch.split(",") if x.strip()]
    attn_seq = [int(x) for x in args.attn_seq.split(",") if x.strip()]
    ops_filter = None if args.ops.strip() == "all" else {x.strip() for x in args.ops.split(",")}

    by_op = build_rows(data, tp=args.tp, gpu=args.gpu, arch=args.arch, dtype=args.dtype,
                       m_values=m_values, attn_batch=attn_batch, attn_seq=attn_seq, eps=args.eps,
                       src_file=args.aiter_model_shapes, ops_filter=ops_filter)

    summary = []
    for op, rows in sorted(by_op.items()):
        merged = dedup(rows)
        path = os.path.join(args.out, op, "shape_ledger.jsonl")
        info = ledger_io.upsert_ledger(path, merged, replace_kinds={SOURCE_KIND})
        summary.append((op, len(merged), info["total"], path))

    print(f"Imported {sum(s[1] for s in summary)} aiter rows across {len(summary)} op(s):")
    for op, n_new, total, path in summary:
        print(f"  {op:18s} +{n_new:3d} aiter rows -> {total:3d} total  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
