"""SGLang serving trace -> weighted rmsnorm shape_ledger rows.

The unique value of a live trace over model-config shapes is the real
*distribution* of per-forward token counts and how often each occurs -- i.e.
production weights. For rmsnorm the kernel shape is (num_tokens, hidden_size):

  * prefill forward: num_tokens = tokens in the prefill batch  (SGLang logs '#new-token')
  * decode forward:  num_tokens = running sequences, 1 tok each (SGLang logs '#running-req')

Two robust input modes (no fragile chrome-trace parsing required):

  --server-log PATH   parse SGLang scheduler stdout: 'Prefill batch. ... #new-token: T'
                      and 'Decode batch. ... #running-req: R' lines -> token-count histogram
  --torch-trace PATH  parse a torch-profiler chrome trace (record_shapes=true): pull the
                      'Input Dims' of rmsnorm/layernorm ops -> (num_tokens, hidden) histogram

Occurrences become weight.occurrences and a normalized traffic_weight, so the
report can compute a production-weighted geomean. Token counts are bucketed
(nearest power-of-two-ish bins) so the ledger stays compact.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

from benchmarks import common
from benchmarks.shape_ledgers import ledger_io

SOURCE_KIND = "sglang_trace"
_NEW_TOKEN = re.compile(r"#new-token:\s*(\d+)")
_RUNNING = re.compile(r"#running-req:\s*(\d+)")
_PREFILL = re.compile(r"[Pp]refill batch")
_DECODE = re.compile(r"[Dd]ecode batch")


def _bucket(n: int) -> int:
    """Bucket a token count to a representative shape (round to a 'nice' value)."""
    if n <= 1:
        return 1
    grid = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
            16384, 32768, 65536, 131072]
    return min(grid, key=lambda g: abs(g - n))


def from_server_log(path: str) -> dict[tuple[str, int], int]:
    """Return {(stage, num_tokens_bucket): occurrences}."""
    hist: Counter = Counter()
    with open(path, errors="ignore") as f:
        for line in f:
            if _PREFILL.search(line):
                m = _NEW_TOKEN.search(line)
                if m:
                    hist[("prefill", _bucket(int(m.group(1))))] += 1
            elif _DECODE.search(line):
                m = _RUNNING.search(line)
                if m:
                    hist[("decode", _bucket(int(m.group(1))))] += 1
    return dict(hist)


def from_torch_trace(path: str, hidden: int) -> dict[tuple[str, int], int]:
    """Parse a chrome trace; bucket rmsnorm/layernorm Input Dims leading dim by hidden match."""
    data = json.load(open(path))
    events = data.get("traceEvents", data) if isinstance(data, dict) else data
    hist: Counter = Counter()
    for ev in events:
        name = str(ev.get("name", "")).lower()
        if "norm" not in name:
            continue
        dims = (ev.get("args", {}) or {}).get("Input Dims") or (ev.get("args", {}) or {}).get("Input dims")
        if not dims:
            continue
        for d in dims:
            if isinstance(d, list) and len(d) >= 2 and d[-1] == hidden:
                ntok = int(d[0]) if len(d) == 2 else int(d[0]) * int(d[1])
                stage = "decode" if ntok <= 8 else "prefill"
                hist[(stage, _bucket(ntok))] += 1
                break
    return dict(hist)


def build_rows(hist: dict, hidden: int, model: str, gpu: str, arch: str,
               extra_dims: list[tuple[str, int]]) -> list[dict]:
    total = sum(hist.values()) or 1
    rows = []
    # main hidden-size rmsnorm, weighted by observed occurrences
    for (stage, ntok), occ in sorted(hist.items()):
        rows.append(_row(model, stage, ntok, hidden, occ, total, gpu, arch,
                         note=f"observed {occ}x in trace ({stage}); num_tokens~{ntok}, hidden={hidden}"))
    # additional per-layer norms (e.g. Qwen3 q/k-norm of size head_dim) at the same token counts
    for label, dim in extra_dims:
        for (stage, ntok), occ in sorted(hist.items()):
            rows.append(_row(model, stage, ntok, dim, occ, total, gpu, arch,
                             note=f"{label} (N={dim}); observed {occ}x ({stage})"))
    return rows


def _row(model, stage, M, N, occ, total, gpu, arch, note) -> dict:
    args = {"M": int(M), "N": int(N), "eps": 1e-5}
    return {
        "op_type": "rmsnorm", "kernel_name": "rmsnorm", "model": model, "stage": stage,
        "gpu": gpu, "arch": arch, "dtype": "bf16", "layout": {"row_major": True}, "args": args,
        "shape_id": common.stable_shape_id(op_type="rmsnorm", model=model, stage=stage,
                                           dtype="bf16", layout={"row_major": True}, args=args),
        "source": {"kind": SOURCE_KIND, "file": None, "input_len": None, "output_len": None,
                   "concurrency": None, "notes": note},
        "weight": {"occurrences": int(occ), "traffic_weight": round(occ / total, 6),
                   "baseline_time_weight": None},
        "baselines_available": ["aiter", "aiter_triton", "triton", "pytorch"], "notes": "",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Import a live SGLang serving trace into weighted rmsnorm shapes")
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--torch-trace", default=None)
    ap.add_argument("--hidden-size", type=int, required=True)
    ap.add_argument("--extra-norm-dim", default=None,
                    help="comma list label:dim for per-layer norms, e.g. qk_norm:128")
    ap.add_argument("--model", default="sglang-trace")
    ap.add_argument("--out", default="benchmarks/examples")
    ap.add_argument("--gpu", default="MI350X")
    ap.add_argument("--arch", default="gfx950")
    args = ap.parse_args(argv)

    if args.server_log:
        hist = from_server_log(args.server_log)
    elif args.torch_trace:
        hist = from_torch_trace(args.torch_trace, args.hidden_size)
    else:
        raise SystemExit("pass --server-log or --torch-trace")
    if not hist:
        raise SystemExit("no rmsnorm/prefill/decode events found in the trace/log")

    extra = []
    if args.extra_norm_dim:
        for part in args.extra_norm_dim.split(","):
            label, _, dim = part.partition(":")
            extra.append((label.strip(), int(dim)))

    rows = build_rows(hist, args.hidden_size, args.model, args.gpu, args.arch, extra)
    path = os.path.join(args.out, "rmsnorm", "shape_ledger.jsonl")
    info = ledger_io.upsert_ledger(path, rows, replace_kinds={SOURCE_KIND})
    print(f"observed {sum(hist.values())} norm forwards across {len(hist)} (stage,tokens) buckets")
    for (stage, ntok), occ in sorted(hist.items()):
        print(f"  {stage:8s} num_tokens~{ntok:7d}  x{occ}")
    print(f"-> +{len(rows)} sglang_trace rows, {info['total']} total in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
