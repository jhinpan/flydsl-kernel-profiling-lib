#!/usr/bin/env python3
"""Current FlyDSL flash-attention benchmark harness for MI350X/gfx950.

This is intentionally separate from the historical multishape provider because
current FlyDSL split the old flash_attn_func into:

  * kernels.flash_attn_generic
  * kernels.flash_attn_gfx950

The harness benchmarks the auto dispatcher and the two implementation families
directly so we can tell whether a shape is helped by the gfx950 dual-wave path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


REPO = Path(__file__).resolve().parents[2]
FLYDSL_LAB = Path(os.environ.get("FLYDSL_LAB", "/sgl-workspace/FlyDSL-lab"))
BUILD_PKGS = FLYDSL_LAB / "build-fly" / "python_packages"
MLIR_LIBS = BUILD_PKGS / "flydsl" / "_mlir" / "_mlir_libs"

for p in (str(BUILD_PKGS), str(FLYDSL_LAB), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

if str(MLIR_LIBS) not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = f"{MLIR_LIBS}:{os.environ.get('LD_LIBRARY_PATH', '')}"

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


UNIFORM_RANGE = (-1.0, 1.0)
DEFAULT_SEED = 123


@dataclass(frozen=True)
class Shape:
    label: str
    B: int
    S: int
    H: int
    Hkv: int
    D: int
    dtype: str
    causal: bool
    stage: str = "synthetic"
    notes: str = ""


@dataclass
class Result:
    label: str
    provider: str
    B: int
    S: int
    H: int
    Hkv: int
    D: int
    dtype: str
    causal: bool
    stage: str
    expected_auto_path: str
    status: str
    us_median: float | None = None
    us_p10: float | None = None
    us_p90: float | None = None
    tflops: float | None = None
    max_err: float | None = None
    mean_err: float | None = None
    min_cos: float | None = None
    correctness: str = "not_checked"
    err: str = ""
    config: str = ""
    notes: str = ""


def setup_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_dtype(dtype: str):
    if dtype == "bf16":
        return torch.bfloat16
    if dtype in ("fp16", "f16"):
        return torch.float16
    raise ValueError(f"unsupported dtype {dtype}")


def fly_dtype(dtype: str) -> str:
    return "f16" if dtype in ("fp16", "f16") else "bf16"


def expected_auto_path(s: Shape) -> str:
    if s.D == 128 and s.dtype in ("bf16", "fp16", "f16") and s.S >= 384 and s.S % 256 == 0:
        return "dualwave_gfx950"
    if s.H >= 32:
        return "generic_m256" if s.B * s.S >= 4096 else "generic_m128"
    return "generic_m128"


def causal_tag(v: bool) -> str:
    return "causal" if v else "nocausal"


def flops(shape: Shape) -> float:
    s_eff = shape.S / 2.0 if shape.causal else float(shape.S)
    return 4.0 * shape.S * s_eff * shape.D * shape.H * shape.B


def tensor_bytes(shape: Shape) -> int:
    elem = 2
    return shape.B * shape.S * shape.D * (shape.H + 2 * shape.Hkv + shape.H) * elem


def can_verify(shape: Shape, max_score_elems: int) -> bool:
    elems = shape.B * shape.H * shape.S * shape.S
    return elems <= max_score_elems


@torch.no_grad()
def reference_attention(q, k, v, causal: bool):
    q_t = q.transpose(1, 2).float()
    k_t = k.transpose(1, 2).float()
    v_t = v.transpose(1, 2).float()
    if q_t.shape[1] != k_t.shape[1]:
        rep = q_t.shape[1] // k_t.shape[1]
        k_t = k_t.repeat_interleave(rep, dim=1)
        v_t = v_t.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
    return out.transpose(1, 2)


def correctness_stats(out, ref, D: int) -> dict[str, float | str]:
    out_f = out.float().reshape(-1)
    ref_f = ref.float().reshape(-1)
    diff = (out_f - ref_f).abs()
    max_err = float(diff.max().item())
    mean_err = float(diff.mean().item())
    cos = F.cosine_similarity(out_f.reshape(-1, D), ref_f.reshape(-1, D), dim=1)
    min_cos = float(cos.min().item())
    ok = max_err < 1e-2 and min_cos > 0.99
    return {
        "max_err": max_err,
        "mean_err": mean_err,
        "min_cos": min_cos,
        "correctness": "pass" if ok else "fail",
    }


def make_inputs(shape: Shape, seed: int) -> dict[str, torch.Tensor]:
    setup_seed(seed)
    dt = torch_dtype(shape.dtype)
    q = torch.empty((shape.B, shape.S, shape.H, shape.D), dtype=dt, device="cuda").uniform_(*UNIFORM_RANGE)
    k = torch.empty((shape.B, shape.S, shape.Hkv, shape.D), dtype=dt, device="cuda").uniform_(*UNIFORM_RANGE)
    v = torch.empty((shape.B, shape.S, shape.Hkv, shape.D), dtype=dt, device="cuda").uniform_(*UNIFORM_RANGE)
    return {"q": q.contiguous(), "k": k.contiguous(), "v": v.contiguous()}


def bench_us(fn: Callable[[], Any], warmup: int, iters: int, repeats: int) -> tuple[float, float, float]:
    samples: list[float] = []
    for _ in range(repeats):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)
    samples.sort()
    median = statistics.median(samples)
    p10 = samples[max(0, int(math.floor(0.10 * (len(samples) - 1))))]
    p90 = samples[min(len(samples) - 1, int(math.ceil(0.90 * (len(samples) - 1))))]
    return median, p10, p90


class Provider:
    def __init__(self, name: str, config: str = ""):
        self.name = name
        self.config = config
        self.cache: dict[Any, Any] = {}

    def supports(self, shape: Shape) -> tuple[bool, str]:
        if shape.S % 128 != 0:
            return False, "seq_len must be divisible by 128"
        if shape.D < 64 or shape.D % 32 != 0:
            return False, "head_dim must be >=64 and divisible by 32"
        if shape.H % shape.Hkv != 0:
            return False, "H must be divisible by Hkv"
        if shape.dtype not in ("bf16", "fp16", "f16"):
            return False, "dtype must be bf16/fp16"
        if self.name.startswith("dualwave"):
            if shape.D != 128:
                return False, "dualwave is D=128 only"
            if shape.S < 384 or shape.S % 256 != 0:
                return False, "dualwave dispatcher contract is S>=384 and S%256==0"
        if self.name == "generic_m256" and shape.D < 128:
            return False, "skipped after observed GPU memory fault for generic_m256 with D<128"
        if self.name == "auto" and shape.D < 128 and expected_auto_path(shape) == "generic_m256":
            return False, "skipped: auto would dispatch to generic_m256, which faulted for D<128"
        if self.name == "aiter_asm" and shape.dtype != "bf16":
            return False, "aiter asm v3 is bf16 only"
        return True, ""

    def _build_flydsl(self, shape: Shape):
        from kernels.flash_attn_generic import build_flash_attn_func_module_primary
        from kernels.flash_attn_gfx950 import build_flash_attn_dualwave_swp_module

        ds = fly_dtype(shape.dtype)
        if self.name == "auto":
            return build_flash_attn_func_module_primary(
                num_heads=shape.H,
                num_kv_heads=shape.Hkv,
                head_dim=shape.D,
                causal=shape.causal,
                dtype_str=ds,
                waves_per_eu=2,
                daz=True,
            )
        if self.name == "generic_m128":
            return build_flash_attn_func_module_primary(
                num_heads=shape.H,
                num_kv_heads=shape.Hkv,
                head_dim=shape.D,
                causal=shape.causal,
                dtype_str=ds,
                waves_per_eu=2,
                flat_work_group_size=256,
                block_m=128,
                daz=True,
            )
        if self.name == "generic_m256":
            return build_flash_attn_func_module_primary(
                num_heads=shape.H,
                num_kv_heads=shape.Hkv,
                head_dim=shape.D,
                causal=shape.causal,
                dtype_str=ds,
                waves_per_eu=2,
                flat_work_group_size=512,
                block_m=256,
                daz=True,
            )
        if self.name == "generic_m128_n32":
            return build_flash_attn_func_module_primary(
                num_heads=shape.H,
                num_kv_heads=shape.Hkv,
                head_dim=shape.D,
                causal=shape.causal,
                dtype_str=ds,
                waves_per_eu=2,
                flat_work_group_size=256,
                block_m=128,
                path_tag="N32",
                daz=True,
            )
        if self.name == "generic_m128_n128":
            return build_flash_attn_func_module_primary(
                num_heads=shape.H,
                num_kv_heads=shape.Hkv,
                head_dim=shape.D,
                causal=shape.causal,
                dtype_str=ds,
                waves_per_eu=2,
                flat_work_group_size=256,
                block_m=128,
                path_tag="N128",
                daz=True,
            )
        if self.name == "dualwave":
            return build_flash_attn_dualwave_swp_module(
                num_heads=shape.H,
                num_kv_heads=shape.Hkv,
                head_dim=shape.D,
                causal=shape.causal,
                dtype_str=ds,
                waves_per_eu=2,
                daz=True,
            )
        if self.name == "dualwave_no_lazy":
            return build_flash_attn_dualwave_swp_module(
                num_heads=shape.H,
                num_kv_heads=shape.Hkv,
                head_dim=shape.D,
                causal=shape.causal,
                dtype_str=ds,
                waves_per_eu=2,
                daz=True,
                dualwave_swp_lazy_rescale=False,
            )
        if self.name == "dualwave_no_setprio":
            return build_flash_attn_dualwave_swp_module(
                num_heads=shape.H,
                num_kv_heads=shape.Hkv,
                head_dim=shape.D,
                causal=shape.causal,
                dtype_str=ds,
                waves_per_eu=2,
                daz=True,
                dualwave_swp_setprio=False,
            )
        if self.name == "dualwave_no_stagger":
            return build_flash_attn_dualwave_swp_module(
                num_heads=shape.H,
                num_kv_heads=shape.Hkv,
                head_dim=shape.D,
                causal=shape.causal,
                dtype_str=ds,
                waves_per_eu=2,
                daz=True,
                dualwave_swp_enable_stagger=False,
            )
        raise KeyError(self.name)

    def run(self, shape: Shape, inputs: dict[str, torch.Tensor], warmup: int, iters: int, repeats: int,
            verify: bool, ref: torch.Tensor | None) -> Result:
        result = Result(
            label=shape.label,
            provider=self.name,
            B=shape.B,
            S=shape.S,
            H=shape.H,
            Hkv=shape.Hkv,
            D=shape.D,
            dtype=shape.dtype,
            causal=shape.causal,
            stage=shape.stage,
            expected_auto_path=expected_auto_path(shape),
            status="unknown",
            config=self.config,
            notes=shape.notes,
        )
        ok, why = self.supports(shape)
        if not ok:
            result.status = "unsupported"
            result.err = why
            return result

        try:
            if self.name.startswith("aiter"):
                fn, get_output = self._aiter_call(shape, inputs)
            else:
                key = (self.name, shape.H, shape.Hkv, shape.D, shape.dtype, shape.causal)
                if key not in self.cache:
                    self.cache[key] = self._build_flydsl(shape)
                exe = self.cache[key]
                o = torch.empty_like(inputs["q"]).contiguous().view(-1)
                qf = inputs["q"].reshape(-1)
                kf = inputs["k"].reshape(-1)
                vf = inputs["v"].reshape(-1)

                def fn():
                    exe(qf, kf, vf, o, shape.B, shape.S, stream=torch.cuda.current_stream())

                def get_output():
                    fn()
                    torch.cuda.synchronize()
                    return o.reshape(shape.B, shape.S, shape.H, shape.D)

            out = get_output()
            if verify and ref is not None:
                stats = correctness_stats(out, ref, shape.D)
                result.max_err = stats["max_err"]  # type: ignore[assignment]
                result.mean_err = stats["mean_err"]  # type: ignore[assignment]
                result.min_cos = stats["min_cos"]  # type: ignore[assignment]
                result.correctness = stats["correctness"]  # type: ignore[assignment]
                if result.correctness != "pass":
                    result.status = "incorrect"
                    return result

            us, p10, p90 = bench_us(fn, warmup=warmup, iters=iters, repeats=repeats)
            result.us_median = us
            result.us_p10 = p10
            result.us_p90 = p90
            result.tflops = flops(shape) / (us * 1e-6) / 1e12
            result.status = "ok"
            return result
        except RuntimeError as e:
            msg = str(e).replace("\n", " ")[:500]
            result.status = "oom" if "out of memory" in msg.lower() else "failed"
            result.err = msg
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return result
        except Exception as e:  # noqa: BLE001
            result.status = "failed"
            result.err = f"{type(e).__name__}: {str(e).replace(chr(10), ' ')[:500]}"
            return result

    def _aiter_call(self, shape: Shape, inputs: dict[str, torch.Tensor]):
        import aiter

        q, k, v = inputs["q"], inputs["k"], inputs["v"]
        scale = 1.0 / math.sqrt(shape.D)
        out = torch.empty_like(q)
        if self.name == "aiter_ck":

            def call():
                return aiter.mha_fwd(
                    q,
                    k,
                    v,
                    0.0,
                    scale,
                    shape.causal,
                    -1,
                    -1,
                    0,
                    True,
                    False,
                    cu_seqlens_q=None,
                    cu_seqlens_kv=None,
                    out=out,
                    bias=None,
                    alibi_slopes=None,
                    q_descale=None,
                    k_descale=None,
                    v_descale=None,
                    gen=None,
                )

        elif self.name == "aiter_asm":

            def call():
                return aiter.fmha_v3_fwd(
                    q,
                    k,
                    v,
                    0.0,
                    scale,
                    shape.causal,
                    -1,
                    -1,
                    True,
                    False,
                    2,
                    out=out,
                    bias=None,
                    alibi_slopes=None,
                    gen=None,
                )

        else:
            raise KeyError(self.name)

        def fn():
            call()

        def get_output():
            ret = call()
            torch.cuda.synchronize()
            if isinstance(ret, (tuple, list)) and ret:
                candidate = ret[0]
                if isinstance(candidate, torch.Tensor) and candidate.shape == out.shape:
                    return candidate
            return out

        return fn, get_output


def smoke_shapes() -> list[Shape]:
    return [
        Shape("small_generic", 1, 256, 8, 8, 128, "bf16", True, "smoke", "below dualwave S gate"),
        Shape("dualwave_gate", 1, 512, 8, 8, 128, "bf16", True, "smoke", "eligible for dualwave"),
        Shape("diagnostic_old", 1, 2048, 32, 32, 128, "bf16", True, "diagnostic", "old ATT primary neighborhood"),
    ]


def full_shapes() -> list[Shape]:
    shapes: list[Shape] = []
    # Boundary and CI-ish shapes: route coverage around S gates and auto M128/M256 threshold.
    for S in (128, 256, 384, 512, 1024, 2048, 4096, 8192):
        B = 1 if S <= 4096 else 2
        shapes.append(Shape(f"boundary_B{B}_S{S}_H32", B, S, 32, 32, 128, "bf16", True, "boundary"))
    # Existing FlyDSL test/default benchmark families.
    for B, S, H, Hkv in (
        (8, 128, 64, 64),
        (8, 256, 64, 64),
        (1, 384, 64, 64),
        (1, 1024, 64, 64),
        (1, 2048, 64, 64),
        (1, 4096, 64, 64),
        (4, 8192, 64, 64),
        (16, 8192, 16, 16),
        (32, 8192, 8, 8),
        (16, 8192, 64, 8),
    ):
        shapes.append(Shape(f"flydsl_default_B{B}_S{S}_H{H}_Hkv{Hkv}", B, S, H, Hkv, 128, "bf16", True, "ci_default"))
    # Model-like prefill/GQA shapes.
    for label, B, S, H, Hkv, D in (
        ("llama3_8b_prefill", 1, 8192, 32, 8, 128),
        ("llama3_8b_batch", 4, 4096, 32, 8, 128),
        ("llama3_70b_prefill", 1, 8192, 64, 8, 128),
        ("llama3_70b_batch", 4, 4096, 64, 8, 128),
        ("qwen_gqa_h28", 4, 4096, 28, 4, 128),
        ("wide_h128", 1, 4096, 128, 8, 128),
    ):
        shapes.append(Shape(label, B, S, H, Hkv, D, "bf16", True, "model_like"))
    # Head-dim and causal variants. D=64/96 force generic-only coverage.
    for D in (64, 96, 128):
        for causal in (True, False):
            shapes.append(Shape(f"D{D}_{causal_tag(causal)}", 2, 2048, 32, 32, D, "bf16", causal, "tile_shape"))
    # fp16 coverage for both kernels/baselines.
    for S in (512, 2048, 8192):
        shapes.append(Shape(f"fp16_S{S}", 1 if S < 8192 else 2, S, 32, 32, 128, "fp16", True, "dtype"))
    return dedupe_shapes(shapes)


def config_shapes() -> list[Shape]:
    return [
        Shape("config_diag_2048", 1, 2048, 32, 32, 128, "bf16", True, "config_sweep"),
        Shape("config_gqa_8192", 4, 8192, 64, 8, 128, "bf16", True, "config_sweep"),
    ]


def dedupe_shapes(shapes: list[Shape]) -> list[Shape]:
    seen = set()
    out = []
    for s in shapes:
        key = (s.B, s.S, s.H, s.Hkv, s.D, s.dtype, s.causal, s.label)
        if key not in seen:
            out.append(s)
            seen.add(key)
    return out


def providers_for_suite(suite: str) -> list[Provider]:
    if suite == "config":
        names = [
            "generic_m128",
            "generic_m128_n32",
            "generic_m128_n128",
            "generic_m256",
            "dualwave",
            "dualwave_no_lazy",
            "dualwave_no_setprio",
            "dualwave_no_stagger",
            "auto",
            "aiter_ck",
            "aiter_asm",
        ]
    elif suite == "smoke":
        names = ["generic_m128", "generic_m256", "dualwave", "auto", "aiter_ck", "aiter_asm"]
    else:
        names = ["generic_m128", "generic_m256", "dualwave", "auto", "aiter_ck", "aiter_asm"]
    return [Provider(n) for n in names]


def write_jsonl(path: Path, rows: list[Result]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[Result]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(Result("", "", 0, 0, 0, 0, 0, "", False, "", "", "").__dict__)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def metadata() -> dict[str, Any]:
    from flydsl.runtime.device import get_rocm_arch

    info = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "gpu_count_visible": torch.cuda.device_count(),
        "rocm_arch": get_rocm_arch(),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
        "flydsl_lab": str(FLYDSL_LAB),
        "flydsl_head": git_head(FLYDSL_LAB),
        "profiling_repo_head": git_head(REPO),
    }
    return info


def git_head(path: Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    except Exception:
        return "unknown"


def summarize(rows: list[Result], meta: dict[str, Any], out: Path) -> None:
    ok_rows = [r for r in rows if r.status == "ok" and r.us_median and r.tflops]
    by_shape: dict[tuple, list[Result]] = {}
    for r in ok_rows:
        key = (r.label, r.B, r.S, r.H, r.Hkv, r.D, r.dtype, r.causal)
        by_shape.setdefault(key, []).append(r)

    lines = [
        "# FlyDSL Flash Attention MI350X Current Benchmark",
        "",
        f"- timestamp_utc: `{meta['timestamp_utc']}`",
        f"- host: `{meta['host']}`",
        f"- GPU: `{meta['gpu']}` visible_count={meta['gpu_count_visible']} arch=`{meta['rocm_arch']}`",
        f"- HIP_VISIBLE_DEVICES: `{meta['hip_visible_devices']}`",
        f"- FlyDSL: `{meta['flydsl_lab']}@{meta['flydsl_head'][:12]}`",
        f"- timing: warm event timing, median of repeats; output allocation excluded for FlyDSL, preallocated `out=` attempted for AITER",
        "",
        "## Coverage",
        "",
        f"- result rows: {len(rows)}",
        f"- ok rows: {len(ok_rows)}",
        f"- failed/unsupported rows: {len(rows) - len(ok_rows)}",
        "",
        "## Best Provider Per Shape",
        "",
        "| shape | B | S | H/Hkv | D | dtype | causal | auto path | best | us | TFLOPS | auto/CK | dualwave/generic_m256 |",
        "|---|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|",
    ]
    for key, group in sorted(by_shape.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][3], kv[0][4], kv[0][5], kv[0][6], kv[0][7])):
        label, B, S, H, Hkv, D, dtype, causal = key
        best = min(group, key=lambda r: r.us_median or 1e30)
        by_provider = {r.provider: r for r in group}
        auto_ck = ratio(by_provider.get("auto"), by_provider.get("aiter_ck"))
        dw_gm256 = ratio(by_provider.get("dualwave"), by_provider.get("generic_m256"))
        lines.append(
            f"| {label} | {B} | {S} | {H}/{Hkv} | {D} | {dtype} | {causal_tag(causal)} | "
            f"{best.expected_auto_path} | {best.provider} | {best.us_median:.1f} | {best.tflops:.1f} | "
            f"{fmt_ratio(auto_ck)} | {fmt_ratio(dw_gm256)} |"
        )

    lines += [
        "",
        "## Provider Geomean Ratios",
        "",
        "Ratios use per-shape pairs only. `>1` means the left provider is faster.",
        "",
    ]
    pairs = [
        ("auto", "aiter_ck"),
        ("auto", "aiter_asm"),
        ("dualwave", "generic_m256"),
        ("dualwave", "aiter_ck"),
        ("generic_m256", "aiter_ck"),
        ("generic_m128", "aiter_ck"),
    ]
    for a, b in pairs:
        vals = []
        for group in by_shape.values():
            bp = {r.provider: r for r in group}
            rr = ratio(bp.get(a), bp.get(b))
            if rr is not None and rr > 0:
                vals.append(rr)
        if vals:
            geo = math.exp(sum(math.log(v) for v in vals) / len(vals))
            lines.append(f"- `{a}` / `{b}` geomean: **{geo:.3f}x** over {len(vals)} shapes")
        else:
            lines.append(f"- `{a}` / `{b}` geomean: n/a")

    lines += [
        "",
        "## Failures And Unsupported Rows",
        "",
        "| shape | provider | status | reason |",
        "|---|---|---|---|",
    ]
    for r in rows:
        if r.status != "ok":
            lines.append(f"| {r.label} | {r.provider} | {r.status} | {r.err[:160]} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ratio(a: Result | None, b: Result | None) -> float | None:
    if not a or not b or not a.us_median or not b.us_median:
        return None
    return b.us_median / a.us_median


def fmt_ratio(v: float | None) -> str:
    return "" if v is None else f"{v:.2f}x"


def run(args) -> int:
    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda is not available")
    torch.cuda.set_device(0)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.suite == "smoke":
        shapes = smoke_shapes()
    elif args.suite == "config":
        shapes = config_shapes()
    else:
        shapes = full_shapes()
    providers = providers_for_suite(args.suite)
    if args.providers:
        wanted = {p.strip() for p in args.providers.split(",") if p.strip()}
        providers = [p for p in providers if p.name in wanted]

    meta = metadata()
    (out_dir / f"metadata_{args.suite}.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    partial_jsonl = out_dir / f"results_{args.suite}.partial.jsonl"
    partial_jsonl.write_text("", encoding="utf-8")
    print(json.dumps(meta, sort_keys=True), flush=True)
    print(f"Running suite={args.suite} shapes={len(shapes)} providers={[p.name for p in providers]}", flush=True)

    rows: list[Result] = []
    total = len(shapes) * len(providers)
    i = 0
    for shape in shapes:
        inputs = make_inputs(shape, args.seed)
        ref = None
        verify = args.verify and can_verify(shape, args.max_verify_score_elems)
        if verify:
            ref = reference_attention(inputs["q"], inputs["k"], inputs["v"], shape.causal).to(torch_dtype(shape.dtype))
            torch.cuda.synchronize()
        for provider in providers:
            i += 1
            print(
                f"[{i}/{total}] {shape.label} B={shape.B} S={shape.S} H={shape.H}/{shape.Hkv} "
                f"D={shape.D} {shape.dtype} {causal_tag(shape.causal)} provider={provider.name}",
                flush=True,
            )
            row = provider.run(shape, inputs, args.warmup, args.iters, args.repeats, verify, ref)
            rows.append(row)
            with partial_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(row), sort_keys=True) + "\n")
            if row.status == "ok":
                print(f"  ok {row.us_median:.1f} us {row.tflops:.1f} TFLOPS correctness={row.correctness}", flush=True)
            else:
                print(f"  {row.status}: {row.err}", flush=True)
        del inputs, ref
        torch.cuda.empty_cache()

    jsonl = out_dir / f"results_{args.suite}.jsonl"
    csv_path = out_dir / f"results_{args.suite}.csv"
    write_jsonl(jsonl, rows)
    write_csv(csv_path, rows)
    summarize(rows, meta, out_dir / f"REPORT_{args.suite}.md")
    print(f"Wrote {jsonl}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {out_dir / f'REPORT_{args.suite}.md'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["smoke", "full", "config"], default="smoke")
    ap.add_argument("--out", default="benchmarks/examples/flash_attn_current")
    ap.add_argument("--providers", default="", help="comma-separated provider subset")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--max-verify-score-elems", type=int, default=128 * 1024 * 1024)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
