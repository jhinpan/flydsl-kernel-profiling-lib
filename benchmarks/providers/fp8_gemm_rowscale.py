"""FP8 row-scale GEMM provider adapters (C = (A*scale_a) @ (B*scale_b)^T).

Kernel: kernels/fp8_gemm_4wave.compile_fp8_gemm_4w / fp8_gemm_8wave.compile_fp8_gemm_8w
(verified against tests/kernels/test_fp8_gemm_rowscale.py). Both are CDNA4-only
(gfx95*). A is (M,K) fp8_e4m3fn row-major, B_T is (N,K) fp8_e4m3fn row-major
(one row per output column), scale_a is (M,) fp32 per-row, scale_b is (N,) fp32
per-row, C is (M,N) bf16. Every provider takes the SAME pre-quantized fp8 A, B,
and fp32 row scales -- no layout conversion (includes_layout_conversion=False).

  * flydsl       -> compile_fp8_gemm_4w / _8w (HipKittens-derived 4/8-wave MFMA
                    16x16x128, XCD swizzle, fp8_e4m3fn in, bf16 out, row scales).
                    NOTE: the 2026-06-01 MI350X sweep flagged this as a config-
                    independent compile-fail ("flyc.compile(): missing
                    _reusable_slot_spec" fast-dispatch path). The adapter is fully
                    wired; if compile still raises, supports() stays True and the
                    failure surfaces in run() / check_correctness as the recorded
                    error (so the row is honest, not silently dropped).
  * pytorch      -> torch._scaled_mm(a_fp8, b_fp8.t(), scale_a=(M,1), scale_b=(1,N),
                    out_dtype=bf16). Also the recipe's named baseline. (The fp32
                    golden lives in ops.Fp8GemmRowscaleOp.reference, not here.)
  * aiter        -> aiter.ops.gemm_op_a8w8.gemm_a8w8 (compiled CK/ASM a8w8; accepts
                    fp8 weights, per-row x_scale (M,1) / w_scale (1,N), bf16 out).
  * aiter_triton -> aiter.ops.triton...gemm_a8w8_per_token_scale (pure Triton,
                    per-token x_scale + per-output-channel w_scale, Y = X @ W^T).
  * gluon        -> aiter.ops.triton.gluon.gemm_a8w8.gemm_a8w8 (Gluon a8w8).
  * hipblaslt    -> torch._scaled_mm path is the hipBLASLt-backed fp8 scaled GEMM
                    on ROCm, but the backend is not Python-selectable and is already
                    raced as 'pytorch'; honest stub here to avoid a duplicate row.
  * ck/aiter_ck/aiter_asm/triton -> honest stubs (no separately-selectable / no
                    standalone non-aiter path on this node).
"""

from __future__ import annotations

from benchmarks import common
from benchmarks.providers.base import ProviderAdapter

# Ledger dtypes this kernel implements. fp8_e4m3 row-scale only (CDNA4 OCP fp8).
_OK_FP8 = {"fp8", "fp8_e4m3"}

# 4-wave: BLOCK_M/N >= 64, %64==0; 8-wave: BLOCK_M >= 128 %128, BLOCK_N >= 256 %256.
# BLOCK_K is fixed at 128 inside both builders, so K must be % 128 == 0.
_BLOCK_K = 128


def _MNK(shape):
    a = shape["args"]
    return int(a["M"]), int(a["N"]), int(a["K"])


def _arch_is_cdna4() -> tuple[bool, str | None]:
    a = common.arch()
    if a is None:
        # No GPU visible at supports() time (CPU import path) -- don't hard-fail;
        # let the runtime gate (flydsl_runtime_ok / launch) decide on-device.
        return True, None
    if "gfx95" not in a:
        return False, f"FP8 row-scale GEMM requires CDNA4 (gfx95*), got {a}"
    return True, None


def _pick_tiles(M, N, use_8w):
    """Validated (BLOCK_M, BLOCK_N) from the test parametrization. 8-wave needs
    BLOCK_N>=256; 4-wave can use 64 for small N. Returns None if no tile divides N."""
    if use_8w:
        cands = [(256, 256), (128, 256)]
    else:
        cands = [(256, 256), (128, 128), (64, 64)]
    for bm, bn in cands:
        if N % bn == 0 and M >= 1:  # M is padded by ceildiv in the launcher
            return bm, bn
    return None


class FlyDSL(ProviderAdapter):
    name = "flydsl"
    includes_allocation = False
    includes_jit = False
    includes_layout_conversion = False

    # default to the 4-wave variant (matches the recipe's primary path); the
    # 8-wave variant is exercised by a separate ledger row via op args use_8w.
    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "fp8_gemm_rowscale":
            return False, "flydsl fp8_gemm_rowscale adapter only implements fp8_gemm_rowscale"
        if shape["dtype"] not in _OK_FP8:
            return False, f"FlyDSL fp8 row-scale GEMM is fp8_e4m3 only, not {shape['dtype']}"
        M, N, K = _MNK(shape)
        if K % _BLOCK_K != 0:
            return False, f"K={K} must be % {_BLOCK_K} == 0 (kernel BLOCK_K is fixed at {_BLOCK_K})"
        use_8w = bool(shape.get("args", {}).get("use_8w", False))
        if _pick_tiles(M, N, use_8w) is None:
            return False, (f"no valid {'8' if use_8w else '4'}-wave tile for M={M},N={N} "
                           f"(need N % BLOCK_N == 0)")
        ok, why = _arch_is_cdna4()
        if not ok:
            return False, why
        ok, why = common.flydsl_runtime_ok()
        return (ok, why) if not ok else (True, None)

    def _build(self, shape, inputs):
        import torch

        M, N, K = inputs["M"], inputs["N"], inputs["K"]
        use_8w = bool(shape.get("args", {}).get("use_8w", False))
        preshuffle = bool(shape.get("args", {}).get("b_preshuffled", False))
        bm, bn = _pick_tiles(M, N, use_8w)
        key = (M, N, K, use_8w, preshuffle, bm, bn)
        if key in self._cache:
            return self._cache[key]

        common.bootstrap_env()
        import flydsl.compiler as flyc
        from kernels.fp8_gemm_4wave import compile_fp8_gemm_4w
        from kernels.fp8_gemm_8wave import compile_fp8_gemm_8w
        from kernels.fp8_gemm_utils import preshuffle_b

        a_q = inputs["a"]                 # (M, K) fp8_e4m3fn
        b_q = inputs["b"]                 # (N, K) fp8_e4m3fn
        scale_a = inputs["scale_a"]       # (M,) fp32
        scale_b = inputs["scale_b"]       # (N,) fp32

        b_kernel = preshuffle_b(b_q) if preshuffle else b_q
        c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")

        if use_8w:
            launch_fn = compile_fp8_gemm_8w(K=K, BLOCK_M=bm, BLOCK_N=bn, b_preshuffled=preshuffle)
            variant = "8wave"
        else:
            launch_fn = compile_fp8_gemm_4w(K=K, BLOCK_M=bm, BLOCK_N=bn,
                                            use_xcd_remap=True, b_preshuffled=preshuffle)
            variant = "4wave(xcd_remap)"

        # fp8 tensors are passed to the kernel as their raw i8 view, flattened.
        def _as_i8(t):
            return t.view(torch.int8) if "float8" in str(t.dtype) else t

        def _args(cc, aa, bb, sa, sb):
            return (
                _as_i8(aa).contiguous().view(-1),
                _as_i8(bb).contiguous().view(-1),
                cc.contiguous().view(-1),
                sa.contiguous().view(-1),
                sb.contiguous().view(-1),
                M,
                N,
                torch.cuda.current_stream(),
            )

        compiled = flyc.compile(launch_fn, *_args(c, a_q, b_kernel, scale_a, scale_b))

        ctx = {
            "compiled": compiled, "c": c, "b_kernel": b_kernel,
            "a_q": a_q, "scale_a": scale_a, "scale_b": scale_b, "_args": _args,
        }
        self._cache[key] = ctx
        self.provider_detail = (
            f"compile_fp8_gemm_{'8w' if use_8w else '4w'}(K={K},BLOCK_M={bm},BLOCK_N={bn},"
            f"b_preshuffled={preshuffle}); {variant} MFMA16x16x128 fp8_e4m3->bf16, "
            f"row scales; flyc.compile static; preallocated C. "
            f"NOTE: MI350X sweep flagged compile-fail (missing _reusable_slot_spec)")
        return ctx

    def run(self, shape, inputs):
        ctx = self._build(shape, inputs)
        c = ctx["c"]
        ctx["compiled"](*ctx["_args"](
            c, ctx["a_q"], ctx["b_kernel"], ctx["scale_a"], ctx["scale_b"]))
        return c


class PyTorch(ProviderAdapter):
    name = "pytorch"
    provider_detail = ("torch._scaled_mm(a_fp8, b_fp8.t(), scale_a=(M,1), scale_b=(1,N), "
                       "out_dtype=bf16) -- recipe baseline (hipBLASLt-backed fp8 scaled GEMM)")
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "fp8_gemm_rowscale":
            return False, "pytorch fp8_gemm_rowscale adapter only implements fp8_gemm_rowscale"
        if shape["dtype"] not in _OK_FP8:
            return False, f"this adapter compares fp8_e4m3 row-scale, not {shape['dtype']}"
        try:
            import torch
            if not hasattr(torch, "_scaled_mm"):
                return False, "torch._scaled_mm unavailable in this torch build"
        except Exception as e:
            return False, f"import torch failed ({type(e).__name__})"
        return True, None

    def run(self, shape, inputs):
        import torch
        M, N = inputs["M"], inputs["N"]
        a_q = inputs["a"]            # (M, K) fp8_e4m3fn
        b_q = inputs["b"]            # (N, K) fp8_e4m3fn
        sa = inputs["scale_a"].view(M, 1).to(torch.float32).contiguous()
        sb = inputs["scale_b"].view(1, N).to(torch.float32).contiguous()
        out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        torch._scaled_mm(a_q, b_q.t(), scale_a=sa, scale_b=sb,
                         out_dtype=torch.bfloat16, out=out)
        return out


class Aiter(ProviderAdapter):
    name = "aiter"
    includes_allocation = True
    includes_layout_conversion = False

    def supports(self, shape):
        if shape.get("op_type") != "fp8_gemm_rowscale":
            return False, "aiter fp8_gemm_rowscale adapter only implements fp8_gemm_rowscale"
        if shape["dtype"] not in _OK_FP8:
            return False, f"aiter gemm_a8w8 wired for fp8_e4m3 row-scale here, not {shape['dtype']}"
        try:
            from aiter.ops.gemm_op_a8w8 import gemm_a8w8  # noqa: F401
        except Exception as e:
            return False, f"import failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def run(self, shape, inputs):
        import torch
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8
        M, N = inputs["M"], inputs["N"]
        # compiled CK/ASM a8w8: XQ (M,K), WQ (N,K), x_scale (M,1), w_scale (1,N), bf16 out.
        x_scale = inputs["scale_a"].view(M, 1).to(torch.float32).contiguous()
        w_scale = inputs["scale_b"].view(1, N).to(torch.float32).contiguous()
        self.provider_detail = ("aiter.ops.gemm_op_a8w8.gemm_a8w8 (compiled CK/ASM a8w8; "
                                "fp8 weights, per-row x_scale/w_scale; backend opaque from Python)")
        return gemm_a8w8(inputs["a"], inputs["b"], x_scale, w_scale,
                         bias=None, dtype=torch.bfloat16)


class AiterTriton(ProviderAdapter):
    name = "aiter_triton"
    provider_detail = ("aiter.ops.triton...gemm_a8w8_per_token_scale (Triton; per-token "
                       "x_scale + per-channel w_scale; Y = X @ W^T)")
    includes_allocation = True
    includes_layout_conversion = False

    def supports(self, shape):
        if shape.get("op_type") != "fp8_gemm_rowscale":
            return False, "aiter_triton fp8_gemm_rowscale adapter only implements fp8_gemm_rowscale"
        if shape["dtype"] not in _OK_FP8:
            return False, f"aiter triton a8w8 per-token wired for fp8_e4m3, not {shape['dtype']}"
        try:
            from aiter.ops.triton.gemm.basic.gemm_a8w8_per_token_scale import (  # noqa: F401
                gemm_a8w8_per_token_scale)
        except Exception as e:
            return False, f"import failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def run(self, shape, inputs):
        import torch
        from aiter.ops.triton.gemm.basic.gemm_a8w8_per_token_scale import gemm_a8w8_per_token_scale
        M, N = inputs["M"], inputs["N"]
        # x (M,K), w (N,K) internally transposed; x_scale (M,1), w_scale (N,1).
        x_scale = inputs["scale_a"].view(M, 1).to(torch.float32).contiguous()
        w_scale = inputs["scale_b"].view(N, 1).to(torch.float32).contiguous()
        with torch.no_grad():
            return gemm_a8w8_per_token_scale(inputs["a"], inputs["b"], x_scale, w_scale,
                                             dtype=torch.bfloat16)


class Gluon(ProviderAdapter):
    name = "gluon"
    provider_detail = "aiter.ops.triton.gluon.gemm_a8w8.gemm_a8w8 (Gluon a8w8; Y = X @ W^T)"
    includes_allocation = True
    includes_layout_conversion = False

    def supports(self, shape):
        if shape.get("op_type") != "fp8_gemm_rowscale":
            return False, "gluon fp8_gemm_rowscale adapter only implements fp8_gemm_rowscale"
        if shape["dtype"] not in _OK_FP8:
            return False, f"gluon a8w8 wired for fp8_e4m3 row-scale here, not {shape['dtype']}"
        try:
            from aiter.ops.triton.gluon.gemm_a8w8 import gemm_a8w8  # noqa: F401
        except Exception as e:
            return False, f"import failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def run(self, shape, inputs):
        import torch
        from aiter.ops.triton.gluon.gemm_a8w8 import gemm_a8w8
        M, N = inputs["M"], inputs["N"]
        x_scale = inputs["scale_a"].view(M, 1).to(torch.float32).contiguous()
        w_scale = inputs["scale_b"].view(1, N).to(torch.float32).contiguous()
        with torch.no_grad():
            return gemm_a8w8(inputs["a"], inputs["b"], x_scale, w_scale,
                             bias=None, dtype=torch.bfloat16)


class _Stub(ProviderAdapter):
    _reason = "not available"

    def supports(self, shape):
        return False, self._reason


class HipBLASLt(_Stub):
    name = "hipblaslt"
    _reason = ("torch._scaled_mm IS the hipBLASLt-backed fp8 scaled GEMM on ROCm, but the "
               "backend is not Python-selectable and is already raced as 'pytorch'; "
               "stubbed here to avoid a duplicate identical row")


class AiterCK(_Stub):
    name = "aiter_ck"
    _reason = ("CK a8w8 backend not separately selectable from Python; folded into 'aiter' "
               "(gemm_a8w8 -> gemm_a8w8_CK)")


class AiterASM(_Stub):
    name = "aiter_asm"
    _reason = ("gemm_a8w8_ASM needs a pre-shuffled weight (shuffle_weight(layout=(32,16))) + "
               "mandatory bias + asm-specific kernel-name lookup; not wired as a fair row-major "
               "row-scale match here (the compiled CK/ASM path is exposed as 'aiter')")


class CK(_Stub):
    name = "ck"
    _reason = "no standalone CK fp8 row-scale gemm adapter on this node (use 'aiter' for compiled CK)"


class Triton(_Stub):
    name = "triton"
    _reason = ("no standalone non-aiter Triton fp8 row-scale GEMM on this node; the only Triton "
               "a8w8 per-token-scale kernel is aiter's (see aiter_triton)")
