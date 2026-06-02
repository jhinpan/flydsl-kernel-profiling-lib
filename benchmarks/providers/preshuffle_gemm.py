"""Preshuffle-GEMM provider adapters (C = (A*scale_a) @ (B*scale_b)^T).

op_type is "preshuffle_gemm" (the harness ledger op_type); the FlyDSL kernel
file is kernels/preshuffle_gemm.py (compile_preshuffle_gemm_a8). Layout contract
verified against kernels/preshuffle_gemm.py + tests/kernels/test_preshuffle_gemm.py:

  * A is (M,K) row-major, per-token-quantized to fp8 (e4m3), scale_a is [M] fp32.
  * B is (N,K) row-major, per-token-quantized to fp8, scale_b is [N] fp32, and
    then PRESHUFFLED to the kernel's MFMA layout via tests.utils.shuffle_weight(
    b_q, layout=(16,16)). The preshuffle is a weight-layout conversion that a
    real serving stack does ONCE at load time -> we do it in make_inputs / the
    build cache, OUTSIDE the timed run(). The FlyDSL adapter sets
    includes_layout_conversion=False because run() launches only the GEMM.
  * C is (M,N), out_dtype bf16 (matches the example test's default).
  * fp8 here is e4m3 (gfx950 OCP float8_e4m3fn; the harness common.torch_dtype
    maps "fp8" to the MI300 _fnuz spelling, so we pick the dtype explicitly).

The fp32 golden DEQUANTIZES the same quantized bits the kernel reads:
  C = (a_q.float()*scale_a) @ (b_q.float()*scale_b)^T  (== test run_torch).
make_inputs builds fp32, quantizes ONCE, and stashes both the fp8 tensors and
the fp32 dequant-of-quant originals so every provider compares against identical
bits (MoeGemmOp quant-input pattern).

Providers:
  * flydsl       -> kernels.preshuffle_gemm.compile_preshuffle_gemm_a8 (MFMA fp8,
                    B preshuffled, per-token scales). bf16/fp16 also compile but
                    this adapter wires the fp8 a8w8 path the recipe + example
                    profile (scale_a/scale_b present).
  * aiter        -> aiter.gemm_a8w8_bpreshuffle (compiled CK/cktile a8w8 with B
                    preshuffle; fp8 only -- asserts WQ.dtype==fp8).
  * pytorch      -> dequant a_q@b_q^T (also the fp32 reference source).
  * triton/aiter_triton/aiter_ck/aiter_asm/ck/gluon/hipblaslt -> honest stubs.

FlyDSL preshuffle_gemm_a8 is fp8/int8/int4/fp16/bf16; this adapter wires the fp8
scaled path (the recipe). Non-fp8 ledger rows are rejected in supports().
"""

from __future__ import annotations

from benchmarks import common
from benchmarks.providers.base import ProviderAdapter

# Only the fp8 a8w8 scaled path is wired here (matches the recipe + example).
_OK_FP8 = {"fp8", "fp8_e4m3"}

# Output dtype for the GEMM. The example test defaults to bf16.
_OUT_DTYPE_STR = "bf16"

# Known-good tile configs, ordered: the validated gfx950 small-M config from the
# example test first, then larger tiles for big shapes. The adapter picks the
# FIRST whose pure-shape divisibility preconditions hold; the rest of the
# kernel's asserts (LDS capacity) surface at compile time inside run().
# Tuple fields match compile_preshuffle_gemm_a8(tile_m, tile_n, tile_k).
_CONFIGS = [
    # --- validated gfx950 configs from tests/kernels/test_preshuffle_gemm.py ---
    dict(tile_m=16, tile_n=64, tile_k=512),
    dict(tile_m=32, tile_n=64, tile_k=512),
    dict(tile_m=64, tile_n=256, tile_k=128),
    dict(tile_m=128, tile_n=128, tile_k=128),
    # --- conservative general tiles ---
    dict(tile_m=16, tile_n=64, tile_k=256),
    dict(tile_m=32, tile_n=128, tile_k=256),
    dict(tile_m=64, tile_n=128, tile_k=256),
    dict(tile_m=128, tile_n=256, tile_k=128),
]

# fp8/int8 -> 1 byte per element.
_ELEM_BYTES = 1
_TOTAL_THREADS = 256
_A_LOAD_BYTES = 16


def _config_ok(M, N, K, cfg) -> bool:
    """Pure-Python check of the SHAPE/TILE preconditions in
    compile_preshuffle_gemm_a8 (fp8, a_elem_vec_pack=1, no GPU). A True here
    means the obvious compile-time asserts pass; the LDS-capacity asserts still
    run on-device."""
    tm, tn, tk = cfg["tile_m"], cfg["tile_n"], cfg["tile_k"]
    # tile_k_bytes % 64 == 0 (compile_preshuffle_gemm_a8 raises otherwise)
    tile_k_bytes = tk * _ELEM_BYTES
    if tile_k_bytes % 64 != 0:
        return False
    # bytes_a_per_tile = tile_m*tile_k*elem_bytes (pack=1) must be %256==0
    bytes_a_per_tile = tm * tk * _ELEM_BYTES
    if bytes_a_per_tile % _TOTAL_THREADS != 0:
        return False
    # bytes_per_thread_a must be a multiple of a_load_bytes (16)
    if (bytes_a_per_tile // _TOTAL_THREADS) % _A_LOAD_BYTES != 0:
        return False
    # output / tiling divisibility: N tiles cleanly into tile_n, K into tile_k.
    if N % tn != 0 or N < tn:
        return False
    if K % tk != 0 or K < tk:
        return False
    # tile_n split across 4 waves -> n_per_wave % 16 == 0 (MFMA 16x16 atom).
    n_per_wave = tn // 4
    if tn % 4 != 0 or n_per_wave % 16 != 0:
        return False
    # tile_m tiled by the 16x16 MFMA atom.
    if tm % 16 != 0:
        return False
    return True


def _pick_config(M, N, K):
    for cfg in _CONFIGS:
        if _config_ok(M, N, K, cfg):
            return cfg
    return None


def _MNK(shape):
    a = shape["args"]
    return int(a["M"]), int(a["N"]), int(a["K"])


class FlyDSL(ProviderAdapter):
    name = "flydsl"
    includes_allocation = False
    includes_jit = False
    # the B preshuffle is a load-time weight conversion done ONCE in the build
    # cache (outside the timed region); run() launches only the GEMM.
    includes_layout_conversion = False

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "preshuffle_gemm":
            return False, "flydsl preshuffle_gemm adapter only implements preshuffle_gemm"
        if shape["dtype"] not in _OK_FP8:
            return False, (f"FlyDSL preshuffle_gemm adapter wires the fp8 a8w8 scaled path; "
                           f"{shape['dtype']} (int8/int4/fp16/bf16/fp4) not wired here")
        M, N, K = _MNK(shape)
        if _pick_config(M, N, K) is None:
            return False, (f"no valid preshuffle_gemm tile config for M={M},N={N},K={K} "
                           "(shape fails kernel asserts: tile_k_bytes%64, "
                           "tile_m*tile_k%256, N%tile_n, K%tile_k)")
        ok, why = common.flydsl_runtime_ok()
        return (ok, why) if not ok else (True, None)

    def _build(self, shape, inputs):
        import torch

        M, N, K = inputs["M"], inputs["N"], inputs["K"]
        key = (M, N, K, shape["dtype"])
        if key in self._cache:
            return self._cache[key]

        common.bootstrap_env()
        import flydsl.compiler as flyc
        from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8
        from tests.utils import shuffle_weight

        cfg = dict(_pick_config(M, N, K))  # checked non-None in supports()
        torch_out_dtype = torch.bfloat16 if _OUT_DTYPE_STR == "bf16" else torch.float16

        # fp8 quantized inputs (shared bits from make_inputs).
        a_q = inputs["a_q"]            # (M,K) fp8
        b_q = inputs["b_q"]            # (N,K) fp8
        scale_a = inputs["scale_a"]    # [M] fp32
        scale_b = inputs["scale_b"]    # [N] fp32

        # --- preshuffle B to the MFMA layout (ONCE, outside the timed run) ---
        b_shuffled = shuffle_weight(b_q.contiguous(), layout=(16, 16))

        def _as_i8(t):
            return t.view(torch.int8) if "float8" in str(t.dtype) else t

        a_flat = _as_i8(a_q.contiguous().view(-1))
        b_flat = _as_i8(b_shuffled.contiguous().view(-1))
        sa_flat = scale_a.contiguous().view(-1)
        sb_flat = scale_b.contiguous().view(-1)
        c_out = torch.zeros((M, N), device="cuda", dtype=torch_out_dtype)
        dummy_bias = torch.empty(0, dtype=torch_out_dtype, device="cuda")

        launch_fn = compile_preshuffle_gemm_a8(
            M=M, N=N, K=K,
            tile_m=cfg["tile_m"], tile_n=cfg["tile_n"], tile_k=cfg["tile_k"],
            in_dtype="fp8", out_dtype=_OUT_DTYPE_STR,
            lds_stage=2, use_cshuffle_epilog=False, epilogue="none",
        )

        # kernel arg order (from test _gemm_args):
        #   (c, a, b, scale_a, scale_b, bias, M, N, stream)
        def _gemm_args(c, a, b, sa, sb):
            return (
                c.contiguous().view(-1),
                a,
                b,
                sa,
                sb,
                dummy_bias,
                M,
                N,
                torch.cuda.current_stream(),
            )

        compiled_fn = flyc.compile(launch_fn, *_gemm_args(c_out, a_flat, b_flat, sa_flat, sb_flat))

        ctx = {
            "compiled_fn": compiled_fn, "c_out": c_out,
            "a_flat": a_flat, "b_flat": b_flat, "sa_flat": sa_flat, "sb_flat": sb_flat,
            "gemm_args": _gemm_args,
        }
        self._cache[key] = ctx
        self.provider_detail = (
            f"compile_preshuffle_gemm_a8(M={M},N={N},K={K},in=fp8,out={_OUT_DTYPE_STR}, "
            f"tile_m={cfg['tile_m']},tile_n={cfg['tile_n']},tile_k={cfg['tile_k']}, "
            f"lds_stage=2); MFMA a8w8, per-token scales; B preshuffle "
            f"(shuffle_weight layout=(16,16)) done once in build cache (untimed)")
        return ctx

    def run(self, shape, inputs):
        ctx = self._build(shape, inputs)
        ctx["compiled_fn"](*ctx["gemm_args"](
            ctx["c_out"], ctx["a_flat"], ctx["b_flat"], ctx["sa_flat"], ctx["sb_flat"]))
        return ctx["c_out"]


class Aiter(ProviderAdapter):
    """aiter.gemm_a8w8_bpreshuffle -- compiled CK/cktile a8w8 GEMM with B preshuffle.

    fp8 only (asserts WQ.dtype==fp8). Same per-token scales + same preshuffled B
    as the FlyDSL path. The preshuffle is built ONCE per shape (outside the timed
    region) -> includes_layout_conversion=False; allocates its own output (Y)."""

    name = "aiter"
    includes_allocation = True
    includes_layout_conversion = False

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "preshuffle_gemm":
            return False, "aiter preshuffle_gemm adapter only implements preshuffle_gemm"
        if shape["dtype"] not in _OK_FP8:
            return False, f"aiter gemm_a8w8_bpreshuffle is fp8 only, not {shape['dtype']}"
        try:
            from aiter import gemm_a8w8_bpreshuffle  # noqa: F401
        except Exception:
            try:
                from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle  # noqa: F401
            except Exception as e:
                return False, f"import failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def _build(self, shape, inputs):
        import torch

        M, N, K = inputs["M"], inputs["N"], inputs["K"]
        key = (M, N, K, shape["dtype"])
        if key not in self._cache:
            from tests.utils import shuffle_weight
            b_shuffled = shuffle_weight(inputs["b_q"].contiguous(), layout=(16, 16))
            self._cache[key] = b_shuffled
        return self._cache[key]

    def run(self, shape, inputs):
        import torch
        try:
            from aiter import gemm_a8w8_bpreshuffle
        except Exception:
            from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle

        b_shuffled = self._build(shape, inputs)
        torch_out_dtype = torch.bfloat16 if _OUT_DTYPE_STR == "bf16" else torch.float16
        self.provider_detail = ("aiter.gemm_a8w8_bpreshuffle (compiled CK/cktile a8w8 "
                                "B-preshuffle; backend chosen from tuned config; B preshuffle "
                                "built once outside timed region)")
        return gemm_a8w8_bpreshuffle(
            inputs["a_q"], b_shuffled, inputs["scale_a"], inputs["scale_b"],
            None, torch_out_dtype)


class PyTorch(ProviderAdapter):
    """Dequant reference: C = (a_q.float()*scale_a) @ (b_q.float()*scale_b)^T.

    Operates on the SAME quantized bits as the kernels, so it is the fp32 golden
    (matches the example test's run_torch). No preshuffle -- B is read in its
    natural (N,K) layout."""

    name = "pytorch"
    provider_detail = "dequant (a_q*scale_a) @ (b_q*scale_b)^T (also the fp32 reference)"
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "preshuffle_gemm":
            return False, "pytorch preshuffle_gemm adapter only implements preshuffle_gemm"
        if shape["dtype"] not in _OK_FP8:
            return False, f"this preshuffle_gemm adapter compares fp8, not {shape['dtype']}"
        return True, None

    def run(self, shape, inputs):
        import torch
        a_f32 = inputs["a_q"].to(torch.float32) * inputs["scale_a"].view(-1, 1)
        b_f32 = inputs["b_q"].to(torch.float32) * inputs["scale_b"].view(-1, 1)
        return torch.mm(a_f32, b_f32.T)


class _Stub(ProviderAdapter):
    _reason = "not available"

    def supports(self, shape):
        return False, self._reason


class AiterTriton(_Stub):
    name = "aiter_triton"
    _reason = ("no Triton a8w8 B-preshuffle GEMM exposed from aiter; the preshuffle "
               "layout is a CK/cktile (and FlyDSL) concept, not the Triton gemm path")


class Triton(_Stub):
    name = "triton"
    _reason = ("no standalone (non-aiter) Triton preshuffle GEMM on this node; "
               "preshuffle is a CK/MFMA weight-layout concept")


class AiterCK(_Stub):
    name = "aiter_ck"
    _reason = ("CK a8w8-bpreshuffle backend is not separately selectable from Python; "
               "it is chosen inside aiter.gemm_a8w8_bpreshuffle (see the 'aiter' provider)")


class AiterASM(_Stub):
    name = "aiter_asm"
    _reason = ("no separately-selectable ASM a8w8-bpreshuffle entrypoint; the ASM path "
               "is dispatched internally by aiter.gemm_a8w8_bpreshuffle when tuned")


class CK(_Stub):
    name = "ck"
    _reason = "no standalone CK preshuffle-GEMM adapter on this node (use 'aiter' for the compiled CK path)"


class Gluon(_Stub):
    name = "gluon"
    _reason = "no Gluon preshuffle-GEMM kernel on this node"


class HipBLASLt(_Stub):
    name = "hipblaslt"
    _reason = ("hipBLASLt has no preshuffled-B fp8 a8w8 path exposed from Python "
               "(dense fp8 GEMM only, non-preshuffled)")
