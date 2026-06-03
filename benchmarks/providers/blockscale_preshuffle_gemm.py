"""Block-Scale Preshuffle GEMM provider adapters (FP8 A8W8, per-block scales).

op_type is "blockscale_preshuffle_gemm". The FlyDSL kernel file is
kernels/blockscale_preshuffle_gemm.py; its builder is
compile_blockscale_preshuffle_gemm(*, M,N,K, tile_m,tile_n,tile_k,
scale_block_k=128, out_dtype, use_async_copy). Recipe + exact call convention
recovered from tests/kernels/test_blockscale_preshuffle_gemm.py.

Layout contract (verified against the kernel + test):
  * x       : [M, K] fp8 (row-major), A operand
  * weight  : [N, K] fp8 (row-major, one row per output column) -> PRESHUFFLED
              with layout=(16,16) before the kernel sees it (b_shuffled)
  * x_scale : [M, scale_k] fp32, per-(token, K-block) scale; scale_k = K//128
  * w_scale : [scale_n, scale_k] fp32, per-(N-block, K-block) scale;
              scale_n = ceil(N/128), scale_k = ceil(K/128)
  * out C   : [M, N] bf16/fp16
  BLOCK_SHAPE = (block_n=128, block_k=128); ScaleBlockM=1.

Because the kernel needs PRESHUFFLED weights + a TRANSPOSED-flattened x_scale
(and aiter needs its own shuffle + a 2-D transposed x_scale), the providers do
their own layout conversion inside run() from the SHARED canonical inputs that
the Op builds once -> includes_layout_conversion=True on both compiled paths.
The Op stores the unshuffled fp8 weight + raw fp32 scales so every provider
gets identical bits; each provider preshuffles/relayouts itself.

Reachable on this node:
  * flydsl       -> kernels.blockscale_preshuffle_gemm.compile_blockscale_preshuffle_gemm
  * aiter        -> aiter.gemm_a8w8_blockscale_bpreshuffle (compiled CK/ASM a8w8 blockscale)
  * pytorch      -> run_torch_blockscale dequant + F.linear (also the fp32 reference)
  * aiter_triton -> aiter.ops.triton.gemm_a8w8_blockscale.gemm_a8w8_blockscale (if present)
  * ck/aiter_ck/aiter_asm/gluon/hipblaslt/triton -> honest stubs (no separately
    selectable / standalone blockscale-bpreshuffle path from Python here).

FlyDSL is fp8-in / bf16|fp16-out only (no other dtype path).
"""

from __future__ import annotations

from benchmarks import common
from benchmarks.providers.base import ProviderAdapter

# Output dtypes the kernel + aiter support; the *input* operands are always fp8.
_OK_OUT = {"bf16", "bfloat16", "fp16", "f16"}
# ledger dtype strings that mean "fp8 in, bf16/fp16 out" for this op
_OK_DTYPE = {"fp8", "fp8_e4m3", "bf16", "bfloat16", "fp16", "f16"}

BLOCK_N = 128
BLOCK_K = 128

# Candidate tiles, copied verbatim from the test's select_tile_config().
_TILE_CANDIDATES = [
    (16, 64, 256),
    (16, 128, 256),
    (32, 64, 128),
    (32, 64, 256),
    (32, 128, 128),
    (32, 128, 256),
    (64, 64, 128),
    (64, 64, 256),
    (64, 128, 128),
    (64, 128, 256),
    (64, 256, 128),
]


def _MNK(shape):
    a = shape["args"]
    return int(a["M"]), int(a["N"]), int(a["K"])


def _out_dtype_str(shape) -> str:
    """ledger dtype/args -> 'bf16' | 'fp16' for the kernel/aiter out_dtype."""
    od = str(shape.get("args", {}).get("out_dtype", "bf16"))
    if od in ("fp16", "f16", "float16"):
        return "fp16"
    return "bf16"


def select_tile_config(M: int, N: int, K: int, scale_block_k: int = BLOCK_K):
    """Auto-select (tile_m, tile_n, tile_k) -- a pure copy of the test heuristic
    so the harness compiles the same tile the test would for a given shape."""

    def _valid(tm, tn, tk):
        return N % tn == 0 and K % tk == 0 and tk % scale_block_k == 0 and tm * tk // 256 >= 16

    valid = [(tm, tn, tk) for tm, tn, tk in _TILE_CANDIDATES if _valid(tm, tn, tk)]
    if not valid:
        return None

    def _score(tm, tn, tk):
        s = 0
        total_blocks = ((M + tm - 1) // tm) * (N // tn)
        s += 15 if total_blocks >= 256 else (10 if total_blocks >= 128 else (5 if total_blocks >= 64 else 0))
        if M <= 48:
            s += 12 if tm == 16 else (8 if tm == 32 else 0)
        elif M <= 128:
            s += 10 if tm == 32 else (6 if tm == 16 else (4 if tm == 64 else 0))
        elif M <= 512:
            s += 12 if tm == 64 else (8 if tm == 32 else 0)
        else:
            s += 12 if tm == 64 else 0
        if M <= 128:
            s += 6 if tn == 64 else (4 if tn == 128 else (2 if tn == 256 else 0))
        else:
            s += 8 if tn == 128 else (4 if tn == 64 else (4 if tn == 256 else 0))
        s += 6 if tk == 128 else 3
        return s

    return max(valid, key=lambda t: _score(*t))


class FlyDSL(ProviderAdapter):
    """compile_blockscale_preshuffle_gemm -> flyc.compile launcher, cached by shape.

    Launcher contract (test line 224-228):
        exe = compile_blockscale_preshuffle_gemm(M,N,K,tile_m,tile_n,tile_k,
              scale_block_k=128, out_dtype=..., use_async_copy=False)
        compiled = flyc.compile(exe, c_out, x, b_shuffled, x_scale_t, w_scale_flat,
                                M, N, stream)
        compiled(c_out, x, b_shuffled, x_scale_t, w_scale_flat, M, N, stream)
    where b_shuffled = shuffle_weight(weight, layout=(16,16)),
          x_scale_t  = x_scale.transpose(0,1).contiguous().view(-1)  (scale_k*M),
          w_scale_flat = w_scale.contiguous().view(-1).
    Preshuffle + scale relayout + output alloc are built ONCE per shape key
    (outside the timed region); run() only relaunches -> includes_allocation=False,
    but the one-time preshuffle is a layout conversion -> includes_layout_conversion=True.
    """

    name = "flydsl"
    includes_allocation = False
    includes_jit = False
    includes_layout_conversion = True

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "blockscale_preshuffle_gemm":
            return False, "flydsl adapter only implements blockscale_preshuffle_gemm"
        if shape["dtype"] not in _OK_DTYPE:
            return False, (f"blockscale_preshuffle_gemm is fp8-in / bf16|fp16-out only; "
                           f"{shape['dtype']} unsupported")
        M, N, K = _MNK(shape)
        if K % BLOCK_K != 0:
            return False, f"K={K} % scale_block_k={BLOCK_K} != 0"
        if select_tile_config(M, N, K) is None:
            return False, (f"no valid tile config for M={M},N={N},K={K} "
                           f"(needs N%tile_n==0, K%tile_k==0, tile_k%{BLOCK_K}==0, "
                           f"tile_m*tile_k//256>=16)")
        ok, why = common.flydsl_runtime_ok()
        return (ok, why) if not ok else (True, None)

    def _build(self, shape, inputs):
        import torch

        M, N, K = inputs["M"], inputs["N"], inputs["K"]
        out_s = _out_dtype_str(shape)
        key = (M, N, K, out_s)
        if key in self._cache:
            return self._cache[key]

        common.bootstrap_env()
        import flydsl.compiler as flyc
        from kernels.blockscale_preshuffle_gemm import compile_blockscale_preshuffle_gemm
        from tests.utils import shuffle_weight

        tile = select_tile_config(M, N, K)  # checked non-None in supports()
        tile_m, tile_n, tile_k = tile
        out_torch = torch.bfloat16 if out_s == "bf16" else torch.float16

        exe = compile_blockscale_preshuffle_gemm(
            M=M, N=N, K=K,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            scale_block_k=BLOCK_K, out_dtype=out_s, use_async_copy=False)

        # --- one-time layout conversion (preshuffle weight, relayout scales) ---
        x = inputs["x_fp8"]                                  # [M, K] fp8
        weight = inputs["weight_fp8"]                        # [N, K] fp8
        b_shuffled = shuffle_weight(weight, layout=(16, 16)).contiguous()
        x_scale_t = inputs["x_scale"].transpose(0, 1).contiguous().view(-1)   # [scale_k*M]
        w_scale_flat = inputs["w_scale"].contiguous().view(-1)                # [scale_n*scale_k]
        c_out = torch.zeros((M, N), device="cuda", dtype=out_torch)

        stream = torch.cuda.current_stream()
        compiled = flyc.compile(exe, c_out, x, b_shuffled, x_scale_t, w_scale_flat,
                                M, N, stream)

        ctx = {
            "compiled": compiled, "c_out": c_out,
            "x": x, "b_shuffled": b_shuffled,
            "x_scale_t": x_scale_t, "w_scale_flat": w_scale_flat,
            "M": M, "N": N,
        }
        self._cache[key] = ctx
        self.provider_detail = (
            f"compile_blockscale_preshuffle_gemm(M={M},N={N},K={K},"
            f"tile=({tile_m}x{tile_n}x{tile_k}),scale_block_k={BLOCK_K},out={out_s}); "
            f"FP8 A8W8 per-block scales (block_n={BLOCK_N},block_k={BLOCK_K}); "
            f"weight preshuffle(16,16) + x_scale transpose-flatten built once")
        return ctx

    def run(self, shape, inputs):
        import torch
        ctx = self._build(shape, inputs)
        ctx["compiled"](ctx["c_out"], ctx["x"], ctx["b_shuffled"],
                        ctx["x_scale_t"], ctx["w_scale_flat"],
                        ctx["M"], ctx["N"], torch.cuda.current_stream())
        return ctx["c_out"]


class Aiter(ProviderAdapter):
    """aiter.gemm_a8w8_blockscale_bpreshuffle (compiled CK/ASM a8w8 blockscale).

    Call convention (test line 257-268):
        b_aiter = aiter_shuffle_weight(weight, layout=(16,16))
        x_scale_t_2d = x_scale.transpose(0,1).contiguous().view(*x_scale.shape)  # [M, scale_k]
        aiter.gemm_a8w8_blockscale_bpreshuffle(x, b_aiter, x_scale_t_2d, w_scale, out_torch_dtype)
    The aiter shuffle + scale layout differ from FlyDSL's, so this path does its
    own one-time conversion -> includes_layout_conversion=True. Backend (CK vs
    ASM vs cktile) is auto-selected inside aiter and not Python-selectable.
    """

    name = "aiter"
    includes_allocation = True
    includes_layout_conversion = True

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "blockscale_preshuffle_gemm":
            return False, "aiter adapter only implements blockscale_preshuffle_gemm"
        if shape["dtype"] not in _OK_DTYPE:
            return False, f"aiter blockscale bpreshuffle is fp8-in / bf16|fp16-out; {shape['dtype']} unsupported"
        try:
            import aiter  # noqa: F401
            from aiter.ops.shuffle import shuffle_weight  # noqa: F401
            if not hasattr(aiter, "gemm_a8w8_blockscale_bpreshuffle"):
                return False, "aiter has no gemm_a8w8_blockscale_bpreshuffle attr"
        except Exception as e:
            return False, f"import aiter failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def _prep(self, shape, inputs):
        import torch

        M, N, K = inputs["M"], inputs["N"], inputs["K"]
        out_torch = torch.bfloat16 if _out_dtype_str(shape) == "bf16" else torch.float16
        key = (M, N, K, out_torch)
        if key in self._cache:
            return self._cache[key]
        from aiter.ops.shuffle import shuffle_weight as aiter_shuffle_weight
        b_aiter = aiter_shuffle_weight(inputs["weight_fp8"], layout=(16, 16))
        # aiter wants x_scale transposed but kept 2-D as [M, scale_k]
        x_scale_t_2d = (inputs["x_scale"].transpose(0, 1).contiguous()
                        .view(*inputs["x_scale"].shape))
        ctx = {"b_aiter": b_aiter, "x_scale_t_2d": x_scale_t_2d, "out_torch": out_torch}
        self._cache[key] = ctx
        return ctx

    def run(self, shape, inputs):
        import aiter
        ctx = self._prep(shape, inputs)
        self.provider_detail = ("aiter.gemm_a8w8_blockscale_bpreshuffle "
                                "(compiled CK/ASM/cktile a8w8 blockscale; backend auto-selected, "
                                "opaque from Python; aiter weight preshuffle(16,16) timed once)")
        return aiter.gemm_a8w8_blockscale_bpreshuffle(
            inputs["x_fp8"], ctx["b_aiter"], ctx["x_scale_t_2d"], inputs["w_scale"],
            ctx["out_torch"])


class PyTorch(ProviderAdapter):
    """Eager torch blockscale dequant + F.linear (run_torch_blockscale). Slow;
    shares the SAME math as the fp32 reference but casts to the output dtype."""

    name = "pytorch"
    provider_detail = ("run_torch_blockscale: dequant x/weight per-block (block=(128,128)) "
                       "then F.linear in fp32")
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "blockscale_preshuffle_gemm":
            return False, "pytorch adapter only implements blockscale_preshuffle_gemm"
        if shape["dtype"] not in _OK_DTYPE:
            return False, f"this adapter handles fp8-in blockscale only, not {shape['dtype']}"
        return True, None

    def run(self, shape, inputs):
        import torch
        from benchmarks import ops as _ops
        out_torch = torch.bfloat16 if _out_dtype_str(shape) == "bf16" else torch.float16
        return _ops.BlockScalePreshuffleGemmOp._torch_blockscale(
            inputs["x_fp8"], inputs["weight_fp8"], inputs["x_scale"], inputs["w_scale"],
            dtype=out_torch)


class _Stub(ProviderAdapter):
    _reason = "not available"

    def supports(self, shape):
        return False, self._reason


class AiterTriton(ProviderAdapter):
    """aiter.ops.triton.gemm_a8w8_blockscale (pure Triton blockscale), if present.

    The Triton path does NOT take a preshuffled weight (it consumes the plain
    [N,K] fp8 weight + 2-D scales), so it relayouts nothing beyond the scale
    transpose convention it expects -> includes_layout_conversion=True kept for
    honesty about the scale relayout. Falls back to an honest skip if the module
    is not present in this aiter build.
    """

    name = "aiter_triton"
    includes_allocation = True
    includes_layout_conversion = True

    def supports(self, shape):
        if shape.get("op_type") != "blockscale_preshuffle_gemm":
            return False, "aiter_triton adapter only implements blockscale_preshuffle_gemm"
        if shape["dtype"] not in _OK_DTYPE:
            return False, f"aiter triton blockscale is fp8-in / bf16|fp16-out; {shape['dtype']} unsupported"
        try:
            from aiter.ops.triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale  # noqa: F401
        except Exception as e:
            return False, (f"aiter.ops.triton.gemm_a8w8_blockscale not importable "
                           f"({type(e).__name__}); not in this aiter build / launch via env.sh")
        return True, None

    def run(self, shape, inputs):
        import torch
        from aiter.ops.triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale
        out_torch = torch.bfloat16 if _out_dtype_str(shape) == "bf16" else torch.float16
        # Triton consumes the plain (un-preshuffled) weight + 2-D scales.
        self.provider_detail = "aiter.ops.triton.gemm_a8w8_blockscale (pure Triton a8w8 blockscale)"
        with torch.no_grad():
            return gemm_a8w8_blockscale(
                inputs["x_fp8"], inputs["weight_fp8"],
                inputs["x_scale"], inputs["w_scale"], dtype=out_torch)


class AiterCK(_Stub):
    name = "aiter_ck"
    _reason = ("CK blockscale-bpreshuffle backend is auto-selected inside "
               "aiter.gemm_a8w8_blockscale_bpreshuffle; not separately selectable -> use 'aiter'")


class AiterASM(_Stub):
    name = "aiter_asm"
    _reason = ("ASM blockscale-bpreshuffle backend is auto-selected inside "
               "aiter.gemm_a8w8_blockscale_bpreshuffle; not separately selectable -> use 'aiter'")


class CK(_Stub):
    name = "ck"
    _reason = "no standalone CK blockscale-bpreshuffle adapter (reached via 'aiter')"


class Triton(_Stub):
    name = "triton"
    _reason = ("no standalone (non-aiter) Triton blockscale GEMM on this node; the only "
               "Triton path is aiter.ops.triton.gemm_a8w8_blockscale (see aiter_triton)")


class Gluon(_Stub):
    name = "gluon"
    _reason = "no Gluon blockscale GEMM kernel on this node"


class HipBLASLt(_Stub):
    name = "hipblaslt"
    _reason = "hipBLASLt has no per-block-scaled fp8 preshuffle-GEMM op"
