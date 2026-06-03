"""Flash-attention provider adapters (non-causal/causal MHA, BSHD layout).

op_type is "flash_attn"; the FlyDSL kernel file is kernels/flash_attn_func.py.
Q/K/V/O are [batch, seq_len, num_heads, head_dim] f16/bf16, out same layout.
sm_scale = 1/sqrt(head_dim). Every provider receives the SAME q,k,v built ONCE
in fp32-then-cast (ops.FlashAttnOp); the fp32 golden is torch SDPA on the
transposed [B,H,seq,hd] view (transpose back to BSHD), matching the FlyDSL
test's pytorch_ref_attention exactly.

Reachable on this node (recipe + source recon):
  * flydsl        -> kernels.flash_attn_func.build_flash_attn_func_module; the
                     compiled launcher takes 1-D FLATTENED q,k,v,o + (B, S). We
                     build+cache the launcher and the flat o buffer by shape key;
                     run() flattens views (zero-copy on contiguous inputs) and
                     launches. output() reshapes o back to BSHD.
  * pytorch       -> F.scaled_dot_product_attention on the [B,H,S,D] transpose
                     (also the fp32 reference source).
  * aiter         -> aiter.mha_fwd (compiled CK fwd; f16/bf16), exact call from
                     tests/kernels/test_flash_attn_func.py:run_aiter_bench("ck").
  * aiter_asm     -> aiter.fmha_v3_fwd (compiled ASM v3 fwd; bf16 only -- the
                     test skips it for f16), how_v3_bf16_cvt=2.
  * aiter_triton  -> aiter.ops.triton.attention.mha.flash_attn_func (pure Triton,
                     BSHD layout, causal= kwarg) -- no layout conversion needed.
  * triton        -> honest stub (the only Triton FA on this node is aiter's,
                     exposed as aiter_triton).
  * aiter_ck      -> honest stub (the CK fwd is exposed as 'aiter'; not a second
                     separately-selectable entrypoint).
  * ck/gluon/hipblaslt -> honest stubs.

Shape constraints from the kernel (flash_attn_func.py:20 + asserts): seq_len %
128 == 0, head_dim % 32 == 0, head_dim >= 64, f16/bf16 only. supports() rejects
out-of-contract shapes so they are skipped, not crashed.
"""

from __future__ import annotations

import math

from benchmarks import common
from benchmarks.providers.base import ProviderAdapter

_OK16 = {"bf16", "bfloat16", "fp16", "f16"}


def _BSHD(shape):
    a = shape["args"]
    return (int(a["batch"]), int(a["seq_len"]), int(a["num_heads"]), int(a["head_dim"]))


def _causal(shape) -> bool:
    return bool(shape.get("args", {}).get("causal", True))


def _shape_ok(B, S, H, D) -> tuple[bool, str | None]:
    if S % 128 != 0:
        return False, f"flash_attn_func requires seq_len % 128 == 0 (got {S})"
    if D % 32 != 0:
        return False, f"flash_attn_func requires head_dim % 32 == 0 (got {D})"
    if D < 64:
        return False, f"flash_attn_func requires head_dim >= 64 (got {D})"
    return True, None


class FlyDSL(ProviderAdapter):
    name = "flydsl"
    includes_allocation = False
    includes_jit = False
    includes_layout_conversion = False

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "flash_attn":
            return False, "flydsl flash_attn adapter only implements flash_attn"
        if shape["dtype"] not in _OK16:
            return False, (f"flash_attn_func is f16/bf16 only "
                           f"(asserts dtype_str in (f16,bf16) for {shape['dtype']})")
        B, S, H, D = _BSHD(shape)
        ok, why = _shape_ok(B, S, H, D)
        if not ok:
            return False, why
        ok, why = common.flydsl_runtime_ok()
        return (ok, why) if not ok else (True, None)

    def _launcher(self, shape, inputs):
        import torch

        B, S, H, D = _BSHD(shape)
        causal = _causal(shape)
        ds = common.FLYDSL_DTYPE[shape["dtype"]]
        sm_scale = 1.0 / math.sqrt(D)
        key = (B, S, H, D, causal, ds)
        if key not in self._cache:
            common.bootstrap_env()
            from kernels.flash_attn_func import build_flash_attn_func_module
            exe = build_flash_attn_func_module(
                num_heads=H,
                head_dim=D,
                causal=causal,
                dtype_str=ds,
                sm_scale=sm_scale,
                waves_per_eu=2,
                daz=True,
            )
            # flat output buffer (BSHD flattened), preallocated outside the timed region
            o_flat = torch.zeros((B * S * H * D,), device="cuda", dtype=inputs["dtype"])
            self._cache[key] = (exe, o_flat)
            self.provider_detail = (
                f"build_flash_attn_func_module(num_heads={H},head_dim={D},"
                f"causal={causal},{ds},sm_scale=1/sqrt({D}),waves_per_eu=2,daz=True); "
                "launcher takes 1-D flattened q,k,v,o + (B,S)"
            )
        return self._cache[key]

    def run(self, shape, inputs):
        import torch
        exe, o_flat = self._launcher(shape, inputs)
        q, k, v = inputs["q"], inputs["k"], inputs["v"]
        B, S, H, D = _BSHD(shape)
        # zero-copy flatten of the shared contiguous BSHD inputs
        q_flat = q.reshape(-1)
        k_flat = k.reshape(-1)
        v_flat = v.reshape(-1)
        # MUST pass the current stream so CUDA-graph capture lands the kernel on
        # the capture stream; the launcher defaults to stream 0 (fx.Stream(None)),
        # which graph capture on a side stream does NOT see -> empty graph ->
        # bogus ~4us replay (huge fake speedup). See common.benchmark_cudagraph.
        exe(q_flat, k_flat, v_flat, o_flat, B, S, stream=torch.cuda.current_stream())
        return o_flat

    def output(self, shape, inputs):
        B, S, H, D = _BSHD(shape)
        return self.run(shape, inputs).reshape(B, S, H, D)


class PyTorch(ProviderAdapter):
    name = "pytorch"
    provider_detail = ("F.scaled_dot_product_attention on [B,H,S,D] transpose "
                       "(transpose back to BSHD); also the fp32 reference")
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "flash_attn":
            return False, "pytorch flash_attn adapter only implements flash_attn"
        if shape["dtype"] not in _OK16:
            return False, f"this flash_attn adapter compares f16/bf16; {shape['dtype']} not supported"
        return True, None

    def run(self, shape, inputs):
        import torch
        import torch.nn.functional as F

        D = _BSHD(shape)[3]
        sm_scale = 1.0 / math.sqrt(D)
        q = inputs["q"].transpose(1, 2)
        k = inputs["k"].transpose(1, 2)
        v = inputs["v"].transpose(1, 2)
        with torch.no_grad():
            out = F.scaled_dot_product_attention(q, k, v, is_causal=_causal(shape), scale=sm_scale)
        return out.transpose(1, 2).contiguous()


class Aiter(ProviderAdapter):
    name = "aiter"
    provider_detail = "aiter.mha_fwd (compiled CK forward; f16/bf16; BSHD)"
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "flash_attn":
            return False, "aiter flash_attn adapter only implements flash_attn"
        if shape["dtype"] not in _OK16:
            return False, f"aiter mha_fwd (CK) validated for f16/bf16, not {shape['dtype']}"
        try:
            import aiter  # noqa: F401
            from aiter import mha_fwd  # noqa: F401
        except Exception as e:
            return False, f"import failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def run(self, shape, inputs):
        import torch
        from aiter import mha_fwd

        D = _BSHD(shape)[3]
        sm_scale = 1.0 / math.sqrt(D)
        with torch.no_grad():
            out = mha_fwd(
                inputs["q"], inputs["k"], inputs["v"],
                0.0,            # dropout_p
                sm_scale,       # softmax_scale
                _causal(shape), # is_causal
                -1, -1,         # window_size_left, window_size_right
                0,              # sink_size
                True,           # return_softmax_lse
                False,          # return_dropout_randval
                None, None, None, None, None, None, None, None, None, None, None,
            )
        # mha_fwd returns (out, softmax_lse, s_dmask, ...); out is BSHD
        return out[0] if isinstance(out, (tuple, list)) else out


class AiterASM(ProviderAdapter):
    name = "aiter_asm"
    provider_detail = "aiter.fmha_v3_fwd (compiled ASM v3 forward; bf16 only; BSHD)"
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "flash_attn":
            return False, "aiter_asm flash_attn adapter only implements flash_attn"
        if shape["dtype"] not in ("bf16", "bfloat16"):
            return False, (f"aiter fmha_v3_fwd (ASM v3) is bf16 only "
                           f"(test skips f16); {shape['dtype']} not supported")
        try:
            import aiter  # noqa: F401
            from aiter import fmha_v3_fwd  # noqa: F401
        except Exception as e:
            return False, f"import failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def run(self, shape, inputs):
        import torch
        from aiter import fmha_v3_fwd

        D = _BSHD(shape)[3]
        sm_scale = 1.0 / math.sqrt(D)
        with torch.no_grad():
            out = fmha_v3_fwd(
                inputs["q"], inputs["k"], inputs["v"],
                0.0,            # dropout_p
                sm_scale,       # softmax_scale
                _causal(shape), # is_causal
                -1, -1,         # window_size_left, window_size_right
                True,           # return_softmax_lse
                False,          # return_dropout_randval
                2,              # how_v3_bf16_cvt
                None, None, None, None, None, None, None,
            )
        return out[0] if isinstance(out, (tuple, list)) else out


class AiterTriton(ProviderAdapter):
    name = "aiter_triton"
    provider_detail = ("aiter.ops.triton.attention.mha.flash_attn_func "
                       "(pure Triton; BSHD; causal= kwarg)")
    includes_allocation = True
    includes_layout_conversion = False

    def supports(self, shape):
        if shape.get("op_type") != "flash_attn":
            return False, "aiter_triton flash_attn adapter only implements flash_attn"
        if shape["dtype"] not in _OK16:
            return False, f"aiter triton flash_attn_func validated for f16/bf16, not {shape['dtype']}"
        try:
            from aiter.ops.triton.attention.mha import flash_attn_func  # noqa: F401
        except Exception as e:
            return False, f"import failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def run(self, shape, inputs):
        import torch
        from aiter.ops.triton.attention.mha import flash_attn_func

        D = _BSHD(shape)[3]
        sm_scale = 1.0 / math.sqrt(D)
        with torch.no_grad():
            out = flash_attn_func(
                inputs["q"], inputs["k"], inputs["v"],
                dropout_p=0.0,
                softmax_scale=sm_scale,
                causal=_causal(shape),
            )
        # returns o, or a tuple when return_lse/return_attn_probs set (default False -> tensor)
        return out[0] if isinstance(out, (tuple, list)) else out


class _Stub(ProviderAdapter):
    _reason = "not available"

    def supports(self, shape):
        return False, self._reason


class Triton(_Stub):
    name = "triton"
    _reason = ("no standalone non-aiter Triton flash-attention on this node; the "
               "only Triton FA is aiter.ops.triton.attention.mha.flash_attn_func "
               "(see the aiter_triton provider)")


class AiterCK(_Stub):
    name = "aiter_ck"
    _reason = "CK fwd is exposed as 'aiter' (aiter.mha_fwd); not a second separately-selectable entrypoint"


class CK(_Stub):
    name = "ck"
    _reason = "no standalone CK flash-attention adapter on this node (use 'aiter' for the compiled CK path)"


class Gluon(_Stub):
    name = "gluon"
    _reason = "no Gluon flash-attention kernel on this node"


class HipBLASLt(_Stub):
    name = "hipblaslt"
    _reason = "hipBLASLt is a GEMM library (no fused attention op)"
