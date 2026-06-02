"""MoE-reduction provider adapters (sum over the topk dim).

op_type is "moe_reduce". The FlyDSL kernel reduces X [tokens, topk, model_dim]
along dim=1 to Y [tokens, model_dim]; it is the non-atomic counterpart used with
compile_moe_gemm2(accumulate=False) to avoid atomic contention in the MoE
stage-2 reduce. Pure bandwidth-bound elementwise reduction.

Layout contract (verified against kernels/moe_gemm_2stage.py:compile_moe_reduction
+ tests/kernels/test_moe_reduce.py): X is [tokens, topk, model_dim] row-major
contiguous, Y is [tokens, model_dim], valid_mask is [tokens, topk] uint8 when
use_mask=True (else an empty (0, topk) uint8 tensor the kernel ignores). Every
provider takes the SAME X (and mask) -- no layout conversion anywhere.

  * flydsl       -> kernels.moe_gemm_2stage.compile_moe_reduction (f16/bf16/f32)
  * pytorch      -> X.sum(dim=1) (masked); also the fp32 reference source
  * triton       -> self-contained inline Triton reduction (no sglang standalone)
  * aiter/aiter_triton/aiter_ck/aiter_asm/ck/gluon/hipblaslt -> honest stubs
    (no separately-callable MoE-reduce op exists; aiter folds this reduce inside
    its fused_moe stage-2, not exposed as a standalone Python entrypoint).

FlyDSL is f16/bf16/f32 only (ValueError otherwise) -> any other dtype is rejected
in supports(). The mask is read from shape["args"]["use_mask"].
"""

from __future__ import annotations

from benchmarks import common
from benchmarks.providers.base import ProviderAdapter

# FlyDSL compile_moe_reduction accepts dtype_str in {f16, bf16, f32}; torch.sum
# and the inline triton reduce handle the same float set.
_OKALL = {"bf16", "bfloat16", "fp16", "f16", "fp32", "f32", "float32", "float16"}


def _dims(shape):
    a = shape["args"]
    return int(a["tokens"]), int(a["topk"]), int(a["model_dim"])


def _use_mask(shape) -> bool:
    return bool(shape.get("args", {}).get("use_mask", False))


class FlyDSL(ProviderAdapter):
    name = "flydsl"
    includes_allocation = False
    includes_jit = False
    includes_layout_conversion = False

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "moe_reduce":
            return False, "flydsl moe_reduce adapter only implements moe_reduce"
        if shape["dtype"] not in common.FLYDSL_DTYPE:
            return False, (f"FlyDSL moe_reduction is f16/bf16/f32 only "
                           f"(ValueError for {shape['dtype']})")
        ok, why = common.flydsl_runtime_ok()
        return (ok, why) if not ok else (True, None)

    def _launcher(self, shape, inputs):
        import torch

        tokens, topk, model_dim = inputs["tokens"], inputs["topk"], inputs["model_dim"]
        ds = common.FLYDSL_DTYPE[shape["dtype"]]
        use_mask = inputs["use_mask"]
        # build/cache keyed by everything the launcher specializes on; the kernel
        # bakes topk/model_dim/dtype/use_mask, tokens is a runtime grid arg.
        key = (topk, model_dim, ds, use_mask, tokens)
        if key not in self._cache:
            common.bootstrap_env()
            from kernels.moe_gemm_2stage import compile_moe_reduction
            reduce_exe = compile_moe_reduction(
                topk=topk, model_dim=model_dim, dtype_str=ds, use_mask=use_mask)
            out = torch.empty((tokens, model_dim), device="cuda", dtype=inputs["dtype"])
            self._cache[key] = (reduce_exe, out)
            self.provider_detail = (
                f"compile_moe_reduction(topk={topk},model_dim={model_dim},{ds},"
                f"use_mask={use_mask}); sum over topk dim; "
                f"BLOCK_SIZE=256,VEC_WIDTH=8 (vec path for f16/bf16, scalar tail)")
        return self._cache[key]

    def run(self, shape, inputs):
        import torch
        reduce_exe, out = self._launcher(shape, inputs)
        # reduce_exe(X, Y, valid_mask, i32_m_tokens, stream) -- positional, exactly
        # as tests/kernels/test_moe_reduce.py launches it.
        reduce_exe(inputs["x"], out, inputs["mask"], inputs["tokens"],
                   torch.cuda.current_stream())
        return out


class PyTorch(ProviderAdapter):
    name = "pytorch"
    provider_detail = "X.sum(dim=1) (masked if use_mask); also the fp32 reference"
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "moe_reduce":
            return False, "pytorch moe_reduce adapter only implements moe_reduce"
        if shape["dtype"] not in _OKALL:
            return False, f"no torch sum path for {shape['dtype']}"
        return True, None

    def run(self, shape, inputs):
        import torch
        x = inputs["x"]
        if inputs["use_mask"]:
            x = x * inputs["mask"].to(torch.bool).unsqueeze(-1)
        return torch.sum(x, dim=1)


class Triton(ProviderAdapter):
    name = "triton"
    provider_detail = "standalone inline Triton topk-reduce (no sglang standalone exists)"
    includes_allocation = True
    _kernel = None

    def supports(self, shape):
        if shape.get("op_type") != "moe_reduce":
            return False, "triton moe_reduce adapter only implements moe_reduce"
        if shape["dtype"] not in _OKALL:
            return False, f"validated fp16/bf16/f32, not {shape['dtype']}"
        try:
            import triton  # noqa: F401
        except Exception as e:
            return False, f"import triton failed ({type(e).__name__})"
        return True, None

    @classmethod
    def _get_kernel(cls):
        if cls._kernel is None:
            import triton
            import triton.language as tl

            @triton.jit
            def _reduce(x_ptr, o_ptr, mask_ptr, n_tokens, topk, model_dim,
                        USE_MASK: tl.constexpr, BLOCK: tl.constexpr):
                # one program per (token, model_dim tile); accumulate over topk in f32.
                tok = tl.program_id(0)
                tile = tl.program_id(1)
                if tok < n_tokens:
                    offs = tile * BLOCK + tl.arange(0, BLOCK)
                    col_mask = offs < model_dim
                    acc = tl.zeros((BLOCK,), dtype=tl.float32)
                    for k in range(0, topk):
                        x = tl.load(x_ptr + (tok * topk + k) * model_dim + offs,
                                    mask=col_mask, other=0.0).to(tl.float32)
                        if USE_MASK:
                            mv = tl.load(mask_ptr + tok * topk + k).to(tl.int32)
                            x = tl.where(mv != 0, x, 0.0)
                        acc += x
                    tl.store(o_ptr + tok * model_dim + offs, acc.to(o_ptr.dtype.element_ty),
                             mask=col_mask)

            cls._kernel = _reduce
        return cls._kernel

    def run(self, shape, inputs):
        import torch
        import triton
        x = inputs["x"]
        tokens, topk, model_dim = inputs["tokens"], inputs["topk"], inputs["model_dim"]
        out = torch.empty((tokens, model_dim), device=x.device, dtype=x.dtype)
        BLOCK = 1024
        grid = (tokens, triton.cdiv(model_dim, BLOCK), 1)
        use_mask = inputs["use_mask"]
        mask = inputs["mask"] if use_mask else x  # dummy ptr when unused
        self._get_kernel()[grid](
            x, out, mask, tokens, topk, model_dim,
            USE_MASK=use_mask, BLOCK=BLOCK)
        return out


class _Stub(ProviderAdapter):
    _reason = "not available"

    def supports(self, shape):
        return False, self._reason


class Aiter(_Stub):
    name = "aiter"
    _reason = ("no standalone compiled aiter MoE-reduce op; aiter folds the topk "
               "reduce inside fused_moe stage-2 (not exposed as a Python entrypoint)")


class AiterTriton(_Stub):
    name = "aiter_triton"
    _reason = ("aiter.ops.triton MoE reduce is internal to e2e_moe; no standalone "
               "topk-reduce Triton op exposed (use the inline 'triton' provider)")


class AiterCK(_Stub):
    name = "aiter_ck"
    _reason = "no separately-selectable CK MoE-reduce from Python"


class AiterASM(_Stub):
    name = "aiter_asm"
    _reason = "no Python-selectable ASM MoE-reduce path in AITER"


class CK(_Stub):
    name = "ck"
    _reason = "no standalone CK MoE-reduce adapter on this node"


class Gluon(_Stub):
    name = "gluon"
    _reason = "no Gluon MoE-reduce kernel on this node"


class HipBLASLt(_Stub):
    name = "hipblaslt"
    _reason = "hipBLASLt is a GEMM library (no reduction op)"
