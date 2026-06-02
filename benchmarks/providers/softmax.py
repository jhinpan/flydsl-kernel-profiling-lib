"""Softmax provider adapters (row softmax over (M,N)).

FlyDSL's vectorized path is dead-coded off (softmax_kernel.py line 104
`if const_expr(False and ...)`), so its kernel always runs the generic scalar
path -- the provider_detail says so. There is no standalone sglang triton
softmax, so the `triton` provider embeds its own tutorial kernel.
"""

from __future__ import annotations

from benchmarks import common
from benchmarks.providers.base import ProviderAdapter

_OKALL = {"bf16", "bfloat16", "fp16", "f16", "fp32", "f32"}


class FlyDSL(ProviderAdapter):
    name = "flydsl"
    includes_allocation = False
    includes_jit = False

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "softmax":
            return False, "flydsl softmax adapter only implements softmax"
        if shape["dtype"] not in common.FLYDSL_DTYPE:
            return False, f"FlyDSL softmax has no {shape['dtype']} path (f32/f16/bf16 only)"
        ok, why = common.flydsl_runtime_ok()
        return (ok, why) if not ok else (True, None)

    def _launcher(self, shape, inputs):
        import torch

        M, N = inputs["M"], inputs["N"]
        ds = common.FLYDSL_DTYPE[shape["dtype"]]
        key = (M, N, ds)
        if key not in self._cache:
            common.bootstrap_env()
            from kernels.softmax_kernel import build_softmax_module
            launch = build_softmax_module(M, N, ds)
            out = torch.empty((M, N), device="cuda", dtype=inputs["dtype"])
            self._cache[key] = (launch, out)
            self.provider_detail = (f"build_softmax_module(M={M},N={N},{ds}); "
                                    "path=generic-scalar (vectorized path dead-coded off, "
                                    "softmax_kernel.py:104 `False and ...`)")
        return self._cache[key]

    def run(self, shape, inputs):
        import torch
        launch, out = self._launcher(shape, inputs)
        launch(inputs["x"], out, inputs["M"], stream=torch.cuda.current_stream())
        return out


class PyTorch(ProviderAdapter):
    name = "pytorch"
    provider_detail = "torch.softmax(dim=-1)"
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "softmax":
            return False, "pytorch softmax adapter only implements softmax"
        if shape["dtype"] not in _OKALL:
            return False, f"no torch softmax path for {shape['dtype']}"
        return True, None

    def run(self, shape, inputs):
        import torch
        return torch.softmax(inputs["x"], dim=-1)


class AiterTriton(ProviderAdapter):
    name = "aiter_triton"
    provider_detail = "aiter.ops.triton.softmax.softmax (online 2-pass Triton)"
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "softmax":
            return False, "aiter_triton softmax adapter only implements softmax"
        if shape["dtype"] not in _OKALL:
            return False, f"validated fp16/bf16/f32, not {shape['dtype']}"
        try:
            from aiter.ops.triton.softmax import softmax  # noqa: F401
        except Exception as e:
            return False, f"import failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def run(self, shape, inputs):
        import torch
        from aiter.ops.triton.softmax import softmax
        with torch.no_grad():
            return softmax(inputs["x"])


class Triton(ProviderAdapter):
    name = "triton"
    provider_detail = "standalone tutorial Triton softmax (inline)"
    includes_allocation = True
    _kernel = None

    def supports(self, shape):
        if shape.get("op_type") != "softmax":
            return False, "triton softmax adapter only implements softmax"
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
            def _softmax(x_ptr, o_ptr, n_rows, n_cols, stride, BLOCK: tl.constexpr):
                row = tl.program_id(0)
                if row < n_rows:
                    offs = tl.arange(0, BLOCK)
                    mask = offs < n_cols
                    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=-float("inf")).to(tl.float32)
                    x = x - tl.max(x, axis=0)
                    e = tl.exp(x)
                    o = e / tl.sum(e, axis=0)
                    tl.store(o_ptr + row * stride + offs, o, mask=mask)

            cls._kernel = _softmax
        return cls._kernel

    def run(self, shape, inputs):
        import torch
        import triton
        x = inputs["x"]
        M, N = inputs["M"], inputs["N"]
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else (8 if BLOCK <= 4096 else 16)
        self._get_kernel()[(M,)](x, out, M, N, x.stride(0), BLOCK=BLOCK, num_warps=num_warps)
        return out


class _Stub(ProviderAdapter):
    _reason = "not available"

    def supports(self, shape):
        return False, self._reason


class Aiter(_Stub):
    name = "aiter"
    _reason = "no standalone compiled aiter softmax op; only aiter.ops.triton.softmax (use aiter_triton)"


class AiterCK(_Stub):
    name = "aiter_ck"
    _reason = "no separately-selectable CK softmax from Python"


class AiterASM(_Stub):
    name = "aiter_asm"
    _reason = "no Python-selectable ASM softmax path in AITER"


class CK(_Stub):
    name = "ck"
    _reason = "no standalone CK softmax adapter on this node"


class Gluon(_Stub):
    name = "gluon"
    _reason = "no Gluon softmax kernel on this node"


class HipBLASLt(_Stub):
    name = "hipblaslt"
    _reason = "hipBLASLt is a GEMM library (no softmax op)"
