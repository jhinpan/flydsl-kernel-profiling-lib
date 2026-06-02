"""LayerNorm provider adapters (all providers co-located, one class each).

baseline_matrix entrypoints select a class, e.g. benchmarks.providers.layernorm:FlyDSL.
LayerNorm mirrors rmsnorm but adds a beta/bias tensor and biased variance; eps is
the hardcoded FlyDSL module constant 1e-5. dtypes f32/f16/bf16 (no fp8).
"""

from __future__ import annotations

from benchmarks import common
from benchmarks.providers.base import ProviderAdapter

_KERNEL_EPS = 1e-5
_OK16 = {"bf16", "bfloat16", "fp16", "f16"}
_OKALL = _OK16 | {"fp32", "f32"}


class FlyDSL(ProviderAdapter):
    name = "flydsl"
    includes_allocation = False
    includes_jit = False

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "layernorm":
            return False, "flydsl layernorm adapter only implements layernorm"
        if shape["dtype"] not in common.FLYDSL_DTYPE:
            return False, f"FlyDSL layernorm has no {shape['dtype']} path (f32/f16/bf16 only)"
        if abs(float(shape["args"].get("eps", _KERNEL_EPS)) - _KERNEL_EPS) > 1e-12:
            return False, f"FlyDSL eps hardcoded {_KERNEL_EPS}"
        ok, why = common.flydsl_runtime_ok()
        return (ok, why) if not ok else (True, None)

    def _launcher(self, shape, inputs):
        import torch

        M, N = inputs["M"], inputs["N"]
        ds = common.FLYDSL_DTYPE[shape["dtype"]]
        key = (M, N, ds)
        if key not in self._cache:
            common.bootstrap_env()
            from kernels.layernorm_kernel import build_layernorm_module
            launch = build_layernorm_module(M, N, ds)
            out = torch.empty((M, N), device="cuda", dtype=inputs["dtype"])
            fast = ds in ("f16", "bf16") and N == 8192
            self._cache[key] = (launch, out)
            self.provider_detail = (f"build_layernorm_module(M={M},N={N},{ds}); "
                                    f"path={'fast-vectorized' if fast else 'generic-scalar'}")
        return self._cache[key]

    def run(self, shape, inputs):
        import torch
        launch, out = self._launcher(shape, inputs)
        launch(inputs["x"], inputs["gamma"], inputs["beta"], out, inputs["M"],
               stream=torch.cuda.current_stream())
        return out


class PyTorch(ProviderAdapter):
    name = "pytorch"
    provider_detail = "torch.nn.functional.layer_norm"
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "layernorm":
            return False, "pytorch layernorm adapter only implements layernorm"
        if shape["dtype"] not in _OKALL:
            return False, f"no torch layer_norm path for {shape['dtype']}"
        return True, None

    def run(self, shape, inputs):
        import torch
        return torch.nn.functional.layer_norm(inputs["x"], (inputs["N"],), inputs["gamma"],
                                              inputs["beta"], eps=inputs["eps"])


class Aiter(ProviderAdapter):
    name = "aiter"
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "layernorm":
            return False, "aiter layernorm adapter only implements layernorm"
        if shape["dtype"] not in _OK16:
            return False, f"aiter.layer_norm validated for fp16/bf16, not {shape['dtype']}"
        try:
            import aiter  # noqa: F401
        except Exception as e:
            return False, f"import aiter failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def run(self, shape, inputs):
        import aiter
        self.provider_detail = "aiter.layer_norm (compiled module_norm; CK/HIP/ASM within module opaque)"
        return aiter.layer_norm(inputs["x"], inputs["gamma"], inputs["beta"], epsilon=inputs["eps"])


class AiterTriton(ProviderAdapter):
    name = "aiter_triton"
    provider_detail = "aiter.ops.triton.norm.layer_norm (Triton)"
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "layernorm":
            return False, "aiter_triton layernorm adapter only implements layernorm"
        if shape["dtype"] not in _OK16:
            return False, f"aiter triton layer_norm validated for fp16/bf16, not {shape['dtype']}"
        try:
            from aiter.ops.triton.norm import layer_norm  # noqa: F401
        except Exception as e:
            return False, f"import failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def run(self, shape, inputs):
        import torch
        from aiter.ops.triton.norm import layer_norm
        with torch.no_grad():
            return layer_norm(inputs["x"], inputs["gamma"], inputs["beta"], inputs["eps"])


class _Stub(ProviderAdapter):
    _reason = "not available"

    def supports(self, shape):
        return False, self._reason


class Triton(_Stub):
    name = "triton"
    _reason = "no standalone sglang triton layernorm on this node (only rmsnorm_onepass exists)"


class AiterCK(_Stub):
    name = "aiter_ck"
    _reason = "CK norm backend not separately selectable from Python; use 'aiter'"


class AiterASM(_Stub):
    name = "aiter_asm"
    _reason = "no Python-selectable ASM layernorm path in AITER"


class CK(_Stub):
    name = "ck"
    _reason = "no standalone CK layernorm adapter on this node"


class Gluon(_Stub):
    name = "gluon"
    _reason = "no Gluon layernorm kernel on this node"


class HipBLASLt(_Stub):
    name = "hipblaslt"
    _reason = "hipBLASLt is GEMM-only (no layernorm op)"
