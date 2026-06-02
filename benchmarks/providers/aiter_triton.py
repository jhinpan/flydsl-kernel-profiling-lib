"""AITER Triton rmsnorm provider (aiter.ops.triton.normalization.rmsnorm.rms_norm).

This is the explicit Triton path (no @compile_ops, pure Triton kernel). It is
2D-only (x is (M,N)) and autograd-wrapped, so we run under no_grad for clean
inference timing.
"""

from __future__ import annotations

from benchmarks.providers.base import ProviderAdapter

_OK_DTYPES = {"bf16", "bfloat16", "fp16", "f16"}


class RmsNormAdapter(ProviderAdapter):
    name = "aiter_triton"
    provider_detail = "aiter.ops.triton.normalization.rmsnorm.rms_norm (Triton)"
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "rmsnorm":
            return False, "aiter_triton adapter only implements rmsnorm"
        if shape["dtype"] not in _OK_DTYPES:
            return False, f"aiter triton rms_norm validated for fp16/bf16, not {shape['dtype']}"
        try:
            from aiter.ops.triton.normalization.rmsnorm import rms_norm  # noqa: F401
        except Exception as e:
            return False, f"import failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def run(self, shape, inputs):
        import torch
        from aiter.ops.triton.normalization.rmsnorm import rms_norm
        with torch.no_grad():
            return rms_norm(inputs["x"], inputs["weight"], inputs["eps"])
