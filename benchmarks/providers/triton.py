"""Standalone Triton rmsnorm provider (sglang one-pass kernel).

Distinct from aiter_triton: this is sglang's self-contained one-pass kernel
(sglang.jit_kernel.diffusion.triton.rmsnorm_onepass.triton_one_pass_rms_norm),
which has no aiter dependency. Requires SGLANG_USE_AITER=0 at import time
(env.sh sets it); handles arbitrary leading dims via internal reshape.
"""

from __future__ import annotations

import os

from benchmarks.providers.base import ProviderAdapter


class RmsNormAdapter(ProviderAdapter):
    name = "triton"
    provider_detail = "sglang triton_one_pass_rms_norm (standalone Triton)"
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "rmsnorm":
            return False, "triton adapter only implements rmsnorm"
        os.environ.setdefault("SGLANG_USE_AITER", "0")
        try:
            from sglang.jit_kernel.diffusion.triton.rmsnorm_onepass import triton_one_pass_rms_norm  # noqa: F401
        except Exception as e:
            return False, f"import failed ({type(e).__name__}: {e})"
        return True, None

    def run(self, shape, inputs):
        from sglang.jit_kernel.diffusion.triton.rmsnorm_onepass import triton_one_pass_rms_norm
        return triton_one_pass_rms_norm(inputs["x"], inputs["weight"], inputs["eps"])
