"""AITER compiled rmsnorm provider (aiter.rms_norm -> module_rmsnorm CK/HIP).

Backend (CK vs hand-written HIP vs ASM) is chosen inside the compiled C++ module
and is NOT observable from Python. The one Python-visible branch: aiter routes to
the *_ck kernel when N > 8192 (or use_model_sensitive_rmsnorm > 0). We record that
branch in provider_detail; the CK/HIP/ASM choice within the module stays opaque
(so this is honestly labeled, never silently "fastest backend").
"""

from __future__ import annotations

from benchmarks.providers.base import ProviderAdapter

_OK_DTYPES = {"bf16", "bfloat16", "fp16", "f16"}


class RmsNormAdapter(ProviderAdapter):
    name = "aiter"
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "rmsnorm":
            return False, "aiter adapter only implements rmsnorm"
        if shape["dtype"] not in _OK_DTYPES:
            return False, f"aiter.rms_norm validated for fp16/bf16, not {shape['dtype']}"
        try:
            import aiter  # noqa: F401
        except Exception as e:
            return False, f"import aiter failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def run(self, shape, inputs):
        import aiter
        N = inputs["N"]
        self.provider_detail = (f"aiter.rms_norm (compiled module_rmsnorm; Python branch "
                                f"{'->_ck (N>8192)' if N > 8192 else 'default (N<=8192)'}; "
                                f"CK/HIP/ASM within module opaque)")
        return aiter.rms_norm(inputs["x"], inputs["weight"], inputs["eps"])
