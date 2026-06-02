"""aiter_asm provider stub: support-detection only (see reason)."""

from __future__ import annotations

from benchmarks.providers.base import ProviderAdapter


class RmsNormAdapter(ProviderAdapter):
    name = "aiter_asm"

    def supports(self, shape):
        return False, "No Python-selectable ASM rmsnorm path in AITER (only gemm_a16w16_asm is ASM-tagged). ASM-vs-HIP within module_rmsnorm is opaque."
