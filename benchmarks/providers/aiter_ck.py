"""aiter_ck provider stub: support-detection only (see reason)."""

from __future__ import annotations

from benchmarks.providers.base import ProviderAdapter


class RmsNormAdapter(ProviderAdapter):
    name = "aiter_ck"

    def supports(self, shape):
        return False, "AITER CK norm backend is not separately selectable from Python; module_rmsnorm picks CK vs HIP vs ASM internally. Use provider 'aiter' (records the >8192 CK branch)."
