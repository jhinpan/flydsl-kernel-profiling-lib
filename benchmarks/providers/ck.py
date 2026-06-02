"""ck provider stub: support-detection only (see reason)."""

from __future__ import annotations

from benchmarks.providers.base import ProviderAdapter


class RmsNormAdapter(ProviderAdapter):
    name = "ck"

    def supports(self, shape):
        return False, "No standalone Composable Kernel rmsnorm adapter on this node; CK rmsnorm is only reachable via the aiter compiled module."
