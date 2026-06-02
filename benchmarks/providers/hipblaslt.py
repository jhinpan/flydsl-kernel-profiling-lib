"""hipblaslt provider stub: support-detection only (see reason)."""

from __future__ import annotations

from benchmarks.providers.base import ProviderAdapter


class RmsNormAdapter(ProviderAdapter):
    name = "hipblaslt"

    def supports(self, shape):
        return False, "hipBLASLt is a GEMM/matmul library; it has no rmsnorm op."
