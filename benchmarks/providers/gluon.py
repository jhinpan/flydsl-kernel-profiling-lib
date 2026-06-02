"""gluon provider stub: support-detection only (see reason)."""

from __future__ import annotations

from benchmarks.providers.base import ProviderAdapter


class RmsNormAdapter(ProviderAdapter):
    name = "gluon"

    def supports(self, shape):
        return False, "No Gluon rmsnorm kernel available on this node."
