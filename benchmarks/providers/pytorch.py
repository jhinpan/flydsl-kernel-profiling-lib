"""PyTorch rmsnorm provider (torch.nn.functional.rms_norm).

Doubles as the default *reference provider*: its dtype-matched output is a fair
baseline, while ops.RmsNormOp.reference() supplies the fp32 golden for correctness.
"""

from __future__ import annotations

from benchmarks.providers.base import ProviderAdapter

_OK_DTYPES = {"bf16", "bfloat16", "fp16", "f16", "fp32", "f32"}


class RmsNormAdapter(ProviderAdapter):
    name = "pytorch"
    provider_detail = "torch.nn.functional.rms_norm"
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "rmsnorm":
            return False, "pytorch adapter only implements rmsnorm"
        if shape["dtype"] not in _OK_DTYPES:
            return False, f"no torch rms_norm path validated for {shape['dtype']}"
        return True, None

    def run(self, shape, inputs):
        import torch
        return torch.nn.functional.rms_norm(inputs["x"], (inputs["N"],), inputs["weight"],
                                            eps=inputs["eps"])
