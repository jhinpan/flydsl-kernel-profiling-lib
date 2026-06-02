"""FlyDSL rmsnorm provider (kernels.rmsnorm_kernel.build_rmsnorm_module).

The launcher is built once per (M,N,dtype) and the output buffer is preallocated
and reused, so neither JIT (excluded via warmup) nor allocation is in the timed
region. eps is hardcoded to 1e-5 inside the FlyDSL kernel; supports() flags a
mismatch. dtypes: f32/f16/bf16 only (no fp8 rmsnorm path).
"""

from __future__ import annotations

from benchmarks import common
from benchmarks.providers.base import ProviderAdapter

_KERNEL_EPS = 1e-5  # hardcoded module constant in rmsnorm_kernel.py


class RmsNormAdapter(ProviderAdapter):
    name = "flydsl"
    includes_allocation = False
    includes_jit = False

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache: dict = {}

    def supports(self, shape):
        if shape.get("op_type") != "rmsnorm":
            return False, "flydsl adapter only implements rmsnorm"
        if shape["dtype"] not in common.FLYDSL_DTYPE:
            return False, f"FlyDSL rmsnorm has no {shape['dtype']} path (f32/f16/bf16 only)"
        ok, reason = common.flydsl_runtime_ok()
        if not ok:
            return False, reason
        if abs(float(shape["args"].get("eps", _KERNEL_EPS)) - _KERNEL_EPS) > 1e-12:
            return False, f"FlyDSL eps is hardcoded {_KERNEL_EPS}, shape wants {shape['args'].get('eps')}"
        return True, None

    def _launcher(self, shape, inputs):
        import torch

        M, N = inputs["M"], inputs["N"]
        ds = common.FLYDSL_DTYPE[shape["dtype"]]
        key = (M, N, ds)
        if key not in self._cache:
            common.bootstrap_env()
            from kernels.rmsnorm_kernel import build_rmsnorm_module
            launch = build_rmsnorm_module(M, N, ds)
            out = torch.empty((M, N), device="cuda", dtype=inputs["dtype"])
            fast = N >= 2048 and N % 2048 == 0 and ds in ("f16", "bf16")
            self._cache[key] = (launch, out)
            self.provider_detail = (f"build_rmsnorm_module(M={M},N={N},{ds}); "
                                    f"path={'fast-vectorized' if fast else 'generic-scalar'}")
        return self._cache[key]

    def run(self, shape, inputs):
        import torch
        launch, out = self._launcher(shape, inputs)
        launch(inputs["x"], inputs["weight"], out, inputs["M"], stream=torch.cuda.current_stream())
        return out
