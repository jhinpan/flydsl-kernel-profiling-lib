"""Per-op canonical inputs, fp32 reference, and roofline model.

This is the single source of truth so that (a) every provider receives the
EXACT same input tensors for a shape (fair comparison), (b) correctness is
judged against one fp32 golden, and (c) effective GB/s / TFLOP/s use one
bytes/flops model. Providers only implement supports()+run(); they never build
their own inputs.

Adding a new op = add one Op subclass + register it.
"""

from __future__ import annotations

from benchmarks import common


def _elem_bytes(dtype: str) -> int:
    return {"fp32": 4, "f32": 4, "fp16": 2, "f16": 2, "bf16": 2, "bfloat16": 2,
            "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1}.get(dtype, 2)


class Op:
    op_type = "base"

    def make_inputs(self, shape: dict, seed: int) -> dict:
        raise NotImplementedError

    def reference(self, shape: dict, inputs: dict):
        """fp32 golden output tensor."""
        raise NotImplementedError

    def bytes_moved(self, shape: dict) -> int:
        return 0

    def flops(self, shape: dict) -> int:
        return 0

    def args_summary(self, shape: dict) -> str:
        return ",".join(f"{k}={v}" for k, v in shape.get("args", {}).items())

    def effective(self, shape: dict, median_us: float | None) -> dict:
        if not median_us or median_us <= 0:
            return {"effective_gbps": None, "effective_tflops": None}
        sec = median_us * 1e-6
        b = self.bytes_moved(shape)
        f = self.flops(shape)
        return {
            "effective_gbps": (b / sec / 1e9) if b else None,
            "effective_tflops": (f / sec / 1e12) if f else None,
        }


class RmsNormOp(Op):
    op_type = "rmsnorm"

    def _MN(self, shape: dict) -> tuple[int, int]:
        a = shape["args"]
        return int(a["M"]), int(a["N"])

    def _eps(self, shape: dict) -> float:
        return float(shape.get("args", {}).get("eps", 1e-5))

    def make_inputs(self, shape: dict, seed: int) -> dict:
        import torch

        M, N = self._MN(shape)
        td = common.torch_dtype(shape["dtype"])
        g = common.make_generator(seed)
        # build in fp32 then cast -> identical bits for every provider
        x = torch.randn((M, N), device="cuda", dtype=torch.float32, generator=g).to(td).contiguous()
        w = torch.rand((N,), device="cuda", dtype=torch.float32, generator=g).to(td).contiguous()
        return {"x": x, "weight": w, "eps": self._eps(shape), "M": M, "N": N, "dtype": td}

    def reference(self, shape: dict, inputs: dict):
        import torch

        x, w, eps = inputs["x"], inputs["weight"], inputs["eps"]
        N = inputs["N"]
        return torch.nn.functional.rms_norm(x.float(), (N,), w.float(), eps=eps)

    def bytes_moved(self, shape: dict) -> int:
        M, N = self._MN(shape)
        e = _elem_bytes(shape["dtype"])
        # read x (M*N) + read weight (N) + write out (M*N)
        return (2 * M * N + N) * e

    def flops(self, shape: dict) -> int:
        M, N = self._MN(shape)
        # x^2 (1) + add to acc (1) + rsqrt(~) + mul rstd (1) + mul weight (1) ~ 4*M*N
        return 4 * M * N

    def args_summary(self, shape: dict) -> str:
        M, N = self._MN(shape)
        return f"M={M},N={N}"


class LayerNormOp(Op):
    op_type = "layernorm"

    def _MN(self, shape):
        a = shape["args"]
        return int(a["M"]), int(a["N"])

    def _eps(self, shape):
        return float(shape.get("args", {}).get("eps", 1e-5))

    def make_inputs(self, shape, seed):
        import torch

        M, N = self._MN(shape)
        td = common.torch_dtype(shape["dtype"])
        g = common.make_generator(seed)
        x = torch.randn((M, N), device="cuda", dtype=torch.float32, generator=g).to(td).contiguous()
        gamma = torch.rand((N,), device="cuda", dtype=torch.float32, generator=g).to(td).contiguous()
        beta = torch.rand((N,), device="cuda", dtype=torch.float32, generator=g).to(td).contiguous()
        return {"x": x, "gamma": gamma, "beta": beta, "eps": self._eps(shape), "M": M, "N": N, "dtype": td}

    def reference(self, shape, inputs):
        import torch

        return torch.nn.functional.layer_norm(inputs["x"].float(), (inputs["N"],),
                                              inputs["gamma"].float(), inputs["beta"].float(), eps=inputs["eps"])

    def bytes_moved(self, shape):
        M, N = self._MN(shape)
        return (2 * M * N + 2 * N) * _elem_bytes(shape["dtype"])

    def flops(self, shape):
        M, N = self._MN(shape)
        return 6 * M * N

    def args_summary(self, shape):
        M, N = self._MN(shape)
        return f"M={M},N={N}"


class SoftmaxOp(Op):
    op_type = "softmax"

    def _MN(self, shape):
        a = shape["args"]
        return int(a["M"]), int(a["N"])

    def make_inputs(self, shape, seed):
        import torch

        M, N = self._MN(shape)
        td = common.torch_dtype(shape["dtype"])
        g = common.make_generator(seed)
        x = torch.randn((M, N), device="cuda", dtype=torch.float32, generator=g).to(td).contiguous()
        return {"x": x, "M": M, "N": N, "dtype": td}

    def reference(self, shape, inputs):
        import torch

        return torch.softmax(inputs["x"].float(), dim=-1)

    def bytes_moved(self, shape):
        M, N = self._MN(shape)
        return 2 * M * N * _elem_bytes(shape["dtype"])

    def flops(self, shape):
        M, N = self._MN(shape)
        return 5 * M * N

    def args_summary(self, shape):
        M, N = self._MN(shape)
        return f"M={M},N={N}"


_REGISTRY: dict[str, Op] = {
    RmsNormOp.op_type: RmsNormOp(),
    LayerNormOp.op_type: LayerNormOp(),
    SoftmaxOp.op_type: SoftmaxOp(),
}


def get_op(op_type: str) -> Op | None:
    return _REGISTRY.get(op_type)


def register(op: Op) -> None:
    _REGISTRY[op.op_type] = op
