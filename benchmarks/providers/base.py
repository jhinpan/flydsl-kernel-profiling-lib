"""Provider adapter contract + registry.

A *provider* is one implementation of a kernel (FlyDSL, AITER-compiled,
AITER-Triton, standalone Triton, PyTorch reference, ...). An *adapter* binds a
provider to one op_type (e.g. rmsnorm). The runner builds inputs ONCE (via the
reference adapter) and hands the SAME inputs dict to every provider so the
comparison is apples-to-apples; an adapter that needs a different layout must
convert inside run() and set includes_layout_conversion=True.

Adapters override the small surface: supports / make_inputs / run / output.
Default check_correctness and benchmark are provided here.
"""

from __future__ import annotations

from typing import Any

from benchmarks import common


class ProviderAdapter:
    # provider id as it appears in benchmark_result.provider (the enum)
    name: str = "base"
    # free-text detail, e.g. backend selected, "backend_unknown", kernel path
    provider_detail: str = ""
    # honest accounting flags for the result row
    includes_allocation: bool = False
    includes_layout_conversion: bool = False
    includes_jit: bool = False

    def __init__(self, op_type: str):
        self.op_type = op_type

    # -- capability ------------------------------------------------------- #
    def supports(self, shape: dict) -> tuple[bool, str | None]:
        """(supported, reason). MUST NOT raise for unsupported shapes."""
        raise NotImplementedError

    # -- inputs ----------------------------------------------------------- #
    def make_inputs(self, shape: dict, seed: int) -> dict:
        """Allocate + initialize inputs deterministically. Only the *reference*
        adapter's make_inputs is called by the runner; the dict is shared."""
        raise NotImplementedError

    # -- execution -------------------------------------------------------- #
    def run(self, shape: dict, inputs: dict) -> Any:
        """Run the kernel. No allocation/JIT in the timed region unless the
        corresponding includes_* flag is set. Returns the raw output object."""
        raise NotImplementedError

    def output(self, shape: dict, inputs: dict):
        """Return the comparable output tensor (the runner upcasts to fp32).
        Default: whatever run() returns."""
        return self.run(shape, inputs)

    # -- correctness ------------------------------------------------------ #
    def check_correctness(self, shape: dict, inputs: dict, reference) -> tuple[bool, str | None]:
        """Compare output() against an fp32 reference with dtype-aware tolerance."""
        import torch

        try:
            out = self.output(shape, inputs)
            if out is None:
                return False, "no output"
            rtol, atol = common.tol_for(shape.get("dtype", "bf16"))
            ref = reference.float()
            got = out.float()
            if got.shape != ref.shape:
                got = got.reshape(ref.shape)
            ok = torch.allclose(got, ref, rtol=rtol, atol=atol)
            if not ok:
                diff = (got - ref).abs()
                denom = ref.abs().clamp_min(1e-12)
                return False, (f"max_abs={diff.max().item():.3e} "
                               f"max_rel={(diff / denom).max().item():.3e} "
                               f"rtol={rtol} atol={atol}")
            return True, None
        except Exception as e:  # pragma: no cover - provider-specific
            return False, f"{type(e).__name__}: {e}"

    # -- timing ----------------------------------------------------------- #
    def benchmark(self, shape: dict, inputs: dict, warmup_iters: int,
                  repeat_iters: int) -> dict:
        return common.benchmark(lambda: self.run(shape, inputs),
                                warmup_iters=warmup_iters, repeat_iters=repeat_iters)


# --------------------------------------------------------------------------- #
# Registry + entrypoint loading
# --------------------------------------------------------------------------- #
def load_entrypoint(entrypoint: str, op_type: str) -> ProviderAdapter:
    """Resolve "pkg.module:Attr" from a baseline_matrix into an adapter instance.

    Attr may be (a) a ProviderAdapter subclass -> instantiated with op_type,
    (b) a factory callable get_adapter(op_type) -> adapter, or
    (c) an already-instantiated adapter (returned as-is)."""
    import importlib

    mod_name, _, attr = entrypoint.partition(":")
    if not attr:
        attr = "Adapter"
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, attr)
    if isinstance(obj, type) and issubclass(obj, ProviderAdapter):
        return obj(op_type)
    if isinstance(obj, ProviderAdapter):
        return obj
    if callable(obj):
        return obj(op_type)
    raise TypeError(f"{entrypoint}: {attr!r} is not a ProviderAdapter / factory")
