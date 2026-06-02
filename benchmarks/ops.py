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


def _safe_torch_dtype(name: str):
    """torch dtype for a ledger dtype, or None for quant dtypes torch can't map
    directly (int8/fp4/mixed) -- providers handle those per-dtype themselves."""
    try:
        return common.torch_dtype(name)
    except KeyError:
        return None


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

    def tolerance(self, shape: dict) -> tuple[float, float]:
        """(rtol, atol) for correctness vs the fp32 reference. Default = dtype-based;
        ops that accumulate (GEMM/MoE over K) override with a looser tol."""
        return common.tol_for(shape.get("dtype", "bf16"))

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


class GemmOp(Op):
    op_type = "gemm"

    def _MNK(self, shape: dict) -> tuple[int, int, int]:
        a = shape["args"]
        return int(a["M"]), int(a["N"]), int(a["K"])

    def make_inputs(self, shape: dict, seed: int) -> dict:
        import torch

        M, N, K = self._MNK(shape)
        td = common.torch_dtype(shape["dtype"])
        g = common.make_generator(seed)
        # build in fp32, uniform(-1,1) (matches the example test), cast -> identical
        # bits for every provider. a:(M,K), b:(N,K) stored transposed; C = A @ B^T.
        a = torch.empty((M, K), device="cuda", dtype=torch.float32).uniform_(-1, 1, generator=g).to(td).contiguous()
        b = torch.empty((N, K), device="cuda", dtype=torch.float32).uniform_(-1, 1, generator=g).to(td).contiguous()
        return {"a": a, "b": b, "M": M, "N": N, "K": K, "dtype": td}

    def reference(self, shape: dict, inputs: dict):
        import torch

        # fp32 golden, exactly the example's run_torch_acc: C = A @ B^T
        return torch.mm(inputs["a"].float(), inputs["b"].float().T)

    def bytes_moved(self, shape: dict) -> int:
        M, N, K = self._MNK(shape)
        e = _elem_bytes(shape["dtype"])
        # read A (M*K) + read B (N*K) + write C (M*N); single-write GEMM convention
        # (SPLIT_K>1 adds zero-init + atomic partials, not counted -- matches example).
        return (M * K + N * K + M * N) * e

    def flops(self, shape: dict) -> int:
        M, N, K = self._MNK(shape)
        return 2 * M * N * K

    def tolerance(self, shape: dict) -> tuple[float, float]:
        # GEMM accumulates over K (cancellation near zero); match FlyDSL's own
        # hgemm_splitk test bar, looser for fp8.
        return (0.2, 0.2) if shape.get("dtype", "bf16").startswith("fp8") else (0.1, 0.1)

    def args_summary(self, shape: dict) -> str:
        M, N, K = self._MNK(shape)
        return f"M={M},N={N},K={K}"

class FusedRopeCacheOp(Op):
    op_type = "fused_rope_cache"

    def _dims(self, shape):
        a = shape["args"]
        return (int(a["head_dim"]), int(a["num_heads"]), int(a["num_kv_heads"]),
                int(a["seq_len"]))

    def make_inputs(self, shape: dict, seed: int) -> dict:
        import torch

        a = shape["args"]
        D, QH, KH, T = self._dims(shape)
        bs = int(a.get("block_size", 16))
        reuse = bool(a.get("reuse_freqs_front_part", True))
        max_pos = int(a.get("max_pos", 8192))
        td = common.torch_dtype(shape["dtype"])
        g = common.make_generator(seed)
        dev = "cuda"

        cols = (D // 2) if reuse else D
        num_blocks = max(32, (T + bs - 1) // bs + 4)

        # All tensors built ONCE in FlyDSL/reference layout (i32 pos/slot, 2-D
        # cos/sin, flash KV caches, fp32 scales=ones) and shared across providers.
        # aiter_triton converts to its layout (i64, 4-D) inside its own run().
        Q = torch.randn((T, QH, D), device=dev, dtype=torch.float32, generator=g).to(td).contiguous()
        K = torch.randn((T, KH, D), device=dev, dtype=torch.float32, generator=g).to(td).contiguous()
        V = torch.randn((T, KH, D), device=dev, dtype=torch.float32, generator=g).to(td).contiguous()
        CosCache = torch.randn((max_pos, cols), device=dev, dtype=torch.float32, generator=g).to(td).contiguous()
        SinCache = torch.randn((max_pos, cols), device=dev, dtype=torch.float32, generator=g).to(td).contiguous()
        Positions = torch.randint(0, max_pos, (T,), device=dev, dtype=torch.int32, generator=g)
        SlotMapping = torch.arange(T, device=dev, dtype=torch.int32)
        KeyCache = torch.zeros((num_blocks, bs, KH, D), device=dev, dtype=td)
        ValueCache = torch.zeros((num_blocks, bs, KH, D), device=dev, dtype=td)
        # KScale/VScale ALWAYS required by the launcher even with apply_scale=False.
        KScale = torch.ones(1, device=dev, dtype=torch.float32)
        VScale = torch.ones(1, device=dev, dtype=torch.float32)

        return {"Q": Q, "K": K, "V": V, "CosCache": CosCache, "SinCache": SinCache,
                "Positions": Positions, "SlotMapping": SlotMapping,
                "KeyCache": KeyCache, "ValueCache": ValueCache,
                "KScale": KScale, "VScale": VScale,
                "T": T, "D": D, "QH": QH, "KH": KH, "dtype": td}

    def reference(self, shape: dict, inputs: dict):
        import torch

        # NeoX RoPE in native dtype (matches HW rounding), then upcast. Comparable
        # tensor = concat([q_out, k_out] flattened to [T, -1]); providers' output()
        # assemble the same way.
        Q, K = inputs["Q"], inputs["K"]
        reuse = bool(shape["args"].get("reuse_freqs_front_part", True))
        cos = inputs["CosCache"][inputs["Positions"].long()].unsqueeze(1).to(Q.dtype)
        sin = inputs["SinCache"][inputs["Positions"].long()].unsqueeze(1).to(Q.dtype)
        if reuse:
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)
        Dh = Q.shape[-1] // 2

        def rot(X):
            x1, x2 = X[..., :Dh], X[..., Dh:]
            return torch.cat([x1 * cos[..., :Dh] - x2 * sin[..., :Dh],
                              x2 * cos[..., Dh:] + x1 * sin[..., Dh:]], dim=-1)

        q_out, k_out = rot(Q), rot(K)
        return torch.cat([q_out.reshape(q_out.shape[0], -1),
                          k_out.reshape(k_out.shape[0], -1)], dim=-1).float()

    def bytes_moved(self, shape: dict) -> int:
        D, QH, KH, T = self._dims(shape)
        reuse = bool(shape["args"].get("reuse_freqs_front_part", True))
        cols = (D // 2) if reuse else D
        e = _elem_bytes(shape["dtype"])
        # Q read+write (2*QH*D) + K read/K_out/KeyCache (3*KH*D) + V read/ValueCache
        # (2*KH*D) + cos+sin reads (2*cols), per token *e; plus i32 pos+slot (8B/tok).
        return e * T * (2 * QH * D + 5 * KH * D + 2 * cols) + 8 * T

    def flops(self, shape: dict) -> int:
        D, QH, KH, T = self._dims(shape)
        # NeoX rotation: 2 mul + 1 add per rotated element; ~negligible (single wave).
        return 3 * T * (QH + KH) * D

    def args_summary(self, shape: dict) -> str:
        D, QH, KH, T = self._dims(shape)
        return f"D={D},QH={QH},KH={KH},T={T}"

    def tolerance(self, shape: dict) -> tuple[float, float]:
        # RoPE is a rotation: outputs contain near-zero elements (q1*cos - q2*sin
        # can cancel), where relative tol is meaningless. Use an atol-dominated
        # bound so a correct kernel with near-zero outputs isn't false-failed
        # (e.g. aiter_triton f16, which otherwise gets excluded and inflates vs-best).
        return (2e-2, 2e-2) if shape.get("dtype", "bf16") in ("bf16", "bfloat16") else (1e-2, 1e-2)


class MoeGemmOp(Op):
    """Full 2-stage MoE (gate+up silu, then down + topk reduce).

    Ledger args: M=tokens, E=experts, Dim1=model_dim, Dim2=2*inter_dim
    (inter_dim=Dim2//2), TopK=topk. Inputs are built ONCE in fp32 and shared
    across providers; the per-provider quantize/cast/preshuffle happens inside
    run(). The fp32 reference is the standard torch MoE and matches both the
    FlyDSL composed output (moe_gemm1 -> requant -> moe_gemm2, routed weight in
    stage2) and aiter.fused_moe (doweight_stage1=False).
    """

    op_type = "moe_gemm"

    def _dims(self, shape: dict):
        a = shape["args"]
        tokens = int(a["M"])
        experts = int(a["E"])
        model_dim = int(a["Dim1"])
        inter_dim = int(a["Dim2"]) // 2
        topk = int(a["TopK"])
        return tokens, experts, model_dim, inter_dim, topk

    def make_inputs(self, shape: dict, seed: int) -> dict:
        import torch

        tokens, experts, model_dim, inter_dim, topk = self._dims(shape)
        g = common.make_generator(seed)
        # build in fp32 once -> every provider gets identical bits, then quantizes/casts itself
        x_fp32 = (torch.randn((tokens, model_dim), device="cuda", dtype=torch.float32, generator=g) * 0.2).contiguous()
        w1_fp32 = (torch.randn((experts, 2 * inter_dim, model_dim), device="cuda", dtype=torch.float32, generator=g) * 0.2).contiguous()
        w2_fp32 = (torch.randn((experts, model_dim, inter_dim), device="cuda", dtype=torch.float32, generator=g)
                   * (0.2 / (inter_dim ** 0.5))).contiguous()
        score = torch.rand((tokens, experts), device="cuda", dtype=torch.float32, generator=g)
        topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
        topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32).contiguous()
        return {
            "x_fp32": x_fp32, "w1_fp32": w1_fp32, "w2_fp32": w2_fp32,
            "topk_ids": topk_ids.to(torch.int32).contiguous(), "topk_weights": topk_weights,
            "M": tokens, "E": experts, "model_dim": model_dim, "inter_dim": inter_dim, "topk": topk,
            # inputs are fp32 + dtype-agnostic; providers quantize per-dtype themselves.
            # map only dtypes torch knows so make_inputs never KeyErrors on int8/fp4/mixed
            # (those become per-provider unsupported, not a shared shape-level failure).
            "dtype_str": shape["dtype"],
            "dtype": _safe_torch_dtype(shape["dtype"]),
        }

    def reference(self, shape: dict, inputs: dict):
        import torch
        import torch.nn.functional as F

        tokens, experts, model_dim, inter_dim, topk = self._dims(shape)
        x = inputs["x_fp32"]
        w1 = inputs["w1_fp32"]
        w2 = inputs["w2_fp32"]
        topk_ids = inputs["topk_ids"].long()
        topk_w = inputs["topk_weights"]
        out = torch.zeros((tokens, model_dim), device="cuda", dtype=torch.float32)
        for e in range(experts):
            idx = (topk_ids == e).nonzero(as_tuple=False)  # [num, 2] (token, slot)
            if idx.numel() == 0:
                continue
            t_idx = idx[:, 0]
            s_idx = idx[:, 1]
            y2 = F.linear(x[t_idx, :], w1[e])          # [num, 2*inter_dim]
            gate = y2[:, :inter_dim]
            up = y2[:, inter_dim:]
            h = F.silu(gate) * up                      # [num, inter_dim]
            y = F.linear(h, w2[e])                     # [num, model_dim]
            y = y * topk_w[t_idx, s_idx].unsqueeze(-1)  # routed weight (applied in stage2)
            out.index_add_(0, t_idx, y)
        return out

    def bytes_moved(self, shape: dict) -> int:
        tokens, experts, model_dim, inter_dim, topk = self._dims(shape)
        eb = _elem_bytes(shape["dtype"])
        e_active = min(experts, tokens * topk)  # weights touched = active experts only
        x_bytes = tokens * model_dim * eb
        w1_bytes = e_active * (2 * inter_dim) * model_dim * eb
        w2_bytes = e_active * model_dim * inter_dim * eb
        out_bytes = tokens * model_dim * 2  # final [tokens, model_dim] f16/bf16
        return x_bytes + w1_bytes + w2_bytes + out_bytes

    def flops(self, shape: dict) -> int:
        tokens, experts, model_dim, inter_dim, topk = self._dims(shape)
        # stage1 2*M*T*(2I)*K + stage2 2*M*T*K*I = 6*M*T*K*I (silu negligible)
        return 6 * tokens * topk * model_dim * inter_dim

    def args_summary(self, shape: dict) -> str:
        tokens, experts, model_dim, inter_dim, topk = self._dims(shape)
        return f"tokens={tokens},E={experts},model_dim={model_dim},inter_dim={inter_dim},topk={topk}"


# add to _REGISTRY dict:


_REGISTRY: dict[str, Op] = {
    RmsNormOp.op_type: RmsNormOp(),
    LayerNormOp.op_type: LayerNormOp(),
    SoftmaxOp.op_type: SoftmaxOp(),
    GemmOp.op_type: GemmOp(),
    FusedRopeCacheOp.op_type: FusedRopeCacheOp(),
    MoeGemmOp.op_type: MoeGemmOp(),
}


def get_op(op_type: str) -> Op | None:
    return _REGISTRY.get(op_type)


def register(op: Op) -> None:
    _REGISTRY[op.op_type] = op
