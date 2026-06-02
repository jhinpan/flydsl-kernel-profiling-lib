import os, sys, json
sys.path[:0] = ['/sgl-workspace/FlyDSL-lab/build-fly/python_packages', '/sgl-workspace/FlyDSL-lab', '/sgl-workspace/flydsl-kernel-profiling']
import torch
from benchmarks import ops
from benchmarks.providers.base import load_entrypoint
shape = json.loads('{"arch": "gfx950", "args": {"M": 1, "N": 7168, "eps": 1e-05}, "baselines_available": ["aiter", "aiter_triton", "triton", "pytorch"], "dtype": "bf16", "gpu": "MI350X", "kernel_name": "rmsnorm", "layout": {"row_major": true}, "model": "DeepSeek-R1", "notes": "", "op_type": "rmsnorm", "shape_id": "sha1:6e14a5f6aebc31c9", "source": {"concurrency": null, "file": "model_shapes.json", "input_len": null, "kind": "aiter_model_shapes", "notes": "M synthesized from default token sweep [1, 32, 256, 2048, 16384] (not in model_shapes.json)", "output_len": null}, "stage": "model_config", "weight": {"baseline_time_weight": null, "occurrences": null, "traffic_weight": null}}')
op = ops.get_op('rmsnorm')
inputs = op.make_inputs(shape, 1234)
ad = load_entrypoint('benchmarks.providers.flydsl:RmsNormAdapter', 'rmsnorm')
ok, why = ad.supports(shape)
assert ok, "provider does not support shape: " + str(why)
for _ in range(10):
    ad.run(shape, inputs)
torch.cuda.synchronize()
for _ in range(50):
    ad.run(shape, inputs)
torch.cuda.synchronize()
