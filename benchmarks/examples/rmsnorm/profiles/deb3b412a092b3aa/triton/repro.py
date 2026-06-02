import os, sys, json
sys.path[:0] = ['/sgl-workspace/FlyDSL-lab/build-fly/python_packages', '/sgl-workspace/FlyDSL-lab', '/sgl-workspace/flydsl-kernel-profiling']
import torch
from benchmarks import ops
from benchmarks.providers.base import load_entrypoint
shape = json.loads('{"arch": "gfx950", "args": {"M": 8, "N": 2560, "eps": 1e-05}, "baselines_available": ["aiter", "aiter_triton", "triton", "pytorch"], "dtype": "bf16", "gpu": "MI350X", "kernel_name": "rmsnorm", "layout": {"row_major": true}, "model": "Qwen3-4B", "notes": "", "op_type": "rmsnorm", "shape_id": "sha1:deb3b412a092b3aa", "source": {"concurrency": 8, "file": null, "input_len": 8192, "kind": "atom_workload", "notes": "derived from ATOM anchor isl=8192 osl=1024 concurrency=8 (decode: M=concurrency); N=hidden_size=2560 from config.json; synthetic anchor -> weights unknown", "output_len": 1024}, "stage": "decode", "weight": {"baseline_time_weight": null, "occurrences": null, "traffic_weight": null}}')
op = ops.get_op('rmsnorm')
inputs = op.make_inputs(shape, 1234)
ad = load_entrypoint('benchmarks.providers.triton:RmsNormAdapter', 'rmsnorm')
ok, why = ad.supports(shape)
assert ok, "provider does not support shape: " + str(why)
for _ in range(10):
    ad.run(shape, inputs)
torch.cuda.synchronize()
for _ in range(50):
    ad.run(shape, inputs)
torch.cuda.synchronize()
