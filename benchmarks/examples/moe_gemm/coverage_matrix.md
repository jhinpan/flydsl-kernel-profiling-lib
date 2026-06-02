# Coverage matrix: moe_gemm

| shape_id | model | stage | dtype | args | flydsl | aiter | aiter_triton | aiter_ck | aiter_asm | ck | triton | gluon | hipblaslt | pytorch | profile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8a2d8ffefecf28b7 | DeepSeek-R1 | model_config | fp4 | tokens=1,E=256,model_dim=7168,inter_dim=256,topk=8 | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| adf71cef3ac6bf36 | DeepSeek-R1 | model_config | fp4 | tokens=32,E=256,model_dim=7168,inter_dim=256,topk=8 | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| a6a2f0f95e1f05eb | DeepSeek-R1 | model_config | fp4 | tokens=256,E=256,model_dim=7168,inter_dim=256,topk=8 | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| be1c8ab60e34c268 | DeepSeek-R1 | model_config | fp4 | tokens=2048,E=256,model_dim=7168,inter_dim=256,topk=8 | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 3cdaa260f5906b4c | DeepSeek-R1 | model_config | fp4 | tokens=16384,E=256,model_dim=7168,inter_dim=256,topk=8 | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 1ffd9226bf013782 | DeepSeek-R1 | model_config | fp8 | tokens=1,E=256,model_dim=7168,inter_dim=256,topk=8 | ok | failed | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 3e1f04d82bff6672 | DeepSeek-R1 | model_config | fp8 | tokens=32,E=256,model_dim=7168,inter_dim=256,topk=8 | ok | failed | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| d48e60cdc80b5fae | DeepSeek-R1 | model_config | fp8 | tokens=256,E=256,model_dim=7168,inter_dim=256,topk=8 | incorrect | failed | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 4608b9916c92f9aa | DeepSeek-R1 | model_config | fp8 | tokens=2048,E=256,model_dim=7168,inter_dim=256,topk=8 | incorrect | failed | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| bd2374bbf6f6789c | DeepSeek-R1 | model_config | fp8 | tokens=16384,E=256,model_dim=7168,inter_dim=256,topk=8 | incorrect | failed | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| a1b2cb5532d1baf5 | Llama4 Maverick | model_config | int8 | tokens=1,E=128,model_dim=5120,inter_dim=1024,topk=1 | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| c9441c3aa1f175f4 | Qwen3-235B-A22B | model_config | int8 | tokens=1,E=128,model_dim=4096,inter_dim=192,topk=8 | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 9c54213b9db66675 | Llama4 Maverick | model_config | int8 | tokens=32,E=128,model_dim=5120,inter_dim=1024,topk=1 | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| abfd97461f3a00b1 | Qwen3-235B-A22B | model_config | int8 | tokens=32,E=128,model_dim=4096,inter_dim=192,topk=8 | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 39fa7510970e7ca1 | Llama4 Maverick | model_config | int8 | tokens=256,E=128,model_dim=5120,inter_dim=1024,topk=1 | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 8153544387e6d54a | Qwen3-235B-A22B | model_config | int8 | tokens=256,E=128,model_dim=4096,inter_dim=192,topk=8 | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 732dc8df330d1b4f | Llama4 Maverick | model_config | int8 | tokens=2048,E=128,model_dim=5120,inter_dim=1024,topk=1 | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 9a8c86686b983e62 | Qwen3-235B-A22B | model_config | int8 | tokens=2048,E=128,model_dim=4096,inter_dim=192,topk=8 | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 81e9230194a615aa | Llama4 Maverick | model_config | int8 | tokens=16384,E=128,model_dim=5120,inter_dim=1024,topk=1 | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 8a266bf003edb36e | Qwen3-235B-A22B | model_config | int8 | tokens=16384,E=128,model_dim=4096,inter_dim=192,topk=8 | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| e5b27ac5ab984165 | GPT-OSS 120B | model_config | mixed_a8w4 | tokens=1,E=128,model_dim=2880,inter_dim=360,topk=4 | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 6aac9cb39fd4adc2 | GPT-OSS 120B | model_config | mixed_a8w4 | tokens=32,E=128,model_dim=2880,inter_dim=360,topk=4 | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 19c813711e7bb465 | GPT-OSS 120B | model_config | mixed_a8w4 | tokens=256,E=128,model_dim=2880,inter_dim=360,topk=4 | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| c42af93f71a8779c | GPT-OSS 120B | model_config | mixed_a8w4 | tokens=2048,E=128,model_dim=2880,inter_dim=360,topk=4 | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 7fd6d8e630c1b2e2 | GPT-OSS 120B | model_config | mixed_a8w4 | tokens=16384,E=128,model_dim=2880,inter_dim=360,topk=4 | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| f006ff8500772932 | synthetic-smallest-pas | synthetic | fp8 | tokens=256,E=4,model_dim=1024,inter_dim=256,topk=2 | ok | failed | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 3417b6c5516ca477 | synthetic-profiled-att | synthetic | fp8 | tokens=32,E=8,model_dim=6144,inter_dim=4096,topk=2 | incorrect | failed | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 2021925b86a88619 | synthetic-profiled-att | synthetic | fp8 | tokens=256,E=8,model_dim=6144,inter_dim=4096,topk=2 | incorrect | failed | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 25b1ea0d5ae33fc1 | synthetic-profiled-att | synthetic | fp8 | tokens=2048,E=8,model_dim=6144,inter_dim=4096,topk=2 | incorrect | failed | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
