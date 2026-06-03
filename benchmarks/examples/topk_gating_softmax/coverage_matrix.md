# Coverage matrix: topk_gating_softmax

| shape_id | model | stage | dtype | args | flydsl | aiter | aiter_triton | aiter_ck | aiter_asm | ck | triton | gluon | hipblaslt | pytorch | profile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2d2eb5d587ba2926 | DeepSeek-R1 | model_config | bf16 | num_tokens=1,num_experts=256,topk=8 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| d9845aa89a6900b7 | DeepSeek-R1 | model_config | bf16 | num_tokens=256,num_experts=256,topk=8 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| f8b5950cdf51f5b7 | DeepSeek-R1 | model_config | bf16 | num_tokens=2048,num_experts=256,topk=8 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 65b627c0d09c4fe4 | DeepSeek-R1 | model_config | bf16 | num_tokens=16384,num_experts=256,topk=8 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 815f175ee43d0f4d | Kimi-K2 | model_config | bf16 | num_tokens=256,num_experts=384,topk=8 | unsupported | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 0a396d08cfabc226 | Kimi-K2 | model_config | bf16 | num_tokens=2048,num_experts=384,topk=8 | unsupported | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 9eaa872543ccc1f9 | Kimi-K2 | model_config | bf16 | num_tokens=16384,num_experts=384,topk=8 | unsupported | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 05d42d0272652291 | Mixtral-8x22B-class | model_config | bf16 | num_tokens=1,num_experts=128,topk=6 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 0e6ddbec4e516f08 | Mixtral-8x22B-class | model_config | bf16 | num_tokens=1024,num_experts=128,topk=6 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 9cf721ce5fd2bd4e | Mixtral-8x22B-class | model_config | fp16 | num_tokens=128,num_experts=128,topk=6 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| caa9e588ae89f9bd | Llama4-class | model_config | bf16 | num_tokens=512,num_experts=64,topk=2 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 02418a3471197492 | Llama4-class | model_config | bf16 | num_tokens=2048,num_experts=64,topk=2 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| dd4f8eb9de4b0d85 | Mixtral-8x7B | model_config | fp32 | num_tokens=256,num_experts=8,topk=2 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 1f55e462fc6f5095 | Mixtral-8x7B | model_config | bf16 | num_tokens=2048,num_experts=8,topk=2 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 3bc1d2b8c02a187c | DeepSeek-R1 | model_config | fp16 | num_tokens=2048,num_experts=256,topk=8 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 442c0f9486760c1d | DeepSeek-R1 | model_config | fp32 | num_tokens=2048,num_experts=256,topk=8 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
