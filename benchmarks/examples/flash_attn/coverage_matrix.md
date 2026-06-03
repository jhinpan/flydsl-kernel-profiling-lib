# Coverage matrix: flash_attn

| shape_id | model | stage | dtype | args | flydsl | aiter | aiter_triton | aiter_ck | aiter_asm | ck | triton | gluon | hipblaslt | pytorch | profile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 82dacb3b0a212b65 | DeepSeek-V3 | model_config | bf16 | B=1,S=2048,H=32,D=128,causal | ok | failed | ok | n/c | ok | n/c | n/c | n/c | n/c | ok |  |
| d640fe378edcd7f3 | DeepSeek-V3 | model_config | bf16 | B=1,S=4096,H=32,D=128,causal | ok | failed | ok | n/c | ok | n/c | n/c | n/c | n/c | ok |  |
| f3ebebabd45dabf2 | DeepSeek-V3 | model_config | bf16 | B=8,S=8192,H=32,D=128,causal | ok | failed | ok | n/c | ok | n/c | n/c | n/c | n/c | ok |  |
| 4fc4a900e5f10c2b | Kimi-K2 | model_config | bf16 | B=1,S=2048,H=16,D=128,causal | ok | failed | ok | n/c | ok | n/c | n/c | n/c | n/c | ok |  |
| efce815bf13d8e72 | Kimi-K2 | model_config | bf16 | B=16,S=8192,H=16,D=128,causal | ok | failed | ok | n/c | ok | n/c | n/c | n/c | n/c | ok |  |
| 9c84ff881813b33a | Qwen3 | model_config | bf16 | B=1,S=2048,H=8,D=128,causal | ok | failed | ok | n/c | ok | n/c | n/c | n/c | n/c | ok |  |
| 490749ede90d52e6 | Qwen3 | model_config | bf16 | B=32,S=8192,H=8,D=128,causal | ok | failed | ok | n/c | ok | n/c | n/c | n/c | n/c | ok |  |
| 9e88d4baea9ba36f | synthetic | synthetic | bf16 | B=1,S=128,H=64,D=128,causal | ok | failed | ok | n/c | ok | n/c | n/c | n/c | n/c | ok |  |
| 26b4834659d534bb | synthetic | synthetic | bf16 | B=8,S=512,H=64,D=128,causal | ok | failed | ok | n/c | ok | n/c | n/c | n/c | n/c | ok |  |
| f5c03ed6860dd1ab | synthetic | synthetic | bf16 | B=1,S=2048,H=64,D=128,causal | ok | failed | ok | n/c | ok | n/c | n/c | n/c | n/c | ok |  |
| 514ac8d2b470172a | synthetic | synthetic | bf16 | B=1,S=8192,H=64,D=128,causal | ok | failed | ok | n/c | ok | n/c | n/c | n/c | n/c | ok |  |
| 35d5a60b1896109e | synthetic | synthetic | bf16 | B=4,S=8192,H=64,D=128,causal | ok | failed | ok | n/c | ok | n/c | n/c | n/c | n/c | ok |  |
| e772c74eee918ecb | synthetic | synthetic | bf16 | B=1,S=2048,H=32,D=128,noncausal | ok | failed | ok | n/c | ok | n/c | n/c | n/c | n/c | ok |  |
| 361714a585636fa1 | synthetic | synthetic | fp16 | B=1,S=4096,H=64,D=128,causal | ok | failed | ok | n/c | unsupported | n/c | n/c | n/c | n/c | ok |  |
| 2ad2497d8929b410 | synthetic | synthetic | fp16 | B=8,S=512,H=64,D=128,noncausal | ok | failed | ok | n/c | unsupported | n/c | n/c | n/c | n/c | ok |  |
| 5ebd91db9af23055 | synthetic | synthetic | bf16 | B=1,S=1024,H=64,D=128,causal | ok | failed | ok | n/c | ok | n/c | n/c | n/c | n/c | ok |  |
