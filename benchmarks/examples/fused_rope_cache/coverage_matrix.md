# Coverage matrix: fused_rope_cache

| shape_id | model | stage | dtype | args | flydsl | aiter | aiter_triton | aiter_ck | aiter_asm | ck | triton | gluon | hipblaslt | pytorch | profile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 00bb5689be0b4786 | Llama4 Maverick | model_config | bf16 | D=128,QH=5,KH=1,T=128 | ok | n/c | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 14ef4753a5c94f31 | Llama3 405B | model_config | bf16 | D=128,QH=16,KH=1,T=1 | ok | n/c | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 1c8b8d5bb34e16d7 | Llama3 8B | model_config | f16 | D=128,QH=4,KH=1,T=1 | ok | n/c | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 209ddde617156680 | Llama3 70B\|Qwen3-235B- | model_config | bf16 | D=128,QH=8,KH=1,T=2048 | ok | n/c | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 26e592cb04548259 | Llama3 405B | model_config | f16 | D=128,QH=16,KH=1,T=2048 | ok | n/c | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 35b821c1816611f9 | Llama4 Maverick | model_config | bf16 | D=128,QH=5,KH=1,T=2048 | ok | n/c | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 4668854b13b751e6 | Llama4 Maverick | model_config | bf16 | D=128,QH=5,KH=1,T=1 | ok | n/c | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 4f4c5e7b89449f29 | Llama3 405B | model_config | f16 | D=128,QH=16,KH=1,T=128 | ok | n/c | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 61acc47fd25a5160 | Llama3 8B | model_config | bf16 | D=128,QH=4,KH=1,T=2048 | ok | n/c | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 62146a3a815641af | GPT-OSS 120B | model_config | bf16 | D=64,QH=8,KH=1,T=2048 | ok | n/c | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 7647f7421dc90f25 | Llama3 8B | model_config | bf16 | D=128,QH=4,KH=1,T=128 | ok | n/c | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 76608634672854a7 | Llama3 8B | model_config | bf16 | D=128,QH=4,KH=1,T=1 | ok | n/c | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 7f1ebfcc08af428f | Llama3 70B\|Qwen3-235B- | model_config | bf16 | D=128,QH=8,KH=1,T=128 | ok | n/c | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 83785412075a98bc | Llama4 Maverick | model_config | f16 | D=128,QH=5,KH=1,T=128 | ok | n/c | incorrect | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 8c6d3ebe65016e09 | GPT-OSS 120B | model_config | bf16 | D=64,QH=8,KH=1,T=1 | ok | n/c | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 8d30de4b7cbe552d | Llama3 405B | model_config | bf16 | D=128,QH=16,KH=1,T=128 | ok | n/c | incorrect | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| b3972184c97e203e | Llama3 405B | model_config | f16 | D=128,QH=16,KH=1,T=1 | ok | n/c | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| bdc8fb7bca6e5be2 | Llama3 405B | model_config | bf16 | D=128,QH=16,KH=1,T=2048 | ok | n/c | incorrect | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| c1c6ff703a51d7bb | Llama3 70B\|Qwen3-235B- | model_config | bf16 | D=128,QH=8,KH=1,T=1 | ok | n/c | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| c1cc2a9b81ecfe48 | GPT-OSS 120B | model_config | f16 | D=64,QH=8,KH=1,T=1 | ok | n/c | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| c8548eee5ea580ea | Llama3 70B\|Qwen3-235B- | model_config | f16 | D=128,QH=8,KH=1,T=128 | ok | n/c | incorrect | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| d26bf61b23023023 | Llama3 70B\|Qwen3-235B- | model_config | f16 | D=128,QH=8,KH=1,T=1 | ok | n/c | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| d346fa810e1afa66 | GPT-OSS 120B | model_config | f16 | D=64,QH=8,KH=1,T=2048 | ok | n/c | incorrect | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| d34cb74800393e83 | Llama4 Maverick | model_config | f16 | D=128,QH=5,KH=1,T=1 | ok | n/c | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| da5500df6f3c49b4 | GPT-OSS 120B | model_config | bf16 | D=64,QH=8,KH=1,T=128 | ok | n/c | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| de0c3addd68726eb | Llama3 70B\|Qwen3-235B- | model_config | f16 | D=128,QH=8,KH=1,T=2048 | ok | n/c | incorrect | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| e852c8e7d0f98113 | Llama3 8B | model_config | f16 | D=128,QH=4,KH=1,T=128 | ok | n/c | incorrect | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| ec68730b29c76e63 | Llama3 8B | model_config | f16 | D=128,QH=4,KH=1,T=2048 | ok | n/c | incorrect | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| ed5e3e177f481fdb | Llama4 Maverick | model_config | f16 | D=128,QH=5,KH=1,T=2048 | ok | n/c | incorrect | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| f82d7cd9b5938dfb | GPT-OSS 120B | model_config | f16 | D=64,QH=8,KH=1,T=128 | ok | n/c | incorrect | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
