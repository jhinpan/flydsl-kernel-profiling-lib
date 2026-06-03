# Coverage matrix: blockscale_preshuffle_gemm

| shape_id | model | stage | dtype | args | flydsl | aiter | aiter_triton | aiter_ck | aiter_asm | ck | triton | gluon | hipblaslt | pytorch | profile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d0d730c9ae49af3a | deepseek-v3 | model_config | fp8 | M=1,N=2112,K=7168,out=bf16 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 94d74839ecfa9c96 | deepseek-v3 | model_config | fp8 | M=33,N=2112,K=7168,out=bf16 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 62d663e018bb2fd5 | deepseek-v3 | model_config | fp8 | M=1024,N=2112,K=7168,out=bf16 | ok | incorrect | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| d91c387b282709f4 | qwen3 | model_config | fp8 | M=16,N=2560,K=2560,out=bf16 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 8ed75c492eafde9d | qwen3 | model_config | fp8 | M=2048,N=2560,K=2560,out=bf16 | ok | incorrect | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| f0e939866b22a68c | deepseek-v3 | model_config | fp8 | M=7,N=3072,K=1536,out=bf16 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| cbfc5a67fceb2b73 | deepseek-v3 | model_config | fp8 | M=64,N=3072,K=1536,out=bf16 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 0cd8d55b558213a0 | deepseek-v3 | model_config | fp8 | M=256,N=3072,K=1536,out=fp16 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| e1a224c8c4c43113 | deepseek-v3 | model_config | fp8 | M=1024,N=3072,K=1536,out=bf16 | ok | incorrect | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 1d7ca5ee6e620373 | deepseek-v3 | model_config | fp8 | M=4096,N=7168,K=2304,out=bf16 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| fdd6453102abf114 | deepseek-v3 | model_config | fp8 | M=16,N=7168,K=2304,out=bf16 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 123853c467e2cdee | deepseek-v3 | model_config | fp8 | M=64,N=7168,K=2304,out=bf16 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| a3247dc61ac2219c | kimi-k2 | model_config | fp8 | M=16,N=7168,K=2304,out=bf16 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| d50289b0074e6137 | kimi-k2 | model_config | fp8 | M=1024,N=7168,K=2304,out=bf16 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
