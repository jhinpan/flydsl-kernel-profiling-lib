# Coverage matrix: preshuffle_gemm

| shape_id | model | stage | dtype | args | flydsl | aiter | aiter_triton | aiter_ck | aiter_asm | ck | triton | gluon | hipblaslt | pytorch | profile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c14f67d461f273da | Qwen | model_config | fp8 | M=16,N=2560,K=2560 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 273ee68e5ed18ebc | Qwen | model_config | fp8 | M=128,N=2560,K=2560 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 193948e8adde9529 | Qwen | model_config | fp8 | M=2048,N=2560,K=2560 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 2e5d48a292eb393e | DeepSeek-V3 | model_config | fp8 | M=16,N=7168,K=7168 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 6e5bdf7cdfda5545 | DeepSeek-V3 | model_config | fp8 | M=128,N=7168,K=2048 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 229c585dffadbcb2 | DeepSeek-V3 | model_config | fp8 | M=2048,N=7168,K=7168 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| e22432840a69c2b2 | Kimi-K2 | model_config | fp8 | M=16,N=7168,K=7168 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 849ade0b90a6d494 | Kimi-K2 | model_config | fp8 | M=256,N=7168,K=2048 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 9ddfbe3552f90b1b | generic | synthetic | fp8 | M=33,N=1024,K=2048 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 0af0a20876f43df3 | generic | synthetic | fp8 | M=5120,N=2048,K=8320 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| f7cc70c962542a20 | generic | synthetic | fp8 | M=16,N=5120,K=8192 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 31b52b190fe26826 | generic | synthetic | fp8 | M=5120,N=5120,K=8320 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 09355b4eb06ddad4 | generic | synthetic | fp8 | M=9728,N=8192,K=8320 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 781f331e2bc6ee94 | generic | synthetic | fp8 | M=4096,N=8192,K=8192 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
