# Coverage matrix: fp8_gemm_rowscale

| shape_id | model | stage | dtype | args | flydsl | aiter | aiter_triton | aiter_ck | aiter_asm | ck | triton | gluon | hipblaslt | pytorch | profile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 99c78a6424d841e3 | deepseek-v3 | model_config | fp8_e4m3 | M=512,N=2112,K=7168 | ok | ok | ok | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| 030af87ed4315638 | deepseek-v3 | model_config | fp8_e4m3 | M=512,N=2112,K=7168,8wave | unsupported | ok | ok | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| 697c92cc550e5ddc | qwen3-32b | model_config | fp8_e4m3 | M=1,N=2560,K=2560 | ok | ok | ok | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| 242a477dbdcff470 | qwen3-32b | model_config | fp8_e4m3 | M=128,N=2560,K=2560 | ok | ok | ok | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| ee8fa6dabd7827fe | qwen3-32b | model_config | fp8_e4m3 | M=4096,N=2560,K=2560 | ok | ok | ok | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| beba39eed8781b41 | kimi-k2 | model_config | fp8_e4m3 | M=5120,N=5120,K=8320 | ok | ok | ok | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| 12dcc89561dbac4c | kimi-k2 | model_config | fp8_e4m3 | M=5120,N=5120,K=8320,8wave | ok | ok | ok | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| 731506f3c22dc9d1 | kimi-k2 | model_config | fp8_e4m3 | M=16,N=5120,K=8320 | ok | ok | ok | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| 4f9b936ff617c4b1 | deepseek-v3 | model_config | fp8_e4m3 | M=256,N=7168,K=7168 | ok | ok | ok | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| cfe55cca9c92e08e | deepseek-v3 | model_config | fp8_e4m3 | M=4096,N=7168,K=7168 | ok | ok | ok | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| 1d6e174ac4a21d2e | deepseek-v3 | model_config | fp8_e4m3 | M=2048,N=7168,K=2048 | ok | ok | ok | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| a32fe385bcf9b632 | deepseek-v3 | model_config | fp8_e4m3 | M=8192,N=7168,K=7168,preshuffle_b | ok | ok | ok | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| bf29462bfe8fe74b | synthetic | synthetic | fp8_e4m3 | M=8192,N=8192,K=8192 | ok | ok | ok | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| 01f8db33c46386cc | synthetic | synthetic | fp8_e4m3 | M=9728,N=8192,K=8320 | ok | ok | ok | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
