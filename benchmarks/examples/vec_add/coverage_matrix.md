# Coverage matrix: vec_add

| shape_id | model | stage | dtype | args | flydsl | aiter | aiter_triton | aiter_ck | aiter_asm | ck | triton | gluon | hipblaslt | pytorch | profile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fed332456993532e | Qwen3 | model_config | fp32 | n=5242880 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 616efdaab57c250f | Qwen3 | model_config | fp32 | n=327680 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 9ad6fc155f5adeef | DeepSeek-V3 | model_config | fp32 | n=14680064 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 9eae19ca99bbcc34 | DeepSeek-V3 | model_config | fp32 | n=917504 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 894f7d58db2f0f10 | Kimi-K2 | model_config | fp32 | n=7340032 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 7df3e2cdfd1c2c10 | micro | synthetic | fp32 | n=1024 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 94e94dcdf44d0352 | micro | synthetic | fp32 | n=262144 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| dc27b39acdd0e886 | micro | synthetic | fp32 | n=4194304 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 5ca8e3e5b2df1591 | micro | synthetic | fp32 | n=10240000 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 7dd9407102edc1d5 | micro | synthetic | fp32 | n=33554432 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| eed4d812a3993809 | micro | synthetic | fp32 | n=67108864 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| fba0b64864740977 | micro | synthetic | fp32 | n=1048576 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
