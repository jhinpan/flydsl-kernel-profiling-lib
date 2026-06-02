# Coverage matrix: quant

| shape_id | model | stage | dtype | args | flydsl | aiter | aiter_triton | aiter_ck | aiter_asm | ck | triton | gluon | hipblaslt | pytorch | profile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 81e851eeb1c6bdf4 | Qwen3 | model_config | fp16 | M=2048,N=2560 | ok | ok | ok | n/c | n/c | n/c | failed | n/c | n/c | ok |  |
| 83d823297978d600 | Qwen3 | model_config | fp16 | M=128,N=2560 | ok | ok | ok | n/c | n/c | n/c | failed | n/c | n/c | ok |  |
| 422f86ffcc8b8536 | flydsl_test | model_config | fp16 | M=2048,N=4096 | ok | ok | ok | n/c | n/c | n/c | failed | n/c | n/c | ok |  |
| 0706e2d15289d4a9 | stress | model_config | fp16 | M=32768,N=4096 | ok | ok | ok | n/c | n/c | n/c | failed | n/c | n/c | ok |  |
| 63b9bff138d5537d | Qwen3 | model_config | fp16 | M=512,N=6144 | ok | ok | ok | n/c | n/c | n/c | failed | n/c | n/c | ok |  |
| 3b280cfc937746bc | DeepSeek-V3 | model_config | fp16 | M=4096,N=7168 | ok | ok | ok | n/c | n/c | n/c | failed | n/c | n/c | ok |  |
| 700e93b0c88b1177 | DeepSeek-V3 | model_config | fp16 | M=256,N=7168 | ok | ok | ok | n/c | n/c | n/c | failed | n/c | n/c | ok |  |
| caa0d75521884b2f | Kimi-K2 | model_config | fp16 | M=16384,N=7168 | ok | ok | ok | n/c | n/c | n/c | failed | n/c | n/c | ok |  |
| 6233f9dd982dfa6c | flydsl_test_default | model_config | fp16 | M=4096,N=8192 | ok | ok | ok | n/c | n/c | n/c | failed | n/c | n/c | ok |  |
| 1cfef37cad1fa0d9 | flydsl_test | model_config | fp16 | M=8192,N=8192 | ok | ok | ok | n/c | n/c | n/c | failed | n/c | n/c | ok |  |
| 5fc92805404f0aac | stress | model_config | fp16 | M=64,N=8192 | ok | ok | ok | n/c | n/c | n/c | failed | n/c | n/c | ok |  |
| 7ab0c2fce01f3dfa | DeepSeek-V3 | model_config | fp16 | M=1024,N=16384 | ok | ok | ok | n/c | n/c | n/c | failed | n/c | n/c | ok |  |
