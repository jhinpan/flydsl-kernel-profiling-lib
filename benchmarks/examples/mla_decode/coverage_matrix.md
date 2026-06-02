# Coverage matrix: mla_decode

| shape_id | model | stage | dtype | args | flydsl | aiter | aiter_triton | aiter_ck | aiter_asm | ck | triton | gluon | hipblaslt | pytorch | profile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1d4176b58df3c3b5 | DeepSeek-R1 | model_config | fp8 | batch=1,ctx_len=63,decode_qlen=1 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| e47c0abcd26502d2 | DeepSeek-R1 | model_config | fp8 | batch=1,ctx_len=64,decode_qlen=1 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 642d4c6255e207d4 | DeepSeek-R1 | model_config | fp8 | batch=1,ctx_len=128,decode_qlen=1 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 5f5c1a4648d1ee52 | DeepSeek-R1 | model_config | fp8 | batch=4,ctx_len=2048,decode_qlen=1 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| b5ca7432e9fb2a16 | DeepSeek-R1 | model_config | fp8 | batch=33,ctx_len=2333,decode_qlen=1 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| e28d150e0f346126 | DeepSeek-R1 | model_config | fp8 | batch=32,ctx_len=8192,decode_qlen=1 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| e6505b5f29af1863 | DeepSeek-V3 | model_config | fp8 | batch=1,ctx_len=1024,decode_qlen=1 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 28dbca9ebcf81171 | DeepSeek-V3 | model_config | fp8 | batch=8,ctx_len=4096,decode_qlen=1 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| ecbb09e446d8f34f | DeepSeek-V3 | model_config | fp8 | batch=16,ctx_len=4096,decode_qlen=1 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| c97f0c6207f40b60 | DeepSeek-V3 | model_config | fp8 | batch=64,ctx_len=2048,decode_qlen=1 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| d4221f511140c391 | DeepSeek-V3 | model_config | fp8 | batch=128,ctx_len=1024,decode_qlen=1 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 4eeca5a1faf1ce32 | Kimi-K2 | model_config | fp8 | batch=2,ctx_len=16384,decode_qlen=1 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 8f06fe7b964fe07e | Kimi-K2 | model_config | fp8 | batch=8,ctx_len=8192,decode_qlen=1 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 341994dfacf2006c | Kimi-K2 | model_config | fp8 | batch=1,ctx_len=32768,decode_qlen=1 | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
