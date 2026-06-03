# Coverage matrix: pa_decode

| shape_id | model | stage | dtype | args | flydsl | aiter | aiter_triton | aiter_ck | aiter_asm | ck | triton | gluon | hipblaslt | pytorch | profile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 96a8a5f2d3c44465 | test_pa normal_accurac | model_config | fp8 | hq=8,hkv=1,d=128,ctx=1027,b=3,ql=1,blk=1024,quant=per_token | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| 1cfdcc7d8cabcaac | test_pa normal_accurac | model_config | fp8 | hq=8,hkv=1,d=128,ctx=1027,b=81,ql=1,blk=1024,quant=per_token | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| 4016a3150b687f40 | test_pa normal_accurac | model_config | fp8 | hq=16,hkv=1,d=128,ctx=1027,b=3,ql=1,blk=1024,quant=per_token | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| 4588f6f9f0034a93 | test_pa normal_accurac | model_config | fp8 | hq=16,hkv=1,d=128,ctx=1027,b=81,ql=1,blk=1024,quant=per_tensor | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| 1cded9b4dd0dd789 | test_pa normal_accurac | model_config | fp8 | hq=8,hkv=1,d=128,ctx=1027,b=3,ql=2,blk=1024,quant=per_token | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| c7ad97f7e36b2ae9 | test_pa normal_accurac | model_config | fp8 | hq=8,hkv=1,d=128,ctx=1027,b=3,ql=4,blk=1024,quant=per_token | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| c9672f23d295bb74 | test_pa normal_accurac | model_config | fp8 | hq=16,hkv=1,d=128,ctx=1027,b=3,ql=4,blk=1024,quant=per_tensor | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| 1aee987fbfa698dc | test_pa normal_accurac | model_config | fp8 | hq=8,hkv=1,d=128,ctx=1027,b=81,ql=2,blk=1024,quant=per_tensor | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| 04cf095672b85d9f | Qwen-like (hidden 2560 | model_config | fp8 | hq=8,hkv=1,d=128,ctx=8192,b=128,ql=1,blk=1024,quant=per_token | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| 643f0de3575a7e01 | sliding-window model ( | model_config | fp8 | hq=16,hkv=1,d=128,ctx=8192,b=128,ql=1,blk=1024,quant=per_token | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| 5d477fe9c5490008 | sliding-window model l | model_config | fp8 | hq=16,hkv=1,d=128,ctx=8192,b=128,ql=4,blk=1024,quant=per_token | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
| b9bc25388f5ffd06 | DeepSeek/Kimi GQA deco | model_config | fp8 | hq=8,hkv=1,d=128,ctx=4096,b=16,ql=1,blk=1024,quant=per_token | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok | n/c | ok |  |
