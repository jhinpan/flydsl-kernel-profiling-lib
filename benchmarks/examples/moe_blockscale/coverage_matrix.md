# Coverage matrix: moe_blockscale

| shape_id | model | stage | dtype | args | flydsl | aiter | aiter_triton | aiter_ck | aiter_asm | ck | triton | gluon | hipblaslt | pytorch | profile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b460de719ae05f2c | deepseek-v3 | model_config | fp8 | tokens=32,E=256,model_dim=7168,inter_dim=2048,topk=8 | incorrect | incorrect | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 227b92812b4bbae3 | deepseek-v3 | model_config | fp8 | tokens=128,E=256,model_dim=7168,inter_dim=2048,topk=8 | incorrect | incorrect | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| d713e98a2bd70ac8 | deepseek-v3 | model_config | fp8 | tokens=2048,E=256,model_dim=7168,inter_dim=2048,topk=8 | incorrect | incorrect | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 4e016f13efe2887c | kimi-k2 | model_config | fp8 | tokens=32,E=384,model_dim=7168,inter_dim=2048,topk=8 | incorrect | incorrect | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| c54f87e855add33c | kimi-k2 | model_config | fp8 | tokens=128,E=384,model_dim=7168,inter_dim=2048,topk=8 | incorrect | incorrect | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 1cf19e9316e20356 | kimi-k2 | model_config | fp8 | tokens=4096,E=384,model_dim=7168,inter_dim=2048,topk=8 | incorrect | incorrect | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| c115cec4c5728c9f | deepseek-v3 | model_config | fp8 | tokens=16,E=8,model_dim=7168,inter_dim=256,topk=2 | incorrect | incorrect | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 64617ad5198d0546 | deepseek-v3 | model_config | fp8 | tokens=32,E=8,model_dim=7168,inter_dim=256,topk=2 | incorrect | incorrect | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 57af6c5af2352cbf | deepseek-v3 | model_config | fp8 | tokens=256,E=256,model_dim=7168,inter_dim=256,topk=8 | incorrect | incorrect | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| a5d3bd10c9e71288 | kimi-k2 | model_config | fp8 | tokens=64,E=384,model_dim=7168,inter_dim=256,topk=8 | incorrect | incorrect | n/c | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 7a3ebe51210ffa29 | deepseek-v3 | model_config | fp8 | tokens=1024,E=256,model_dim=7168,inter_dim=2048,topk=8 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| f2d47f12b81acc6f | kimi-k2 | model_config | fp8 | tokens=512,E=384,model_dim=7168,inter_dim=2048,topk=8 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
