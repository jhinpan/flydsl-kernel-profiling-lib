# Coverage matrix: moe_reduce

| shape_id | model | stage | dtype | args | flydsl | aiter | aiter_triton | aiter_ck | aiter_asm | ck | triton | gluon | hipblaslt | pytorch | profile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 063793830f351f49 | deepseek-v3 | model_config | bf16 | tokens=1,topk=8,model_dim=7168 | ok | n/c | n/c | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 56b3359e032ecc1f | deepseek-v3 | model_config | bf16 | tokens=5,topk=8,model_dim=7168 | ok | n/c | n/c | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 3924162f55e276e0 | deepseek-v3 | model_config | bf16 | tokens=65,topk=8,model_dim=7168 | ok | n/c | n/c | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| c4b5d10f09b96ebc | deepseek-v3 | model_config | bf16 | tokens=32769,topk=8,model_dim=7168 | ok | n/c | n/c | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 5dc6f2b3a647fedf | kimi-k2 | model_config | bf16 | tokens=1,topk=8,model_dim=7168 | ok | n/c | n/c | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 2112d067e59ddf58 | kimi-k2 | model_config | bf16 | tokens=65,topk=8,model_dim=7168 | ok | n/c | n/c | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 6662fe54d5168824 | kimi-k2 | model_config | bf16 | tokens=16384,topk=8,model_dim=7168 | ok | n/c | n/c | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| b5dd612339da787d | qwen3-moe | model_config | bf16 | tokens=1,topk=8,model_dim=2560 | ok | n/c | n/c | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 9273cf88983b9bbb | qwen3-moe | model_config | bf16 | tokens=65,topk=8,model_dim=2560 | ok | n/c | n/c | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 9f390b91920815dd | qwen3-moe | model_config | bf16 | tokens=16384,topk=8,model_dim=2560 | ok | n/c | n/c | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 394f2a520eea1023 | ep-k6 | model_config | f16 | tokens=5,topk=6,model_dim=5120 | ok | n/c | n/c | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| f60be19bbad5b7cc | ep-k6 | model_config | f16 | tokens=65,topk=6,model_dim=5120 | ok | n/c | n/c | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| b58aef58c49e53e2 | ep-k6 | model_config | f16 | tokens=16384,topk=6,model_dim=5120 | ok | n/c | n/c | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 791ebddaea875060 | deepseek-v3 | model_config | f16 | tokens=129,topk=8,model_dim=7168,mask | ok | n/c | n/c | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 6df0462d66689bea | ep-k6 | model_config | f16 | tokens=129,topk=6,model_dim=5120,mask | ok | n/c | n/c | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| eda71c12d1c15757 | deepseek-v3 | model_config | f32 | tokens=5,topk=8,model_dim=7168 | ok | n/c | n/c | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
