# Coverage matrix: gemm

| shape_id | model | stage | dtype | args | flydsl | aiter | aiter_triton | aiter_ck | aiter_asm | ck | triton | gluon | hipblaslt | pytorch | profile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2a239e8c7cb60a18 | Qwen3-235B-A22B | model_config | bf16 | M=1,N=128,K=4096 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| 4cbab47416333528 | GPT-OSS 120B | model_config | bf16 | M=1,N=128,K=2880 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| 987b3009773ac115 | Llama4 Maverick | model_config | bf16 | M=1,N=128,K=5120 | ok | incorrect | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| 419c1180f1fd2f73 | Qwen3-235B-A22B | model_config | bf16 | M=32,N=128,K=4096 | ok | incorrect | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| cdd29bc0a59c9864 | Llama4 Maverick | model_config | bf16 | M=32,N=128,K=5120 | ok | incorrect | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| d4209ebef4a1c6dd | GPT-OSS 120B | model_config | bf16 | M=32,N=128,K=2880 | ok | incorrect | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| 0297901a84887d0d | GPT-OSS 120B | model_config | bf16 | M=256,N=128,K=2880 | ok | incorrect | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| 28043498098d1bfe | Llama4 Maverick | model_config | bf16 | M=256,N=128,K=5120 | ok | incorrect | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| df627ce32ab44717 | Qwen3-235B-A22B | model_config | bf16 | M=256,N=128,K=4096 | ok | incorrect | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| bf21aa5e3ac004fb | Llama4 Maverick | model_config | bf16 | M=2048,N=128,K=5120 | ok | incorrect | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| d5fb0d346572eea7 | GPT-OSS 120B | model_config | bf16 | M=2048,N=128,K=2880 | ok | incorrect | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| f1f8490da3374966 | Qwen3-235B-A22B | model_config | bf16 | M=2048,N=128,K=4096 | ok | incorrect | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| 45cbcfd18820e3c9 | Qwen3-235B-A22B | model_config | bf16 | M=16384,N=128,K=4096 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| ea893d1fc5a2b40f | Llama4 Maverick | model_config | bf16 | M=16384,N=128,K=5120 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| f59d7962fa5dd3db | GPT-OSS 120B | model_config | bf16 | M=16384,N=128,K=2880 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| 55fdf3dcb9d71b2a | DeepSeek-R1 | model_config | bf16 | M=1,N=256,K=7168 | ok | incorrect | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| df905e6d25fffdd7 | DeepSeek-R1 | model_config | bf16 | M=32,N=256,K=7168 | ok | incorrect | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| 5c34bb2994714f4a | DeepSeek-R1 | model_config | bf16 | M=256,N=256,K=7168 | ok | incorrect | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| e1fcce2b4e143067 | DeepSeek-R1 | model_config | bf16 | M=2048,N=256,K=7168 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| 4050f1cf89fdd36e | DeepSeek-R1 | model_config | bf16 | M=16384,N=256,K=7168 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| 9f622e958eb2bf1a | DeepSeek-R1 | model_config | fp4 | M=1,N=512,K=7168 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| bae3026147bc8178 | DeepSeek-R1 | model_config | fp4 | M=32,N=512,K=7168 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 68cbc2041880c05c | DeepSeek-R1 | model_config | fp4 | M=256,N=512,K=7168 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| ee5426eadcdb8011 | DeepSeek-R1 | model_config | fp4 | M=2048,N=512,K=7168 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 9c38a591e32759ef | DeepSeek-R1 | model_config | fp4 | M=16384,N=512,K=7168 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 108a479eac6483ff | DeepSeek-R1 | model_config | fp8 | M=1,N=512,K=7168 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| cdd6d1000d2e6a6a | DeepSeek-R1 | model_config | fp8 | M=32,N=512,K=7168 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 52afd51436fc9c5a | DeepSeek-R1 | model_config | fp8 | M=256,N=512,K=7168 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 8a353290457a38f0 | DeepSeek-R1 | model_config | fp8 | M=2048,N=512,K=7168 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 5df6b8fb8bce3fa7 | DeepSeek-R1 | model_config | fp8 | M=16384,N=512,K=7168 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| bb103450134e438c | GPT-OSS 120B | model_config | bf16 | M=1,N=640,K=2880 | ok | incorrect | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| 013bda0bc2a519e5 | GPT-OSS 120B | model_config | bf16 | M=32,N=640,K=2880 | ok | incorrect | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| a9b21ba9ed22540d | GPT-OSS 120B | model_config | bf16 | M=256,N=640,K=2880 | ok | incorrect | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| 6dc6d860d8d12f39 | GPT-OSS 120B | model_config | bf16 | M=2048,N=640,K=2880 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| bbda08cb1f4d72c4 | GPT-OSS 120B | model_config | bf16 | M=16384,N=640,K=2880 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| b9c0c4bd506c7786 | Llama3 8B | model_config | fp4 | M=1,N=768,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 3e1856e2706590f8 | Llama3 8B | model_config | fp4 | M=32,N=768,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 51839834812727b3 | Llama3 8B | model_config | fp4 | M=256,N=768,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 80ccd469426d6cb4 | Llama3 8B | model_config | fp4 | M=2048,N=768,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 4f52ab6b46241a20 | Llama3 8B | model_config | fp4 | M=16384,N=768,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 05d117c5ec8c5a40 | Llama3 8B | model_config | int8 | M=1,N=768,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 202542c238b68057 | Llama3 8B | model_config | int8 | M=32,N=768,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 7919662ed2095dd2 | Llama3 8B | model_config | int8 | M=256,N=768,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| c283d61ba33d8d3d | Llama3 8B | model_config | int8 | M=2048,N=768,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| c51f72315d40d5ee | Llama3 8B | model_config | int8 | M=16384,N=768,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| a2f5648a9e896a4a | Llama4 Maverick | model_config | int8 | M=1,N=896,K=5120 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 7d63118c45aafea0 | Llama4 Maverick | model_config | int8 | M=32,N=896,K=5120 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 4068b9fb8719c6c6 | Llama4 Maverick | model_config | int8 | M=256,N=896,K=5120 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 662e663a53f8cd23 | Llama4 Maverick | model_config | int8 | M=2048,N=896,K=5120 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 460621e8d9a1a5c2 | Llama4 Maverick | model_config | int8 | M=16384,N=896,K=5120 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| f8e7909a1da7c573 | Qwen3-235B-A22B | model_config | int8 | M=1,N=1152,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 171e3ce35405645c | Qwen3-235B-A22B | model_config | int8 | M=32,N=1152,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| c61a70626aa05ffe | Qwen3-235B-A22B | model_config | int8 | M=256,N=1152,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 14b68f1d5f616e96 | Qwen3-235B-A22B | model_config | int8 | M=2048,N=1152,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| bec5e92c9f21de00 | Qwen3-235B-A22B | model_config | int8 | M=16384,N=1152,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 878600a0f98b967a | Llama3 70B | model_config | fp4 | M=1,N=1280,K=8192 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 6f9428cb6ded9b0b | Llama3 70B | model_config | fp4 | M=32,N=1280,K=8192 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| d6185e5652e3b2fc | Llama3 70B | model_config | fp4 | M=256,N=1280,K=8192 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| ff9150940309702f | Llama3 70B | model_config | fp4 | M=2048,N=1280,K=8192 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 0277be6e333c0bd4 | Llama3 70B | model_config | fp4 | M=16384,N=1280,K=8192 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| c2d859ac25b4b112 | Llama3 70B | model_config | int8 | M=1,N=1280,K=8192 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| e2cf5421cb791b81 | Llama3 70B | model_config | int8 | M=32,N=1280,K=8192 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 40b156e793a4dd0f | Llama3 70B | model_config | int8 | M=256,N=1280,K=8192 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 30415feab4657cd7 | Llama3 70B | model_config | int8 | M=2048,N=1280,K=8192 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 81efcbccba31dd68 | Llama3 70B | model_config | int8 | M=16384,N=1280,K=8192 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 05da07cbdc807e96 | Llama4 Maverick | model_config | int8 | M=1,N=2048,K=5120 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 6210066e811325c8 | Llama4 Maverick | model_config | int8 | M=32,N=2048,K=5120 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| c76c4e199fd28faf | Llama4 Maverick | model_config | int8 | M=256,N=2048,K=5120 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 3311f294dbc963db | Llama4 Maverick | model_config | int8 | M=2048,N=2048,K=5120 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| be8fff83baafaec1 | Llama4 Maverick | model_config | int8 | M=16384,N=2048,K=5120 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| f25bbc38df83eab6 | DeepSeek-R1 | model_config | fp4 | M=1,N=2112,K=7168 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 09cc109067f4ba8e | DeepSeek-R1 | model_config | fp4 | M=32,N=2112,K=7168 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| a23067e409a1e5ca | DeepSeek-R1 | model_config | fp4 | M=256,N=2112,K=7168 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 8d1608be1e0a9ca0 | DeepSeek-R1 | model_config | fp4 | M=2048,N=2112,K=7168 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 127703be17930bf8 | DeepSeek-R1 | model_config | fp4 | M=16384,N=2112,K=7168 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 35969198bf3c8cdf | DeepSeek-R1 | model_config | fp8 | M=1,N=2112,K=7168 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 288045e2faf77147 | DeepSeek-R1 | model_config | fp8 | M=32,N=2112,K=7168 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 16ccc91befb51977 | DeepSeek-R1 | model_config | fp8 | M=256,N=2112,K=7168 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| e3fae06b1d7860b8 | DeepSeek-R1 | model_config | fp8 | M=2048,N=2112,K=7168 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 4b79444b0ba741cc | DeepSeek-R1 | model_config | fp8 | M=16384,N=2112,K=7168 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 4c17246e916c81d6 | Llama3 405B | model_config | fp4 | M=1,N=2304,K=16384 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 7c59c6758fdb7a44 | Llama3 405B | model_config | fp4 | M=32,N=2304,K=16384 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 69eee0498ebc4ca2 | Llama3 405B | model_config | fp4 | M=256,N=2304,K=16384 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 28cc8c9d1ab846b2 | Llama3 405B | model_config | fp4 | M=2048,N=2304,K=16384 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 7a6d666e63bc7901 | Llama3 405B | model_config | fp4 | M=16384,N=2304,K=16384 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 388bef125497edbc | Llama3 405B | model_config | int8 | M=1,N=2304,K=16384 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| ea1481a523cdcca4 | Llama3 405B | model_config | int8 | M=32,N=2304,K=16384 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| b5b82876ecd52ae5 | Llama3 405B | model_config | int8 | M=256,N=2304,K=16384 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| ac1a0784c47e992f | Llama3 405B | model_config | int8 | M=2048,N=2304,K=16384 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 93911aed1546bbfe | Llama3 405B | model_config | int8 | M=16384,N=2304,K=16384 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 3787bb45c64a1f1b | GPT-OSS 120B | model_config | bf16 | M=1,N=2880,K=512 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| 750b6061085e17d7 | GPT-OSS 120B | model_config | bf16 | M=32,N=2880,K=512 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| a89dd3d2d18435ef | GPT-OSS 120B | model_config | bf16 | M=256,N=2880,K=512 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| 66109d32bebad3d0 | GPT-OSS 120B | model_config | bf16 | M=2048,N=2880,K=512 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| 7ad98039c1a92ea9 | GPT-OSS 120B | model_config | bf16 | M=16384,N=2880,K=512 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | ok | ok |  |
| a146f4859c1246cc | DeepSeek-R1 | model_config | fp4 | M=1,N=3072,K=1536 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 59234f0162d46d81 | DeepSeek-R1 | model_config | fp4 | M=32,N=3072,K=1536 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| ccaa2ade8ead3678 | DeepSeek-R1 | model_config | fp4 | M=256,N=3072,K=1536 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 6e4d0e7bc9016653 | DeepSeek-R1 | model_config | fp4 | M=2048,N=3072,K=1536 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 5e95472fc48b92c4 | DeepSeek-R1 | model_config | fp4 | M=16384,N=3072,K=1536 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| edfa2d3873c3bc5a | DeepSeek-R1 | model_config | fp8 | M=1,N=3072,K=1536 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 936a7d056f651d0b | DeepSeek-R1 | model_config | fp8 | M=32,N=3072,K=1536 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| a7b53952808b0e43 | DeepSeek-R1 | model_config | fp8 | M=256,N=3072,K=1536 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 26c1245031419cb8 | DeepSeek-R1 | model_config | fp8 | M=2048,N=3072,K=1536 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 90fe1f91d8c4703f | DeepSeek-R1 | model_config | fp8 | M=16384,N=3072,K=1536 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| aebb5689192a928f | Llama3 8B | model_config | fp4 | M=1,N=3584,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 4835c06d203a00a7 | Llama3 8B | model_config | fp4 | M=32,N=3584,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 9020af8234c1393a | Llama3 8B | model_config | fp4 | M=256,N=3584,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| b49979e8569c576f | Llama3 8B | model_config | fp4 | M=2048,N=3584,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 54cd42f5bc2e5c14 | Llama3 8B | model_config | fp4 | M=16384,N=3584,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 1f539488fbdeb7ce | Llama3 8B | model_config | int8 | M=1,N=3584,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| a554462f0a3ae3be | Llama3 8B | model_config | int8 | M=32,N=3584,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 42b804e6fef54b06 | Llama3 8B | model_config | int8 | M=256,N=3584,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| be573a1174a46796 | Llama3 8B | model_config | int8 | M=2048,N=3584,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 541dc813e45f30cd | Llama3 8B | model_config | int8 | M=16384,N=3584,K=4096 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 1636ba56ed452a56 | Llama3 8B | model_config | fp4 | M=1,N=4096,K=1792 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 4184e4a3be872744 | DeepSeek-R1\|Llama3 8B | model_config | fp4 | M=1,N=4096,K=512 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 96d25addd8ce427e | DeepSeek-R1\|Llama3 8B | model_config | fp4 | M=32,N=4096,K=512 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| a8405452409d2a29 | Llama3 8B | model_config | fp4 | M=32,N=4096,K=1792 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| be930eac5a0598c3 | DeepSeek-R1\|Llama3 8B | model_config | fp4 | M=256,N=4096,K=512 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| edd65183e1af62fb | Llama3 8B | model_config | fp4 | M=256,N=4096,K=1792 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 529b37b15e3836e4 | Llama3 8B | model_config | fp4 | M=2048,N=4096,K=1792 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| e3ce6a85e58773bc | DeepSeek-R1\|Llama3 8B | model_config | fp4 | M=2048,N=4096,K=512 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 0081ec0528ccd762 | Llama3 8B | model_config | fp4 | M=16384,N=4096,K=1792 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 654e686c6feb5c8e | DeepSeek-R1\|Llama3 8B | model_config | fp4 | M=16384,N=4096,K=512 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 9b18bdf8f3a561c2 | DeepSeek-R1 | model_config | fp8 | M=1,N=4096,K=512 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 71b1d6d46e27eb55 | DeepSeek-R1 | model_config | fp8 | M=32,N=4096,K=512 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 17aaebe940a6d56f | DeepSeek-R1 | model_config | fp8 | M=256,N=4096,K=512 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| c0616c72c0542e6c | DeepSeek-R1 | model_config | fp8 | M=2048,N=4096,K=512 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 05901d0ee31339ae | DeepSeek-R1 | model_config | fp8 | M=16384,N=4096,K=512 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 2ae86e0695655660 | Llama4 Maverick | model_config | int8 | M=1,N=4096,K=5120 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 3f01e3328bfcdbfe | Llama3 8B | model_config | int8 | M=1,N=4096,K=512 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 9c69ca979120ca38 | Qwen3-235B-A22B | model_config | int8 | M=1,N=4096,K=1024 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| f7f5c062f58752b4 | Llama3 8B | model_config | int8 | M=1,N=4096,K=1792 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 04c1cf5cb98b9032 | Llama4 Maverick | model_config | int8 | M=32,N=4096,K=5120 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 4c39b1f881d0b7c7 | Llama3 8B | model_config | int8 | M=32,N=4096,K=1792 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 69afdfda735add20 | Llama3 8B | model_config | int8 | M=32,N=4096,K=512 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| f6a7fd9460d6c755 | Qwen3-235B-A22B | model_config | int8 | M=32,N=4096,K=1024 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 03117f215ed9171a | Llama3 8B | model_config | int8 | M=256,N=4096,K=512 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 31f910b683aaeafb | Llama4 Maverick | model_config | int8 | M=256,N=4096,K=5120 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| dd8ee8500416a1c2 | Qwen3-235B-A22B | model_config | int8 | M=256,N=4096,K=1024 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| fc194a331d37cdfc | Llama3 8B | model_config | int8 | M=256,N=4096,K=1792 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 0290292d034016ab | Llama4 Maverick | model_config | int8 | M=2048,N=4096,K=5120 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 19d5942da0acae21 | Llama3 8B | model_config | int8 | M=2048,N=4096,K=512 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 1e8fbc39c132ac51 | Llama3 8B | model_config | int8 | M=2048,N=4096,K=1792 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| b894b3846b75ddcf | Qwen3-235B-A22B | model_config | int8 | M=2048,N=4096,K=1024 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 3b572dbbc7bab937 | Qwen3-235B-A22B | model_config | int8 | M=16384,N=4096,K=1024 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 9bd3b4565006702e | Llama3 8B | model_config | int8 | M=16384,N=4096,K=512 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| b1a5a9073d3084c0 | Llama3 8B | model_config | int8 | M=16384,N=4096,K=1792 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| cb46ee1afd3ae23a | Llama4 Maverick | model_config | int8 | M=16384,N=4096,K=5120 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 274521d9b4bae9f5 | DeepSeek-R1 | model_config | fp4 | M=1,N=4608,K=7168 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 5ad6c26cfaf401ba | DeepSeek-R1 | model_config | fp4 | M=32,N=4608,K=7168 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 14ccd011db95d6c1 | DeepSeek-R1 | model_config | fp4 | M=256,N=4608,K=7168 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 08a0d66e1b16c6a4 | DeepSeek-R1 | model_config | fp4 | M=2048,N=4608,K=7168 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 4060244c1e70141e | DeepSeek-R1 | model_config | fp4 | M=16384,N=4608,K=7168 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 8ee2ac2730e8e370 | DeepSeek-R1 | model_config | fp8 | M=1,N=4608,K=7168 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| b4d7791177123082 | DeepSeek-R1 | model_config | fp8 | M=32,N=4608,K=7168 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| d789e3e3bd9eaace | DeepSeek-R1 | model_config | fp8 | M=256,N=4608,K=7168 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| ba66230142a88eb0 | DeepSeek-R1 | model_config | fp8 | M=2048,N=4608,K=7168 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 20eca4e6e67650c2 | DeepSeek-R1 | model_config | fp8 | M=16384,N=4608,K=7168 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 279e2bcfd4f525e7 | Llama4 Maverick | model_config | int8 | M=1,N=5120,K=2048 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 9390be9603d7c983 | Llama4 Maverick | model_config | int8 | M=1,N=5120,K=1024 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| c739e1749430ef2b | Llama4 Maverick | model_config | int8 | M=1,N=5120,K=640 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 071e9ca58e3b6490 | Llama4 Maverick | model_config | int8 | M=32,N=5120,K=1024 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 7ff66a46ac0ebdb2 | Llama4 Maverick | model_config | int8 | M=32,N=5120,K=640 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| fa6974a3e0a58daa | Llama4 Maverick | model_config | int8 | M=32,N=5120,K=2048 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 3810ed6d3b265fd9 | Llama4 Maverick | model_config | int8 | M=256,N=5120,K=2048 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| a0d2f70b6deed9e0 | Llama4 Maverick | model_config | int8 | M=256,N=5120,K=1024 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| b54dde58fb87f5b1 | Llama4 Maverick | model_config | int8 | M=256,N=5120,K=640 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 0f1173c28836eb1d | Llama4 Maverick | model_config | int8 | M=2048,N=5120,K=2048 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 6e5434af8b6e2100 | Llama4 Maverick | model_config | int8 | M=2048,N=5120,K=1024 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 8106f4ab55aa45c3 | Llama4 Maverick | model_config | int8 | M=2048,N=5120,K=640 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 224bee7ee15bb75a | Llama4 Maverick | model_config | int8 | M=16384,N=5120,K=1024 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 31671b17d1060aa7 | Llama4 Maverick | model_config | int8 | M=16384,N=5120,K=2048 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| ed68c6bffe923792 | Llama4 Maverick | model_config | int8 | M=16384,N=5120,K=640 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 26ae8891f0fabc12 | DeepSeek-R1 | model_config | fp4 | M=1,N=7168,K=256 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 453864b9337ca985 | Llama3 70B | model_config | fp4 | M=1,N=7168,K=8192 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| e255f09b03904883 | DeepSeek-R1 | model_config | fp4 | M=1,N=7168,K=2304 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| e513d7b9a278f002 | DeepSeek-R1 | model_config | fp4 | M=1,N=7168,K=2048 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 61f922be1d9562cc | Llama3 70B | model_config | fp4 | M=32,N=7168,K=8192 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| a002824029842412 | DeepSeek-R1 | model_config | fp4 | M=32,N=7168,K=256 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| a0ba994ec627dd30 | DeepSeek-R1 | model_config | fp4 | M=32,N=7168,K=2048 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| f75e8426a2704425 | DeepSeek-R1 | model_config | fp4 | M=32,N=7168,K=2304 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 40f2ed6aab8a0b98 | DeepSeek-R1 | model_config | fp4 | M=256,N=7168,K=2048 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 9e1075a3e44f9ff0 | Llama3 70B | model_config | fp4 | M=256,N=7168,K=8192 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| a554cedf965408f2 | DeepSeek-R1 | model_config | fp4 | M=256,N=7168,K=2304 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| d93a1a6137e2e106 | DeepSeek-R1 | model_config | fp4 | M=256,N=7168,K=256 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 1b16fce5266b4dac | DeepSeek-R1 | model_config | fp4 | M=2048,N=7168,K=2304 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 3460aaa565d6d06a | DeepSeek-R1 | model_config | fp4 | M=2048,N=7168,K=2048 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 4781200dc4b17e44 | Llama3 70B | model_config | fp4 | M=2048,N=7168,K=8192 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 8a047aa46e62dca1 | DeepSeek-R1 | model_config | fp4 | M=2048,N=7168,K=256 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 2a4b4dcde71d8765 | DeepSeek-R1 | model_config | fp4 | M=16384,N=7168,K=256 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 45b9354169dbeb05 | DeepSeek-R1 | model_config | fp4 | M=16384,N=7168,K=2304 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 7d3845403f72555b | Llama3 70B | model_config | fp4 | M=16384,N=7168,K=8192 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| b509eae53bada20b | DeepSeek-R1 | model_config | fp4 | M=16384,N=7168,K=2048 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 032fda58dd525ef2 | DeepSeek-R1 | model_config | fp8 | M=1,N=7168,K=2048 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 0e6d07405b926221 | DeepSeek-R1 | model_config | fp8 | M=1,N=7168,K=256 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| cffd6bebfcbc6f7b | DeepSeek-R1 | model_config | fp8 | M=1,N=7168,K=2304 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 0e2c167e4cf577bf | DeepSeek-R1 | model_config | fp8 | M=32,N=7168,K=2048 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 5b84770781fa1a1b | DeepSeek-R1 | model_config | fp8 | M=32,N=7168,K=2304 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 7ce9cdbbb565c763 | DeepSeek-R1 | model_config | fp8 | M=32,N=7168,K=256 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| b065f7d343dc7763 | DeepSeek-R1 | model_config | fp8 | M=256,N=7168,K=2048 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| b7f164cd42fc785b | DeepSeek-R1 | model_config | fp8 | M=256,N=7168,K=2304 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| cb67f95dc61ee608 | DeepSeek-R1 | model_config | fp8 | M=256,N=7168,K=256 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 0bc6767b4111665c | DeepSeek-R1 | model_config | fp8 | M=2048,N=7168,K=256 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 5e177d7f42435fe2 | DeepSeek-R1 | model_config | fp8 | M=2048,N=7168,K=2048 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 6f77a0e943f66806 | DeepSeek-R1 | model_config | fp8 | M=2048,N=7168,K=2304 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| 8cfcd1ab1b09b5b2 | DeepSeek-R1 | model_config | fp8 | M=16384,N=7168,K=2304 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| e2880f125d20db37 | DeepSeek-R1 | model_config | fp8 | M=16384,N=7168,K=2048 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| e462dba2dc9a719e | DeepSeek-R1 | model_config | fp8 | M=16384,N=7168,K=256 | unsupported | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | unsupported | unsupported |  |
| bfe9dc5d99965580 | Llama3 70B | model_config | int8 | M=1,N=7168,K=8192 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 49acc553048c9944 | Llama3 70B | model_config | int8 | M=32,N=7168,K=8192 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| e32d332bc6c36746 | Llama3 70B | model_config | int8 | M=256,N=7168,K=8192 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 5de9dffbfdb3dc0e | Llama3 70B | model_config | int8 | M=2048,N=7168,K=8192 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| e1275b5452c445d3 | Llama3 70B | model_config | int8 | M=16384,N=7168,K=8192 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 647603acebf5e5fb | Llama3 70B | model_config | fp4 | M=1,N=8192,K=3584 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 9305835c44a6a354 | Llama3 70B | model_config | fp4 | M=1,N=8192,K=1024 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 7f9297ebc0fb18d6 | Llama3 70B | model_config | fp4 | M=32,N=8192,K=1024 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| db1d5db475d5a025 | Llama3 70B | model_config | fp4 | M=32,N=8192,K=3584 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 9065d155735c404c | Llama3 70B | model_config | fp4 | M=256,N=8192,K=3584 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| a8293db450769146 | Llama3 70B | model_config | fp4 | M=256,N=8192,K=1024 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 234cdb1153990366 | Llama3 70B | model_config | fp4 | M=2048,N=8192,K=3584 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 7a517041ad8dbc6d | Llama3 70B | model_config | fp4 | M=2048,N=8192,K=1024 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 3408705b7bcd79b9 | Llama3 70B | model_config | fp4 | M=16384,N=8192,K=3584 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| eac8322a4d58df9d | Llama3 70B | model_config | fp4 | M=16384,N=8192,K=1024 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 67efd50fdbd7220a | Llama3 70B | model_config | int8 | M=1,N=8192,K=3584 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 7a0c25bb808716b1 | Llama3 70B | model_config | int8 | M=1,N=8192,K=1024 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 4dcd95b9de18469b | Llama3 70B | model_config | int8 | M=32,N=8192,K=3584 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 7c7956658438f41c | Llama3 70B | model_config | int8 | M=32,N=8192,K=1024 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 1d382007dd2a6e5b | Llama3 70B | model_config | int8 | M=256,N=8192,K=1024 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| f96fef858542dce8 | Llama3 70B | model_config | int8 | M=256,N=8192,K=3584 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 3ac2b100eae49d01 | Llama3 70B | model_config | int8 | M=2048,N=8192,K=3584 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| ba6562cbbee8ca93 | Llama3 70B | model_config | int8 | M=2048,N=8192,K=1024 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 2cb9357c542ce0f4 | Llama3 70B | model_config | int8 | M=16384,N=8192,K=1024 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 828f73cb088e080b | Llama3 70B | model_config | int8 | M=16384,N=8192,K=3584 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| d255bc649a9f9a06 | Llama3 405B | model_config | fp4 | M=1,N=13312,K=16384 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| cb0ca05ddee29cb3 | Llama3 405B | model_config | fp4 | M=32,N=13312,K=16384 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| d67269dd9dc44b0e | Llama3 405B | model_config | fp4 | M=256,N=13312,K=16384 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| e0085a3f31dd6f04 | Llama3 405B | model_config | fp4 | M=2048,N=13312,K=16384 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 752dd48c0748e4fb | Llama3 405B | model_config | fp4 | M=16384,N=13312,K=16384 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 2da5c2889a38e0b4 | Llama3 405B | model_config | int8 | M=1,N=13312,K=16384 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 8010db7c55a938e9 | Llama3 405B | model_config | int8 | M=32,N=13312,K=16384 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 6f1edc3125a82f90 | Llama3 405B | model_config | int8 | M=256,N=13312,K=16384 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 089eadc7122e754e | Llama3 405B | model_config | int8 | M=2048,N=13312,K=16384 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 352d34f3584b2e73 | Llama3 405B | model_config | int8 | M=16384,N=13312,K=16384 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 14ddadd2f7f2cd9e | Llama3 405B | model_config | fp4 | M=1,N=16384,K=6656 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 75762ac4f21e7ab6 | Llama3 405B | model_config | fp4 | M=1,N=16384,K=2048 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 5ccb1596df03a540 | Llama3 405B | model_config | fp4 | M=32,N=16384,K=2048 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 66bb7bf934e8f83b | Llama3 405B | model_config | fp4 | M=32,N=16384,K=6656 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 1e1cf79232d41b48 | Llama3 405B | model_config | fp4 | M=256,N=16384,K=6656 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 550254ebe2244e27 | Llama3 405B | model_config | fp4 | M=256,N=16384,K=2048 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 0e427c9ae020d4a6 | Llama3 405B | model_config | fp4 | M=2048,N=16384,K=2048 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| fb6a5d36092fd92d | Llama3 405B | model_config | fp4 | M=2048,N=16384,K=6656 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 56bf59202b379716 | Llama3 405B | model_config | fp4 | M=16384,N=16384,K=6656 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 967c3c53fdaf2f1f | Llama3 405B | model_config | fp4 | M=16384,N=16384,K=2048 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 26a5bec46f0fe27e | Llama3 405B | model_config | int8 | M=1,N=16384,K=6656 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 68f0532f42c9c561 | Llama3 405B | model_config | int8 | M=1,N=16384,K=2048 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| b9598b98a9cea8cb | Llama3 405B | model_config | int8 | M=32,N=16384,K=6656 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| ce7dc7a9a6c15a7a | Llama3 405B | model_config | int8 | M=32,N=16384,K=2048 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 20e0026bfc46b49f | Llama3 405B | model_config | int8 | M=256,N=16384,K=2048 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| da3a6c97a32df617 | Llama3 405B | model_config | int8 | M=256,N=16384,K=6656 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 5428d8393492fe44 | Llama3 405B | model_config | int8 | M=2048,N=16384,K=6656 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| e7343f7002b2fe7f | Llama3 405B | model_config | int8 | M=2048,N=16384,K=2048 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 15eb6f24780d7e3d | Llama3 405B | model_config | int8 | M=16384,N=16384,K=2048 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
| 673177fc42015cf5 | Llama3 405B | model_config | int8 | M=16384,N=16384,K=6656 | failed | failed | failed | failed | failed | failed | failed | failed | failed | failed |  |
