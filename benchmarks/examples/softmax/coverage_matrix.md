# Coverage matrix: softmax

| shape_id | model | stage | dtype | args | flydsl | aiter | aiter_triton | aiter_ck | aiter_asm | ck | triton | gluon | hipblaslt | pytorch | profile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a5bf9702ded2560e | diagnostic | diagnostic | bf16 | M=32768,N=8192 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 1e91328ae9d6866a | Qwen3-235B-A22B | model_config | bf16 | M=1,N=128 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 9eff93301f0f89de | Qwen3-235B-A22B | model_config | bf16 | M=32,N=128 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| bba9b30e38d8ad3f | Qwen3-235B-A22B | model_config | bf16 | M=256,N=128 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| c09cb65d913c45a8 | Qwen3-235B-A22B | model_config | bf16 | M=2048,N=128 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| b1d5ed1727661e90 | Qwen3-235B-A22B | model_config | bf16 | M=16384,N=128 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 33e7cd1217bb4b70 | DeepSeek-R1 | model_config | bf16 | M=1,N=512 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| b81c0661acfd39c6 | DeepSeek-R1 | model_config | bf16 | M=32,N=512 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 25787cb90da27c49 | DeepSeek-R1 | model_config | bf16 | M=256,N=512 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| bee65e5d8a76b185 | DeepSeek-R1 | model_config | bf16 | M=2048,N=512 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 23c610f2c5243f00 | DeepSeek-R1 | model_config | bf16 | M=16384,N=512 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| c626afaee4a7ff06 | DeepSeek-R1 | model_config | bf16 | M=1,N=1536 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 76dc424b8bfdfcff | DeepSeek-R1 | model_config | bf16 | M=32,N=1536 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 5cae3fb3f5cb7abe | DeepSeek-R1 | model_config | bf16 | M=256,N=1536 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 11655fd608a6aee2 | DeepSeek-R1 | model_config | bf16 | M=2048,N=1536 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 9369dd80d1115a27 | DeepSeek-R1 | model_config | bf16 | M=16384,N=1536 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 72e90e5c092aa1a8 | GPT-OSS 120B | model_config | bf16 | M=1,N=2880 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| bc0a2c2d4f761bf7 | GPT-OSS 120B | model_config | bf16 | M=32,N=2880 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 6e119c324b880b99 | GPT-OSS 120B | model_config | bf16 | M=256,N=2880 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 6264f394c18aa2bf | GPT-OSS 120B | model_config | bf16 | M=2048,N=2880 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| a1145aee11159b20 | GPT-OSS 120B | model_config | bf16 | M=16384,N=2880 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 072cf98e02509dc8 | Llama3 8B\|Qwen3-235B-A | model_config | bf16 | M=1,N=4096 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 42a94b6ea3feda58 | Llama3 8B\|Qwen3-235B-A | model_config | bf16 | M=32,N=4096 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| d607de10a2cbcc38 | Llama3 8B\|Qwen3-235B-A | model_config | bf16 | M=256,N=4096 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| cc76da29fc786a64 | Llama3 8B\|Qwen3-235B-A | model_config | bf16 | M=2048,N=4096 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 42895e92d4baa134 | Llama3 8B\|Qwen3-235B-A | model_config | bf16 | M=16384,N=4096 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| d2eb7087d4e4bb50 | Llama4 Maverick | model_config | bf16 | M=1,N=5120 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 29ebd34af7e29840 | Llama4 Maverick | model_config | bf16 | M=32,N=5120 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 7da2be364c342fdb | Llama4 Maverick | model_config | bf16 | M=256,N=5120 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 9ea7aabae7cb1f71 | Llama4 Maverick | model_config | bf16 | M=2048,N=5120 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| e19eed1bdd96c96d | Llama4 Maverick | model_config | bf16 | M=16384,N=5120 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 95fd546e12ccfbf5 | DeepSeek-R1 | model_config | bf16 | M=1,N=7168 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 82bc146850c86573 | DeepSeek-R1 | model_config | bf16 | M=32,N=7168 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 3ea658a4cf41c55e | DeepSeek-R1 | model_config | bf16 | M=256,N=7168 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 84aebd676b7d133f | DeepSeek-R1 | model_config | bf16 | M=2048,N=7168 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 5c5c2614ff1c792a | DeepSeek-R1 | model_config | bf16 | M=16384,N=7168 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| f829befc3a731cf8 | Llama3 70B | model_config | bf16 | M=1,N=8192 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 9242b5b6920cfc95 | Llama3 70B | model_config | bf16 | M=32,N=8192 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 4db40d42b9ed5acc | Llama3 70B | model_config | bf16 | M=256,N=8192 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| ee778c0bb10c74a8 | Llama3 70B | model_config | bf16 | M=2048,N=8192 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 73a48aae302e3dda | Llama3 70B | model_config | bf16 | M=16384,N=8192 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 559831497cdf9ead | Llama3 405B | model_config | bf16 | M=1,N=16384 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| f607416f529bdcd4 | Llama3 405B | model_config | bf16 | M=32,N=16384 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| cbfee112089c823c | Llama3 405B | model_config | bf16 | M=256,N=16384 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 99805401293914ef | Llama3 405B | model_config | bf16 | M=2048,N=16384 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 591658171432af01 | Llama3 405B | model_config | bf16 | M=16384,N=16384 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 2c7262bd0360b3ba | synthetic | synthetic | bf16 | M=1,N=2047 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| f22847a1227a4f58 | synthetic | synthetic | bf16 | M=4096,N=2047 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 1baa3700ba7e9d2b | synthetic | synthetic | bf16 | M=1,N=2048 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| a7af78c908491ebe | synthetic | synthetic | bf16 | M=4096,N=2048 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 64517bc341eab7f6 | synthetic | synthetic | bf16 | M=1,N=2049 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 0cec046725533a2f | synthetic | synthetic | bf16 | M=4096,N=2049 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 49a02abeb0617c95 | synthetic | synthetic | bf16 | M=1,N=3000 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 4e5f593e1adac240 | synthetic | synthetic | bf16 | M=4096,N=3000 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| f04cc125a042b29c | synthetic | synthetic | bf16 | M=1,N=4095 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 7bd5ce9aef53bf90 | synthetic | synthetic | bf16 | M=4096,N=4095 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 2bda598e6a8db212 | synthetic | synthetic | bf16 | M=1,N=4096 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 684a5a2b7379f1bb | synthetic | synthetic | bf16 | M=4096,N=4096 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| b10bfda3b0eb4c30 | synthetic | synthetic | bf16 | M=1,N=4097 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 1d2cafbf561e95ef | synthetic | synthetic | bf16 | M=4096,N=4097 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| dd9838f41947a9fc | synthetic | synthetic | bf16 | M=1,N=5333 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 48eefeb21361712e | synthetic | synthetic | bf16 | M=4096,N=5333 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 4f4314c935f57e21 | synthetic | synthetic | bf16 | M=1,N=8191 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| c94f7396b8116fb6 | synthetic | synthetic | bf16 | M=4096,N=8191 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 1dafdbf06f2e3fde | synthetic | synthetic | bf16 | M=1,N=8192 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| a66381b83aa0aa02 | synthetic | synthetic | bf16 | M=4096,N=8192 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 1e6128d733849123 | synthetic | synthetic | bf16 | M=131072,N=8192 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| ea7104c40c98fcea | synthetic | synthetic | f16 | M=4096,N=8192 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 925f0144eb137f42 | synthetic | synthetic | f32 | M=4096,N=8192 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 26158cdf328c2112 | synthetic | synthetic | bf16 | M=1,N=8193 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 537cbe0ad0dbe6fe | synthetic | synthetic | bf16 | M=4096,N=8193 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| f8d6182ccc90509d | synthetic | synthetic | bf16 | M=1,N=12288 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| ce16fc69eba2bf79 | synthetic | synthetic | bf16 | M=4096,N=12288 | ok | n/c | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
