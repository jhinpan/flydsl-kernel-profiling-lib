# Coverage matrix: layernorm

| shape_id | model | stage | dtype | args | flydsl | aiter | aiter_triton | aiter_ck | aiter_asm | ck | triton | gluon | hipblaslt | pytorch | profile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 564840640b0a9b26 | diagnostic | diagnostic | bf16 | M=32768,N=8192 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 62cc2457634c4152 | Qwen3-235B-A22B | model_config | bf16 | M=1,N=128 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 9a885d60be71367d | Qwen3-235B-A22B | model_config | bf16 | M=32,N=128 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 0e95d110d2f0dd31 | Qwen3-235B-A22B | model_config | bf16 | M=256,N=128 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| bc6cfc6835c0fb74 | Qwen3-235B-A22B | model_config | bf16 | M=2048,N=128 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 4d21d9d84daa5d73 | Qwen3-235B-A22B | model_config | bf16 | M=16384,N=128 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 5e856bc73f2ff9d2 | DeepSeek-R1 | model_config | bf16 | M=1,N=512 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 7eb20d4e8b9112cf | DeepSeek-R1 | model_config | bf16 | M=32,N=512 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 7d74d258777d2008 | DeepSeek-R1 | model_config | bf16 | M=256,N=512 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 6085ae45b695a465 | DeepSeek-R1 | model_config | bf16 | M=2048,N=512 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| d6b7ce77a7a73192 | DeepSeek-R1 | model_config | bf16 | M=16384,N=512 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| dcbf256332136add | DeepSeek-R1 | model_config | bf16 | M=1,N=1536 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 757f9fcc23581668 | DeepSeek-R1 | model_config | bf16 | M=32,N=1536 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| a16f38807b37c49f | DeepSeek-R1 | model_config | bf16 | M=256,N=1536 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| fe1f1818e3fbae1a | DeepSeek-R1 | model_config | bf16 | M=2048,N=1536 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| f5733623704636db | DeepSeek-R1 | model_config | bf16 | M=16384,N=1536 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| a971833de3e228f9 | GPT-OSS 120B | model_config | bf16 | M=1,N=2880 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| f86b0fb667396ed8 | GPT-OSS 120B | model_config | bf16 | M=32,N=2880 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 8304fea2c3ef922b | GPT-OSS 120B | model_config | bf16 | M=256,N=2880 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| e0760021e8ad96b6 | GPT-OSS 120B | model_config | bf16 | M=2048,N=2880 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| fbcf3b4e166264b6 | GPT-OSS 120B | model_config | bf16 | M=16384,N=2880 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 2fe5f0e7a68e407f | Llama3 8B\|Qwen3-235B-A | model_config | bf16 | M=1,N=4096 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 33e1ede4786db394 | Llama3 8B\|Qwen3-235B-A | model_config | bf16 | M=32,N=4096 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| f16dfd27c4b517c7 | Llama3 8B\|Qwen3-235B-A | model_config | bf16 | M=256,N=4096 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 8d5d42eb4b9cf080 | Llama3 8B\|Qwen3-235B-A | model_config | bf16 | M=2048,N=4096 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 9e42d1db03df47cb | Llama3 8B\|Qwen3-235B-A | model_config | bf16 | M=16384,N=4096 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 5f8cbc3efb1386df | Llama4 Maverick | model_config | bf16 | M=1,N=5120 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 44674e4f5f889420 | Llama4 Maverick | model_config | bf16 | M=32,N=5120 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| b32f5eec26e2b0c4 | Llama4 Maverick | model_config | bf16 | M=256,N=5120 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 04e58a4566c3b7bb | Llama4 Maverick | model_config | bf16 | M=2048,N=5120 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 4f11b772f5043501 | Llama4 Maverick | model_config | bf16 | M=16384,N=5120 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 3b711643bbd687da | DeepSeek-R1 | model_config | bf16 | M=1,N=7168 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 401f80163fd0130a | DeepSeek-R1 | model_config | bf16 | M=32,N=7168 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| b2a09d1895a6f06c | DeepSeek-R1 | model_config | bf16 | M=256,N=7168 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 1c00cc71498441df | DeepSeek-R1 | model_config | bf16 | M=2048,N=7168 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 78b5276f5a595b65 | DeepSeek-R1 | model_config | bf16 | M=16384,N=7168 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 4eec94911c4e3c06 | Llama3 70B | model_config | bf16 | M=1,N=8192 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| afcc9fc024f38015 | Llama3 70B | model_config | bf16 | M=32,N=8192 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| d3313f6cb0469ebc | Llama3 70B | model_config | bf16 | M=256,N=8192 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 67ef7c3054c1f6bb | Llama3 70B | model_config | bf16 | M=2048,N=8192 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| c2e2a8848298194d | Llama3 70B | model_config | bf16 | M=16384,N=8192 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| b7e3109ced3eef94 | Llama3 405B | model_config | bf16 | M=1,N=16384 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 23b9ebe075ff39e0 | Llama3 405B | model_config | bf16 | M=32,N=16384 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| b6c1931f6e994030 | Llama3 405B | model_config | bf16 | M=256,N=16384 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 71f0d7db538e60d3 | Llama3 405B | model_config | bf16 | M=2048,N=16384 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| d467084d24c68668 | Llama3 405B | model_config | bf16 | M=16384,N=16384 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 84e8a3366e2c2b13 | synthetic | synthetic | bf16 | M=1,N=2047 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 851c8a6aa3212c8e | synthetic | synthetic | bf16 | M=4096,N=2047 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 1141591ad70d754e | synthetic | synthetic | bf16 | M=1,N=2048 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| e1d4f4954e864690 | synthetic | synthetic | bf16 | M=4096,N=2048 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| ebdccfefe01191d3 | synthetic | synthetic | bf16 | M=1,N=2049 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 16dadfc20f58f1d3 | synthetic | synthetic | bf16 | M=4096,N=2049 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 3e114ac3885283bf | synthetic | synthetic | bf16 | M=1,N=3000 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 633327d733a681fd | synthetic | synthetic | bf16 | M=4096,N=3000 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 797fe6c79af06228 | synthetic | synthetic | bf16 | M=1,N=4095 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 3d7f8324a06aef0f | synthetic | synthetic | bf16 | M=4096,N=4095 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 6e20d3aa10af1395 | synthetic | synthetic | bf16 | M=1,N=4096 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 39283e794eef402d | synthetic | synthetic | bf16 | M=4096,N=4096 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 2cad8d14dde95cad | synthetic | synthetic | bf16 | M=1,N=4097 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| ce7d1075969888e5 | synthetic | synthetic | bf16 | M=4096,N=4097 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 4d47acce69d5f2ab | synthetic | synthetic | bf16 | M=1,N=5333 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| ac9ebba272b08527 | synthetic | synthetic | bf16 | M=4096,N=5333 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| c4db60451baa4941 | synthetic | synthetic | bf16 | M=1,N=8191 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| c7c480a301fcf83b | synthetic | synthetic | bf16 | M=4096,N=8191 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| e5fec8299714907a | synthetic | synthetic | bf16 | M=1,N=8192 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 100a0100041547e9 | synthetic | synthetic | bf16 | M=4096,N=8192 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 0dbaa83d97976f77 | synthetic | synthetic | bf16 | M=131072,N=8192 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 7c2d1d3ef4a6fc4c | synthetic | synthetic | f16 | M=4096,N=8192 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| ae6bb190a2c95060 | synthetic | synthetic | f32 | M=4096,N=8192 | ok | unsupported | unsupported | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| 9020d4494c70b307 | synthetic | synthetic | bf16 | M=1,N=8193 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| c2bb80090949d089 | synthetic | synthetic | bf16 | M=4096,N=8193 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| fa8656cbcc9db5c0 | synthetic | synthetic | bf16 | M=1,N=12288 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
| b5533984340bf6c0 | synthetic | synthetic | bf16 | M=4096,N=12288 | ok | ok | ok | n/c | n/c | n/c | n/c | n/c | n/c | ok |  |
