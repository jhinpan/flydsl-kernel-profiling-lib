# Coverage matrix: rmsnorm

| shape_id | model | stage | dtype | args | flydsl | aiter | aiter_triton | aiter_ck | aiter_asm | ck | triton | gluon | hipblaslt | pytorch | profile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 28efa5ce3a8d2dde | Qwen3-4B | decode | bf16 | M=1,N=128 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| abc99eae90130d87 | Qwen3-4B | decode | bf16 | M=2,N=128 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 7e978c38f2a2b2fb | Qwen3-4B | decode | bf16 | M=4,N=128 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 4eee69dbad657f36 | Qwen3-4B | decode | bf16 | M=16,N=128 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 45e95924617b7c3b | Qwen3-4B | decode | bf16 | M=64,N=128 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 4939bcc8bb687232 | Qwen3-4B | decode | bf16 | M=256,N=128 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| d484847f76d8f64d | DeepSeek-R1 | decode | bf16 | M=1,N=512 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| dfa7ea5380b4cb59 | DeepSeek-R1 | decode | bf16 | M=4,N=512 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 0fac6993d53a245b | DeepSeek-R1 | decode | bf16 | M=8,N=512 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 26d855e5cd31a6b9 | DeepSeek-R1 | decode | bf16 | M=64,N=512 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 8481d854c219882d | DeepSeek-R1 | decode | bf16 | M=256,N=512 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 8f7f9c4afc1f1190 | DeepSeek-R1 | decode | bf16 | M=1,N=1536 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| b7681f6c5ef89b2d | DeepSeek-R1 | decode | bf16 | M=4,N=1536 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 3497a37892a75980 | DeepSeek-R1 | decode | bf16 | M=8,N=1536 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| cd984e749ef26488 | DeepSeek-R1 | decode | bf16 | M=64,N=1536 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 690428797e218ab9 | DeepSeek-R1 | decode | bf16 | M=256,N=1536 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| deb3b412a092b3aa | Qwen3-4B | decode | bf16 | M=8,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| e497e54a4bd401da | Qwen3-4B | decode | bf16 | M=32,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| d497f1fe07ad8689 | Qwen3-4B | decode | bf16 | M=128,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 6035e2d3c0035e28 | Qwen3-4B | decode | bf16 | M=512,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 75c51cdcc3d94779 | Qwen3-4B | decode | bf16 | M=1024,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| bd27b3e9def3f2aa | Qwen3-4B | decode | bf16 | M=1,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 42d8b5bb9491d947 | Qwen3-4B | decode | bf16 | M=2,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| a28405710bee535d | Qwen3-4B | decode | bf16 | M=4,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| e88853a848f5ac1f | Qwen3-4B | decode | bf16 | M=16,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| f912dec6a13916d9 | Qwen3-4B | decode | bf16 | M=64,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 1a804a1472742bec | Qwen3-4B | decode | bf16 | M=256,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| ef6357b58ed55a68 | DeepSeek-R1 | decode | bf16 | M=1,N=7168 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 4bf4a209ed613f2a | DeepSeek-R1 | decode | bf16 | M=4,N=7168 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 0c08faa8cc5e9c55 | DeepSeek-R1 | decode | bf16 | M=8,N=7168 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 0f20ed486b23e2d2 | DeepSeek-R1 | decode | bf16 | M=64,N=7168 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 947ca7f0c2825b5f | DeepSeek-R1 | decode | bf16 | M=256,N=7168 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| cf5630ea92ae83e4 | diagnostic | diagnostic | bf16 | M=32768,N=8192 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 7744cb58890ae22a | Qwen3-235B-A22B | model_config | bf16 | M=1,N=128 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| aa111141e2e58f09 | Qwen3-235B-A22B | model_config | bf16 | M=32,N=128 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 2a2efa37b52cd4d6 | Qwen3-235B-A22B | model_config | bf16 | M=256,N=128 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 7be062f79c1f3e04 | Qwen3-235B-A22B | model_config | bf16 | M=2048,N=128 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| cf10c3af753e37df | Qwen3-235B-A22B | model_config | bf16 | M=16384,N=128 | failed | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 9d6bd70ffb77208e | DeepSeek-R1 | model_config | bf16 | M=1,N=512 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| f841e2b035ed19c6 | DeepSeek-R1 | model_config | bf16 | M=32,N=512 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 96e5b2877ed22036 | DeepSeek-R1 | model_config | bf16 | M=256,N=512 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 03a2f14fd88d3c75 | DeepSeek-R1 | model_config | bf16 | M=2048,N=512 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| ed6cbfc28c29fad6 | DeepSeek-R1 | model_config | bf16 | M=16384,N=512 | failed | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 0a5272d05a252344 | DeepSeek-R1 | model_config | bf16 | M=1,N=1536 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| c6f0f11d61617561 | DeepSeek-R1 | model_config | bf16 | M=32,N=1536 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 11368359d79bf509 | DeepSeek-R1 | model_config | bf16 | M=256,N=1536 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 038dcce015315fcf | DeepSeek-R1 | model_config | bf16 | M=2048,N=1536 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| b727e898e240606f | DeepSeek-R1 | model_config | bf16 | M=16384,N=1536 | failed | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| af5255f54bb86176 | GPT-OSS 120B | model_config | bf16 | M=1,N=2880 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 5bcedecfd463aa71 | GPT-OSS 120B | model_config | bf16 | M=32,N=2880 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 4dc7b16c4ed05e1a | GPT-OSS 120B | model_config | bf16 | M=256,N=2880 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| a81a42745ff39a2b | GPT-OSS 120B | model_config | bf16 | M=2048,N=2880 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| c79ce11ecbae7357 | GPT-OSS 120B | model_config | bf16 | M=16384,N=2880 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 345dffc14e8d1df0 | Llama3 8B\|Qwen3-235B-A | model_config | bf16 | M=1,N=4096 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 79a852447a1444de | Llama3 8B\|Qwen3-235B-A | model_config | bf16 | M=32,N=4096 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 2a2a2c4bc7461276 | Llama3 8B\|Qwen3-235B-A | model_config | bf16 | M=256,N=4096 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| ef1dc621eec4a3b4 | Llama3 8B\|Qwen3-235B-A | model_config | bf16 | M=2048,N=4096 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| e82f1fc28364e8fb | Llama3 8B\|Qwen3-235B-A | model_config | bf16 | M=16384,N=4096 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 610dd85cb62c7979 | Llama4 Maverick | model_config | bf16 | M=1,N=5120 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| cb10c4ee4e740fd2 | Llama4 Maverick | model_config | bf16 | M=32,N=5120 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| b155bfa94a449032 | Llama4 Maverick | model_config | bf16 | M=256,N=5120 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| cd7c706821398e69 | Llama4 Maverick | model_config | bf16 | M=2048,N=5120 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 0644a453d16b3a35 | Llama4 Maverick | model_config | bf16 | M=16384,N=5120 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 6e14a5f6aebc31c9 | DeepSeek-R1 | model_config | bf16 | M=1,N=7168 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 0ba9931601f3f183 | DeepSeek-R1 | model_config | bf16 | M=32,N=7168 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 0dee51d54829c0e9 | DeepSeek-R1 | model_config | bf16 | M=256,N=7168 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| f171e19248a35e8f | DeepSeek-R1 | model_config | bf16 | M=2048,N=7168 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| f82ed0c1e19e6bad | DeepSeek-R1 | model_config | bf16 | M=16384,N=7168 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 497265f1a3a6b8aa | Llama3 70B | model_config | bf16 | M=1,N=8192 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| e0245c5f49c5f605 | Llama3 70B | model_config | bf16 | M=32,N=8192 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 5f3965934331ecc4 | Llama3 70B | model_config | bf16 | M=256,N=8192 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| b3968dc22115aafb | Llama3 70B | model_config | bf16 | M=2048,N=8192 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 4b5ecd76d1bd17d9 | Llama3 70B | model_config | bf16 | M=16384,N=8192 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| f5e99168502397a2 | Llama3 405B | model_config | bf16 | M=1,N=16384 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 95fbc8d1ccb724e5 | Llama3 405B | model_config | bf16 | M=32,N=16384 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 016722709498033f | Llama3 405B | model_config | bf16 | M=256,N=16384 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 89c39948cad71b41 | Llama3 405B | model_config | bf16 | M=2048,N=16384 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 9e0c412ab792829e | Llama3 405B | model_config | bf16 | M=16384,N=16384 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 5df28e6152de8f91 | Qwen3-4B | prefill | bf16 | M=1,N=128 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| ede1bb8d9ab9a17c | Qwen3-4B | prefill | bf16 | M=2,N=128 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 5b50fcdaa0efc15c | Qwen3-4B | prefill | bf16 | M=4,N=128 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 12bb24f051c0113f | Qwen3-4B | prefill | bf16 | M=8,N=128 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 7906598a7a26100a | Qwen3-4B | prefill | bf16 | M=16,N=128 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| f754411d01b7ee91 | Qwen3-4B | prefill | bf16 | M=256,N=128 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| c2dfc7381a10ee2d | Qwen3-4B | prefill | bf16 | M=512,N=128 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| edc2d038fb764ebc | Qwen3-4B | prefill | bf16 | M=1024,N=128 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| f520ddc020498084 | Qwen3-4B | prefill | bf16 | M=2048,N=128 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| a132d2449f892362 | Qwen3-4B | prefill | bf16 | M=4096,N=128 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 6c22330de8581ef8 | Qwen3-4B | prefill | bf16 | M=8192,N=128 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 0a60f5baf30e8457 | Qwen3-4B | prefill | bf16 | M=16384,N=128 | failed | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 1b94569505cbcf5a | DeepSeek-R1 | prefill | bf16 | M=1,N=512 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 3c1b97647ce69d83 | DeepSeek-R1 | prefill | bf16 | M=8,N=512 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 092a23809d3e17a9 | DeepSeek-R1 | prefill | bf16 | M=64,N=512 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 5c23b7292b6d52d1 | DeepSeek-R1 | prefill | bf16 | M=512,N=512 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| b29a9bdcfb340206 | DeepSeek-R1 | prefill | bf16 | M=1024,N=512 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| e7aaa32fa441aa2a | DeepSeek-R1 | prefill | bf16 | M=2048,N=512 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 91c45b17a0547175 | DeepSeek-R1 | prefill | bf16 | M=4096,N=512 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 72f16280eba63818 | DeepSeek-R1 | prefill | bf16 | M=8192,N=512 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 7d7f355bf15c5480 | DeepSeek-R1 | prefill | bf16 | M=16384,N=512 | failed | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 5adced960477f24b | DeepSeek-R1 | prefill | bf16 | M=1,N=1536 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 9f5300e8cfef27f4 | DeepSeek-R1 | prefill | bf16 | M=8,N=1536 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| ebd1a6ebd6b5fcd3 | DeepSeek-R1 | prefill | bf16 | M=64,N=1536 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 3258adb891dd49c9 | DeepSeek-R1 | prefill | bf16 | M=512,N=1536 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 3b5f98730e96dc28 | DeepSeek-R1 | prefill | bf16 | M=1024,N=1536 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 5e58d8f75446b8f3 | DeepSeek-R1 | prefill | bf16 | M=2048,N=1536 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 2c19570b3e2f3dbd | DeepSeek-R1 | prefill | bf16 | M=4096,N=1536 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 125c9de57b3b380e | DeepSeek-R1 | prefill | bf16 | M=8192,N=1536 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| b1421e5b1c2d33d2 | DeepSeek-R1 | prefill | bf16 | M=16384,N=1536 | failed | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 575e4fde4a4ba867 | Qwen3-4B | prefill | bf16 | M=32768,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 67a5ee8759dacf57 | Qwen3-4B | prefill | bf16 | M=65536,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 04bd63f00e94fea9 | Qwen3-4B | prefill | bf16 | M=131072,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 79e8785ac835441a | Qwen3-4B | prefill | bf16 | M=1,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 88718f81d4333e25 | Qwen3-4B | prefill | bf16 | M=2,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 038585f86663eb0b | Qwen3-4B | prefill | bf16 | M=4,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 517b2be4eeb1d657 | Qwen3-4B | prefill | bf16 | M=8,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 99f1db5081ea0e6b | Qwen3-4B | prefill | bf16 | M=16,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 210ce0150fe8c2d2 | Qwen3-4B | prefill | bf16 | M=256,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 61f4ab8f7fbe7027 | Qwen3-4B | prefill | bf16 | M=512,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 22e64b56f93c5009 | Qwen3-4B | prefill | bf16 | M=1024,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 92bca3bb02c1d4a6 | Qwen3-4B | prefill | bf16 | M=2048,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 9c929420ce4269cb | Qwen3-4B | prefill | bf16 | M=4096,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 8560e65eaba030e2 | Qwen3-4B | prefill | bf16 | M=8192,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 7f3edd5d5cdd647b | Qwen3-4B | prefill | bf16 | M=16384,N=2560 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 9052ae0ffc3af0d2 | DeepSeek-R1 | prefill | bf16 | M=1,N=7168 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 2d4c3a41fb0880c2 | DeepSeek-R1 | prefill | bf16 | M=8,N=7168 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| d760c9b8f170a05c | DeepSeek-R1 | prefill | bf16 | M=64,N=7168 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| a1b99928a5f67264 | DeepSeek-R1 | prefill | bf16 | M=512,N=7168 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| a2e1f70a9393e451 | DeepSeek-R1 | prefill | bf16 | M=1024,N=7168 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 002de95fd2e9506c | DeepSeek-R1 | prefill | bf16 | M=2048,N=7168 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 16df035650d9cd1b | DeepSeek-R1 | prefill | bf16 | M=4096,N=7168 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 46fc90092d0addde | DeepSeek-R1 | prefill | bf16 | M=8192,N=7168 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| f19a0944f94cfeef | DeepSeek-R1 | prefill | bf16 | M=16384,N=7168 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 9d563c9b29ce206a | synthetic | synthetic | bf16 | M=1,N=2047 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| bea98ed7a9541e73 | synthetic | synthetic | bf16 | M=4096,N=2047 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| a775bd02fbd45477 | synthetic | synthetic | bf16 | M=1,N=2048 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| eda8cb1079cd849d | synthetic | synthetic | bf16 | M=4096,N=2048 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| dc5ea01b37e285f6 | synthetic | synthetic | bf16 | M=1,N=2049 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 180a73c791fe88f4 | synthetic | synthetic | bf16 | M=4096,N=2049 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 5777bda73bc7211a | synthetic | synthetic | bf16 | M=1,N=3000 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 1e6185ff5bf07d5f | synthetic | synthetic | bf16 | M=4096,N=3000 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| a65af0b359292b20 | synthetic | synthetic | bf16 | M=1,N=4095 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 42a2e085fb02f458 | synthetic | synthetic | bf16 | M=4096,N=4095 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| a88689944834ffc4 | synthetic | synthetic | bf16 | M=1,N=4096 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| fa9c407aa053f1ac | synthetic | synthetic | bf16 | M=4096,N=4096 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 3537fb04aa324c1b | synthetic | synthetic | bf16 | M=1,N=4097 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 5cd17c68f2832a25 | synthetic | synthetic | bf16 | M=4096,N=4097 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 93f963649e7636bb | synthetic | synthetic | bf16 | M=1,N=5333 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 83066af325dd870c | synthetic | synthetic | bf16 | M=4096,N=5333 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| c7dae6b0f9a69ea7 | synthetic | synthetic | bf16 | M=1,N=8191 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 49a45fbce23d447b | synthetic | synthetic | bf16 | M=4096,N=8191 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| a8a013688c432ada | synthetic | synthetic | bf16 | M=1,N=8192 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 66ec55459ec1da60 | synthetic | synthetic | bf16 | M=4096,N=8192 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| ed68cffd5f161976 | synthetic | synthetic | bf16 | M=131072,N=8192 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 450d963df81af266 | synthetic | synthetic | f16 | M=4096,N=8192 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 73ef6a7abaed15fe | synthetic | synthetic | f32 | M=4096,N=8192 | ok | unsupported | unsupported | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 3086fa7a19b9f22f | synthetic | synthetic | bf16 | M=1,N=8193 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 6cc420baf32b358f | synthetic | synthetic | bf16 | M=4096,N=8193 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| afdbde4bd583d860 | synthetic | synthetic | bf16 | M=1,N=12288 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
| 01de2c864a0fd201 | synthetic | synthetic | bf16 | M=4096,N=12288 | ok | ok | ok | n/c | n/c | n/c | ok | n/c | n/c | ok |  |
