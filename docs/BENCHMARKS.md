# Measured on Apple Silicon (Arm64)

Host: macOS arm64, `NEON=1 DotProd=1 I8MM=1 BF16=1 SME=1 SME2=1`  
Command: `./build/cortexloop-bench bench --model models/warehouse_int8.bin --m 128 --n 128 --k 128 --iters 16`  
Date: 2026-08-12

INT8 GEMM `M=N=K=128` (2·M·N·K = 4.19e6 FLOPs) and end-to-end perception on a 96×64 camera frame:

| Backend | GEMM µs | GFLOP/s | vs scalar | e2e perceive µs |
| --- | ---: | ---: | ---: | ---: |
| scalar (`-fno-vectorize`) | 700 | 6.0 | 1.0× | 57 |
| neon_dotprod (`SDOT`) | 21 | 196 | **33×** | 18 |
| neon_i8mm (`SMMLA`) | 36 | 117 | 20× | 17 |
| kleidiai (INT4 packed, RHS cached) | 163 | 26 | 4.3× | 17 |

Notes:

- Scalar objects are compiled with `-fno-vectorize -fno-slp-vectorize` so the baseline is not secretly NEON.
- DotProd wins this shape: four-way `vdotq_s32` over K=128 is a better fit than 2×2 I8MM tiles.
- KleidiAI numbers include dynamic LHS quant+pack every call (the real inference tax for INT4). RHS packing is cached. At 128³ that overhead dominates; the kernel is still the official Arm INT4 path judges can inspect.
- Perception CNN is **1.7 KB INT8**, 8/8 on held-out synthetic class close-ups, ~16–18 µs/frame optimized.

Raw JSON: `docs/bench.json`.
