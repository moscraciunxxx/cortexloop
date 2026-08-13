# Devpost submission copy

**Project name:** CortexLoop

**Tagline:** Closed-loop Physical AI on Arm — sense, plan, act — with 33× INT8 GEMM vs scalar.

**Track:** Physical AI

**Built with:** Arm NEON, FEAT_DotProd (SDOT), FEAT_I8MM (SMMLA), Arm KleidiAI, C11/C++17, Python 3.12

## Project Overview

CortexLoop is a warehouse robot that actually closes the loop: a simulated RGB camera feeds an INT8 CNN running on Arm CPU kernels, a planner turns class labels into pick/haul/dock/avoid actions, and differential-drive control moves the machine. The point is not another chatbot screenshot. It is Physical AI as an optimization problem — model size, kernel selection, and control-rate latency — with numbers you can reproduce on any Arm64 laptop.

It should win because the Arm work is real and readable: a non-vectorized scalar baseline, hand-written `SDOT` / `SMMLA` GEMM, optional KleidiAI INT4 micro-kernels, runtime feature detection (NEON, DotProd, I8MM, SME2), and a live HUD where judges toggle backends and watch microseconds change.

## Functionality / Output

- Native library `libcortexloop` — INT8 CNN + Arm kernels
- Trained artifact `models/warehouse_int8.bin` (1.7 KB)
- Live dashboard at `http://127.0.0.1:8765` — map, camera, mission, bench table
- JSON microbenchmarks (`python -m cortexloop bench`)
- On Apple Silicon: **33×** INT8 GEMM (NEON DotProd vs scalar), **3.4×** end-to-end perception, 8/8 class accuracy on the synthetic eval set

## Setup Instructions

Arm64 macOS or Linux. `clang`, `make` or `cmake`, Python 3.10+, numpy.

```bash
make -j                 # NEON / I8MM (no network)
# or: cmake -S . -B build -DCORTEXLOOP_WITH_KLEIDIAI=ON && cmake --build build -j
python -m cortexloop train    # if you want to regenerate weights
python -m cortexloop bench
python -m cortexloop demo     # http://127.0.0.1:8765
```

Full steps: repository README.md. Track mapping: docs/TRACK.md. Numbers: docs/BENCHMARKS.md.

## Video notes (optional, < 3 min)

1. Dashboard boot, robot collecting red boxes.
2. Toggle scalar → neon_dotprod; HUD latency drops.
3. Run GEMM bench; show 33× table.
4. Flash `native/src/gemm.c` SDOT loop.
