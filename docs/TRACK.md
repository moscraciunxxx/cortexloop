# Track mapping — Physical AI

Submission track: **Track 1 · Physical AI**

Official definition: AI that perceives real-world (or simulated) sensor inputs and produces actions that affect a machine in a physical environment.

## Eligibility checklist

| Requirement | CortexLoop |
| --- | --- |
| On-device / on-robot / nearby edge inference | INT8 CNN runs locally in `libcortexloop` on Arm64 CPU |
| Real or simulated sensor data | Simulated RGB camera + pose (IMU-like heading) |
| Actions / control signals | Forward velocity, yaw rate, pick, place |
| Robotics / autonomy stack | A* navigation + mission policy + collision/energy model |
| Arm-based platform | Apple Silicon / Arm64 Linux (Neoverse, Cortex-A, Graviton-class) |

## Optimization evidence

- Model size: INT8 weights vs FP32 (export in `cortexloop/train.py`)
- Model speed: `python -m cortexloop bench`
- Arm-specific: NEON bilinear resize, SDOT GEMM, I8MM SMMLA, optional KleidiAI INT4
- Developer experience: `python -m cortexloop demo`

## Required write-up fields

**Overview.** CortexLoop is a closed-loop warehouse robot for Arm. It exists to prove that Physical AI is an optimization problem: perception kernels, quantization, and control-rate latency — not just a chatbot on a phone.

**Functionality / output.** A runnable simulator + native library + INT8 model + JSON benchmarks + live HUD. Output artifacts: `models/warehouse_int8.bin`, bench JSON, dashboard.

**Setup.** See README.md. Arm64 host, CMake, Python 3.10+, numpy. No proprietary robot required.
