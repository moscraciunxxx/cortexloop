#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
cmake -S . -B build -DCORTEXLOOP_WITH_KLEIDIAI="${1:-ON}"
cmake --build build -j
echo "built: build/libcortexloop.*  build/cortexloop-bench"
