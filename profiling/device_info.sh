#!/usr/bin/env bash
set -euo pipefail

PROFILING_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$PROFILING_DIR/results/$(date +%Y-%m-%d)}"
mkdir -p "$RESULTS_DIR"

OUT="$RESULTS_DIR/device_info.txt"

{
    echo "=== PLATFORM ==="
    uname -a
    echo ""
    echo "=== OS ==="
    cat /etc/os-release 2>/dev/null || echo "N/A"
    echo ""
    echo "=== CPU ==="
    lscpu | grep -E "Model name|Socket|Core|Thread|CPU MHz|CPU max|Architecture"
    echo ""
    echo "=== MEMORY ==="
    free -h
    echo ""
    echo "=== NVIDIA DRIVER ==="
    nvidia-smi --query-gpu=driver_version --format=csv,noheader
    echo ""
    echo "=== CUDA TOOLKIT ==="
    nvcc --version
    echo ""
    echo "=== GPU DEVICE ==="
    nvidia-smi --query-gpu=name,pci.bus_id,compute_cap,memory.total,memory.free,clocks.max.graphics,clocks.max.mem,power.limit,temperature.gpu --format=csv
    echo ""
    echo "=== GPU FULL QUERY ==="
    nvidia-smi -q
} > "$OUT"

echo "Device info written to $OUT"
