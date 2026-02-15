#!/usr/bin/env bash
set -euo pipefail

PROFILING_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$PROFILING_DIR/.." && pwd)"

export PROFILING_TS="${PROFILING_TS:-$(date +%Y-%m-%d)}"
export RESULTS_DIR="$PROFILING_DIR/results/$PROFILING_TS"
mkdir -p "$RESULTS_DIR"

echo "=== Profiling run: $PROFILING_TS ==="

echo ""
echo "========================================="
echo "[1/6] Collecting device information"
echo "========================================="
bash "$PROFILING_DIR/device_info.sh"

echo ""
echo "========================================="
echo "[2/6] Generating particle samples"
echo "========================================="
bash "$PROFILING_DIR/generate_samples.sh"

echo ""
echo "========================================="
echo "[3/6] Building profiling binary"
echo "========================================="
bash "$PROFILING_DIR/build_profile.sh"

echo ""
echo "========================================="
echo "[4/6] Running scalability benchmarks"
echo "========================================="
bash "$PROFILING_DIR/benchmark_scalability.sh"

echo ""
echo "========================================="
echo "[5/6] Running Nsight Systems profiling"
echo "========================================="
bash "$PROFILING_DIR/profile_nsys.sh"

echo ""
echo "========================================="
echo "[6/6] Running Nsight Compute profiling"
echo "========================================="
bash "$PROFILING_DIR/profile_ncu.sh"

echo ""
echo "========================================="
echo "Generating plots"
echo "========================================="
RESULTS_DIR="$RESULTS_DIR" python3 "$PROFILING_DIR/generate_plots.py"

echo ""
echo "========================================="
echo "All profiling complete."
echo "========================================="
echo "Results: $RESULTS_DIR/"
