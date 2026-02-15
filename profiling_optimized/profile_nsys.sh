#!/usr/bin/env bash
set -euo pipefail

PROFILING_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$PROFILING_DIR/.." && pwd)"
BINARY="$PROJECT_ROOT/builddir_optimized/nbody"
SAMPLES_DIR="$PROFILING_DIR/samples"
RESULTS_DIR="${RESULTS_DIR:-$PROFILING_DIR/results/$(date +%Y-%m-%d)}"
NSYS_DIR="$RESULTS_DIR/nsys"
mkdir -p "$NSYS_DIR"

N_VALUES=(64 128 256 512 1024 2048 4096 8192)

for N in "${N_VALUES[@]}"; do
    echo "=== Nsight Systems: N=$N ==="
    WORK_DIR=$(mktemp -d)
    cp "$SAMPLES_DIR/particles_${N}.txt" "$WORK_DIR/particles.txt"
    mkdir -p "$WORK_DIR/vtk"
    cd "$WORK_DIR"

    nsys profile \
        --output="$NSYS_DIR/nsys_N${N}" \
        --force-overwrite=true \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --stats=true \
        --export=sqlite,json \
        -- "$BINARY" \
        2>&1 | tee "$NSYS_DIR/nsys_N${N}_log.txt" \
        || true

    cd "$PROFILING_DIR"

    nsys stats "$NSYS_DIR/nsys_N${N}.nsys-rep" \
        --report gputrace \
        --format csv \
        --output "$NSYS_DIR/nsys_N${N}_gputrace" \
        2>/dev/null || true

    nsys stats "$NSYS_DIR/nsys_N${N}.nsys-rep" \
        --report cudaapisum \
        --format csv \
        --output "$NSYS_DIR/nsys_N${N}_cudaapisum" \
        2>/dev/null || true

    nsys stats "$NSYS_DIR/nsys_N${N}.nsys-rep" \
        --report gpukernsum \
        --format csv \
        --output "$NSYS_DIR/nsys_N${N}_gpukernsum" \
        2>/dev/null || true

    nsys stats "$NSYS_DIR/nsys_N${N}.nsys-rep" \
        --report gpumemtimesum \
        --format csv \
        --output "$NSYS_DIR/nsys_N${N}_gpumemtimesum" \
        2>/dev/null || true

    nsys stats "$NSYS_DIR/nsys_N${N}.nsys-rep" \
        --report gpumemsizesum \
        --format csv \
        --output "$NSYS_DIR/nsys_N${N}_gpumemsizesum" \
        2>/dev/null || true

    rm -rf "$WORK_DIR"
done

echo "Nsight Systems profiling complete. Results in $NSYS_DIR"
