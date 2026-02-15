#!/usr/bin/env bash
set -euo pipefail

PROFILING_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$PROFILING_DIR/.." && pwd)"
BINARY="$PROJECT_ROOT/builddir_optimized/nbody"
SAMPLES_DIR="$PROFILING_DIR/samples"
RESULTS_DIR="${RESULTS_DIR:-$PROFILING_DIR/results/$(date +%Y-%m-%d)}"
SCALE_DIR="$RESULTS_DIR/scalability"
mkdir -p "$SCALE_DIR"

N_VALUES=(64 128 256 512 1024 2048 4096 8192)
REPEATS=3

CSV="$SCALE_DIR/scalability.csv"
echo "N,run,wall_time_s" > "$CSV"

for N in "${N_VALUES[@]}"; do
    for R in $(seq 1 $REPEATS); do
        echo "=== N=$N run=$R ==="
        WORK_DIR=$(mktemp -d)
        cp "$SAMPLES_DIR/particles_${N}.txt" "$WORK_DIR/particles.txt"
        mkdir -p "$WORK_DIR/vtk"
        cd "$WORK_DIR"

        START=$(date +%s%N)
        "$BINARY" > /dev/null 2>&1
        END=$(date +%s%N)

        WALL_NS=$((END - START))
        WALL_S=$(python3 -c "print(f'{$WALL_NS/1000000000:.6f}')")

        cd "$PROFILING_DIR"
        rm -rf "$WORK_DIR"

        echo "$N,$R,$WALL_S" >> "$CSV"
    done
done

echo "N,mean_wall_time_s,stddev_wall_time_s" > "$SCALE_DIR/scalability_summary.csv"
python3 -c "
import csv
from collections import defaultdict
import math
data = defaultdict(list)
with open('$CSV') as f:
    r = csv.DictReader(f)
    for row in r:
        data[int(row['N'])].append(float(row['wall_time_s']))
for n in sorted(data.keys()):
    vals = data[n]
    mean = sum(vals)/len(vals)
    std = math.sqrt(sum((v-mean)**2 for v in vals)/max(len(vals)-1,1))
    print(f'{n},{mean:.6f},{std:.6f}')
" >> "$SCALE_DIR/scalability_summary.csv"

echo "Scalability data written to $SCALE_DIR"
