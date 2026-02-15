#!/usr/bin/env bash
set -euo pipefail

PROFILING_DIR="$(cd "$(dirname "$0")" && pwd)"
SAMPLES_DIR="$PROFILING_DIR/samples"
mkdir -p "$SAMPLES_DIR"

N_VALUES=(64 128 256 512 1024 2048 4096 8192)

for N in "${N_VALUES[@]}"; do
    OUT="$SAMPLES_DIR/particles_${N}.txt"
    if [ -f "$OUT" ]; then
        continue
    fi
    python3 -c "
import numpy as np
N=$N
rng = np.random.default_rng(42)
x = rng.uniform(-1, 1, N)
y = rng.uniform(-1, 1, N)
z = rng.uniform(-1, 1, N)
vx = vy = vz = np.zeros(N)
m = np.ones(N)
q = rng.uniform(-1.602176634e-18, 1.602176634e-18, N)
with open('$OUT', 'w') as f:
    f.write(f'{N}\n')
    for i in range(N):
        f.write(f'{x[i]:.8e} {y[i]:.8e} {z[i]:.8e} {vx[i]:.8e} {vy[i]:.8e} {vz[i]:.8e} {m[i]:.8e} {q[i]:.8e}\n')
"
    echo "Generated $OUT"
done

echo "All samples generated in $SAMPLES_DIR"
