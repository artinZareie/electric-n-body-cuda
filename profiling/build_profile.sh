#!/usr/bin/env bash
set -euo pipefail

PROFILING_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$PROFILING_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/builddir_profile"

if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
fi

cd "$PROJECT_ROOT"
meson setup "$BUILD_DIR" \
    --buildtype=release \
    -Doptimization=3 \
    -Ddebug=false

cd "$BUILD_DIR"

sed -i 's/--generate-code=arch=compute_80,code=sm_80/--generate-code=arch=compute_80,code=sm_80 -lineinfo/g' build.ninja

ninja

echo "Profile binary built at $BUILD_DIR/nbody"
