#!/usr/bin/env bash
set -euo pipefail

PROFILING_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$PROFILING_DIR/.." && pwd)"
BINARY="$PROJECT_ROOT/builddir_profile/nbody"
SAMPLES_DIR="$PROFILING_DIR/samples"
RESULTS_DIR="${RESULTS_DIR:-$PROFILING_DIR/results/$(date +%Y-%m-%d)}"
NCU_DIR="$RESULTS_DIR/ncu"
mkdir -p "$NCU_DIR"

N_VALUES_FULL=(512 1024 2048 4096)
N_VALUES_DETAIL=(1024 4096)

METRICS_BASIC="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__occupancy_limit_blocks,\
launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,\
launch__occupancy_limit_warps,\
sm__maximum_warps_per_active_cycle_pct,\
sm__warps_active.avg.per_cycle_active,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum,\
lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st.sum,\
sm__inst_executed.sum,\
sm__inst_executed_pipe_fp32.sum,\
gpu__time_duration.sum,\
gpu__time_active.sum"

for N in "${N_VALUES_FULL[@]}"; do
    echo "=== Nsight Compute (full): N=$N ==="
    WORK_DIR=$(mktemp -d)
    cp "$SAMPLES_DIR/particles_${N}.txt" "$WORK_DIR/particles.txt"
    mkdir -p "$WORK_DIR/vtk"
    cd "$WORK_DIR"

    ncu \
        --set full \
        --force-overwrite \
        --target-processes all \
        --replay-mode kernel \
        --kernel-name-base function \
        --launch-skip 0 \
        --launch-count 3 \
        --export "$NCU_DIR/ncu_N${N}_full" \
        --csv \
        "$BINARY" \
        2>&1 | tee "$NCU_DIR/ncu_N${N}_full.csv"

    cd "$PROFILING_DIR"
    rm -rf "$WORK_DIR"
done

for N in "${N_VALUES_DETAIL[@]}"; do
    echo "=== Nsight Compute (metrics): N=$N ==="
    WORK_DIR=$(mktemp -d)
    cp "$SAMPLES_DIR/particles_${N}.txt" "$WORK_DIR/particles.txt"
    mkdir -p "$WORK_DIR/vtk"
    cd "$WORK_DIR"

    ncu \
        --metrics "$METRICS_BASIC" \
        --force-overwrite \
        --target-processes all \
        --replay-mode kernel \
        --kernel-name-base function \
        --launch-skip 0 \
        --launch-count 3 \
        --export "$NCU_DIR/ncu_N${N}_metrics" \
        --csv \
        "$BINARY" \
        2>&1 | tee "$NCU_DIR/ncu_N${N}_metrics.csv"

    cd "$PROFILING_DIR"
    rm -rf "$WORK_DIR"
done

for N in "${N_VALUES_DETAIL[@]}"; do
    echo "=== Nsight Compute (sections): N=$N ==="
    WORK_DIR=$(mktemp -d)
    cp "$SAMPLES_DIR/particles_${N}.txt" "$WORK_DIR/particles.txt"
    mkdir -p "$WORK_DIR/vtk"
    cd "$WORK_DIR"

    ncu \
        --section Occupancy \
        --section MemoryWorkloadAnalysis \
        --section MemoryWorkloadAnalysis_Chart \
        --section MemoryWorkloadAnalysis_Tables \
        --section LaunchStats \
        --section SpeedOfLight \
        --section SpeedOfLight_RooflineChart \
        --section ComputeWorkloadAnalysis \
        --section InstructionStats \
        --section SchedulerStats \
        --section WarpStateStats \
        --force-overwrite \
        --target-processes all \
        --replay-mode kernel \
        --kernel-name-base function \
        --launch-skip 0 \
        --launch-count 3 \
        --export "$NCU_DIR/ncu_N${N}_sections" \
        --csv \
        "$BINARY" \
        2>&1 | tee "$NCU_DIR/ncu_N${N}_sections.csv"

    cd "$PROFILING_DIR"
    rm -rf "$WORK_DIR"
done

echo "Nsight Compute profiling complete. Results in $NCU_DIR"
