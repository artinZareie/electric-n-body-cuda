# Profiling Report — Electric N-Body CUDA Simulation

**GPU:** NVIDIA GeForce RTX 3060 (Ampere, SM 8.6, 28 SMs, 12 GB GDDR6)  
**DRAM BW (peak):** ~360 GB/s | **FP32 peak:** ~12.7 TFLOPS  
**Profiled with:** Nsight Systems (nsys) + Nsight Compute (ncu)  
**Date:** 2026-02-15

---

## 1. Time Breakdown per Kernel

All three kernels run per timestep on the default stream, sequentially.

### N = 1024 (1000 steps)

| Kernel | Total (ms) | Avg (µs) | Time % |
|---|---|---|---|
| `compute_force` | 50.4 | 50.4 | 45.4% |
| `force_reduction` | 56.6 | 56.6 | 50.9% |
| `update_particle_states` | 4.2 | 4.2 | 3.8% |

### N = 4096

| Kernel | Total (ms) | Avg (µs) | Time % |
|---|---|---|---|
| `compute_force` | 825.4 | 825.4 | 50.8% |
| `force_reduction` | 793.8 | 793.8 | 48.9% |
| `update_particle_states` | 4.2 | 4.2 | 0.3% |

### N = 8192

| Kernel | Total (ms) | Avg (ms) | Time % |
|---|---|---|---|
| `compute_force` | 3649 | 3.65 | 52.2% |
| `force_reduction` | 3340 | 3.34 | 47.8% |
| `update_particle_states` | 4.4 | 0.004 | 0.1% |

**Takeaway:** `compute_force` and `force_reduction` split time almost 50/50. `update_particle_states` is negligible. At large N, `compute_force` starts to dominate because it's O(N²) compute while `force_reduction` is O(N²) memory-bound.

---

## 2. Scalability

| N | Wall Time (s) | Scaling Factor |
|---|---|---|
| 64 | 0.30 | (baseline, launch-overhead dominated) |
| 128 | 0.24 | — |
| 256 | 0.25 | — |
| 512 | 0.29 | — |
| 1024 | 0.37 | — |
| 2048 | 0.72 | 1.94× from 1024 |
| 4096 | 2.03 | 2.82× from 2048 |
| 8192 | 7.69 | 3.79× from 4096 |

The expected scaling is ~4× per doubling of N (since both `compute_force` and `force_reduction` are O(N²)). We see ~3.8× from 4096$\rightarrow$8192, confirming quadratic behavior. Below N=1024, kernel launch overhead and fixed costs dominate.

---

## 3. Kernel Analysis: `compute_force`

**Launch config:** block (16,16), grid (N/16, N/16) $\rightarrow$ 1M threads for N=1024.

### NCU Metrics (N=1024)

| Metric | Value |
|---|---|
| Duration | 53 µs |
| DRAM Throughput | 87–89% of peak |
| SM Throughput | 57% |
| Achieved Occupancy | 85% (theoretical 100%) |
| Registers/thread | 34 |
| L1 Hit Rate | 54.7% |
| L2 Hit Rate | 99.8% |
| IPC (active) | 2.32 |
| Warp Cycles Per Issued Instr | 17.4 |

### NCU Metrics (N=4096)

| Metric | Value |
|---|---|
| Duration | 827 µs |
| DRAM Throughput | 93% of peak |
| SM Throughput | 58% |
| Achieved Occupancy | 89.5% |
| L1 Global Load | 352 MB read |
| L1 Global Store | 268 MB written |
| L2 miss on load | 25 MB (low — good cache reuse) |

### What NCU Says

- **Bottleneck: DRAM bandwidth.** At 87–93% peak DRAM throughput, this kernel is memory-bound.
- **SM compute is only 57–58% utilized** — the SMs are stalled waiting for memory most of the time.
- **Uncoalesced global loads:** only 24 of 32 bytes per sector are used on average. The SoA particle layout (`particles.x[y]`, `particles.y[y]`, ...) causes each thread to load from 8 separate float arrays. With a 2D thread block, the inner-loop access to particle `x` (indexed by `tx`) is strided across warps in the y-dimension. This wastes ~25% of memory bandwidth.
- **FP32 instructions not fused:** NCU reports 393K fused (FMA) vs 753K non-fused (separate FADD+FMUL). Converting to FMA could improve FP throughput by ~33%.
- **Occupancy limited by registers** (34 regs/thread $\rightarrow$ max 6 blocks/SM). Not critical since achieved is already 85%, but reducing register pressure slightly could help.

---

## 4. Kernel Analysis: `force_reduction`

**Launch config:** block 256, grid N $\rightarrow$ one block per particle row.

### NCU Metrics (N=1024)

| Metric | Value |
|---|---|
| Duration | 57 µs |
| DRAM Throughput | 89% of peak |
| SM Throughput | 30% |
| Achieved Occupancy | 94.6% |
| L1 Hit Rate | 0.03% |
| L2 Hit Rate | 0.36% |
| Memory Throughput | 312 GB/s |
| Warp Cycles Per Issued Instr | 50 |

### What NCU Says

- **Pure memory-bound kernel.** SM throughput is only 30% — almost all time is spent waiting for DRAM reads.
- **Zero cache reuse.** L1 hit rate is 0.03%, L2 hit rate is 0.36%. Each row of the N×N `float4` force matrix is read once and never reused. This makes sense: `force_reduction` is a row-sum over the matrix.
- **Shared memory bank conflicts:** 3.1-way average bank conflict on shared stores during the reduction tree, wasting ~37% of shared memory wavefronts.
- **Uncoalesced global stores:** only 16 of 32 bytes per L2 sector used for stores. Writing `net_forces[row] = sdata[0]` — since only thread 0 writes, the `float4` (16 bytes) wastes the other 16 bytes of the sector.
- **Scheduler stalls:** 82.7% of warp stall time is L1TEX scoreboard dependency — waiting for global memory loads. Only 0.32 eligible warps per scheduler per cycle (needs ~2+ to hide latency well).
- **Thread divergence in reduction:** avg 22.5 active threads per warp vs 31.9 allocated, because the `for (s = blockDim/2; ...)` tree halves participation each step.

### NCU Metrics (N=4096)

| Metric | Value |
|---|---|
| Duration | 797 µs |
| DRAM Throughput | 96.6% of peak |
| SM Throughput | 11% |
| Achieved Occupancy | 98.8% |

At N=4096, DRAM is pegged at 96.6% — there's almost no room left. SM utilization drops to 11%. This kernel is entirely DRAM-limited.

---

## 5. Kernel Analysis: `update_particle_states`

**Launch config:** block 256, grid ceil(N/256) $\rightarrow$ only 4 blocks for N=1024.

### NCU Metrics (N=1024)

| Metric | Value |
|---|---|
| Duration | 5.6 µs |
| DRAM Throughput | 2.7% |
| SM Throughput | 0.8% |
| Achieved Occupancy | 16.4% |
| Waves Per SM | 0.02 |

**Problem:** Grid size is way too small (4 blocks for 28 SMs). Most SMs sit idle. Not a real bottleneck since this kernel takes <0.1% of total time, but it could matter if the rest gets faster.

---

## 6. Memory Transfer Analysis

Per-step, there's a D2H copy for VTK rendering (`output_interval = 1`).

| N | D2H Total (MB) | D2H Count | D2H Total Time (ms) |
|---|---|---|---|
| 1024 | 32.8 | 8016 | 7.9 |
| 4096 | 131.3 | 8016 | 11.2 |
| 8192 | 262.7 | 8016 | 16.2 |

These are 8 small copies per frame (one per SoA array: x, y, z, vx, vy, vz, m, q). The total transfer time is small relative to kernel time (7.9 ms vs 111 ms for N=1024), but it still forces a sync point.

---

## 7. The N×N Force Matrix: The Real Problem

The architecture allocates an N×N `float4` matrix on GPU:

- N=1024: 1024² × 16 bytes = **16 MB**
- N=4096: 4096² × 16 bytes = **256 MB**
- N=8192: 8192² × 16 bytes = **1 GB**

This is the root cause of all bandwidth problems:

1. `compute_force` writes the entire N×N matrix $\rightarrow$ 256 MB written at N=4096
2. `force_reduction` reads the entire N×N matrix $\rightarrow$ 256 MB read at N=4096
3. Total DRAM traffic per step: **~512 MB** at N=4096, **~2 GB** at N=8192

At 360 GB/s peak DRAM bandwidth, reading+writing 2 GB takes ~5.5 ms minimum. Measured total is ~7 ms, which is consistent with ~93% efficiency plus overhead.

---

## 8. Where to Improve Memory Bandwidth

### 8.1 Eliminate the force matrix — use tiled direct summation

Instead of writing N² forces to DRAM and reading them back, compute the force **and reduce it** in a single kernel using shared memory tiling (the classic "nbody tile" approach):

```
for each tile of particles:
    load tile into shared memory
    compute partial forces from this tile
    accumulate into thread-local register sum
write final net_force to global memory
```

This reduces DRAM traffic from O(N²) to O(N) per step — just reading particle data and writing N net forces. For N=4096, this cuts memory from ~512 MB to ~0.5 MB per step.

**Expected speedup: 2–4×** or more, since you remove the force_reduction kernel entirely and cut DRAM pressure drastically.

### 8.2 Fix memory coalescing in `compute_force`

The inner loop iterates `x` across the thread's x-dimension, reading `particles.x[x]`, `particles.y[x]`, etc. With block (16,16):
- Threads in the same warp have consecutive `tx` values (first 16 threads have same `ty`, varying `tx`)
- Each load of `particles.x[tx]` is coalesced within a half-warp
- But loading 8 separate arrays (x, y, z, vx, vy, vz, m, q) means 8 separate memory transactions per particle

If using tiled shared memory, you load each tile once and reuse it N/tile_size times, which amortizes the cost.

### 8.3 Shared memory bank conflicts in `force_reduction`

The 3.1-way bank conflict comes from writing `sdata[tid] = sum` where `sum` is `float4` (16 bytes). Consecutive threads write to consecutive `float4` slots, which spans 4 banks per element. Since bank width is 4 bytes, thread 0 hits banks 0-3, thread 1 hits banks 4-7, etc. This is actually not the worst pattern, but the reduction tree causes conflicts when stride `s` aligns badly.

Fixing this is moot if you eliminate the force matrix (see 8.1).

### 8.4 FMA (fused multiply-add) usage in `compute_force`

NCU reports a 33% FP32 improvement opportunity by fusing FADD+FMUL into FMA. The compiler isn't fusing because of floating-point precision flags. Try:

- Compile with `--use_fast_math` or `--fmad=true`
- Manually use `fmaf()` for critical paths
- Use `__fmul_rn` / `__fadd_rn` only where precision matters

---

## 9. Would CUDA Streams Help?

### Current situation
All three kernels run on the default stream, sequentially:
```
compute_force $\rightarrow$ force_reduction $\rightarrow$ update_particle_states $\rightarrow$ [D2H copy for render]
```

### Where streams help

**Streams help overlap independent work.** Here's what could overlap:

1. **Overlap D2H copy with next step's compute:** After computing step N, copy results to host while already starting step N+1's `compute_force`. This hides the 1–2 ms copy latency behind GPU compute.

2. **Overlap VTK file I/O with GPU work:** Write VTK files on a CPU thread while the GPU runs the next step.

### Where streams DON'T help

- **The three kernels within a step are sequential by dependency.** `force_reduction` depends on `compute_force`'s output. `update_particle_states` depends on `force_reduction`. You can't overlap these.
- **Memory bandwidth is already saturated.** At 93–97% peak DRAM utilization, running two memory-bound kernels on different streams won't go faster — they'd compete for the same bandwidth.

### Verdict

Streams give a **small gain** (hiding D2H copy latency, ~5–10% at N≥4096). They won't fix the main bottleneck. The N×N force matrix elimination (section 8.1) is worth far more.

---

## 10. Summary of Recommendations

| Priority | Change | Expected Impact |
|---|---|---|
| **1 (high)** | Replace force matrix with tiled shared-memory reduction — fuse `compute_force` + `force_reduction` into one kernel | **2–4× speedup.** Cuts DRAM from O(N²) to O(N). Removes `force_reduction` entirely. |
| **2 (medium)** | Enable FMA fusion (`--use_fast_math` or `--fmad=true`) | ~15–20% compute improvement in `compute_force` |
| **3 (low)** | Use CUDA streams to overlap D2H copies with next-step compute | ~5–10% hiding of copy latency |
| **4 (low)** | Increase `update_particle_states` grid size (already negligible) | Marginal — only matters if other kernels get much faster |
| **5 (if needed)** | Reduce VTK output frequency (`output_interval`) | Less D2H traffic, fewer disk writes |

The tiled approach is the single biggest win. The current design pays the full N×N DRAM cost twice per step (write + read). A tiled kernel reads particle data (~O(N) per tile pass) from shared memory and accumulates forces in registers, writing only the N-element result.
