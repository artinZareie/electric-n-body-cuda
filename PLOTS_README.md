# Performance Comparison Visualizations

This directory contains comprehensive performance comparisons between the original and optimized N-body simulation implementations.

## Generated Plots

### 1. Complete Overview: `profiling_comparison_plots.png`
A 6-panel visualization showing:
- **Wall Time Comparison** (log-log scale)
- **Speedup Factor** (bar chart)
- **Absolute Time Saved** (seconds)
- **Kernel Timing Breakdown** (N=4096)
- **Scaling Factor** (per doubling of N)
- **Improvement Percentage**

### 2. Detailed Speedup: `speedup_chart.png`
High-resolution speedup chart with annotations showing:
- Speedup factor for each N value
- Peak performance (4.30× at N=8192)
- Regions where optimization helps most

### 3. Wall Time Comparison: `wall_time_comparison.png`
High-resolution timing plot with:
- Original vs optimized wall times (log scale)
- Error bars showing standard deviation
- Shaded confidence regions
- Time savings annotation

## Key Results Summary

| N | Original | Optimized | Speedup | Time Saved |
|---|----------|-----------|---------|------------|
| 64 | 0.287s | 0.305s | 0.94× | -0.018s |
| 128 | 0.231s | 0.248s | 0.93× | -0.017s |
| 256 | 0.246s | 0.267s | 0.92× | -0.020s |
| 512 | 0.279s | 0.309s | 0.90× | -0.031s |
| 1024 | 0.366s | 0.379s | 0.97× | -0.013s |
| 2048 | 0.713s | 0.530s | **1.34×** | **+0.182s** |
| 4096 | 2.019s | 0.837s | **2.41×** | **+1.182s** |
| 8192 | 7.643s | 1.777s | **4.30×** | **+5.866s** |

## Reproducing the Plots

To regenerate all plots:

```bash
python3 create_comparison_plots.py
```

Requirements:
- Python 3.6+
- matplotlib
- numpy

Install dependencies:
```bash
pip install matplotlib numpy
```

## Profiling Data Sources

**Original Version:**
- Directory: `profiling_original/results/original/`
- Build: `builddir_original/`
- Branch: `main`

**Optimized Version:**
- Directory: `profiling_optimized/results/2026-02-15/`
- Build: `builddir_optimized/`
- Branch: `optimizations`

## Interpretation

### Why Small N Shows Slowdown
At N ≤ 1024, the simulation is overhead-limited:
- Kernel launch overhead dominates
- Force matrix fits in L2 cache (no memory pressure)
- Tile processing adds overhead without benefit

### Why Large N Shows Major Speedup
At N ≥ 2048, the simulation becomes memory-bound:
- Force matrix (N²) exceeds cache capacity
- DRAM bandwidth saturated in original version
- Optimized version reduces memory traffic from O(N²) to O(N)
- Result: 4.30× speedup at N=8192

### Kernel-Level Improvements (N=4096)
- Original: 823.7ms (compute_force) + 794.7ms (force_reduction) = 1618.4ms
- Optimized: 457.9ms (compute_force_tiled) = 457.9ms
- **Improvement: 3.53× faster kernels**
- Eliminated force_reduction entirely (49% of original time)
- Single tiled kernel is 1.80× faster than old compute_force

## Related Documents

- `PERFORMANCE_COMPARISON.md` - Detailed written analysis
- `profiling_optimized/IMPROVEMENT_REPORT.md` - Implementation details
- `profiling/REPORT.md` - Original profiling analysis

---

**Generated:** February 15, 2026  
**GPU:** NVIDIA GeForce RTX 3060 (Ampere)
