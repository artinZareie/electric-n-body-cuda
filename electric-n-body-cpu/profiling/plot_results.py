#!/usr/bin/env python3
"""
plot_results.py — Generate profiling graphs for the CPU N-Body simulation.

Reads CSV files from profiling/results/ and produces publication-quality
plots saved to profiling/graphs/.

Graphs produced:
  1. Execution time vs particle count (log-log)  — the main comparison chart
  2. Time breakdown (force vs integrate vs I/O)   — stacked bar
  3. OpenMP thread scaling & speedup              — multi-line
  4. GFLOPS vs particle count                     — efficiency metric  
  5. Throughput (particles/sec) vs particle count  — throughput metric
  6. I/O overhead comparison                      — with vs without VTK
  7. Computational complexity verification         — O(N²) fit

All graphs use consistent styling and a "CPU" label so CUDA results
can be overlaid later for direct comparison.

Usage:
    cd electric-n-body-cpu
    python3 profiling/plot_results.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import csv

# ── Configuration ─────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
GRAPHS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
os.makedirs(GRAPHS_DIR, exist_ok=True)

# Consistent styling
CPU_COLOR = "#2196F3"       # Blue for CPU
CPU_COLOR2 = "#1565C0"      # Darker blue
FORCE_COLOR = "#E53935"     # Red for force computation  
INTEGRATE_COLOR = "#43A047" # Green for integration
IO_COLOR = "#FF9800"        # Orange for I/O
BG_COLOR = "#FAFAFA"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": BG_COLOR,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})


def read_csv(filename):
    """Read a CSV file into a list of dicts with numeric conversion."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"  ⚠ Missing: {path}")
        return []
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            converted = {}
            for k, v in row.items():
                try:
                    converted[k] = float(v)
                except (ValueError, TypeError):
                    converted[k] = v
            rows.append(converted)
    return rows


def save_fig(fig, name):
    """Save figure as both PNG and PDF."""
    for ext in ["png", "pdf"]:
        path = os.path.join(GRAPHS_DIR, f"{name}.{ext}")
        fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {name}.png / .pdf")


# ═══════════════════════════════════════════════════════════════════
# Graph 1: Execution Time vs Particle Count (log-log)
# ═══════════════════════════════════════════════════════════════════
def plot_scaling_particles(data):
    if not data:
        return
    N = [r["n_particles"] for r in data]
    avg_step = [r["avg_step_ms"] for r in data]
    avg_force = [r["avg_force_ms"] for r in data]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(N, avg_step, "o-", color=CPU_COLOR, linewidth=2.2,
              markersize=7, label="CPU Total Step", zorder=5)
    ax.loglog(N, avg_force, "s--", color=FORCE_COLOR, linewidth=1.8,
              markersize=6, label="CPU Force Computation", zorder=4)

    # O(N²) reference line
    if len(N) >= 2:
        N_arr = np.array(N, dtype=float)
        ref = avg_step[0] * (N_arr / N_arr[0]) ** 2
        ax.loglog(N_arr, ref, ":", color="gray", linewidth=1.5,
                  label="O(N²) reference", zorder=2)

    ax.set_xlabel("Number of Particles (N)")
    ax.set_ylabel("Average Time per Step (ms)")
    ax.set_title("CPU N-Body: Execution Time vs Problem Size")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    # Add grid
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.15)

    save_fig(fig, "01_scaling_particles")


# ═══════════════════════════════════════════════════════════════════
# Graph 2: Time Breakdown — Stacked Bar Chart
# ═══════════════════════════════════════════════════════════════════
def plot_time_breakdown(data):
    if not data:
        return
    N = [int(r["n_particles"]) for r in data]
    force = [r["avg_force_ms"] for r in data]
    integrate = [r["avg_integrate_ms"] for r in data]

    x = np.arange(len(N))
    width = 0.6

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars_force = ax.bar(x, force, width, label="Force Computation (O(N²))",
                        color=FORCE_COLOR, edgecolor="white", linewidth=0.5)
    bars_int   = ax.bar(x, integrate, width, bottom=force,
                        label="Integration (O(N))",
                        color=INTEGRATE_COLOR, edgecolor="white", linewidth=0.5)

    # Add percentage labels on bars
    for i in range(len(N)):
        total = force[i] + integrate[i]
        if total > 0:
            pct_force = force[i] / total * 100
            if force[i] > 0.5:  # only label if visible
                ax.text(x[i], force[i] / 2, f"{pct_force:.0f}%",
                        ha="center", va="center", fontsize=9, fontweight="bold",
                        color="white")

    ax.set_xlabel("Number of Particles (N)")
    ax.set_ylabel("Average Time per Step (ms)")
    ax.set_title("CPU N-Body: Time Breakdown per Step")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in N], rotation=45)
    ax.legend(loc="upper left")

    save_fig(fig, "02_time_breakdown")


# ═══════════════════════════════════════════════════════════════════
# Graph 3: OpenMP Thread Scaling & Speedup
# ═══════════════════════════════════════════════════════════════════
def plot_thread_scaling(data):
    if not data:
        return

    # Group by particle count
    sizes = sorted(set(int(r["n_particles"]) for r in data))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sizes)))

    for idx, N in enumerate(sizes):
        subset = [r for r in data if int(r["n_particles"]) == N]
        subset.sort(key=lambda r: r["n_threads"])
        threads = [int(r["n_threads"]) for r in subset]
        times = [r["avg_step_ms"] for r in subset]

        # Execution time
        ax1.plot(threads, times, "o-", color=colors[idx], linewidth=2,
                 markersize=6, label=f"N={N}")

        # Speedup (relative to 1 thread)
        t1 = times[0]
        speedup = [t1 / t for t in times]
        ax2.plot(threads, speedup, "o-", color=colors[idx], linewidth=2,
                 markersize=6, label=f"N={N}")

    # Ideal scaling line
    max_t = max(int(r["n_threads"]) for r in data)
    ideal_t = list(range(1, max_t + 1))
    ax2.plot(ideal_t, ideal_t, "--", color="gray", linewidth=1.5,
             label="Ideal (linear)", zorder=1)

    ax1.set_xlabel("Number of OpenMP Threads")
    ax1.set_ylabel("Average Time per Step (ms)")
    ax1.set_title("Execution Time vs Threads")
    ax1.legend()

    ax2.set_xlabel("Number of OpenMP Threads")
    ax2.set_ylabel("Speedup (T₁ / Tₙ)")
    ax2.set_title("Parallel Speedup")
    ax2.legend()

    fig.suptitle("CPU N-Body: OpenMP Thread Scaling", fontsize=14, y=1.02)
    fig.tight_layout()

    save_fig(fig, "03_thread_scaling")


# ═══════════════════════════════════════════════════════════════════
# Graph 4: GFLOPS vs Particle Count
# ═══════════════════════════════════════════════════════════════════
def plot_gflops(data):
    if not data:
        return
    N = [int(r["n_particles"]) for r in data]
    gflops = [r["gflops_force"] for r in data]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(N, gflops, "o-", color=CPU_COLOR, linewidth=2.2,
            markersize=7, label="CPU (OpenMP)", zorder=5)

    ax.set_xlabel("Number of Particles (N)")
    ax.set_ylabel("GFLOPS (force computation)")
    ax.set_title("CPU N-Body: Computational Throughput")
    ax.legend()
    ax.set_xscale("log", base=2)

    # Annotate peak
    peak_idx = np.argmax(gflops)
    ax.annotate(f"Peak: {gflops[peak_idx]:.2f} GFLOPS",
                xy=(N[peak_idx], gflops[peak_idx]),
                xytext=(15, 15), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=10, fontweight="bold")

    save_fig(fig, "04_gflops")


# ═══════════════════════════════════════════════════════════════════
# Graph 5: Throughput (Particles/sec) vs Particle Count
# ═══════════════════════════════════════════════════════════════════
def plot_throughput(data):
    if not data:
        return
    N = [int(r["n_particles"]) for r in data]
    throughput = [r["particles_per_sec"] for r in data]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.semilogy(N, throughput, "o-", color=CPU_COLOR, linewidth=2.2,
                markersize=7, label="CPU (OpenMP)", zorder=5)

    ax.set_xlabel("Number of Particles (N)")
    ax.set_ylabel("Particles Processed / Second")
    ax.set_title("CPU N-Body: Simulation Throughput")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    save_fig(fig, "05_throughput")


# ═══════════════════════════════════════════════════════════════════
# Graph 6: I/O Overhead — Compute Only vs With VTK
# ═══════════════════════════════════════════════════════════════════
def plot_io_overhead(data):
    if not data:
        return

    # Separate compute-only (vtk_ms ≈ 0) from VTK rows
    compute_only = [r for r in data if r["avg_vtk_ms"] < 0.001]
    with_vtk     = [r for r in data if r["avg_vtk_ms"] >= 0.001]

    if not compute_only or not with_vtk:
        return

    # Match by particle count
    co_dict = {int(r["n_particles"]): r for r in compute_only}
    vtk_dict = {int(r["n_particles"]): r for r in with_vtk}
    common_N = sorted(set(co_dict.keys()) & set(vtk_dict.keys()))

    if not common_N:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(common_N))
    width = 0.35

    compute_times = [co_dict[n]["avg_step_ms"] for n in common_N]
    total_with_io = [vtk_dict[n]["avg_step_ms"] for n in common_N]
    io_times      = [vtk_dict[n]["avg_vtk_ms"] for n in common_N]

    ax1.bar(x - width/2, compute_times, width, label="Compute Only",
            color=CPU_COLOR, edgecolor="white")
    ax1.bar(x + width/2, total_with_io, width, label="With VTK I/O",
            color=IO_COLOR, edgecolor="white")

    ax1.set_xlabel("Number of Particles (N)")
    ax1.set_ylabel("Average Time per Step (ms)")
    ax1.set_title("I/O Overhead: Total Step Time")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(n) for n in common_N], rotation=45)
    ax1.legend()

    # I/O as percentage of total
    io_pct = [io_times[i] / total_with_io[i] * 100 if total_with_io[i] > 0 else 0
              for i in range(len(common_N))]
    ax2.bar(x, io_pct, 0.6, color=IO_COLOR, edgecolor="white")
    ax2.set_xlabel("Number of Particles (N)")
    ax2.set_ylabel("VTK I/O as % of Total Step Time")
    ax2.set_title("I/O Overhead Fraction")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(n) for n in common_N], rotation=45)

    for i, pct in enumerate(io_pct):
        ax2.text(i, pct + 1, f"{pct:.1f}%", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("CPU N-Body: I/O Overhead Analysis", fontsize=14, y=1.02)
    fig.tight_layout()

    save_fig(fig, "06_io_overhead")


# ═══════════════════════════════════════════════════════════════════
# Graph 7: O(N²) Complexity Verification
# ═══════════════════════════════════════════════════════════════════
def plot_complexity(data):
    if not data:
        return

    N = np.array([r["n_particles"] for r in data])
    force_ms = np.array([r["avg_force_ms"] for r in data])

    # Fit log(time) = a * log(N) + b $\rightarrow$ time = C * N^a
    log_N = np.log(N)
    log_T = np.log(force_ms)
    coeffs = np.polyfit(log_N, log_T, 1)
    exponent = coeffs[0]
    C = np.exp(coeffs[1])

    N_fit = np.linspace(N.min(), N.max(), 200)
    T_fit = C * N_fit ** exponent

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(N, force_ms, "o", color=FORCE_COLOR, markersize=8,
              label="Measured force time", zorder=5)
    ax.loglog(N_fit, T_fit, "-", color="black", linewidth=2,
              label=f"Fit: T = {C:.2e} · N^{exponent:.2f}", zorder=4)

    ax.set_xlabel("Number of Particles (N)")
    ax.set_ylabel("Average Force Computation Time (ms)")
    ax.set_title(f"Complexity Verification — Measured Exponent: {exponent:.2f} (expected: 2.0)")
    ax.legend(loc="upper left")

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    save_fig(fig, "07_complexity_verification")


# ═══════════════════════════════════════════════════════════════════
# Graph 8: Parallel Efficiency
# ═══════════════════════════════════════════════════════════════════
def plot_parallel_efficiency(data):
    if not data:
        return

    sizes = sorted(set(int(r["n_particles"]) for r in data))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sizes)))

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for idx, N in enumerate(sizes):
        subset = [r for r in data if int(r["n_particles"]) == N]
        subset.sort(key=lambda r: r["n_threads"])
        threads = [int(r["n_threads"]) for r in subset]
        times = [r["avg_step_ms"] for r in subset]

        t1 = times[0]
        efficiency = [(t1 / (t * threads[i])) * 100 for i, t in enumerate(times)]

        ax.plot(threads, efficiency, "o-", color=colors[idx], linewidth=2,
                markersize=6, label=f"N={N}")

    ax.axhline(y=100, color="gray", linestyle="--", linewidth=1.5, label="Ideal (100%)")
    ax.set_xlabel("Number of OpenMP Threads")
    ax.set_ylabel("Parallel Efficiency (%)")
    ax.set_title("CPU N-Body: Parallel Efficiency (T₁ / (p · Tₚ) × 100)")
    ax.legend()
    ax.set_ylim(0, 120)

    save_fig(fig, "08_parallel_efficiency")


# ═══════════════════════════════════════════════════════════════════
# Summary Table (text output)
# ═══════════════════════════════════════════════════════════════════
def print_summary_table(data):
    if not data:
        return
    print("\n  ┌─────────┬──────────┬───────────┬──────────┬──────────┬───────────┐")
    print("  │    N    │ Avg Step │ Avg Force │ Avg Intg │  GFLOPS  │   P/sec   │")
    print("  ├─────────┼──────────┼───────────┼──────────┼──────────┼───────────┤")
    for r in data:
        print(f"  │ {int(r['n_particles']):>7} │ {r['avg_step_ms']:>7.2f}ms │ "
              f"{r['avg_force_ms']:>8.2f}ms │ {r['avg_integrate_ms']:>7.3f}ms │ "
              f"{r['gflops_force']:>8.4f} │ {r['particles_per_sec']:>9.0f} │")
    print("  └─────────┴──────────┴───────────┴──────────┴──────────┴───────────┘")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    print("═══ CPU N-Body Profiling — Graph Generation ═══")
    print(f"  Reading from: {RESULTS_DIR}")
    print(f"  Saving to:    {GRAPHS_DIR}\n")

    # Load data
    scaling = read_csv("scaling_particles.csv")
    threads = read_csv("scaling_threads.csv")
    io_data = read_csv("io_overhead.csv")

    if not scaling:
        print("ERROR: No scaling data found. Run profiling/run_profiling.sh first.")
        sys.exit(1)

    # Generate all graphs
    print("Generating graphs...")
    plot_scaling_particles(scaling)
    plot_time_breakdown(scaling)
    plot_gflops(scaling)
    plot_throughput(scaling)
    plot_complexity(scaling)

    if threads:
        plot_thread_scaling(threads)
        plot_parallel_efficiency(threads)

    if io_data:
        plot_io_overhead(io_data)

    # Print summary
    print_summary_table(scaling)

    print(f"\n  All graphs saved to {GRAPHS_DIR}/")
    print("  Files: 01_scaling_particles, 02_time_breakdown, 03_thread_scaling,")
    print("         04_gflops, 05_throughput, 06_io_overhead,")
    print("         07_complexity_verification, 08_parallel_efficiency")


if __name__ == "__main__":
    main()
