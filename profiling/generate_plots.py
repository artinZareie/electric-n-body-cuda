#!/usr/bin/env python3
import csv
import os
import glob
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

NSYS_DIR = os.path.join(RESULTS_DIR, "nsys")
NCU_DIR = os.path.join(RESULTS_DIR, "ncu")
SCALE_DIR = os.path.join(RESULTS_DIR, "scalability")


def _save(fig, name):
    fig.savefig(os.path.join(PLOTS_DIR, f"{name}.png"), dpi=200)
    fig.savefig(os.path.join(PLOTS_DIR, f"{name}.svg"))
    plt.close(fig)


def _find_nsys(pattern):
    results = {}
    for f in sorted(glob.glob(os.path.join(NSYS_DIR, pattern))):
        m = re.search(r'N(\d+)', os.path.basename(f))
        if m:
            results[int(m.group(1))] = f
    return results


def _find_ncu(pattern):
    results = {}
    for f in sorted(glob.glob(os.path.join(NCU_DIR, pattern))):
        m = re.search(r'N(\d+)', os.path.basename(f))
        if m:
            results[int(m.group(1))] = f
    return results


def plot_scalability():
    p = os.path.join(SCALE_DIR, "scalability_summary.csv")
    if not os.path.exists(p):
        return
    N, mu, sd = [], [], []
    with open(p) as f:
        for row in csv.DictReader(f):
            N.append(int(row["N"]))
            mu.append(float(row["mean_wall_time_s"]))
            sd.append(float(row["stddev_wall_time_s"]))
    N, mu, sd = np.array(N), np.array(mu), np.array(sd)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].errorbar(N, mu, yerr=sd, fmt='o-', capsize=4, color='tab:blue')
    axes[0].set_xlabel("N"); axes[0].set_ylabel("Wall time (s)"); axes[0].set_title("Wall Time vs N"); axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(N, mu, yerr=sd, fmt='o-', capsize=4, color='tab:red')
    axes[1].set_xscale('log', base=2); axes[1].set_yscale('log')
    axes[1].set_xlabel("N"); axes[1].set_ylabel("Wall time (s)"); axes[1].set_title("Wall Time vs N (log-log)"); axes[1].grid(True, alpha=0.3, which='both')

    if len(N) > 2:
        ln, lt = np.log2(N), np.log2(mu)
        c = np.polyfit(ln, lt, 1)
        axes[2].plot(N, mu, 'o', color='tab:blue', label='Measured')
        axes[2].plot(N, 2**np.polyval(c, ln), '--', color='tab:red', label=f'Fit: O(N^{c[0]:.2f})')
        axes[2].set_xscale('log', base=2); axes[2].set_yscale('log')
        axes[2].set_xlabel("N"); axes[2].set_ylabel("Wall time (s)"); axes[2].set_title("Scaling Exponent")
        axes[2].legend(); axes[2].grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    _save(fig, "scalability")


def plot_scalability_raw():
    p = os.path.join(SCALE_DIR, "scalability.csv")
    if not os.path.exists(p):
        return
    data = defaultdict(list)
    with open(p) as f:
        for row in csv.DictReader(f):
            data[int(row["N"])].append(float(row["wall_time_s"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in sorted(data):
        ax.scatter([n]*len(data[n]), data[n], alpha=0.6, s=30)
    ax.set_xlabel("N"); ax.set_ylabel("Wall time (s)"); ax.set_title("All Runs (Raw)")
    ax.set_xscale('log', base=2); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "scalability_raw")


def plot_nsys_kernel_time():
    files = _find_nsys("*kernsum*kern_sum*.csv")
    if not files:
        return
    kernel_times = {}
    for N, f in files.items():
        with open(f) as fh:
            for row in csv.DictReader(fh):
                name = row.get("Name", "").split("(")[0].strip()
                total = float(row.get("Total Time (ns)", "0").replace(",", ""))
                kernel_times.setdefault(N, {})[name] = kernel_times.get(N, {}).get(name, 0) + total

    all_k = sorted({k for d in kernel_times.values() for k in d})
    N_vals = sorted(kernel_times)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(N_vals))
    w = 0.8 / max(len(all_k), 1)
    for i, k in enumerate(all_k):
        ax.bar(x + i*w, [kernel_times.get(n, {}).get(k, 0)/1e6 for n in N_vals], w, label=k[:40])
    ax.set_xlabel("N"); ax.set_ylabel("Time (ms)"); ax.set_title("GPU Kernel Time by N")
    ax.set_xticks(x + w*len(all_k)/2); ax.set_xticklabels([str(n) for n in N_vals])
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    _save(fig, "nsys_kernel_time")


def plot_nsys_kernel_time_stacked():
    files = _find_nsys("*kernsum*kern_sum*.csv")
    if not files:
        return
    kernel_times = {}
    for N, f in files.items():
        with open(f) as fh:
            for row in csv.DictReader(fh):
                name = row.get("Name", "").split("(")[0].strip()
                total = float(row.get("Total Time (ns)", "0").replace(",", ""))
                kernel_times.setdefault(N, {})[name] = kernel_times.get(N, {}).get(name, 0) + total

    all_k = sorted({k for d in kernel_times.values() for k in d})
    N_vals = sorted(kernel_times)
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(N_vals))
    for k in all_k:
        vals = np.array([kernel_times.get(n, {}).get(k, 0)/1e6 for n in N_vals])
        ax.bar(range(len(N_vals)), vals, bottom=bottom, label=k[:40])
        bottom += vals
    ax.set_xlabel("N"); ax.set_ylabel("Total GPU Time (ms)"); ax.set_title("Kernel Time Breakdown (Stacked)")
    ax.set_xticks(range(len(N_vals))); ax.set_xticklabels([str(n) for n in N_vals])
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    _save(fig, "nsys_kernel_time_stacked")


def plot_nsys_api():
    files = _find_nsys("*apisum*api_sum*.csv")
    if not files:
        return
    api_data = {}
    for N, f in files.items():
        with open(f) as fh:
            for row in csv.DictReader(fh):
                name = row.get("Name", "")
                total = float(row.get("Total Time (ns)", "0").replace(",", ""))
                api_data.setdefault(N, {})[name] = total

    all_a = sorted({k for d in api_data.values() for k in d})
    N_vals = sorted(api_data)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(N_vals))
    w = 0.8 / max(len(all_a), 1)
    for i, a in enumerate(all_a):
        ax.bar(x + i*w, [api_data.get(n, {}).get(a, 0)/1e6 for n in N_vals], w, label=a[:30])
    ax.set_xlabel("N"); ax.set_ylabel("Time (ms)"); ax.set_title("CUDA API Time by N")
    ax.set_xticks(x + w*len(all_a)/2); ax.set_xticklabels([str(n) for n in N_vals])
    ax.legend(fontsize=5, loc='upper left'); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    _save(fig, "nsys_cuda_api")


def plot_ncu_sol():
    files = _find_ncu("*full.csv")
    if not files:
        return
    sol_data = {}
    for N, f in files.items():
        with open(f) as fh:
            for row in csv.DictReader(fh):
                if row.get("Section Name") != "GPU Speed Of Light Throughput":
                    continue
                kern = row.get("Kernel Name", "").split("(")[0].strip()
                metric = row.get("Metric Name", "")
                val_str = row.get("Metric Value", "0").replace(",", "")
                unit = row.get("Metric Unit", "")
                if "%" not in unit:
                    continue
                try:
                    val = float(val_str)
                except ValueError:
                    continue
                key = f"{kern}: {metric}"
                sol_data.setdefault(N, {})[key] = val

    if not sol_data:
        return
    all_m = sorted({k for d in sol_data.values() for k in d})
    N_vals = sorted(sol_data)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(N_vals))
    w = 0.8 / max(len(all_m), 1)
    for i, m in enumerate(all_m):
        ax.bar(x + i*w, [sol_data.get(n, {}).get(m, 0) for n in N_vals], w, label=m[:50])
    ax.set_xlabel("N"); ax.set_ylabel("% of Peak"); ax.set_title("Speed of Light (% of Peak Throughput)")
    ax.set_xticks(x + w*len(all_m)/2); ax.set_xticklabels([str(n) for n in N_vals])
    ax.legend(fontsize=5, loc='upper left', ncol=2); ax.set_ylim(0, 100); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    _save(fig, "ncu_speed_of_light")


def plot_ncu_occupancy():
    files = _find_ncu("*full.csv")
    if not files:
        return
    occ = {}
    for N, f in files.items():
        with open(f) as fh:
            for row in csv.DictReader(fh):
                if row.get("Section Name") != "Occupancy":
                    continue
                kern = row.get("Kernel Name", "").split("(")[0].strip()
                metric = row.get("Metric Name", "")
                val_str = row.get("Metric Value", "0").replace(",", "")
                try:
                    val = float(val_str)
                except ValueError:
                    continue
                key = f"{kern}: {metric}"
                occ.setdefault(N, {})[key] = val

    if not occ:
        return
    all_m = sorted({k for d in occ.values() for k in d})
    N_vals = sorted(occ)
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(N_vals))
    w = 0.8 / max(len(all_m), 1)
    for i, m in enumerate(all_m):
        ax.bar(x + i*w, [occ.get(n, {}).get(m, 0) for n in N_vals], w, label=m[:60])
    ax.set_xlabel("N"); ax.set_ylabel("Value"); ax.set_title("Occupancy Metrics by N")
    ax.set_xticks(x + w*len(all_m)/2); ax.set_xticklabels([str(n) for n in N_vals])
    ax.legend(fontsize=5, loc='upper left', ncol=2); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    _save(fig, "ncu_occupancy")


def plot_ncu_memory_metrics():
    files = _find_ncu("*metrics.csv")
    if not files:
        return

    BYTE_METRICS = {"dram__bytes_read.sum", "dram__bytes_write.sum",
                    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
                    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum",
                    "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum",
                    "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st.sum"}

    UNIT_SCALE = {"byte": 1, "Kbyte": 1024, "Mbyte": 1024**2, "Gbyte": 1024**3}

    mem = {}
    for N, f in files.items():
        with open(f) as fh:
            for row in csv.DictReader(fh):
                mname = row.get("Metric Name", "")
                if mname not in BYTE_METRICS:
                    continue
                kern = row.get("Kernel Name", "").split("(")[0].strip()
                val_str = row.get("Metric Value", "0").replace(",", "")
                unit = row.get("Metric Unit", "byte")
                try:
                    val = float(val_str) * UNIT_SCALE.get(unit, 1) / (1024**2)
                except ValueError:
                    continue
                short_metric = mname.replace("__", " ").replace(".sum", "").replace("_", " ")
                key = f"{kern}: {short_metric}"
                mem.setdefault(N, {})[key] = val

    if not mem:
        return
    all_m = sorted({k for d in mem.values() for k in d})
    N_vals = sorted(mem)
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(N_vals))
    w = 0.8 / max(len(all_m), 1)
    for i, m in enumerate(all_m):
        ax.bar(x + i*w, [mem.get(n, {}).get(m, 0) for n in N_vals], w, label=m[:55])
    ax.set_xlabel("N"); ax.set_ylabel("MB (per launch)"); ax.set_title("Memory Bytes by Kernel and N")
    ax.set_xticks(x + w*len(all_m)/2); ax.set_xticklabels([str(n) for n in N_vals])
    ax.legend(fontsize=5, loc='upper left', ncol=2); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    _save(fig, "ncu_memory_bytes")


def plot_ncu_throughput():
    files = _find_ncu("*metrics.csv")
    if not files:
        return
    TP_METRICS = {"sm__throughput.avg.pct_of_peak_sustained_elapsed",
                  "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                  "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"}
    tp = {}
    for N, f in files.items():
        with open(f) as fh:
            for row in csv.DictReader(fh):
                mname = row.get("Metric Name", "")
                if mname not in TP_METRICS:
                    continue
                kern = row.get("Kernel Name", "").split("(")[0].strip()
                val_str = row.get("Metric Value", "0").replace(",", "")
                try:
                    val = float(val_str)
                except ValueError:
                    continue
                key = f"{kern}: {mname.split('__')[0]}"
                tp.setdefault(N, {})[key] = val

    if not tp:
        return
    all_m = sorted({k for d in tp.values() for k in d})
    N_vals = sorted(tp)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(N_vals))
    w = 0.8 / max(len(all_m), 1)
    for i, m in enumerate(all_m):
        ax.bar(x + i*w, [tp.get(n, {}).get(m, 0) for n in N_vals], w, label=m[:50])
    ax.set_xlabel("N"); ax.set_ylabel("% of Peak"); ax.set_title("Throughput (% of Peak)")
    ax.set_xticks(x + w*len(all_m)/2); ax.set_xticklabels([str(n) for n in N_vals])
    ax.legend(fontsize=6); ax.set_ylim(0, 100); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    _save(fig, "ncu_throughput")


def plot_memory_footprint():
    p = os.path.join(SCALE_DIR, "scalability_summary.csv")
    if not os.path.exists(p):
        return
    N_vals = []
    with open(p) as f:
        for row in csv.DictReader(f):
            N_vals.append(int(row["N"]))
    if not N_vals:
        return
    N_arr = np.array(N_vals, dtype=np.float64)
    forces = N_arr**2 * 16 / 1024**2
    net = N_arr * 16 / 1024**2
    parts = N_arr * 8 * 4 / 1024**2
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(N_vals, parts, net, forces,
                 labels=['Particles (8×float)', 'Net Forces (float4)', 'Forces Matrix (N²×float4)'], alpha=0.7)
    ax.axhline(y=12288, color='red', linestyle='--', label='VRAM Limit (12 GB)')
    ax.set_xlabel("N"); ax.set_ylabel("GPU Memory (MB)"); ax.set_title("Estimated GPU Memory Footprint")
    ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.legend(); ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    _save(fig, "memory_footprint")


def plot_theoretical():
    N = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(N, N**2/N[0]**2, 'o-', label='O(N²) - compute_force')
    ax.plot(N, N*np.log2(N)/(N[0]*np.log2(N[0])), 's-', label='O(N log N) - force_reduction')
    ax.plot(N, N/N[0], '^-', label='O(N) - update_states')
    ax.set_xlabel("N"); ax.set_ylabel("Relative cost"); ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_title("Theoretical Kernel Complexity"); ax.legend(); ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    _save(fig, "theoretical_complexity")


if __name__ == "__main__":
    plot_scalability()
    plot_scalability_raw()
    plot_nsys_kernel_time()
    plot_nsys_kernel_time_stacked()
    plot_nsys_api()
    plot_ncu_sol()
    plot_ncu_occupancy()
    plot_ncu_memory_metrics()
    plot_ncu_throughput()
    plot_memory_footprint()
    plot_theoretical()
    print(f"Plots saved to {PLOTS_DIR}")
