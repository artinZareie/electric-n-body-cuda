#!/usr/bin/env python3
"""
Generate side-by-side comparison plots for original vs optimized performance
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'original': '#d62728', 'optimized': '#2ca02c'}

def read_scalability_data(csv_path):
    """Read scalability summary CSV file"""
    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row['N'])
            data[n] = {
                'mean': float(row['mean_wall_time_s']),
                'stddev': float(row['stddev_wall_time_s'])
            }
    return data

def main():
    # Read data
    original = read_scalability_data('profiling_original/results/original/scalability/scalability_summary.csv')
    optimized = read_scalability_data('profiling_optimized/results/2026-02-15/scalability/scalability_summary.csv')
    
    n_values = sorted(original.keys())
    
    # Extract arrays for plotting
    orig_times = [original[n]['mean'] for n in n_values]
    orig_std = [original[n]['stddev'] for n in n_values]
    opt_times = [optimized[n]['mean'] for n in n_values]
    opt_std = [optimized[n]['stddev'] for n in n_values]
    speedups = [original[n]['mean'] / optimized[n]['mean'] for n in n_values]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    
    # ============== Plot 1: Wall Time Comparison (log scale) ==============
    ax1 = plt.subplot(2, 3, 1)
    ax1.errorbar(n_values, orig_times, yerr=orig_std, marker='o', linewidth=2, 
                 markersize=8, label='Original', color=colors['original'], capsize=5)
    ax1.errorbar(n_values, opt_times, yerr=opt_std, marker='s', linewidth=2, 
                 markersize=8, label='Optimized', color=colors['optimized'], capsize=5)
    ax1.set_xlabel('Number of Particles (N)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Wall Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('Wall Time Comparison', fontsize=13, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xticks(n_values)
    ax1.set_xticklabels([str(n) for n in n_values], rotation=45)
    
    # ============== Plot 2: Speedup vs N ==============
    ax2 = plt.subplot(2, 3, 2)
    colors_bars = [colors['optimized'] if s > 1 else colors['original'] for s in speedups]
    bars = ax2.bar(range(len(n_values)), speedups, color=colors_bars, alpha=0.7, edgecolor='black')
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='No speedup')
    ax2.set_xlabel('Number of Particles (N)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Speedup Factor', fontsize=11, fontweight='bold')
    ax2.set_title('Speedup (Original / Optimized)', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(n_values)))
    ax2.set_xticklabels([str(n) for n in n_values], rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=10)
    
    # Add value labels on bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        height = bar.get_height()
        label = f'{speedup:.2f}×'
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # ============== Plot 3: Absolute Time Difference ==============
    ax3 = plt.subplot(2, 3, 3)
    time_saved = [orig_times[i] - opt_times[i] for i in range(len(n_values))]
    colors_bars2 = [colors['optimized'] if t > 0 else colors['original'] for t in time_saved]
    bars2 = ax3.bar(range(len(n_values)), time_saved, color=colors_bars2, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('Number of Particles (N)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Time Saved (seconds)', fontsize=11, fontweight='bold')
    ax3.set_title('Absolute Time Improvement', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(len(n_values)))
    ax3.set_xticklabels([str(n) for n in n_values], rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, saved) in enumerate(zip(bars2, time_saved)):
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{saved:.2f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # ============== Plot 4: Kernel Breakdown (N=4096) ==============
    ax4 = plt.subplot(2, 3, 4)
    
    # Original kernel times (from profiling data)
    original_kernels = ['compute_force', 'force_reduction', 'update_particle']
    original_times_4096 = [823.7, 794.7, 4.2]  # ms
    
    # Optimized kernel times
    optimized_kernels = ['compute_force_tiled', 'update_particle']
    optimized_times_4096 = [457.9, 2.5]  # ms
    
    # Create grouped bar chart
    x_pos = np.arange(max(len(original_kernels), len(optimized_kernels)))
    width = 0.35
    
    # Pad optimized to same length
    opt_padded = optimized_times_4096 + [0] * (len(original_kernels) - len(optimized_kernels))
    
    bars1 = ax4.bar(x_pos - width/2, original_times_4096, width, label='Original', 
                    color=colors['original'], alpha=0.7, edgecolor='black')
    bars2 = ax4.bar(x_pos + width/2, opt_padded, width, label='Optimized',
                    color=colors['optimized'], alpha=0.7, edgecolor='black')
    
    ax4.set_xlabel('Kernel', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Time per 1000 steps (ms)', fontsize=11, fontweight='bold')
    ax4.set_title('Kernel Timing Breakdown (N=4096)', fontsize=13, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(['compute_force', 'force_reduction', 'update_particle'], rotation=30, ha='right')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add total time annotations
    ax4.text(0.98, 0.98, f'Original Total: {sum(original_times_4096):.1f} ms',
             transform=ax4.transAxes, ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.text(0.98, 0.88, f'Optimized Total: {sum(optimized_times_4096):.1f} ms',
             transform=ax4.transAxes, ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # ============== Plot 5: Scaling Factor ==============
    ax5 = plt.subplot(2, 3, 5)
    
    # Calculate scaling factors (time ratio per doubling of N)
    orig_scaling = []
    opt_scaling = []
    transitions = []
    
    for i in range(1, len(n_values)):
        orig_ratio = orig_times[i] / orig_times[i-1]
        opt_ratio = opt_times[i] / opt_times[i-1]
        orig_scaling.append(orig_ratio)
        opt_scaling.append(opt_ratio)
        transitions.append(f'{n_values[i-1]}→{n_values[i]}')
    
    x_pos = np.arange(len(transitions))
    width = 0.35
    
    bars1 = ax5.bar(x_pos - width/2, orig_scaling, width, label='Original',
                    color=colors['original'], alpha=0.7, edgecolor='black')
    bars2 = ax5.bar(x_pos + width/2, opt_scaling, width, label='Optimized',
                    color=colors['optimized'], alpha=0.7, edgecolor='black')
    
    ax5.axhline(y=4.0, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='O(N²) theoretical')
    ax5.set_xlabel('N Transition', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Scaling Factor', fontsize=11, fontweight='bold')
    ax5.set_title('Time Scaling per Doubling of N', fontsize=13, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(transitions, rotation=45, ha='right', fontsize=9)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # ============== Plot 6: Improvement Percentage ==============
    ax6 = plt.subplot(2, 3, 6)
    
    improvements = [(orig_times[i] - opt_times[i]) / orig_times[i] * 100 for i in range(len(n_values))]
    colors_bars3 = [colors['optimized'] if imp > 0 else colors['original'] for imp in improvements]
    
    bars3 = ax6.bar(range(len(n_values)), improvements, color=colors_bars3, alpha=0.7, edgecolor='black')
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax6.set_xlabel('Number of Particles (N)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax6.set_title('Performance Improvement Percentage', fontsize=13, fontweight='bold')
    ax6.set_xticks(range(len(n_values)))
    ax6.set_xticklabels([str(n) for n in n_values], rotation=45)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars3, improvements)):
        height = bar.get_height()
        if abs(height) > 1:  # Only show if significant
            label = f'{imp:+.1f}%'
            y_pos = height + (2 if height > 0 else -4)
            ax6.text(bar.get_x() + bar.get_width()/2., y_pos,
                    label, ha='center', va='bottom' if height > 0 else 'top', 
                    fontsize=9, fontweight='bold')
    
    # Add overall title
    fig.suptitle('Performance Comparison: Original vs Optimized N-Body Simulation', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_path = 'profiling_comparison_plots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'✓ Saved comparison plots to: {output_path}')
    
    # Also save individual high-res versions
    
    # High-res speedup chart
    fig2, ax = plt.subplots(figsize=(10, 6))
    colors_bars = [colors['optimized'] if s > 1 else colors['original'] for s in speedups]
    bars = ax.bar(range(len(n_values)), speedups, color=colors_bars, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='No speedup (1.0×)')
    ax.set_xlabel('Number of Particles (N)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speedup Factor (Original / Optimized)', fontsize=14, fontweight='bold')
    ax.set_title('Performance Speedup: Original vs Optimized', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(n_values)))
    ax.set_xticklabels([str(n) for n in n_values], fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=12, loc='upper left')
    
    # Add value labels on bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        height = bar.get_height()
        label = f'{speedup:.2f}×'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.08,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add annotation for best result
    max_speedup_idx = speedups.index(max(speedups))
    max_speedup = speedups[max_speedup_idx]
    ax.annotate(f'Peak: {max_speedup:.2f}× faster\n({n_values[max_speedup_idx]} particles)',
                xy=(max_speedup_idx, max_speedup),
                xytext=(max_speedup_idx - 1.5, max_speedup + 0.5),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', linewidth=2))
    
    plt.tight_layout()
    plt.savefig('speedup_chart.png', dpi=300, bbox_inches='tight')
    print(f'✓ Saved high-res speedup chart to: speedup_chart.png')
    
    # High-res wall time comparison
    fig3, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_values, orig_times, marker='o', linewidth=3, markersize=10, 
            label='Original', color=colors['original'], linestyle='-')
    ax.plot(n_values, opt_times, marker='s', linewidth=3, markersize=10,
            label='Optimized', color=colors['optimized'], linestyle='-')
    ax.fill_between(n_values, 
                     [orig_times[i] - orig_std[i] for i in range(len(n_values))],
                     [orig_times[i] + orig_std[i] for i in range(len(n_values))],
                     alpha=0.2, color=colors['original'])
    ax.fill_between(n_values,
                     [opt_times[i] - opt_std[i] for i in range(len(n_values))],
                     [opt_times[i] + opt_std[i] for i in range(len(n_values))],
                     alpha=0.2, color=colors['optimized'])
    ax.set_xlabel('Number of Particles (N)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Wall Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('Wall Time Comparison: Original vs Optimized', fontsize=16, fontweight='bold', pad=20)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(n) for n in n_values], fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=13, loc='upper left')
    
    # Add time savings annotation at N=8192
    time_saved_8192 = orig_times[-1] - opt_times[-1]
    ax.annotate(f'Time saved: {time_saved_8192:.2f}s\n({time_saved_8192/orig_times[-1]*100:.1f}% faster)',
                xy=(n_values[-1], opt_times[-1]),
                xytext=(n_values[-2], (orig_times[-1] + opt_times[-1])/2),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3', linewidth=2))
    
    plt.tight_layout()
    plt.savefig('wall_time_comparison.png', dpi=300, bbox_inches='tight')
    print(f'✓ Saved high-res wall time comparison to: wall_time_comparison.png')
    
    print('\n' + '='*60)
    print('  All plots generated successfully!')
    print('='*60)
    print('\nGenerated files:')
    print('  1. profiling_comparison_plots.png   (6-panel overview)')
    print('  2. speedup_chart.png                (detailed speedup)')
    print('  3. wall_time_comparison.png         (detailed timing)')
    print('='*60 + '\n')

if __name__ == '__main__':
    main()
