#!/usr/bin/env python3
"""
Plot benchmark results for CUDA video convolution performance analysis.
"""

import matplotlib.pyplot as plt
import io

# Benchmark data
block_sizes = [4, 8, 16, 32]
fps = [1081.92, 2012.21, 2522.61, 1687.68]
times = [0.501882, 0.269852, 0.215253, 0.321742]

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines with markers
ax.plot(block_sizes, fps, marker='o', linewidth=2.5, markersize=10, color='#2E86AB', label='Frames/sec')
ax.axvline(x=16, color='#A23B72', linestyle='--', linewidth=1.5, alpha=0.7, label='Peak (blockDimSize=16)')

# Annotations
for i, (bs, f) in enumerate(zip(block_sizes, fps)):
    ax.annotate(f'{f:.0f}', xy=(bs, f), xytext=(0, 10), textcoords='offset points', 
                ha='center', fontsize=10, fontweight='bold')

# Labels and formatting
ax.set_xlabel('Block Dimension Size (blockDimSize)', fontsize=12, fontweight='bold')
ax.set_ylabel('Throughput (Frames per Second)', fontsize=12, fontweight='bold')
ax.set_title('CUDA Video Convolution Performance vs. Block Size\n(A100 GPU, 1920×1080, 543 frames)', 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='best')
ax.set_xticks(block_sizes)

# Add secondary y-axis for time
ax2 = ax.twinx()
ax2.plot(block_sizes, times, marker='s', linewidth=2.5, markersize=10, color='#F18F01', alpha=0.6, label='Execution Time')
ax2.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold', color='#F18F01')
ax2.tick_params(axis='y', labelcolor='#F18F01')

# Layout and save
plt.tight_layout()
plt.savefig('benchmark_plot.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved as benchmark_plot.png")
plt.close()

# Print summary table
print("\n" + "="*60)
print("BENCHMARK RESULTS SUMMARY")
print("="*60)
print(f"{'Block Size':<15} {'FPS':<20} {'Time (s)':<15}")
print("-"*60)
for bs, f, t in zip(block_sizes, fps, times):
    print(f"{bs:<15} {f:<20.2f} {t:<15.6f}")
print("="*60)
print(f"\nPeak Performance: {max(fps):.2f} FPS at blockDimSize=16")
print(f"Peak Speedup vs blockDimSize=8: {max(fps)/fps[1]:.2f}x")
print(f"Speedup vs blockDimSize=4: {max(fps)/fps[0]:.2f}x\n")
