#!/usr/bin/env python3
"""
Compare scaling factors between local and Petals BLOOM-560m inference
"""

import matplotlib.pyplot as plt
import numpy as np

# Results from local inference
local_results = {
    'batch_sizes': [1, 2, 4, 8, 16],
    'total_tokens_sec': [16.01, 12.92, 20.69, 58.81, 98.56],
    'per_prompt_tokens_sec': [16.01, 6.46, 5.17, 7.35, 6.16]
}

# Results from Petals inference (from your previous run)
petals_results = {
    'batch_sizes': [1, 2, 4, 8, 16],
    'total_tokens_sec': [2.61, 4.28, 7.59, 10.68, 15.15],
    'per_prompt_tokens_sec': [2.61, 2.14, 1.90, 1.33, 0.95]
}

def calculate_scaling_factors(results):
    """Calculate scaling factors relative to batch size 1"""
    baseline_total = results['total_tokens_sec'][0]
    baseline_per_prompt = results['per_prompt_tokens_sec'][0]
    
    total_scaling = [total / baseline_total for total in results['total_tokens_sec']]
    per_prompt_scaling = [per_prompt / baseline_per_prompt for per_prompt in results['per_prompt_tokens_sec']]
    
    return total_scaling, per_prompt_scaling

def calculate_efficiency_ratios(results):
    """Calculate efficiency ratios (actual scaling / expected scaling)"""
    batch_sizes = results['batch_sizes']
    total_scaling, per_prompt_scaling = calculate_scaling_factors(results)
    
    # Expected scaling is linear with batch size
    expected_scaling = batch_sizes
    
    # Efficiency ratio = actual scaling / expected scaling
    total_efficiency = [actual / expected for actual, expected in zip(total_scaling, expected_scaling)]
    per_prompt_efficiency = [actual / expected for actual, expected in zip(per_prompt_scaling, expected_scaling)]
    
    return total_efficiency, per_prompt_efficiency

# Calculate scaling factors
local_total_scaling, local_per_prompt_scaling = calculate_scaling_factors(local_results)
petals_total_scaling, petals_per_prompt_scaling = calculate_scaling_factors(petals_results)

# Calculate efficiency ratios
local_total_efficiency, local_per_prompt_efficiency = calculate_efficiency_ratios(local_results)
petals_total_efficiency, petals_per_prompt_efficiency = calculate_efficiency_ratios(petals_results)

# Print comparison table
print("=" * 80)
print("BATCHING SCALING FACTOR COMPARISON")
print("=" * 80)
print(f"{'Batch':<6} {'Local Total':<12} {'Local Per-Prompt':<16} {'Petals Total':<12} {'Petals Per-Prompt':<16}")
print(f"{'Size':<6} {'Scaling':<12} {'Scaling':<16} {'Scaling':<12} {'Scaling':<16}")
print("-" * 80)

for i, batch_size in enumerate(local_results['batch_sizes']):
    print(f"{batch_size:<6} {local_total_scaling[i]:<12.2f} {local_per_prompt_scaling[i]:<16.2f} "
          f"{petals_total_scaling[i]:<12.2f} {petals_per_prompt_scaling[i]:<16.2f}")

print("\n" + "=" * 80)
print("EFFICIENCY RATIOS (Actual/Expected Scaling)")
print("=" * 80)
print(f"{'Batch':<6} {'Local Total':<12} {'Local Per-Prompt':<16} {'Petals Total':<12} {'Petals Per-Prompt':<16}")
print(f"{'Size':<6} {'Efficiency':<12} {'Efficiency':<16} {'Efficiency':<12} {'Efficiency':<16}")
print("-" * 80)

for i, batch_size in enumerate(local_results['batch_sizes']):
    print(f"{batch_size:<6} {local_total_efficiency[i]:<12.2f} {local_per_prompt_efficiency[i]:<16.2f} "
          f"{petals_total_efficiency[i]:<12.2f} {petals_per_prompt_efficiency[i]:<16.2f}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

# Find best performing batch sizes
local_best_total = local_results['batch_sizes'][np.argmax(local_total_scaling)]
petals_best_total = petals_results['batch_sizes'][np.argmax(petals_total_scaling)]

print(f"Local best total scaling: {max(local_total_scaling):.2f}x at batch size {local_best_total}")
print(f"Petals best total scaling: {max(petals_total_scaling):.2f}x at batch size {petals_best_total}")

# Compare efficiency
local_avg_efficiency = np.mean(local_total_efficiency[1:])  # Exclude batch size 1
petals_avg_efficiency = np.mean(petals_total_efficiency[1:])

print(f"\nLocal average efficiency: {local_avg_efficiency:.2f} (should be 1.0 for perfect scaling)")
print(f"Petals average efficiency: {petals_avg_efficiency:.2f} (should be 1.0 for perfect scaling)")

# Create comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Total throughput scaling
ax1.plot(local_results['batch_sizes'], local_total_scaling, 'o-', label='Local', linewidth=2, markersize=8)
ax1.plot(petals_results['batch_sizes'], petals_total_scaling, 's-', label='Petals', linewidth=2, markersize=8)
ax1.plot(local_results['batch_sizes'], local_results['batch_sizes'], '--', label='Ideal (linear)', alpha=0.5)
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Total Throughput Scaling Factor')
ax1.set_title('Total Throughput Scaling')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Per-prompt scaling
ax2.plot(local_results['batch_sizes'], local_per_prompt_scaling, 'o-', label='Local', linewidth=2, markersize=8)
ax2.plot(petals_results['batch_sizes'], petals_per_prompt_scaling, 's-', label='Petals', linewidth=2, markersize=8)
ax2.axhline(y=1.0, color='red', linestyle='--', label='Ideal (constant)', alpha=0.5)
ax2.set_xlabel('Batch Size')
ax2.set_ylabel('Per-Prompt Scaling Factor')
ax2.set_title('Per-Prompt Performance Scaling')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Efficiency ratios
ax3.plot(local_results['batch_sizes'], local_total_efficiency, 'o-', label='Local', linewidth=2, markersize=8)
ax3.plot(petals_results['batch_sizes'], petals_total_efficiency, 's-', label='Petals', linewidth=2, markersize=8)
ax3.axhline(y=1.0, color='red', linestyle='--', label='Perfect efficiency', alpha=0.5)
ax3.set_xlabel('Batch Size')
ax3.set_ylabel('Efficiency Ratio')
ax3.set_title('Total Throughput Efficiency')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Per-prompt efficiency
ax4.plot(local_results['batch_sizes'], local_per_prompt_efficiency, 'o-', label='Local', linewidth=2, markersize=8)
ax4.plot(petals_results['batch_sizes'], petals_per_prompt_efficiency, 's-', label='Petals', linewidth=2, markersize=8)
ax4.axhline(y=1.0, color='red', linestyle='--', label='Perfect efficiency', alpha=0.5)
ax4.set_xlabel('Batch Size')
ax4.set_ylabel('Efficiency Ratio')
ax4.set_title('Per-Prompt Efficiency')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scaling_factor_comparison.png', dpi=300, bbox_inches='tight')
print(f"\nScaling comparison graph saved as 'scaling_factor_comparison.png'")

# Key insights
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

if local_avg_efficiency > petals_avg_efficiency:
    print("✓ Local inference shows better batching efficiency than Petals")
else:
    print("✗ Petals shows better batching efficiency than local inference")

if max(local_total_scaling) > max(petals_total_scaling):
    print("✓ Local inference achieves higher maximum scaling")
else:
    print("✗ Petals achieves higher maximum scaling")

print(f"\nLocal inference is {local_results['total_tokens_sec'][0] / petals_results['total_tokens_sec'][0]:.1f}x faster for single inference")
print(f"Local inference is {max(local_results['total_tokens_sec']) / max(petals_results['total_tokens_sec']):.1f}x faster at best batch size") 