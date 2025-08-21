import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse

def load_paired_second_moment_data(coarsening_ratio=4):
    """Load and compute paired deltas for second moment errors"""
    
    aposteriori_all_path = "results/aposteriori_all.pkl"
    if not os.path.exists(aposteriori_all_path):
        raise FileNotFoundError(f"Data file not found: {aposteriori_all_path}")
    
    with open(aposteriori_all_path, "rb") as f:
        aposteriori_all = pickle.load(f)
    
    print(f"Available configurations: {list(aposteriori_all.keys())}")
    
    # Find all configurations for the given coarsening ratio
    configs = {}
    for key in aposteriori_all.keys():
        if f"_r{coarsening_ratio}_" in key and "box" in key and "pbc" in key:
            # Extract stencil size
            parts = key.split('_')
            s_part = [p for p in parts if p.startswith('s')]
            if s_part:
                stencil_size = int(s_part[0][1:])  # Remove 's' prefix
                configs[stencil_size] = key
    
    if not configs:
        raise ValueError(f"No data found for coarsening ratio {coarsening_ratio}")
    
    print(f"Found stencil sizes for r={coarsening_ratio}: {sorted(configs.keys())}")
    
    # Compute paired deltas for each stencil size
    paired_results = {}
    
    for stencil_size in sorted(configs.keys()):
        key = configs[stencil_size]
        data = aposteriori_all[key]
        
        baseline_values = data['second_moment_baseline_all']
        ours_values = data['second_moment_ours_all']
        
        # Ensure we have the same number of samples
        n_samples = min(len(baseline_values), len(ours_values))
        if n_samples == 0:
            print(f"Warning: No valid samples for stencil {stencil_size}")
            continue
        
        # Compute paired deltas: Δ = (NN error) - (baseline error)
        deltas = []
        successes = []
        for i in range(n_samples):
            baseline_val = baseline_values[i]
            ours_val = ours_values[i]
            
            # Skip NaN values
            if not (np.isnan(baseline_val) or np.isnan(ours_val)):
                # Convert to error magnitudes
                baseline_error_mag = abs(baseline_val)
                ours_error_mag = abs(ours_val)
                
                # Delta of error magnitudes
                delta = ours_error_mag - baseline_error_mag  # negative = NN helps
                deltas.append(delta)
                successes.append(ours_error_mag < baseline_error_mag)
        
        if deltas:
            paired_results[stencil_size] = {
                'deltas': deltas,
                'mean_delta': np.mean(deltas),
                'std_delta': np.std(deltas),
                'n_samples': len(deltas),
                'success_rate': sum(successes) / len(successes) * 100
            }
            
            print(f"Stencil {stencil_size}: {len(deltas)} valid samples")
            print(f"  Mean Δ: {np.mean(deltas):.6e} ± {np.std(deltas):.6e}")
            print(f"  NN helped in {sum(successes)}/{len(deltas)} cases ({paired_results[stencil_size]['success_rate']:.1f}%)")
        else:
            print(f"Warning: No valid deltas for stencil {stencil_size}")
    
    return paired_results

def load_paired_second_moment_data_relative(coarsening_ratio=4):
    """Load and compute paired deltas using RELATIVE (percentage) errors"""
    
    aposteriori_all_path = "results/aposteriori_all.pkl"
    if not os.path.exists(aposteriori_all_path):
        raise FileNotFoundError(f"Data file not found: {aposteriori_all_path}")
    
    with open(aposteriori_all_path, "rb") as f:
        aposteriori_all = pickle.load(f)
    
    print(f"Available configurations: {list(aposteriori_all.keys())}")
    
    # Find all configurations for the given coarsening ratio
    configs = {}
    for key in aposteriori_all.keys():
        if f"_r{coarsening_ratio}_" in key and "box" in key and "pbc" in key:
            parts = key.split('_')
            s_part = [p for p in parts if p.startswith('s')]
            if s_part:
                stencil_size = int(s_part[0][1:])
                configs[stencil_size] = key
    
    if not configs:
        raise ValueError(f"No data found for coarsening ratio {coarsening_ratio}")
    
    print(f"Found stencil sizes for r={coarsening_ratio}: {sorted(configs.keys())}")
    
    # Compute paired relative error deltas for each stencil size
    paired_results = {}
    
    for stencil_size in sorted(configs.keys()):
        key = configs[stencil_size]
        data = aposteriori_all[key]
        
        baseline_signed = data['second_moment_baseline_all']
        ours_signed = data['second_moment_ours_all']
        
        # Check if truth values are available
        truth_signed = data.get('second_moment_truth_all', None)
        if truth_signed is None:
            print(f"Warning: No truth values found for {stencil_size}. Trying to compute from baseline...")
            # If no explicit truth values, you might need to compute them differently
            # This is a placeholder - you'll need to adjust based on your data structure
            print(f"  Skipping relative error calculation for stencil {stencil_size}")
            continue
        
        # Pairwise filter NaNs
        baseline_signed, ours_signed, truth_signed = _pairwise_filter_3(
            baseline_signed, ours_signed, truth_signed)
        
        if len(baseline_signed) == 0:
            print(f"Warning: No valid samples for stencil {stencil_size}")
            continue
        
        # Compute RELATIVE errors (as percentages)
        relative_errors_baseline = []
        relative_errors_ours = []
        valid_indices = []
        
        for i in range(len(baseline_signed)):
            truth_val = truth_signed[i]
            baseline_val = baseline_signed[i] 
            ours_val = ours_signed[i]
            
            # Skip if truth is too close to zero (avoid division by zero)
            if abs(truth_val) < 1e-10:
                print(f"    Skipping IC {i}: truth value too close to zero ({truth_val})")
                continue
                
            # Convert errors to relative errors (as percentages)
            # baseline_val and ours_val are ALREADY errors, just normalize them

            # |truth^2 - baseline^2| / |truth^2| * 100
            baseline_rel_error = abs(baseline_val) / abs(truth_val) * 100
            ours_rel_error = abs(ours_val) / abs(truth_val) * 100
            
            relative_errors_baseline.append(baseline_rel_error)
            relative_errors_ours.append(ours_rel_error)
            valid_indices.append(i)
        
        if len(relative_errors_baseline) == 0:
            print(f"Warning: No valid relative errors for stencil {stencil_size}")
            continue
        
        # Success: NN has lower relative error
        successes = [ours < baseline for baseline, ours in 
                     zip(relative_errors_baseline, relative_errors_ours)]
        
        paired_results[stencil_size] = {
            'n_samples': len(relative_errors_baseline),
            'success_rate': sum(successes) / len(successes) * 100,
            'mean_baseline_percent': np.mean(relative_errors_baseline),
            'mean_ours_percent': np.mean(relative_errors_ours),
            'std_baseline_percent': np.std(relative_errors_baseline),
            'std_ours_percent': np.std(relative_errors_ours),
            'relative_errors_baseline': relative_errors_baseline,
            'relative_errors_ours': relative_errors_ours,
            'valid_indices': valid_indices
        }
        
        print(f"Stencil {stencil_size}: {len(relative_errors_baseline)} valid samples")
        print(f"  Baseline error: {np.mean(relative_errors_baseline):.1f}% ± {np.std(relative_errors_baseline):.1f}%")
        print(f"  NN error: {np.mean(relative_errors_ours):.1f}% ± {np.std(relative_errors_ours):.1f}%") 
        print(f"  NN helped in {sum(successes)}/{len(successes)} cases ({sum(successes)/len(successes)*100:.1f}%)")
    
    return paired_results

def _pairwise_filter_3(a, b, c):
    """Remove NaNs from three arrays simultaneously"""
    a, b, c = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(c, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b) | np.isnan(c))
    return a[mask], b[mask], c[mask]

def plot_paired_second_moment_comparison(coarsening_ratio=4):
    """Create bar plot showing paired second moment delta comparison"""
    
    try:
        paired_results = load_paired_second_moment_data(coarsening_ratio)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    if not paired_results:
        print("No valid data to plot")
        return
    
    # Prepare data for plotting
    stencil_sizes = sorted(paired_results.keys())
    means = [paired_results[s]['mean_delta'] for s in stencil_sizes]
    stds = [paired_results[s]['std_delta'] for s in stencil_sizes]
    success_rates = [paired_results[s]['success_rate'] for s in stencil_sizes]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(stencil_sizes))
    
    # Create bars with error bars
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                  color='green', edgecolor='darkgreen', linewidth=1.5,
                  label='Δ Second Moment Error')
    
    # Add horizontal line at y=0 (no improvement)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1,
               label='No improvement (Δ=0)')
    
    # Customize plot
    ax.set_xlabel('Stencil Size', fontsize=12)
    ax.set_ylabel('Δ Second Moment Error\n(NN - Baseline)', fontsize=12)
    ax.set_title(f'Paired Comparison: Second Moment Error Improvement\n(Coarsening Ratio r={coarsening_ratio}, Negative = NN Helps)', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stencil_sizes)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add value labels on bars
    for i, (bar, mean, std, success_rate) in enumerate(zip(bars, means, stds, success_rates)):
        height = bar.get_height()
        
        # Main value label
        ax.annotate(f'{mean:.2e}\n±{std:.2e}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3 if height >= 0 else -35),
                   textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=9, fontweight='bold')
        
        # Success rate label
        ax.annotate(f'{success_rate:.0f}% improved',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 25 if height >= 0 else -55),
                   textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=8, style='italic', color='blue')
    
    plt.tight_layout()
    
    # Save plot
    stencil_str = "_".join([str(s) for s in stencil_sizes])
    filename = f'paired_second_moment_r{coarsening_ratio}_{stencil_str}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved as: {filename}")
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"PAIRED SECOND MOMENT ANALYSIS SUMMARY (r={coarsening_ratio})")
    print(f"{'='*60}")
    
    for s in stencil_sizes:
        data = paired_results[s]
        print(f"\nStencil Size {s}:")
        print(f"  Mean Δ: {data['mean_delta']:.6e} ± {data['std_delta']:.6e}")
        print(f"  Samples: {data['n_samples']}")
        print(f"  Success Rate: {data['success_rate']:.1f}% (NN helped)")
        
        # Statistical significance (rough estimate)
        if data['n_samples'] > 1:
            t_stat = abs(data['mean_delta']) / (data['std_delta'] / np.sqrt(data['n_samples']))
            print(f"  |t-statistic|: {t_stat:.2f} {'(likely significant)' if t_stat > 2 else '(not significant)'}")

def plot_paired_second_moment_comparison_relative(coarsening_ratio=4):
    """Create bar plot showing baseline vs NN relative second moment errors side by side"""
    
    try:
        paired_results = load_paired_second_moment_data_relative(coarsening_ratio)
    except Exception as e:
        print(f"Error loading relative data: {e}")
        return
    
    if not paired_results:
        print("No valid data to plot")
        return
    
    # Prepare data for plotting
    stencil_sizes = sorted(paired_results.keys())
    baseline_means = [paired_results[s]['mean_baseline_percent'] for s in stencil_sizes]
    ours_means = [paired_results[s]['mean_ours_percent'] for s in stencil_sizes]
    baseline_stds = [paired_results[s]['std_baseline_percent'] for s in stencil_sizes]
    ours_stds = [paired_results[s]['std_ours_percent'] for s in stencil_sizes]
    success_rates = [paired_results[s]['success_rate'] for s in stencil_sizes]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(stencil_sizes))
    width = 0.35  # Width of bars
    
    # Create side-by-side bars
    bars1 = ax.bar(x_pos - width/2, baseline_means, width, yerr=baseline_stds, 
                   capsize=5, alpha=0.8, color='red', edgecolor='darkred', 
                   linewidth=1.5, label='Baseline Model')
    
    bars2 = ax.bar(x_pos + width/2, ours_means, width, yerr=ours_stds, 
                   capsize=5, alpha=0.8, color='blue', edgecolor='darkblue', 
                   linewidth=1.5, label='Neural Network Model')
    
    # Customize plot
    ax.set_xlabel('Stencil Size', fontsize=12)
    ax.set_ylabel('Relative Second Moment Error (%)', fontsize=12)
    ax.set_title(f'Comparison: Baseline vs NN Relative Second Moment Errors\n(Coarsening Ratio r={coarsening_ratio}, Lower = Better)', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stencil_sizes)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add value labels on bars
    for i, (bar1, bar2, base_mean, base_std, ours_mean, ours_std, success_rate) in enumerate(
        zip(bars1, bars2, baseline_means, baseline_stds, ours_means, ours_stds, success_rates)):
        
        # Baseline bar label
        ax.annotate(f'{base_mean:.1f}%\n±{base_std:.1f}%',
                   xy=(bar1.get_x() + bar1.get_width()/2, bar1.get_height()),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=9, fontweight='bold', color='darkred')
        
        # NN bar label  
        ax.annotate(f'{ours_mean:.1f}%\n±{ours_std:.1f}%',
                   xy=(bar2.get_x() + bar2.get_width()/2, bar2.get_height()),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=9, fontweight='bold', color='darkblue')
        
        # Success rate label between bars
        max_height = max(bar1.get_height() + base_std, bar2.get_height() + ours_std)
        ax.annotate(f'{success_rate:.0f}% improved',
                   xy=(x_pos[i], max_height),
                   xytext=(0, 15),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=10, style='italic', color='green', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    stencil_str = "_".join([str(s) for s in stencil_sizes])
    filename = f'relative_second_moment_comparison_r{coarsening_ratio}_{stencil_str}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved as: {filename}")
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"RELATIVE SECOND MOMENT ERROR COMPARISON (r={coarsening_ratio})")
    print(f"{'='*60}")
    
    for s in stencil_sizes:
        data = paired_results[s]
        improvement = data['mean_baseline_percent'] - data['mean_ours_percent']
        print(f"\nStencil Size {s}:")
        print(f"  Baseline Error: {data['mean_baseline_percent']:.1f}% ± {data['std_baseline_percent']:.1f}%")
        print(f"  NN Error: {data['mean_ours_percent']:.1f}% ± {data['std_ours_percent']:.1f}%")
        print(f"  Improvement: {improvement:.1f} percentage points")
        print(f"  Samples: {data['n_samples']}")
        print(f"  Success Rate: {data['success_rate']:.1f}% (NN helped)")

def main():
    """Main function to run paired second moment analysis"""
    parser = argparse.ArgumentParser(description='Paired comparison analysis for second moment errors')
    parser.add_argument('-r', '--coarsening_ratio', type=int, default=4, choices=[2, 4, 8],
                        help='Coarsening ratio to analyze (2, 4, or 8). Default: 4')
    parser.add_argument('--method', type=str, default='absolute', choices=['absolute', 'relative'],
                        help='Analysis method: absolute (raw differences) or relative (percentage errors). Default: absolute')
    
    args = parser.parse_args()
    
    print(f"Running paired second moment analysis for coarsening ratio: {args.coarsening_ratio}")
    print(f"Method: {args.method}")
    
    if args.method == 'absolute':
        plot_paired_second_moment_comparison(args.coarsening_ratio)
    elif args.method == 'relative':
        plot_paired_second_moment_comparison_relative(args.coarsening_ratio)

if __name__ == "__main__":
    main()