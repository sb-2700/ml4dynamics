import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse

def get_stencil_sizes_for_coarsening_ratio(coarsening_ratio):
    """Get the appropriate stencil sizes for a given coarsening ratio
    
    Args:
        coarsening_ratio: Coarsening ratio (2, 4, or 8)
    
    Returns:
        List of odd stencil sizes to test for this coarsening ratio
    """
    if coarsening_ratio == 2:
        return [3, 5, 7, 9]
    elif coarsening_ratio == 4:
        return [5, 7, 9, 11]
    elif coarsening_ratio == 8:
        return [9, 11, 13, 15, 17, 19]
    else:
        raise ValueError(f"Unsupported coarsening ratio: {coarsening_ratio}. Use 2, 4, or 8.")

def compare_stencil_fields(coarsening_ratio=4, stencil_sizes=None):
    """Compare filtered fields and correction stresses across different stencil sizes for box filter
    
    Args:
        coarsening_ratio: Coarsening ratio (2, 4, or 8)
        stencil_sizes: List of stencil sizes to compare. If None, uses default for coarsening ratio
    """
    
    if stencil_sizes is None:
        stencil_sizes = get_stencil_sizes_for_coarsening_ratio(coarsening_ratio)
    
    # File paths for each stencil size (box filter only) with coarsening ratio
    # Assuming naming convention: data/ks/pbc_nu1.0_c0.0_n10_r{r}_box_s{stencil_size}.h5
    stencil_files = {}
    for size in stencil_sizes:
        stencil_files[size] = f"data/ks/pbc_nu1.0_c0.0_n10_r{coarsening_ratio}_box_s{size}.h5"
    
    print(f"Comparing box filter with stencil sizes: {stencil_sizes} at coarsening ratio: {coarsening_ratio}")
    
    # Load datasets dynamically
    data = {}
    for stencil_size in stencil_sizes:
        file_path = stencil_files[stencil_size]
        if not os.path.exists(file_path):
            print(f"Warning: Dataset not found: {file_path}")
            continue
            
        with h5py.File(file_path, "r") as f:
            data[stencil_size] = {
                'filtered_field': f["data"]["inputs"][:],
                'filter_stress': f["data"]["outputs_filter"][:],
                'correction_stress': f["data"]["outputs_correction"][:]
            }
    
    # Update stencil_sizes to only include available data
    available_stencils = list(data.keys())
    if len(available_stencils) < 2:
        raise ValueError(f"Need at least 2 stencil sizes with data. Found: {available_stencils}")
    
    stencil_sizes = available_stencils
    print(f"Available stencil sizes: {stencil_sizes}")
    
    # Print shapes
    for stencil_size in stencil_sizes:
        print(f"Stencil {stencil_size} filtered field shape: {data[stencil_size]['filtered_field'].shape}")
    
    # Reshape the data back to (simulations, time_steps, spatial_points, channels)
    num_sims = 10  # from config case_num
    time_steps_per_sim = data[stencil_sizes[0]]['filtered_field'].shape[0] // num_sims
    
    for stencil_size in stencil_sizes:
        for key in data[stencil_size]:
            data[stencil_size][key] = data[stencil_size][key].reshape(num_sims, time_steps_per_sim, -1, 1)
    
    print(f"Reshaped to: {data[stencil_sizes[0]]['filtered_field'].shape}")
    
    # Calculate global min/max for consistent color scales
    print("\nCalculating global color scale ranges...")
    
    sim_idx = 0  # First simulation
    
    # Global ranges for each data type
    field_min = float('inf')
    field_max = float('-inf')
    filter_stress_min = float('inf') 
    filter_stress_max = float('-inf')
    correction_stress_min = float('inf')
    correction_stress_max = float('-inf')
    
    # Find global ranges across all stencil sizes
    for stencil_size in stencil_sizes:
        field_data = data[stencil_size]['filtered_field'][sim_idx, :, :, 0]
        filter_stress_data = data[stencil_size]['filter_stress'][sim_idx, :, :, 0]
        correction_stress_data = data[stencil_size]['correction_stress'][sim_idx, :, :, 0]
        
        field_min = min(field_min, np.min(field_data))
        field_max = max(field_max, np.max(field_data))
        filter_stress_min = min(filter_stress_min, np.min(filter_stress_data))
        filter_stress_max = max(filter_stress_max, np.max(filter_stress_data))
        correction_stress_min = min(correction_stress_min, np.min(correction_stress_data))
        correction_stress_max = max(correction_stress_max, np.max(correction_stress_data))
    
    print(f"Filtered field range: [{field_min:.3f}, {field_max:.3f}]")
    print(f"Filter stress range: [{filter_stress_min:.3e}, {filter_stress_max:.3e}]")
    print(f"Correction stress range: [{correction_stress_min:.3e}, {correction_stress_max:.3e}]")
    
    # Create visualizations - only individual stencil results
    n_stencils = len(stencil_sizes)
    
    fig, axes = plt.subplots(3, n_stencils, figsize=(5*n_stencils, 12))
    
    # Handle single column case
    if n_stencils == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot individual stencil results with consistent color scales
    for i, stencil_size in enumerate(stencil_sizes):
        # Row 1: Filtered fields (consistent scale)
        im = axes[0,i].imshow(data[stencil_size]['filtered_field'][sim_idx, :, :, 0].T, 
                            aspect='auto', cmap='RdBu_r', vmin=field_min, vmax=field_max)
        axes[0,i].set_title(f'Box Filter (s={stencil_size})\nFiltered Field')
        if i == 0:
            axes[0,i].set_ylabel('Space')
        plt.colorbar(im, ax=axes[0,i])
        
        # Row 2: Filter stresses (consistent scale)
        im = axes[1,i].imshow(data[stencil_size]['filter_stress'][sim_idx, :, :, 0].T, 
                            aspect='auto', cmap='RdBu_r', vmin=filter_stress_min, vmax=filter_stress_max)
        axes[1,i].set_title(f'Box Filter (s={stencil_size})\nFilter Stress')
        if i == 0:
            axes[1,i].set_ylabel('Space')
        plt.colorbar(im, ax=axes[1,i])
        
        # Row 3: Correction stresses (consistent scale)
        im = axes[2,i].imshow(data[stencil_size]['correction_stress'][sim_idx, :, :, 0].T, 
                            aspect='auto', cmap='RdBu_r', vmin=correction_stress_min, vmax=correction_stress_max)
        axes[2,i].set_title(f'Box Filter (s={stencil_size})\nCorrection Stress')
        if i == 0:
            axes[2,i].set_ylabel('Space')
        axes[2,i].set_xlabel('Time')
        plt.colorbar(im, ax=axes[2,i])
    
    plt.tight_layout()
    stencil_str = "_".join([str(s) for s in stencil_sizes])
    plt.savefig(f'stencil_comparison_box_r{coarsening_ratio}_{stencil_str}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Return the data for further analysis
    return {'data': data}

def compare_stencil_errors(coarsening_ratio=4, stencil_sizes=None):
    """Compare a priori and a posteriori errors across different stencil sizes for box filter
    
    Args:
        coarsening_ratio: Coarsening ratio (2, 4, or 8)
        stencil_sizes: List of stencil sizes to compare. If None, uses default for coarsening ratio
    """
    
    if stencil_sizes is None:
        stencil_sizes = get_stencil_sizes_for_coarsening_ratio(coarsening_ratio)
    
    print(f"\n=== LOADING ERROR METRICS FOR BOX FILTER ===")
    print(f"Comparing stencil sizes: {stencil_sizes} at coarsening ratio: {coarsening_ratio}")
    
    # Load a priori errors from pickle file
    train_losses_path = "results/train_losses.pkl"
    if os.path.exists(train_losses_path):
        with open(train_losses_path, "rb") as f:
            train_losses = pickle.load(f)
        print(f"A priori losses loaded: {list(train_losses.keys())}")
    else:
        print(f"Warning: {train_losses_path} not found")
        train_losses = {}
    
    # Load a posteriori metrics from pickle file
    aposteriori_path = "results/aposteriori_metrics.pkl"
    if os.path.exists(aposteriori_path):
        with open(aposteriori_path, "rb") as f:
            aposteriori_metrics = pickle.load(f)
        print(f"A posteriori metrics loaded: {list(aposteriori_metrics.keys())}")
    else:
        print(f"Warning: {aposteriori_path} not found")
        aposteriori_metrics = {}
    
    # Helper function to find the correct key for a stencil size with coarsening ratio
    def find_stencil_key(stencil_size, data_dict, coarsening_ratio):
        # New format with coarsening ratio: box_pbc_r{r}_s{stencil_size}
        key = f"box_pbc_r{coarsening_ratio}_s{stencil_size}"
        if key in data_dict:
            return key
        
        # Fallback: look for partial matches that include the stencil size and coarsening ratio
        for key in data_dict.keys():
            if (str(stencil_size) in key and "box" in key and 
                f"r{coarsening_ratio}" in key and "pbc" in key):
                return key
        
        return None
    
    # Filter out stencil sizes that don't have data
    available_stencils = []
    for s in stencil_sizes:
        # Check if we have either a priori or a posteriori data
        train_key = find_stencil_key(s, train_losses, coarsening_ratio)
        apost_key = find_stencil_key(s, aposteriori_metrics, coarsening_ratio)
        
        if train_key or apost_key:
            available_stencils.append(s)
        else:
            print(f"Warning: No data found for stencil size {s} with box filter")
    
    if len(available_stencils) < 2:
        print(f"Insufficient data for comparison. Available stencils: {available_stencils}")
        return
    
    stencil_sizes = available_stencils
    print(f"Available stencil sizes: {stencil_sizes}")
    
    # Create error comparison plots
    n_stencils = len(stencil_sizes)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: A Priori Training Loss vs Stencil Size
    train_vals = []
    available_train_stencils = []
    
    for s in stencil_sizes:
        key = find_stencil_key(s, train_losses, coarsening_ratio)
        if key:
            loss_data = train_losses[key]
            if isinstance(loss_data, dict):
                # Use relative MSE if available, otherwise fallback to absolute MSE
                if 'rel_mse_mean' in loss_data:
                    mean_val = loss_data.get('rel_mse_mean', float('nan'))
                else:
                    # Fallback to absolute MSE
                    mean_val = loss_data.get('mean', float('nan'))
                train_vals.append(mean_val)
            else:
                # Old format (single value)
                train_vals.append(loss_data)
            available_train_stencils.append(s)
        else:
            print(f"No a priori data for stencil {s}")
    
    if available_train_stencils:
        bars = ax1.bar(available_train_stencils, train_vals, 
                      alpha=0.7, color='skyblue', edgecolor='navy', linewidth=1.5)
        ax1.set_xlabel('Stencil Size')
        # Check if we're using relative MSE or absolute MSE
        if train_losses and any('rel_mse_mean' in v for v in train_losses.values() if isinstance(v, dict)):
            ax1.set_ylabel('A Priori Relative MSE')
            ax1.set_title('A Priori Relative MSE vs Stencil Size\n(Box Filter)')
        else:
            ax1.set_ylabel('A Priori Loss (MSE)')
            ax1.set_title('A Priori Loss vs Stencil Size\n(Box Filter)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on top of bars
        for bar, y in zip(bars, train_vals):
            height = bar.get_height()
            if y < 1e-2:
                ax1.annotate(f'{y:.2e}', xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords="offset points", 
                            ha='center', va='bottom', fontsize=8)
            else:
                ax1.annotate(f'{y:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords="offset points", 
                            ha='center', va='bottom', fontsize=8)
    
    # Plots 2-4: A Posteriori Metrics vs Stencil Size
    metric_info = [
        ("L2 Error", ax2, "l2"),
        ("1st Moment Error", ax3, "first_moment"), 
        ("2nd Moment Error", ax4, "second_moment"),
    ]
    
    for title, ax, metric in metric_info:
        baseline_vals = []
        ours_vals = []
        baseline_errs = []
        ours_errs = []
        available_apost_stencils = []
        
        for s in stencil_sizes:
            key = find_stencil_key(s, aposteriori_metrics, coarsening_ratio)
            if key:
                data = aposteriori_metrics[key]
                
                # Baseline values
                baseline_val = data.get(f"{metric}_baseline", float('nan'))
                baseline_std = data.get(f"{metric}_baseline_std", 0.0)
                
                # Ours values
                ours_val = data.get(f"{metric}_ours", float('nan'))
                ours_std = data.get(f"{metric}_ours_std", 0.0)
                
                if not (np.isnan(baseline_val) and np.isnan(ours_val)):
                    baseline_vals.append(baseline_val)
                    ours_vals.append(ours_val)
                    baseline_errs.append(baseline_std if not np.isnan(baseline_std) and baseline_std > 0 else 0)
                    ours_errs.append(ours_std if not np.isnan(ours_std) and ours_std > 0 else 0)
                    available_apost_stencils.append(s)
        
        if available_apost_stencils:
            # Use absolute values for all metrics
            baseline_vals_plot = [abs(x) for x in baseline_vals]
            ours_vals_plot = [abs(x) for x in ours_vals]
            ylabel = f'|{title}|'
            title_suffix = ", Absolute Values"
            
            # Create grouped bar chart
            x_pos = np.arange(len(available_apost_stencils))
            width = 0.35
            
            # Plot baseline and ours bars side by side
            bars1 = ax.bar(x_pos - width/2, baseline_vals_plot, width, yerr=baseline_errs, 
                          capsize=5, alpha=0.7, label='Baseline (No NN)', 
                          color='gray', edgecolor='black', linewidth=1)
            bars2 = ax.bar(x_pos + width/2, ours_vals_plot, width, yerr=ours_errs, 
                          capsize=5, alpha=0.7, label='With NN', 
                          color='blue', edgecolor='darkblue', linewidth=1)
            
            ax.set_xlabel('Stencil Size')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{title} vs Stencil Size\n(Box Filter{title_suffix})')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(available_apost_stencils)
            ax.grid(True, alpha=0.3)
            
            # Set scale: log for 1st moment only, linear for L2 and 2nd moment
            if metric == "first_moment":
                ax.set_yscale('log')
            else:
                ax.set_yscale('linear')
            
            ax.legend()
            
            # Add improvement percentages above the "With NN" bars
            for i, (bar, baseline, ours) in enumerate(zip(bars2, baseline_vals, ours_vals)):
                if not (np.isnan(baseline) or np.isnan(ours)) and baseline != 0:
                    improvement = (baseline - ours) / baseline * 100
                    height = bar.get_height()
                    ax.annotate(f'{improvement:+.1f}%', 
                               xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 3), textcoords="offset points", 
                               ha='center', va='bottom', fontsize=7, color='blue')
    
    plt.suptitle(f'Box Filter: A Priori and A Posteriori Metrics vs Stencil Size (r={coarsening_ratio})', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    stencil_str = "_".join([str(s) for s in stencil_sizes])
    plt.savefig(f'stencil_error_comparison_box_r{coarsening_ratio}_{stencil_str}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Print numerical comparison
    print(f"\n=== NUMERICAL COMPARISON (BOX FILTER) ===")
    print("A Priori (Training Loss):")
    for s in available_train_stencils:
        key = find_stencil_key(s, train_losses, coarsening_ratio)
        if key:
            loss_data = train_losses[key]
            if isinstance(loss_data, dict):
                mean_val = loss_data.get('mean', float('nan'))
                std_val = loss_data.get('std', float('nan'))
                print(f"  Stencil {s}: {mean_val:.6e} ± {std_val:.6e}")
            else:
                print(f"  Stencil {s}: {loss_data:.6e}")
    
    print("\nA Posteriori Metrics:")
    for metric in ["l2", "first_moment", "second_moment"]:
        print(f"\n{metric.upper()}:")
        for s in stencil_sizes:
            key = find_stencil_key(s, aposteriori_metrics, coarsening_ratio)
            if key:
                data = aposteriori_metrics[key]
                
                baseline_val = data.get(f"{metric}_baseline", float('nan'))
                baseline_std = data.get(f"{metric}_baseline_std", 0.0)
                ours_val = data.get(f"{metric}_ours", float('nan'))
                ours_std = data.get(f"{metric}_ours_std", 0.0)
                
                improvement = (baseline_val - ours_val) / baseline_val * 100 if not np.isnan(baseline_val) and not np.isnan(ours_val) else float('nan')
                
                print(f"  Stencil {s}:")
                print(f"    Baseline: {baseline_val:.6e} ± {baseline_std:.6e}")
                print(f"    Ours:     {ours_val:.6e} ± {ours_std:.6e}")
                print(f"    Improvement: {improvement:.1f}%")
            else:
                print(f"  Stencil {s}: No data found")

def main(coarsening_ratio=4, stencil_sizes=None):
    """Run both stencil field comparison and error comparison for box filter
    
    Args:
        coarsening_ratio: Coarsening ratio (2, 4, or 8)
        stencil_sizes: List of stencil sizes to compare. If None, uses default for coarsening ratio
    """
    
    if stencil_sizes is None:
        stencil_sizes = get_stencil_sizes_for_coarsening_ratio(coarsening_ratio)
    
    print(f"=== BOX FILTER STENCIL SIZE COMPARISON (r={coarsening_ratio}) ===")
    print(f"Stencil sizes: {stencil_sizes}")
    
    # Field and stress comparison
    print("\n=== FILTERED FIELD AND SGS STRESS COMPARISON ===")
    try:
        field_results = compare_stencil_fields(coarsening_ratio, stencil_sizes)
    except Exception as e:
        print(f"Error in field comparison: {e}")
        field_results = None
    
    # Error metrics comparison
    print("\n" + "="*60)
    print("=== ERROR METRICS COMPARISON ===")
    try:
        compare_stencil_errors(coarsening_ratio, stencil_sizes)
    except Exception as e:
        print(f"Error in metrics comparison: {e}")
    
    return field_results

def test_all_coarsening_ratios():
    """Test stencil sizes across all coarsening ratios with their respective stencil ranges"""
    coarsening_ratios = [2, 4, 8]
    
    for r in coarsening_ratios:
        print(f"\n{'='*100}")
        print(f"TESTING COARSENING RATIO: {r}")
        stencil_sizes = get_stencil_sizes_for_coarsening_ratio(r)
        print(f"STENCIL SIZES: {stencil_sizes}")
        print(f"{'='*100}")
        
        try:
            main(r)
        except Exception as e:
            print(f"Error testing coarsening ratio {r}: {e}")
            continue

# Run the comparison for box filter
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare stencil sizes for box filter at different coarsening ratios')
    parser.add_argument('-r', '--coarsening_ratio', type=int, default=4, choices=[2, 4, 8],
                        help='Coarsening ratio to test (2, 4, or 8). Default: 4')
    parser.add_argument('--all', action='store_true', 
                        help='Test all coarsening ratios (2, 4, 8)')
    parser.add_argument('-s', '--stencil_sizes', type=int, nargs='+', 
                        help='Custom stencil sizes to test (e.g., -s 5 7 9). If not provided, uses defaults for the coarsening ratio')
    
    args = parser.parse_args()
    
    if args.all:
        # Test all coarsening ratios
        print("Testing all coarsening ratios...")
        test_all_coarsening_ratios()
    else:
        # Test specific coarsening ratio
        coarsening_ratio = args.coarsening_ratio
        stencil_sizes = args.stencil_sizes
        
        if stencil_sizes is None:
            default_stencils = get_stencil_sizes_for_coarsening_ratio(coarsening_ratio)
            print(f"Testing r={coarsening_ratio} with default stencil sizes {default_stencils}")
        else:
            print(f"Testing r={coarsening_ratio} with custom stencil sizes {stencil_sizes}")
        
        results = main(coarsening_ratio=coarsening_ratio, stencil_sizes=stencil_sizes)
