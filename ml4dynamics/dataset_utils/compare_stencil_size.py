import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def compare_stencil_fields(stencil_sizes=[3, 5, 7, 9, 11], filter_type="gaussian"):
    """Compare filtered fields and correction stresses across different stencil sizes
    
    Args:
        stencil_sizes: List of stencil sizes to compare
        filter_type: Type of filter to use ("box", "gaussian", "spectral")
    """
    
    # Define file paths for each stencil size
    # Assuming naming convention: data/ks/pbc_nu1.0_c0.0_n10_{filter_type}_s{stencil_size}.h5
    stencil_files = {}
    for size in stencil_sizes:
        stencil_files[size] = f"data/ks/pbc_nu1.0_c0.0_n10_{filter_type}_s{size}.h5"
    
    print(f"Comparing {filter_type} filter with stencil sizes: {stencil_sizes}")
    
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
    
    # Calculate pairwise differences (using smallest stencil as reference)
    ref_stencil = min(stencil_sizes)
    differences = {}
    
    for stencil_size in stencil_sizes:
        if stencil_size != ref_stencil:
            diff_key = f"s{stencil_size}_vs_s{ref_stencil}"
            differences[diff_key] = {
                'field_diff': data[stencil_size]['filtered_field'] - data[ref_stencil]['filtered_field'],
                'filter_stress_diff': data[stencil_size]['filter_stress'] - data[ref_stencil]['filter_stress'],
                'correction_stress_diff': data[stencil_size]['correction_stress'] - data[ref_stencil]['correction_stress']
            }
    
    # Print statistics for each comparison
    for diff_key, diffs in differences.items():
        print(f"\n=== {diff_key.upper()} COMPARISON ===")
        
        field_diff = diffs['field_diff']
        print(f"FILTERED FIELD: RMS diff = {np.sqrt(np.mean(field_diff**2)):.6e}")
        
        filter_stress_diff = diffs['filter_stress_diff']
        print(f"FILTER STRESS: RMS diff = {np.sqrt(np.mean(filter_stress_diff**2)):.6e}")
        
        correction_stress_diff = diffs['correction_stress_diff']
        print(f"CORRECTION STRESS: RMS diff = {np.sqrt(np.mean(correction_stress_diff**2)):.6e}")
    
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
    
    # Also calculate ranges for differences
    field_diff_min = float('inf')
    field_diff_max = float('-inf')
    filter_diff_min = float('inf')
    filter_diff_max = float('-inf')
    correction_diff_min = float('inf')
    correction_diff_max = float('-inf')
    
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
    
    # Find global ranges for differences
    for diff_key, diffs in differences.items():
        field_diff_data = diffs['field_diff'][sim_idx, :, :, 0]
        filter_diff_data = diffs['filter_stress_diff'][sim_idx, :, :, 0]
        correction_diff_data = diffs['correction_stress_diff'][sim_idx, :, :, 0]
        
        field_diff_min = min(field_diff_min, np.min(field_diff_data))
        field_diff_max = max(field_diff_max, np.max(field_diff_data))
        filter_diff_min = min(filter_diff_min, np.min(filter_diff_data))
        filter_diff_max = max(filter_diff_max, np.max(filter_diff_data))
        correction_diff_min = min(correction_diff_min, np.min(correction_diff_data))
        correction_diff_max = max(correction_diff_max, np.max(correction_diff_data))
    
    print(f"Filtered field range: [{field_min:.3f}, {field_max:.3f}]")
    print(f"Filter stress range: [{filter_stress_min:.3e}, {filter_stress_max:.3e}]")
    print(f"Correction stress range: [{correction_stress_min:.3e}, {correction_stress_max:.3e}]")
    print(f"Field diff range: [{field_diff_min:.3e}, {field_diff_max:.3e}]")
    print(f"Filter diff range: [{filter_diff_min:.3e}, {filter_diff_max:.3e}]")
    print(f"Correction diff range: [{correction_diff_min:.3e}, {correction_diff_max:.3e}]")
    
    # Create visualizations
    n_stencils = len(stencil_sizes)
    n_comparisons = len(differences)
    n_cols = n_stencils + n_comparisons
    
    fig, axes = plt.subplots(3, n_cols, figsize=(5*n_cols, 12))
    
    # Handle single column case
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot individual stencil results with consistent color scales
    for i, stencil_size in enumerate(stencil_sizes):
        # Row 1: Filtered fields (consistent scale)
        im = axes[0,i].imshow(data[stencil_size]['filtered_field'][sim_idx, :, :, 0].T, 
                            aspect='auto', cmap='RdBu_r', vmin=field_min, vmax=field_max)
        axes[0,i].set_title(f'Stencil {stencil_size}\nFiltered Field')
        if i == 0:
            axes[0,i].set_ylabel('Space')
        plt.colorbar(im, ax=axes[0,i])
        
        # Row 2: Filter stresses (consistent scale)
        im = axes[1,i].imshow(data[stencil_size]['filter_stress'][sim_idx, :, :, 0].T, 
                            aspect='auto', cmap='RdBu_r', vmin=filter_stress_min, vmax=filter_stress_max)
        axes[1,i].set_title(f'Stencil {stencil_size}\nFilter Stress')
        if i == 0:
            axes[1,i].set_ylabel('Space')
        plt.colorbar(im, ax=axes[1,i])
        
        # Row 3: Correction stresses (consistent scale)
        im = axes[2,i].imshow(data[stencil_size]['correction_stress'][sim_idx, :, :, 0].T, 
                            aspect='auto', cmap='RdBu_r', vmin=correction_stress_min, vmax=correction_stress_max)
        axes[2,i].set_title(f'Stencil {stencil_size}\nCorrection Stress')
        if i == 0:
            axes[2,i].set_ylabel('Space')
        axes[2,i].set_xlabel('Time')
        plt.colorbar(im, ax=axes[2,i])
    
    # Plot difference comparisons with consistent scales for differences
    for i, (diff_key, diffs) in enumerate(differences.items()):
        col_idx = n_stencils + i
        
        # Row 1: Field differences (consistent diff scale)
        im = axes[0,col_idx].imshow(diffs['field_diff'][sim_idx, :, :, 0].T, 
                                  aspect='auto', cmap='RdBu_r', vmin=field_diff_min, vmax=field_diff_max)
        axes[0,col_idx].set_title(f'Field Diff: {diff_key.replace("_vs_", " vs ")}')
        plt.colorbar(im, ax=axes[0,col_idx])
        
        # Row 2: Filter stress differences (consistent diff scale)
        im = axes[1,col_idx].imshow(diffs['filter_stress_diff'][sim_idx, :, :, 0].T, 
                                  aspect='auto', cmap='RdBu_r', vmin=filter_diff_min, vmax=filter_diff_max)
        axes[1,col_idx].set_title(f'Filter Stress Diff: {diff_key.replace("_vs_", " vs ")}')
        plt.colorbar(im, ax=axes[1,col_idx])
        
        # Row 3: Correction stress differences (consistent diff scale)
        im = axes[2,col_idx].imshow(diffs['correction_stress_diff'][sim_idx, :, :, 0].T, 
                                  aspect='auto', cmap='RdBu_r', vmin=correction_diff_min, vmax=correction_diff_max)
        axes[2,col_idx].set_title(f'Correction Stress Diff: {diff_key.replace("_vs_", " vs ")}')
        axes[2,col_idx].set_xlabel('Time')
        plt.colorbar(im, ax=axes[2,col_idx])
    
    plt.tight_layout()
    stencil_str = "_".join([str(s) for s in stencil_sizes])
    plt.savefig(f'stencil_comparison_{filter_type}_{stencil_str}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Return the data for further analysis
    result = {
        'data': data, 
        'differences': differences
    }
    return result

def compare_stencil_errors(stencil_sizes=[3, 5, 7, 9, 11], filter_type="gaussian"):
    """Compare a priori and a posteriori errors across different stencil sizes
    
    Args:
        stencil_sizes: List of stencil sizes to compare
        filter_type: Type of filter used ("box", "gaussian", "spectral")
    """
    
    print(f"\n=== LOADING ERROR METRICS FOR {filter_type.upper()} FILTER ===")
    print(f"Comparing stencil sizes: {stencil_sizes}")
    
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
    
    # Helper function to find the correct key for a stencil size
    def find_stencil_key(stencil_size, filter_type, data_dict):
        # Try different key formats
        possible_keys = [
            f"{filter_type}_s{stencil_size}_pbc",  # New format with stencil and BC
            f"{filter_type}_s{stencil_size}",      # Format with stencil
            f"{filter_type}_{stencil_size}_pbc",   # Alternative format
            f"{filter_type}_{stencil_size}",       # Alternative format
            f"s{stencil_size}_{filter_type}_pbc",  # Another format
            f"s{stencil_size}_{filter_type}",      # Another format
        ]
        
        for key in possible_keys:
            if key in data_dict:
                return key
        
        # Fallback: look for partial matches
        for key in data_dict.keys():
            if str(stencil_size) in key and filter_type in key:
                return key
        
        return None
    
    # Filter out stencil sizes that don't have data
    available_stencils = []
    for s in stencil_sizes:
        # Check if we have either a priori or a posteriori data
        train_key = find_stencil_key(s, filter_type, train_losses)
        apost_key = find_stencil_key(s, filter_type, aposteriori_metrics)
        
        if train_key or apost_key:
            available_stencils.append(s)
        else:
            print(f"Warning: No data found for stencil size {s} with {filter_type} filter")
    
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
    train_errs = []
    available_train_stencils = []
    
    for s in stencil_sizes:
        key = find_stencil_key(s, filter_type, train_losses)
        if key:
            loss_data = train_losses[key]
            if isinstance(loss_data, dict):
                # New format with mean/std
                mean_val = loss_data.get('mean', float('nan'))
                std_val = loss_data.get('std', 0.0)
                train_vals.append(mean_val)
                train_errs.append(std_val if not np.isnan(std_val) and std_val > 0 else 0)
            else:
                # Old format (single value)
                train_vals.append(loss_data)
                train_errs.append(0.0)
            available_train_stencils.append(s)
        else:
            print(f"No a priori data for stencil {s}")
    
    if available_train_stencils:
        ax1.errorbar(available_train_stencils, train_vals, yerr=train_errs, 
                    marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
        ax1.set_xlabel('Stencil Size')
        ax1.set_ylabel('A Priori Loss (MSE)')
        ax1.set_title(f'A Priori Loss vs Stencil Size\n({filter_type.capitalize()} Filter)')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Add value labels
        for x, y in zip(available_train_stencils, train_vals):
            ax1.annotate(f'{y:.2e}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
    
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
            key = find_stencil_key(s, filter_type, aposteriori_metrics)
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
            # Plot baseline and ours on same axis
            ax.errorbar(available_apost_stencils, baseline_vals, yerr=baseline_errs, 
                       marker='s', capsize=5, capthick=2, linewidth=2, markersize=8, 
                       label='Baseline (No NN)', color='gray', alpha=0.8)
            ax.errorbar(available_apost_stencils, ours_vals, yerr=ours_errs, 
                       marker='o', capsize=5, capthick=2, linewidth=2, markersize=8,
                       label='With NN', color='blue')
            
            ax.set_xlabel('Stencil Size')
            ax.set_ylabel(f'{title}')
            ax.set_title(f'{title} vs Stencil Size\n({filter_type.capitalize()} Filter)')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            ax.legend()
            
            # Add improvement percentages
            for i, (x, baseline, ours) in enumerate(zip(available_apost_stencils, baseline_vals, ours_vals)):
                if not (np.isnan(baseline) or np.isnan(ours)) and baseline > 0:
                    improvement = (baseline - ours) / baseline * 100
                    ax.annotate(f'{improvement:+.1f}%', (x, ours), 
                               textcoords="offset points", xytext=(0,-20), 
                               ha='center', fontsize=7, color='blue')
    
    plt.suptitle(f'Stencil Size Comparison: A Priori and A Posteriori Metrics\n({filter_type.capitalize()} Filter)', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    stencil_str = "_".join([str(s) for s in stencil_sizes])
    plt.savefig(f'stencil_error_comparison_{filter_type}_{stencil_str}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Print numerical comparison
    print(f"\n=== NUMERICAL COMPARISON ({filter_type.upper()} FILTER) ===")
    print("A Priori (Training Loss):")
    for s in available_train_stencils:
        key = find_stencil_key(s, filter_type, train_losses)
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
            key = find_stencil_key(s, filter_type, aposteriori_metrics)
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

def main(stencil_sizes=[3, 5, 7, 9, 11], filter_type="gaussian"):
    """Run both stencil field comparison and error comparison
    
    Args:
        stencil_sizes: List of stencil sizes to compare
        filter_type: Type of filter to use ("box", "gaussian", "spectral")
    """
    print(f"=== STENCIL SIZE COMPARISON FOR {filter_type.upper()} FILTER ===")
    print(f"Stencil sizes: {stencil_sizes}")
    
    # Field and stress comparison
    print("\n=== FILTERED FIELD AND SGS STRESS COMPARISON ===")
    try:
        field_results = compare_stencil_fields(stencil_sizes, filter_type)
    except Exception as e:
        print(f"Error in field comparison: {e}")
        field_results = None
    
    # Error metrics comparison
    print("\n" + "="*60)
    print("=== ERROR METRICS COMPARISON ===")
    try:
        compare_stencil_errors(stencil_sizes, filter_type)
    except Exception as e:
        print(f"Error in metrics comparison: {e}")
    
    return field_results

# Example usage functions
def compare_gaussian_stencils(stencil_sizes=[3, 5, 7, 9, 11]):
    """Compare Gaussian filter across different stencil sizes"""
    return main(stencil_sizes, "gaussian")

def compare_box_stencils(stencil_sizes=[3, 5, 7, 9, 11]):
    """Compare Box filter across different stencil sizes"""
    return main(stencil_sizes, "box")

def compare_spectral_stencils(stencil_sizes=[3, 5, 7, 9, 11]):
    """Compare Spectral filter across different stencil sizes"""
    return main(stencil_sizes, "spectral")

# Run the comparison
if __name__ == "__main__":
    # Default comparison - Gaussian filter with common stencil sizes
    results = compare_gaussian_stencils([3, 5, 7, 9, 11])
    
    # Uncomment to compare other filter types:
    # results = compare_box_stencils([3, 5, 7, 9, 11])
    # results = compare_spectral_stencils([3, 5, 7, 9, 11])
