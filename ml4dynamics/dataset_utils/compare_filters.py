import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def compare_filtered_fields(filters=["box", "gaussian"]):
    """Compare filtered fields from the KS datasets
    
    Args:
        filters: List of filter types to compare. Options: ["box", "gaussian", "spectral"]
                Can be any subset of these filters.
    """
    
    # Define file paths for each filter type
   # filter_files = {
    #    "box": "data/ks/dnbc_nu1.0_c1.6_n10_box_s7.h5",
   #     "gaussian": "data/ks/dnbc_nu1.0_c1.6_n10_gaussian.h5", 
    #    "spectral": "data/ks/dnbc_nu1.0_c1.6_n10_spectral.h5"
   # }
    
    # Alternative files (commented out for now)
    filter_files = {
        "box": "data/ks/pbc_nu1.0_c0.0_n10_box_s7.h5",
        "gaussian": "data/ks/pbc_nu1.0_c0.0_n10_gaussian.h5",
        "spectral": "data/ks/pbc_nu1.0_c0.0_n10_spectral.h5"  # Add this when available
        }
    
    # Validate filters
    valid_filters = ["box", "gaussian", "spectral"]
    filters = [f for f in filters if f in valid_filters]
    if len(filters) < 2:
        raise ValueError(f"Need at least 2 filters to compare. Valid options: {valid_filters}")
    
    print(f"Comparing filters: {filters}")
    
    # Load datasets dynamically
    data = {}
    for filter_name in filters:
        file_path = filter_files[filter_name]
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found: {file_path}")
            
        with h5py.File(file_path, "r") as f:
            data[filter_name] = {
                'filtered_field': f["data"]["inputs"][:],
                'filter_stress': f["data"]["outputs_filter"][:],
                'correction_stress': f["data"]["outputs_correction"][:]
            }
    
    # Print shapes
    for filter_name in filters:
        print(f"{filter_name.capitalize()} filtered field shape: {data[filter_name]['filtered_field'].shape}")
    
    # Reshape the data back to (simulations, time_steps, spatial_points, channels)
    num_sims = 10  # from config case_num
    time_steps_per_sim = data[filters[0]]['filtered_field'].shape[0] // num_sims
    
    for filter_name in filters:
        for key in data[filter_name]:
            original_shape = data[filter_name][key].shape
            data[filter_name][key] = data[filter_name][key].reshape(num_sims, time_steps_per_sim, -1, 1)
    
    print(f"Reshaped to: {data[filters[0]]['filtered_field'].shape}")
    
    # Print individual filter statistics
    print("\n=== INDIVIDUAL FILTER STATISTICS ===")
    for filter_name in filters:
        print(f"\n{filter_name.upper()} FILTER:")
        
        # Filtered field stats
        field_data = data[filter_name]['filtered_field']
        print(f"  Filtered Field:")
        print(f"    Range: [{np.min(field_data):.6f}, {np.max(field_data):.6f}]")
        print(f"    RMS: {np.sqrt(np.mean(field_data**2)):.6f}")
        
        # Filter stress stats  
        filter_stress_data = data[filter_name]['filter_stress']
        print(f"  Filter Stress:")
        print(f"    Range: [{np.min(filter_stress_data):.6f}, {np.max(filter_stress_data):.6f}]")
        print(f"    RMS: {np.sqrt(np.mean(filter_stress_data**2)):.6f}")
        
        # Correction stress stats
        correction_stress_data = data[filter_name]['correction_stress']
        print(f"  Correction Stress:")
        print(f"    Range: [{np.min(correction_stress_data):.6f}, {np.max(correction_stress_data):.6f}]")
        print(f"    RMS: {np.sqrt(np.mean(correction_stress_data**2)):.6f}")
        print(f"    Mean absolute magnitude: {np.mean(np.abs(correction_stress_data)):.6f}")
    
    # Calculate pairwise differences (using first filter as reference)
    ref_filter = filters[0]
    differences = {}
    
    for i, filter_name in enumerate(filters[1:], 1):
        diff_key = f"{filter_name}_vs_{ref_filter}"
        differences[diff_key] = {
            'field_diff': data[filter_name]['filtered_field'] - data[ref_filter]['filtered_field'],
            'filter_stress_diff': data[filter_name]['filter_stress'] - data[ref_filter]['filter_stress'],
            'correction_stress_diff': data[filter_name]['correction_stress'] - data[ref_filter]['correction_stress']
        }
    
    # Print statistics for each comparison
    for diff_key, diffs in differences.items():
        print(f"\n=== {diff_key.upper()} COMPARISON ===")
        
        print("FILTERED FIELD COMPARISON:")
        field_diff = diffs['field_diff']
        print(f"  Max absolute difference: {np.max(np.abs(field_diff)):.6f}")
        print(f"  RMS difference: {np.sqrt(np.mean(field_diff**2)):.6f}")
        print(f"  Mean difference: {np.mean(field_diff):.6f}")
        
        print("FILTER STRESS COMPARISON:")
        filter_stress_diff = diffs['filter_stress_diff']
        print(f"  Max absolute difference: {np.max(np.abs(filter_stress_diff)):.6f}")
        print(f"  RMS difference: {np.sqrt(np.mean(filter_stress_diff**2)):.6f}")
        print(f"  Mean difference: {np.mean(filter_stress_diff):.6f}")
        
        print("CORRECTION STRESS COMPARISON:")
        correction_stress_diff = diffs['correction_stress_diff']
        print(f"  Max absolute difference: {np.max(np.abs(correction_stress_diff)):.6f}")
        print(f"  RMS difference: {np.sqrt(np.mean(correction_stress_diff**2)):.6f}")
        print(f"  Mean difference: {np.mean(correction_stress_diff):.6f}")
    
    # Calculate global min/max for consistent color scales
    print("Calculating global color scale ranges...")
    
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
    
    # Find global ranges across all filters
    for filter_name in filters:
        # Individual filter ranges
        field_data = data[filter_name]['filtered_field'][sim_idx, :, :, 0]
        filter_stress_data = data[filter_name]['filter_stress'][sim_idx, :, :, 0]
        correction_stress_data = data[filter_name]['correction_stress'][sim_idx, :, :, 0]
        
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
    
    # Create visualizations - adapt to number of filters
    n_filters = len(filters)
    n_comparisons = len(differences)
    
    # For visualization, we'll show all filters + differences
    n_cols = n_filters + n_comparisons
    fig, axes = plt.subplots(3, n_cols, figsize=(5*n_cols, 12))
    
    # Handle single column case
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot individual filter results with consistent color scales
    for i, filter_name in enumerate(filters):
        # Row 1: Filtered fields (consistent scale)
        im = axes[0,i].imshow(data[filter_name]['filtered_field'][sim_idx, :, :, 0].T, 
                            aspect='auto', cmap='RdBu_r', vmin=field_min, vmax=field_max)
        axes[0,i].set_title(f'{filter_name.capitalize()} Filtered Field')
        if i == 0:
            axes[0,i].set_ylabel('Space')
        plt.colorbar(im, ax=axes[0,i])
        
        # Row 2: Filter stresses (consistent scale)
        im = axes[1,i].imshow(data[filter_name]['filter_stress'][sim_idx, :, :, 0].T, 
                            aspect='auto', cmap='RdBu_r', vmin=filter_stress_min, vmax=filter_stress_max)
        axes[1,i].set_title(f'{filter_name.capitalize()} Filter Stress')
        if i == 0:
            axes[1,i].set_ylabel('Space')
        plt.colorbar(im, ax=axes[1,i])
        
        # Row 3: Correction stresses (consistent scale)
        im = axes[2,i].imshow(data[filter_name]['correction_stress'][sim_idx, :, :, 0].T, 
                            aspect='auto', cmap='RdBu_r', vmin=correction_stress_min, vmax=correction_stress_max)
        axes[2,i].set_title(f'{filter_name.capitalize()} Correction Stress')
        if i == 0:
            axes[2,i].set_ylabel('Space')
        axes[2,i].set_xlabel('Time')
        plt.colorbar(im, ax=axes[2,i])
    
    # Plot difference comparisons with consistent scales for differences
    for i, (diff_key, diffs) in enumerate(differences.items()):
        col_idx = n_filters + i
        
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
    filter_names_str = "_".join(filters)
    plt.savefig(f'filter_comparison_{filter_names_str}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Time series comparison at a specific spatial location
    plt.figure(figsize=(12, 8))
    mid_point = data[filters[0]]['filtered_field'].shape[2] // 2  # Middle spatial point
    
    # Determine subplot layout based on number of filters
    n_plots = min(4, len(filters) + 1)  # Max 4 subplots for readability
    rows = 2
    cols = 2
    
    # Plot 1: Filtered fields time series
    plt.subplot(rows, cols, 1)
    for filter_name in filters:
        plt.plot(data[filter_name]['filtered_field'][sim_idx, :, mid_point, 0], 
                label=filter_name.capitalize(), alpha=0.7)
    plt.title(f'Filtered Field at Spatial Point {mid_point}')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Filter stress time series
    plt.subplot(rows, cols, 2)
    for filter_name in filters:
        plt.plot(data[filter_name]['filter_stress'][sim_idx, :, mid_point, 0], 
                label=filter_name.capitalize(), alpha=0.7)
    plt.title('Filter Stress Time Series')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Correction stress time series
    plt.subplot(rows, cols, 3)
    for filter_name in filters:
        plt.plot(data[filter_name]['correction_stress'][sim_idx, :, mid_point, 0], 
                label=filter_name.capitalize(), alpha=0.7)
    plt.title('Correction Stress Time Series')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: First difference time series (if we have differences)
    if differences:
        plt.subplot(rows, cols, 4)
        first_diff_key = list(differences.keys())[0]
        plt.plot(differences[first_diff_key]['field_diff'][sim_idx, :, mid_point, 0], 
                label=f'Field Diff: {first_diff_key.replace("_vs_", " vs ")}')
        plt.title('Field Difference Time Series')
        plt.xlabel('Time Steps')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'filter_timeseries_comparison_{filter_names_str}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Return the data for further analysis if needed
    result = {'data': data, 'differences': differences}
    return result

def compare_errors(filters=["box", "gaussian"]):
    """Compare a priori and a posteriori errors between different filter types
    
    Args:
        filters: List of filter types to compare. Options: ["box", "gaussian", "spectral"]
    """
    
    print("\n=== LOADING ERROR METRICS ===")
    
    # Define file paths for each filter type (needed for parsing BC and stencil info)
    filter_files = {
        "box": "data/ks/pbc_nu1.0_c0.0_n10_box_s7.h5",
        "gaussian": "data/ks/pbc_nu1.0_c0.0_n10_gaussian.h5",
        "spectral": "data/ks/pbc_nu1.0_c0.0_n10_spectral.h5"
    }
    
    # Validate filters
    valid_filters = ["box", "gaussian", "spectral"]
    filters = [f for f in filters if f in valid_filters]
    if len(filters) < 2:
        raise ValueError(f"Need at least 2 filters to compare. Valid options: {valid_filters}")
    
    print(f"Comparing error metrics for filters: {filters}")
    
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
    
    # Filter out any requested filters that don't have data
    # Note: Keys now include boundary condition, so we need to check for filter_bc format
    available_filters = []
    for f in filters:
        # Check for both old format (just filter name) and new format (filter_bc)
        filter_found = False
        for key in list(train_losses.keys()) + list(aposteriori_metrics.keys()):
            if key == f or key.startswith(f + "_"):
                filter_found = True
                break
        
        if filter_found:
            available_filters.append(f)
        else:
            print(f"Warning: No data found for filter '{f}'")
    
    if len(available_filters) < 2:
        print(f"Insufficient data for comparison. Available: {available_filters}")
        return
    
    filters = available_filters
    
    # Helper function to find the correct key for a filter by parsing the file path
    def find_filter_key(filter_name, data_dict, file_path):
        """Find the correct key by parsing the file path to extract filter type, BC, and stencil size"""
        import re
        
        # Parse the file path to extract components
        # Example: "data/ks/pbc_nu1.0_c0.0_n10_box_s7.h5"
        # Extract: BC (pbc/dnbc), filter_type (box/gaussian/spectral), stencil_size (s7)
        
        filename = file_path.split('/')[-1]  # Get just the filename
        
        # Extract boundary condition
        if filename.startswith('pbc_'):
            bc = 'pbc'
        elif filename.startswith('dnbc_'):
            bc = 'dnbc'
        else:
            # Fallback to trying both
            bc = None
        
        # Extract stencil size (for box filter)
        stencil_match = re.search(r'_s(\d+)\.h5$', filename)
        stencil_size = stencil_match.group(1) if stencil_match else None
        
        # Construct the expected key based on parsed information
        if bc and stencil_size and filter_name == "box":
            # Box filter with BC and stencil size: "box_pbc_s7"
            expected_key = f"{filter_name}_{bc}_s{stencil_size}"
        elif bc and filter_name in ["gaussian", "spectral"]:
            # Gaussian/spectral with BC: "gaussian_pbc"
            expected_key = f"{filter_name}_{bc}"
        else:
            # Fallback to old format
            expected_key = filter_name
        
        # Check if the expected key exists
        if expected_key in data_dict:
            return expected_key
        
        # If not found, try some fallbacks
        fallback_keys = [
            filter_name,  # Old format
            f"{filter_name}_{bc}" if bc else None,  # With BC only
            f"{filter_name}_pbc",  # Try PBC
            f"{filter_name}_dnbc",  # Try DNBC
        ]
        
        for key in fallback_keys:
            if key and key in data_dict:
                return key
        
        return None
    
    # Create error comparison plots - adapt layout to number of filters
    n_filters = len(filters)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: A Priori Training Loss Comparison
    train_vals = []
    print("\n=== A PRIORI KEY SELECTION ===")
    for f in filters:
        # Find the correct key for this filter (handles both old and new format)
        file_path = filter_files.get(f, "")
        key = find_filter_key(f, train_losses, file_path)
        print(f"Filter '{f}': Using key '{key}' from train_losses")
        if key and isinstance(train_losses.get(key), dict):
            # Use relative MSE if available, otherwise fallback to absolute MSE
            loss_data = train_losses[key]
            if 'rel_mse_mean' in loss_data:
                train_vals.append(loss_data.get('rel_mse_mean', float('nan')))
            else:
                # Fallback to absolute MSE
                train_vals.append(loss_data.get('mean', float('nan')))
        elif key:
            # Old format (single value)
            train_vals.append(train_losses.get(key, float('nan')))
        else:
            train_vals.append(float('nan'))
    
    # Use different colors for different numbers of filters
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum'][:n_filters]
    bars1 = ax1.bar(filters, train_vals, color=colors)
    # Check if we're using relative MSE or absolute MSE for the title
    if train_losses and any('rel_mse_mean' in v for v in train_losses.values() if isinstance(v, dict)):
        ax1.set_title('A Priori Relative MSE')
        ax1.set_ylabel('Relative MSE')
    else:
        ax1.set_title('A Priori Loss')
        ax1.set_ylabel('Training Loss')
    
    # Add value labels on bars - shorter format
    for i, (bar, val) in enumerate(zip(bars1, train_vals)):
        if not np.isnan(val):
            ax1.text(bar.get_x() + bar.get_width()/2, val, f"{val:.1e}", 
                    ha='center', va='bottom', fontsize=7)
    
    # Plots 2-4: A Posteriori Metrics (baseline and ours for each filter)
    metric_groups = [
        ("L2 Error", ax2, "l2"),
        ("1st Moment", ax3, "first_moment"), 
        ("2nd Moment", ax4, "second_moment"),
    ]
    
    print("\n=== A POSTERIORI KEY SELECTION ===")
    for title, ax, metric in metric_groups:
        # Create bars: baseline1, ours1, baseline2, ours2, ...
        n_bars = n_filters * 2  # baseline + ours for each filter
        x = np.arange(n_bars)
        width = 0.6
        
        values = []
        stds = []
        labels = []
        bar_colors = []
        
        for i, filter_name in enumerate(filters):
            # Find the correct key for this filter (handles both old and new format)
            file_path = filter_files.get(filter_name, "")
            key = find_filter_key(filter_name, aposteriori_metrics, file_path)
            if i == 0:  # Only print once per filter to avoid repetition
                print(f"Filter '{filter_name}': Using key '{key}' from aposteriori_metrics")
            filter_data = aposteriori_metrics.get(key, {}) if key else {}
            
            # Baseline values
            baseline_val = filter_data.get(f"{metric}_baseline", float('nan'))
            baseline_std = filter_data.get(f"{metric}_baseline_std", 0.0)
            values.extend([baseline_val])
            stds.extend([baseline_std])
            labels.extend([f'{filter_name.capitalize()}\nBaseline'])
            bar_colors.extend(['lightgray'])
            
            # Ours values  
            ours_val = filter_data.get(f"{metric}_ours", float('nan'))
            ours_std = filter_data.get(f"{metric}_ours_std", 0.0)
            values.extend([ours_val])
            stds.extend([ours_std])
            labels.extend([f'{filter_name.capitalize()}'])
            bar_colors.extend([colors[i]])
        
        # Use absolute values for plotting to ensure positive values
        values_plot = [abs(x) if not np.isnan(x) else x for x in values]
        
        # Use std directly for error bars (not SEM)
        errs = [std if not np.isnan(std) and std > 0 else 0 for std in stds]
        
        bars = ax.bar(x, values_plot, width, color=bar_colors, alpha=0.7, yerr=errs, capsize=3)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_title(f'{title} (Magnitude)')
        ax.set_ylabel(f'|{title}|')
        
        # Add value labels on bars (using absolute values)
        for i, (bar, val) in enumerate(zip(bars, values_plot)):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, val, f"{val:.1e}", 
                       ha='center', va='bottom', fontsize=6)
    
    plt.suptitle(f'Filter Comparison: A Priori and A Posteriori Metrics ({", ".join([f.capitalize() for f in filters])})', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])  # Leave more space for title and labels
    
    filter_names_str = "_".join(filters)
    plt.savefig(f'filter_error_comparison_{filter_names_str}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Print numerical comparison
    print("\n=== NUMERICAL COMPARISON ===")
    print("A Priori (Training Loss):")
    for f in filters:
        file_path = filter_files.get(f, "")
        key = find_filter_key(f, train_losses, file_path)
        if key:
            print(f"  {f}: Using key '{key}'")  # Debug: show which key is selected
            loss_data = train_losses.get(key, {})
            if isinstance(loss_data, dict):
                # New format with mean/std
                mean_val = loss_data.get('rel_mse_mean', loss_data.get('mean', float('nan')))
                print(f"    Value: {mean_val:.6e}")
            else:
                # Old format (single value)
                print(f"    Value: {loss_data:.6e}")
        else:
            print(f"  {f}: No data found")
    
    print("\nA Posteriori Metrics:")
    for metric in ["l2", "first_moment", "second_moment"]:
        print(f"\n{metric.upper()}:")
        for f in filters:
            file_path = filter_files.get(f, "")
            key = find_filter_key(f, aposteriori_metrics, file_path)
            if key:
                print(f"  {f}: Using key '{key}'")  # Debug: show which key is selected
                data = aposteriori_metrics[key]
                
                # Use original field names (no _mean suffix)
                baseline_val = data.get(f"{metric}_baseline", float('nan'))
                baseline_std = data.get(f"{metric}_baseline_std", 0.0)
                ours_val = data.get(f"{metric}_ours", float('nan'))
                ours_std = data.get(f"{metric}_ours_std", 0.0)
                
                improvement = (baseline_val - ours_val) / baseline_val * 100 if not np.isnan(baseline_val) and not np.isnan(ours_val) else float('nan')
                
                print(f"  {f}:")
                print(f"    Baseline: {baseline_val:.6e} ± {baseline_std:.6e}")
                print(f"    Ours:     {ours_val:.6e} ± {ours_std:.6e}")
                print(f"    Improvement: {improvement:.1f}%")
            else:
                print(f"  {f}: No data found")

def main(field_filters=["box", "gaussian"], error_filters=None):
    """Run both filtered field comparison and error comparison
    
    Args:
        field_filters: List of filters for field comparison. Options: ["box", "gaussian", "spectral"]
        error_filters: List of filters for error comparison. If None, uses field_filters.
    """
    if error_filters is None:
        error_filters = field_filters
        
    print("=== FILTERED FIELD AND SGS STRESS COMPARISON ===")
    print(f"Field comparison filters: {field_filters}")
    field_results = compare_filtered_fields(field_filters)
    
    print("\n" + "="*60)
    print("=== ERROR METRICS COMPARISON ===")
    print(f"Error comparison filters: {error_filters}")
    compare_errors(error_filters)
    
    return field_results

# Example usage functions
def compare_all_three():
    """Compare all three filter types: box, gaussian, and spectral"""
    return main(["box", "gaussian", "spectral"])

def compare_box_gaussian():
    """Compare just box and gaussian filters"""
    return main(["box", "gaussian"])

def compare_box_spectral():
    """Compare just box and spectral filters"""
    return main(["box", "spectral"])

def compare_gaussian_spectral():
    """Compare just gaussian and spectral filters"""
    return main(["gaussian", "spectral"])

# Run the comparison
if __name__ == "__main__":
    # Default comparison (box vs gaussian)
    # results = main()
    
    #results = compare_all_three()  # Compare all three filters
    results = compare_box_spectral()  # Compare box and spectral filters