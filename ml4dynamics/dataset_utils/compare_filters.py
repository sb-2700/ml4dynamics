import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def compare_filtered_fields():
    """Compare box and Gaussian filtered fields from the KS datasets"""
    
    # Load both datasets
    box_file = "data/ks/dnbc_nu1.0_c1.6_n10_box.h5"
    gaussian_file = "data/ks/dnbc_nu1.0_c1.6_n10_gaussian.h5"
    
    with h5py.File(box_file, "r") as f_box:
        box_filtered_field = f_box["data"]["inputs"][:]  
        box_filter_stress = f_box["data"]["outputs_filter"][:]
        box_correction_stress = f_box["data"]["outputs_correction"][:]
        
    with h5py.File(gaussian_file, "r") as f_gauss:
        gauss_filtered_field = f_gauss["data"]["inputs"][:]  
        gauss_filter_stress = f_gauss["data"]["outputs_filter"][:]
        gauss_correction_stress = f_gauss["data"]["outputs_correction"][:]
    
    print(f"Box filtered field shape: {box_filtered_field.shape}")
    print(f"Gaussian filtered field shape: {gauss_filtered_field.shape}")
    
    # Reshape the data back to (simulations, time_steps, spatial_points, channels)
    num_sims = 10  # from config case_num
    time_steps_per_sim = box_filtered_field.shape[0] // num_sims  # e.g., 40000 // 10 = 4000
    
    box_filtered_field = box_filtered_field.reshape(num_sims, time_steps_per_sim, -1, 1)
    gauss_filtered_field = gauss_filtered_field.reshape(num_sims, time_steps_per_sim, -1, 1)
    box_filter_stress = box_filter_stress.reshape(num_sims, time_steps_per_sim, -1, 1)
    gauss_filter_stress = gauss_filter_stress.reshape(num_sims, time_steps_per_sim, -1, 1)
    box_correction_stress = box_correction_stress.reshape(num_sims, time_steps_per_sim, -1, 1)
    gauss_correction_stress = gauss_correction_stress.reshape(num_sims, time_steps_per_sim, -1, 1)
    
    print(f"Reshaped box filtered field shape: {box_filtered_field.shape}")
    print(f"Reshaped gaussian filtered field shape: {gauss_filtered_field.shape}")
    
    # Calculate differences
    field_diff = gauss_filtered_field - box_filtered_field
    filter_stress_diff = gauss_filter_stress - box_filter_stress
    correction_stress_diff = gauss_correction_stress - box_correction_stress
    
    # Print statistics
    print("\n=== FILTERED FIELD COMPARISON ===")
    print(f"Max absolute difference: {np.max(np.abs(field_diff)):.6f}")
    print(f"RMS difference: {np.sqrt(np.mean(field_diff**2)):.6f}")
    print(f"Mean difference: {np.mean(field_diff):.6f}")
    print(f"Std of difference: {np.std(field_diff):.6f}")
    
    print("\n=== FILTER STRESS COMPARISON ===")
    print(f"Max absolute difference: {np.max(np.abs(filter_stress_diff)):.6f}")
    print(f"RMS difference: {np.sqrt(np.mean(filter_stress_diff**2)):.6f}")
    print(f"Mean difference: {np.mean(filter_stress_diff):.6f}")
    
    print("\n=== CORRECTION STRESS COMPARISON ===")
    print(f"Max absolute difference: {np.max(np.abs(correction_stress_diff)):.6f}")
    print(f"RMS difference: {np.sqrt(np.mean(correction_stress_diff**2)):.6f}")
    print(f"Mean difference: {np.mean(correction_stress_diff):.6f}")
    
    # Create visualizations
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # First simulation, all time steps
    sim_idx = 0
    
    # Row 1: Filtered fields
    im1 = axes[0,0].imshow(box_filtered_field[sim_idx, :, :, 0].T, aspect='auto', cmap='RdBu_r')
    axes[0,0].set_title('Box Filtered Field')
    axes[0,0].set_ylabel('Space')
    plt.colorbar(im1, ax=axes[0,0])
    
    im2 = axes[0,1].imshow(gauss_filtered_field[sim_idx, :, :, 0].T, aspect='auto', cmap='RdBu_r')
    axes[0,1].set_title('Gaussian Filtered Field')
    plt.colorbar(im2, ax=axes[0,1])
    
    im3 = axes[0,2].imshow(field_diff[sim_idx, :, :, 0].T, aspect='auto', cmap='RdBu_r')
    axes[0,2].set_title('Difference (Gauss - Box)')
    plt.colorbar(im3, ax=axes[0,2])
    
    # Row 2: Filter stresses
    im4 = axes[1,0].imshow(box_filter_stress[sim_idx, :, :, 0].T, aspect='auto', cmap='RdBu_r')
    axes[1,0].set_title('Box Filter Stress')
    axes[1,0].set_ylabel('Space')
    plt.colorbar(im4, ax=axes[1,0])
    
    im5 = axes[1,1].imshow(gauss_filter_stress[sim_idx, :, :, 0].T, aspect='auto', cmap='RdBu_r')
    axes[1,1].set_title('Gaussian Filter Stress')
    plt.colorbar(im5, ax=axes[1,1])
    
    im6 = axes[1,2].imshow(filter_stress_diff[sim_idx, :, :, 0].T, aspect='auto', cmap='RdBu_r')
    axes[1,2].set_title('Filter Stress Difference')
    plt.colorbar(im6, ax=axes[1,2])
    
    # Row 3: Correction stresses
    im7 = axes[2,0].imshow(box_correction_stress[sim_idx, :, :, 0].T, aspect='auto', cmap='RdBu_r')
    axes[2,0].set_title('Box Correction Stress')
    axes[2,0].set_ylabel('Space')
    axes[2,0].set_xlabel('Time')
    plt.colorbar(im7, ax=axes[2,0])
    
    im8 = axes[2,1].imshow(gauss_correction_stress[sim_idx, :, :, 0].T, aspect='auto', cmap='RdBu_r')
    axes[2,1].set_title('Gaussian Correction Stress')
    axes[2,1].set_xlabel('Time')
    plt.colorbar(im8, ax=axes[2,1])
    
    im9 = axes[2,2].imshow(correction_stress_diff[sim_idx, :, :, 0].T, aspect='auto', cmap='RdBu_r')
    axes[2,2].set_title('Correction Stress Difference')
    axes[2,2].set_xlabel('Time')
    plt.colorbar(im9, ax=axes[2,2])
    
    plt.tight_layout()
    plt.savefig('filter_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Time series comparison at a specific spatial location
    plt.figure(figsize=(12, 8))
    mid_point = box_filtered_field.shape[2] // 2  # Middle spatial point
    
    plt.subplot(2, 2, 1)
    plt.plot(box_filtered_field[sim_idx, :, mid_point, 0], 'b-', label='Box', alpha=0.7)
    plt.plot(gauss_filtered_field[sim_idx, :, mid_point, 0], 'r-', label='Gaussian', alpha=0.7)
    plt.title(f'Filtered Field at Spatial Point {mid_point}')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(field_diff[sim_idx, :, mid_point, 0], 'g-')
    plt.title('Filtered Field Difference')
    plt.xlabel('Time Steps')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(box_filter_stress[sim_idx, :, mid_point, 0], 'b-', label='Box', alpha=0.7)
    plt.plot(gauss_filter_stress[sim_idx, :, mid_point, 0], 'r-', label='Gaussian', alpha=0.7)
    plt.title('Filter Stress Time Series')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(box_correction_stress[sim_idx, :, mid_point, 0], 'b-', label='Box', alpha=0.7)
    plt.plot(gauss_correction_stress[sim_idx, :, mid_point, 0], 'r-', label='Gaussian', alpha=0.7)
    plt.title('Correction Stress Time Series')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('filter_timeseries_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Return the data for further analysis if needed
    return {
        'box_filtered_field': box_filtered_field,
        'gauss_filtered_field': gauss_filtered_field,
        'field_diff': field_diff,
        'box_filter_stress': box_filter_stress,
        'gauss_filter_stress': gauss_filter_stress,
        'filter_stress_diff': filter_stress_diff,
        'box_correction_stress': box_correction_stress,
        'gauss_correction_stress': gauss_correction_stress,
        'correction_stress_diff': correction_stress_diff
    }

def compare_errors():
    """Compare a priori and a posteriori errors between different filter types"""
    
    print("\n=== LOADING ERROR METRICS ===")
    
    # Load a priori errors from pickle file
    train_losses_path = "results/train_losses.pkl"
    if os.path.exists(train_losses_path):
        with open(train_losses_path, "rb") as f:
            train_losses = pickle.load(f)
        print(f"A priori losses loaded: {train_losses}")
    else:
        print(f"Warning: {train_losses_path} not found")
        train_losses = {}
    
    # Load a posteriori metrics from pickle file
    aposteriori_path = "results/aposteriori_metrics.pkl"
    if os.path.exists(aposteriori_path):
        with open(aposteriori_path, "rb") as f:
            aposteriori_metrics = pickle.load(f)
        print(f"A posteriori metrics loaded: {aposteriori_metrics}")
    else:
        print(f"Warning: {aposteriori_path} not found")
        aposteriori_metrics = {}
    
    # Create error comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: A Priori Training Loss Comparison
    filters = ["box", "gaussian"]  # Focus on box vs gaussian
    train_vals = [train_losses.get(f, float('nan')) for f in filters]
    
    bars1 = ax1.bar(filters, train_vals, color=['skyblue', 'lightcoral'])
    ax1.set_title('A Priori Training Loss Comparison')
    ax1.set_ylabel('Training Loss')
    ax1.set_yscale('log')  # Use log scale for better visualization
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, train_vals)):
        if not np.isnan(val):
            ax1.text(bar.get_x() + bar.get_width()/2, val, f"{val:.2e}", 
                    ha='center', va='bottom', fontsize=10)
    
    # Plots 2-4: A Posteriori Metrics (baseline vs ours for box and gaussian)
    metric_groups = [
        ("l2_baseline", "l2_ours", "L2 Error", ax2),
        ("first_moment_baseline", "first_moment_ours", "1st Moment Error", ax3),
        ("second_moment_baseline", "second_moment_ours", "2nd Moment Error", ax4),
    ]
    
    x = np.arange(len(filters))
    width = 0.35  # Width of bars
    
    for base_key, ours_key, title, ax in metric_groups:
        baseline_vals = [aposteriori_metrics.get(f, {}).get(base_key, float('nan')) for f in filters]
        ours_vals = [aposteriori_metrics.get(f, {}).get(ours_key, float('nan')) for f in filters]
        
        bars_baseline = ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='gray', alpha=0.7)
        bars_ours = ax.bar(x + width/2, ours_vals, width, label='Ours', color=['skyblue', 'lightcoral'])
        
        ax.set_xticks(x)
        ax.set_xticklabels(filters)
        ax.set_title(title)
        ax.set_ylabel('Error')
        ax.legend()
        ax.set_yscale('log')  # Use log scale for better visualization
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars_baseline, baseline_vals)):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, val, f"{val:.2e}", 
                       ha='center', va='bottom', fontsize=8, rotation=45)
        
        for i, (bar, val) in enumerate(zip(bars_ours, ours_vals)):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, val, f"{val:.2e}", 
                       ha='center', va='bottom', fontsize=8, rotation=45)
    
    plt.suptitle('Filter Error Comparison: A Priori and A Posteriori Metrics')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('filter_error_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print numerical comparison
    print("\n=== NUMERICAL COMPARISON ===")
    print("A Priori (Training Loss):")
    for f in filters:
        val = train_losses.get(f, float('nan'))
        print(f"  {f}: {val:.6e}")
    
    print("\nA Posteriori Metrics:")
    for metric in ["l2", "first_moment", "second_moment"]:
        print(f"\n{metric.upper()}:")
        for f in filters:
            if f in aposteriori_metrics:
                baseline = aposteriori_metrics[f].get(f"{metric}_baseline", float('nan'))
                ours = aposteriori_metrics[f].get(f"{metric}_ours", float('nan'))
                improvement = (baseline - ours) / baseline * 100 if not np.isnan(baseline) and not np.isnan(ours) else float('nan')
                print(f"  {f}: Baseline={baseline:.6e}, Ours={ours:.6e}, Improvement={improvement:.1f}%")

def main():
    """Run both filtered field comparison and error comparison"""
    print("=== FILTERED FIELD AND SGS STRESS COMPARISON ===")
    field_results = compare_filtered_fields()
    
    print("\n" + "="*60)
    print("=== ERROR METRICS COMPARISON ===")
    compare_errors()
    
    return field_results

# Run the comparison
if __name__ == "__main__":
    results = main()