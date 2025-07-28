#!/usr/bin/env python3
"""
Test script to compare manual Gaussian filter implementation against OpenCV
"""
import numpy as np
import jax.numpy as jnp
import cv2
import matplotlib.pyplot as plt
from ml4dynamics.dataset_utils.dataset_utils import _create_gaussian_filter

def test_gaussian_filter_comparison():
    """Compare manual Gaussian filter against OpenCV implementation"""
    
    # Test parameters
    N1 = 64  # Fine grid size
    r = 4    # Coarsening factor
    N2 = N1 // r  # Coarse grid size (16)
    BC = "Dirichlet-Neumann"  # Non-periodic for now
    
    print(f"Testing with N1={N1}, N2={N2}, r={r}, BC={BC}")
    
    # Create test signal (1D)
    x = np.linspace(0, 1, N1)
    test_signal = np.sin(2 * np.pi * x) + 0.5 * np.sin(8 * np.pi * x) + 0.2 * np.random.randn(N1)
    
    print(f"Test signal shape: {test_signal.shape}")
    
    # 1. Manual Gaussian filter
    print("\n=== Testing Manual Gaussian Filter ===")
    try:
        res_op = _create_gaussian_filter(N1, N2, r, BC)
        print(f"res_op shape: {res_op.shape}")
        print(f"res_op row sums (should be ~1): {np.sum(res_op, axis=1)}")
        
        # Apply manual filter
        manual_result = res_op @ test_signal
        print(f"Manual result shape: {manual_result.shape}")
        print(f"Manual result range: [{np.min(manual_result):.6f}, {np.max(manual_result):.6f}]")
        
    except Exception as e:
        print(f"ERROR in manual filter: {e}")
        return
    
    # 2. OpenCV Gaussian filter (approximate equivalent)
    print("\n=== Testing OpenCV Gaussian Filter ===")
    
    # Calculate equivalent sigma for OpenCV
    sigma = r / 2  # This matches your manual implementation
    kernel_size = int(6 * sigma + 1)  # This matches your stencil_size
    if kernel_size % 2 == 0:
        kernel_size += 1  # OpenCV needs odd kernel size
    
    print(f"OpenCV parameters: sigma={sigma}, kernel_size={kernel_size}")
    
    # Apply OpenCV filter
    opencv_filtered = cv2.GaussianBlur(test_signal.astype(np.float32), 
                                       (kernel_size, 1), 
                                       sigma, 
                                       borderType=cv2.BORDER_REFLECT)
    
    # Downsample to match coarse grid
    opencv_result = opencv_filtered[::r]  # Simple downsampling
    
    print(f"OpenCV result shape: {opencv_result.shape}")
    print(f"OpenCV result range: [{np.min(opencv_result):.6f}, {np.max(opencv_result):.6f}]")
    
    # 3. Compare results
    print("\n=== Comparison ===")
    
    if manual_result.shape != opencv_result.shape:
        print(f"WARNING: Shape mismatch! Manual: {manual_result.shape}, OpenCV: {opencv_result.shape}")
        min_len = min(len(manual_result), len(opencv_result))
        manual_result = manual_result[:min_len]
        opencv_result = opencv_result[:min_len]
    
    # Calculate differences
    abs_diff = np.abs(manual_result - opencv_result)
    rel_diff = abs_diff / (np.abs(opencv_result) + 1e-12)
    
    print(f"Max absolute difference: {np.max(abs_diff):.6f}")
    print(f"Mean absolute difference: {np.mean(abs_diff):.6f}")
    print(f"Max relative difference: {np.max(rel_diff):.6f}")
    print(f"Mean relative difference: {np.mean(rel_diff):.6f}")
    
    # Check if they match within tolerance
    tolerance = 1e-2  # 1% tolerance
    matches = np.allclose(manual_result, opencv_result, rtol=tolerance, atol=tolerance)
    print(f"Results match within {tolerance} tolerance: {matches}")
    
    # 4. Visualize
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Original signal
    plt.subplot(2, 3, 1)
    plt.plot(x, test_signal, 'b-', linewidth=1)
    plt.title('Original Signal')
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.grid(True)
    
    # Plot 2: Manual filter result
    plt.subplot(2, 3, 2)
    x_coarse = np.linspace(0, 1, len(manual_result))
    plt.plot(x_coarse, manual_result, 'ro-', linewidth=2, markersize=4)
    plt.title('Manual Gaussian Filter')
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.grid(True)
    
    # Plot 3: OpenCV result
    plt.subplot(2, 3, 3)
    plt.plot(x_coarse, opencv_result, 'go-', linewidth=2, markersize=4)
    plt.title('OpenCV Gaussian Filter')
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.grid(True)
    
    # Plot 4: Overlay comparison
    plt.subplot(2, 3, 4)
    plt.plot(x_coarse, manual_result, 'ro-', linewidth=2, markersize=4, label='Manual')
    plt.plot(x_coarse, opencv_result, 'go-', linewidth=2, markersize=4, label='OpenCV')
    plt.title('Overlay Comparison')
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Plot 5: Absolute difference
    plt.subplot(2, 3, 5)
    plt.plot(x_coarse, abs_diff, 'mo-', linewidth=2, markersize=4)
    plt.title('Absolute Difference')
    plt.xlabel('x')
    plt.ylabel('|Manual - OpenCV|')
    plt.grid(True)
    
    # Plot 6: Filter weights visualization
    plt.subplot(2, 3, 6)
    # Show a few rows of the filter matrix
    for i in [0, N2//4, N2//2, 3*N2//4, N2-1]:
        if i < res_op.shape[0]:
            plt.plot(res_op[i, :], label=f'Row {i}')
    plt.title('Manual Filter Weights (selected rows)')
    plt.xlabel('Fine grid index')
    plt.ylabel('Weight')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/sassan/ml4dynamics/gaussian_filter_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 5. Check for potential issues
    print("\n=== Diagnostic Checks ===")
    
    # Check for NaN or Inf in filter matrix
    if np.any(np.isnan(res_op)) or np.any(np.isinf(res_op)):
        print("ERROR: res_op contains NaN or Inf values!")
    else:
        print("✓ res_op is finite")
    
    # Check row normalization
    row_sums = np.sum(res_op, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        print(f"WARNING: Row sums not normalized! Range: [{np.min(row_sums):.6f}, {np.max(row_sums):.6f}]")
    else:
        print("✓ Row sums are normalized")
    
    # Check for negative weights
    if np.any(res_op < 0):
        print("WARNING: Filter contains negative weights!")
        print(f"Min weight: {np.min(res_op):.6f}")
    else:
        print("✓ All weights are non-negative")
    
    # Check condition number
    cond_num = np.linalg.cond(res_op)
    print(f"Filter condition number: {cond_num:.2e}")
    if cond_num > 1e12:
        print("WARNING: Filter is poorly conditioned!")
    else:
        print("✓ Filter is well-conditioned")
    
    return matches, manual_result, opencv_result

def test_periodic_case():
    """Test periodic boundary conditions"""
    print("\n" + "="*50)
    print("TESTING PERIODIC BOUNDARY CONDITIONS")
    print("="*50)
    
    N1 = 64
    r = 4
    N2 = N1 // r
    BC = "periodic"
    
    # Create periodic test signal
    x = np.linspace(0, 2*np.pi, N1)
    test_signal = np.sin(x) + 0.3 * np.sin(3*x)  # Should be exactly periodic
    
    print(f"Testing with N1={N1}, N2={N2}, r={r}, BC={BC}")
    
    try:
        res_op = _create_gaussian_filter(N1, N2, r, BC)
        manual_result = res_op @ test_signal
        
        print(f"Manual result shape: {manual_result.shape}")
        print(f"Row sums: {np.sum(res_op, axis=1)}")
        
        # For periodic case, we can check if filtering preserves DC component
        dc_original = np.mean(test_signal)
        dc_filtered = np.mean(manual_result)
        print(f"DC component preservation: {dc_original:.6f} -> {dc_filtered:.6f}")
        
        return True
        
    except Exception as e:
        print(f"ERROR in periodic filter: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Gaussian Filter Implementation")
    print("="*50)
    
    # Test non-periodic case
    try:
        matches, manual, opencv = test_gaussian_filter_comparison()
        print(f"\nNon-periodic test passed: {matches}")
    except Exception as e:
        print(f"Non-periodic test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test periodic case
    try:
        periodic_ok = test_periodic_case()
        print(f"Periodic test passed: {periodic_ok}")
    except Exception as e:
        print(f"Periodic test failed: {e}")
        import traceback
        traceback.print_exc()
