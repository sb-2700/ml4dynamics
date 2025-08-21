"""
Finite-Time Lyapunov Exponent (FTLE) Analysis for KS Equation

This module provides tools to compute FTLE for the Kuramoto-Sivashinsky equation
at different coarsening ratios, comparing baseline and NN-corrected solvers.

FTLE measures sensitivity to initial conditions by tracking how nearby trajectories
diverge exponentially over time.
"""

import os
import pickle
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from box import Box

# Import required functions from existing utils
from ml4dynamics.utils.utils import (
    create_fine_coarse_simulator,
    prepare_unet_train_state
)


def debug_model_loading(model_path: str, config_dict: dict) -> None:
    """
    Debug function to check what's in the model file and test loading.
    """
    print(f"\n{'='*60}")
    print("MODEL LOADING DEBUG")
    print(f"{'='*60}")
    
    # Check if file exists and get size
    if not os.path.exists(model_path):
        print(f"❌ File does not exist: {model_path}")
        return
    
    file_size = os.path.getsize(model_path)
    print(f"✅ File exists: {model_path}")
    print(f"   File size: {file_size:,} bytes")
    
    # Try to load the pickle file
    try:
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        print(f"✅ Pickle file loaded successfully")
        print(f"   Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"   Dictionary keys: {list(data.keys())}")
            for key, value in data.items():
                print(f"     {key}: {type(value)}")
        else:
            print(f"   Data shape/info: {getattr(data, 'shape', 'no shape attribute')}")
            
    except Exception as e:
        print(f"❌ Failed to load pickle file: {e}")
        return
    
    # Try to create a train state with the loaded data
    try:
        print(f"\n🔄 Testing train state creation...")
        train_state, _ = prepare_unet_train_state(
            config_dict, 
            model_path,
            is_global=True,
            is_training=False
        )
        print(f"✅ Train state created successfully")
        print(f"   Train state type: {type(train_state)}")
        print(f"   Has params: {hasattr(train_state, 'params')}")
        print(f"   Has batch_stats: {hasattr(train_state, 'batch_stats')}")
        
        # Try a simple forward pass
        config = Box(config_dict)
        N_coarse = config.sim.n // 4  # Use r=4 for testing
        test_input = jnp.ones((1, N_coarse, 1))
        
        print(f"\n🔄 Testing forward pass...")
        print(f"   Test input shape: {test_input.shape}")
        
        @jax.jit
        def test_forward(x):
            y_pred, _ = train_state.apply_fn_with_bn(
                {"params": train_state.params, "batch_stats": train_state.batch_stats},
                x, is_training=False
            )
            return y_pred
        
        output = test_forward(test_input)
        print(f"✅ Forward pass successful")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{float(jnp.min(output)):.6f}, {float(jnp.max(output)):.6f}]")
        
    except Exception as e:
        print(f"❌ Train state creation/forward pass failed: {e}")
        import traceback
        traceback.print_exc()


def solve_coarse_ks_ftle(u0: np.ndarray, t_eval: np.ndarray, config_dict: dict, 
                        forward_fn: callable = None, model_type: str = "baseline") -> np.ndarray:
    """
    Solve KS equation on coarse grid for FTLE analysis.
    
    Parameters:
    -----------
    u0 : np.ndarray
        Initial condition on coarse grid
    t_eval : np.ndarray
        Time points to evaluate
    config_dict : dict
        Configuration dictionary
    forward_fn : callable, optional
        Neural network forward function (for "ours" model)
    model_type : str
        "baseline" or "ours"
    
    Returns:
    --------
    trajectory : np.ndarray
        Array of shape (len(t_eval), len(u0))
    """
    try:
        config = Box(config_dict)
        
        # Create coarse simulator
        _, coarse_model = create_fine_coarse_simulator(config_dict)
        
        # Set up trajectory storage
        trajectory = np.zeros((len(t_eval), len(u0)))
        
        # Initial condition
        x = jnp.array(u0)
        trajectory[0] = np.array(x)
        
        # Determine progress reporting frequency (report at most 2 times)
        n_steps = len(t_eval)
        report_every = max(1, n_steps // 2)
        
        if model_type == "baseline":
            # Pure coarse solver
            for i in range(1, len(t_eval)):
                if i % report_every == 0 or i == len(t_eval) - 1:
                    print(f"        Step {i}/{len(t_eval)}")
                x = coarse_model.CN_FEM(x)
                trajectory[i] = np.array(x)  # Convert to numpy immediately
                
        elif model_type == "ours":
            # NN-corrected solver
            if forward_fn is None:
                raise ValueError("forward_fn required for 'ours' model")
                
            for i in range(1, len(t_eval)):
                if i % report_every == 0 or i == len(t_eval) - 1:
                    print(f"        Step {i}/{len(t_eval)} (NN)")
                x_next = coarse_model.CN_FEM(x)
                correction = forward_fn(x.reshape(1, -1, 1))[0, :, 0]
                x = x_next + correction * coarse_model.dt
                trajectory[i] = np.array(x)  # Convert to numpy immediately
        return trajectory
        
    except Exception as e:
        print(f"      ❌ Error in solve_coarse_ks_ftle: {e}")
        raise


def compute_ftle_single_ic(u0: np.ndarray, config_dict: dict, forward_fn: callable = None,
                          model_type: str = "baseline", eps: float = 1e-7, 
                          early_frac: float = 0.25, t_end: float = 50.0) -> dict:
    """
    Compute FTLE for a single initial condition.
    
    Parameters:
    -----------
    u0 : np.ndarray
        Initial condition on coarse grid
    config_dict : dict
        Configuration dictionary
    forward_fn : callable, optional
        Neural network forward function
    model_type : str
        "baseline" or "ours"
    eps : float
        Perturbation magnitude
    early_frac : float
        Fraction of time for exponential fit
    t_end : float
        End time for FTLE analysis
    
    Returns:
    --------
    result : dict
        Dictionary with 'lambda', 'r2', 'doubling_time', 'success'
    """
    config = Box(config_dict)
    L = config.sim.L
    N_coarse = len(u0)
    dx = L / N_coarse
    
    # Time array (much shorter than full simulation for FTLE)
    n_steps = int(t_end / config.sim.dt)
    t_eval = np.linspace(0, t_end, n_steps)
    
    # Generate perturbation (fixed seed for reproducibility)
    rng = random.PRNGKey(42)
    eta = random.normal(rng, u0.shape)
    eta = eta / (np.linalg.norm(np.array(eta)) + 1e-30)  # Use NumPy norm
    u0_eps = u0 + eps * eta
    
    try:
        # Solve both trajectories
        U = solve_coarse_ks_ftle(u0, t_eval, config_dict, forward_fn, model_type)
        U_eps = solve_coarse_ks_ftle(u0_eps, t_eval, config_dict, forward_fn, model_type)
        
        # Compute separation over time (with continuous L2 norm)
        d = np.array([np.linalg.norm(U[k] - U_eps[k]) * np.sqrt(dx) 
                     for k in range(len(t_eval))])
        
        # Fit exponential growth in early window
        end_idx = max(12, int(early_frac * len(t_eval)))
        if end_idx >= len(t_eval):
            end_idx = len(t_eval) - 1
            
        t_win = t_eval[:end_idx]
        d_win = np.maximum(d[:end_idx], 1e-30)
        
        # Linear fit to log(d) vs t
        A = np.vstack([np.ones_like(t_win), t_win]).T
        log_d = np.log(d_win)
        a, b = np.linalg.lstsq(A, log_d, rcond=None)[0]
        
        # Compute R²
        log_d_pred = a + b * t_win
        ss_res = np.sum((log_d - log_d_pred)**2)
        ss_tot = np.sum((log_d - log_d.mean())**2) + 1e-15
        r2 = 1.0 - ss_res / ss_tot
        
        # Doubling time
        thr = 2.0 * eps
        doubling_idx = np.argmax(d >= thr)
        doubling_time = np.inf
        if d[doubling_idx] >= thr:
            doubling_time = float(t_eval[doubling_idx] - t_eval[0])
        
        return {
            'lambda': float(b),  # FTLE = slope
            'r2': float(r2),
            'doubling_time': doubling_time,
            'success': True
        }
        
    except Exception as e:
        print(f"FTLE computation failed: {e}")
        return {
            'lambda': np.nan,
            'r2': np.nan,
            'doubling_time': np.inf,
            'success': False
        }


def run_ftle_analysis(config_dict: dict, model_path: str = None, 
                     r_list: list = [2, 4, 8], n_ics: int = 10, 
                     t_end: float = 50.0, debug: bool = False) -> dict:
    """
    Run complete FTLE analysis for baseline and (optionally) NN-corrected models.
    
    Parameters:
    -----------
    config_dict : dict
        Configuration dictionary
    model_path : str, optional
        Path to trained model file
    r_list : list
        List of coarsening ratios to analyze
    n_ics : int
        Number of initial conditions
    t_end : float
        End time for FTLE analysis
    debug : bool
        Enable debugging output
    
    Returns:
    --------
    results : dict
        Nested dictionary with results[model_type][r] containing FTLE statistics
    """
    print(f"\n{'='*60}")
    print("FTLE ANALYSIS")
    print(f"{'='*60}")
    
    # Debug model loading if requested
    if debug and model_path:
        debug_model_loading(model_path, config_dict)
    
    config = Box(config_dict)
    L = config.sim.L
    results = {}
    
    # Prepare models to test
    models_to_test = ["baseline"]
    forward_fn = None
    
    # Try to load NN model if path provided
    if model_path and os.path.exists(model_path):
        try:
            print(f"\n🔄 Loading NN model from: {model_path}")
            train_state, _ = prepare_unet_train_state(
                config_dict, 
                model_path,
                is_global=True,
                is_training=False
            )
            
            @jax.jit
            def forward_fn(x):
                y_pred, _ = train_state.apply_fn_with_bn(
                    {"params": train_state.params, "batch_stats": train_state.batch_stats},
                    x, is_training=False
                )
                return y_pred
            
            models_to_test.append("ours")
            print(f"✅ NN model loaded successfully")
            
        except Exception as e:
            print(f"❌ Could not load NN model: {e}")
            print("Running baseline-only analysis")
            if debug:
                import traceback
                traceback.print_exc()
    else:
        if model_path:
            print(f"❌ Model file not found: {model_path}")
        print("Running baseline-only analysis")
    
    # Run FTLE analysis for each model and coarsening ratio
    for model_type in models_to_test:
        results[model_type] = {}
        print(f"\nModel: {model_type}")
        
        for r in r_list:
            print(f"  Coarsening ratio r={r}")
            
            # Update config for this coarsening ratio
            config_r = config_dict.copy()
            config_r['sim']['rx'] = r
            N_coarse = config.sim.n // r
            
            lambdas = []
            r2s = []
            doubling_times = []
            
            for ic_idx in range(n_ics):
                print(f"    IC {ic_idx+1}/{n_ics}")
                
                # Use same IC generation as in eval_a_posteriori
                fixed_key = random.PRNGKey(2000 + ic_idx)
                r0 = random.uniform(fixed_key) * 20 + 44
                x_coarse = np.linspace(0, L - L/N_coarse, N_coarse)
                u0 = np.exp(-(x_coarse - r0)**2 / r0**2 * 4)
                
                # Compute FTLE for this IC
                result = compute_ftle_single_ic(
                    u0, config_r, forward_fn, model_type, 
                    eps=1e-7, early_frac=0.25, t_end=t_end
                )
                
                lambdas.append(result['lambda'])
                r2s.append(result['r2'])
                doubling_times.append(result['doubling_time'])
            
            # Filter valid results
            lambdas = np.array(lambdas)
            r2s = np.array(r2s)
            doubling_times = np.array(doubling_times)
            
            valid_mask = ~np.isnan(lambdas)
            n_valid = np.sum(valid_mask)
            
            if n_valid > 0:
                valid_lambdas = lambdas[valid_mask]
                valid_r2s = r2s[valid_mask]
                
                results[model_type][r] = {
                    'lambdas': lambdas,
                    'mean': float(valid_lambdas.mean()),
                    'std': float(valid_lambdas.std(ddof=1) if len(valid_lambdas) > 1 else 0.0),
                    'r2_mean': float(valid_r2s.mean()),
                    'r2_min': float(valid_r2s.min()),
                    'doubling_times': doubling_times,
                    'n_valid': int(n_valid),
                    'n_total': len(lambdas)
                }
                
                mean_val = valid_lambdas.mean()
                std_val = valid_lambdas.std(ddof=1) if len(valid_lambdas) > 1 else 0.0
                print(f"      λ = {mean_val:.4f} ± {std_val:.4f}")
                print(f"      R² = {valid_r2s.mean():.2f} (min: {valid_r2s.min():.2f})")
                print(f"      Valid: {n_valid}/{len(lambdas)}")
                
                # Warn if many fits are poor
                poor_fits = np.sum(valid_r2s < 0.7)
                if poor_fits > len(valid_r2s) * 0.5:
                    print(f"      Warning: {poor_fits}/{len(valid_r2s)} fits have R² < 0.7")
                    print(f"      Consider reducing t_end or eps")
                    
            else:
                results[model_type][r] = {
                    'lambdas': lambdas,
                    'mean': np.nan,
                    'std': np.nan,
                    'r2_mean': np.nan,
                    'r2_min': np.nan,
                    'doubling_times': doubling_times,
                    'n_valid': 0,
                    'n_total': len(lambdas)
                }
                print(f"      No valid results")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/ftle_analysis.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Print summary
    print(f"\n{'='*60}")
    print("FTLE SUMMARY")
    print(f"{'='*60}")
    for model_type in models_to_test:
        print(f"\n{model_type.upper()} MODEL:")
        print(f"{'r':<3} {'FTLE Mean':<12} {'± Std':<12} {'R² Mean':<8} {'Valid/Total':<12}")
        print("-" * 50)
        for r in r_list:
            res = results[model_type][r]
            if not np.isnan(res['mean']):
                print(f"{r:<3} {res['mean']:<12.4f} {res['std']:<12.4f} "
                      f"{res['r2_mean']:<8.2f} {res['n_valid']}/{res['n_total']}")
            else:
                print(f"{r:<3} {'NaN':<12} {'NaN':<12} {'NaN':<8} {res['n_valid']}/{res['n_total']}")
    
    return results


def print_ftle_interpretation(results: dict, r_list: list = [2, 4, 8]) -> None:
    """
    Print interpretation of FTLE results.
    
    Parameters:
    -----------
    results : dict
        Results from run_ftle_analysis
    r_list : list
        List of coarsening ratios analyzed
    """
    print(f"\n{'='*60}")
    print("FTLE INTERPRETATION")
    print(f"{'='*60}")
    
    print("\nWhat FTLE tells us:")
    print("• Higher FTLE = More chaotic (sensitive to initial conditions)")
    print("• Lower FTLE = Less chaotic (more stable)")
    print("• Expected: FTLE increases with coarsening ratio (r)")
    
    if "baseline" in results and "ours" in results:
        print("\nBaseline vs NN comparison:")
        for r in r_list:
            if (r in results["baseline"] and r in results["ours"] and 
                not np.isnan(results["baseline"][r]["mean"]) and 
                not np.isnan(results["ours"][r]["mean"])):
                
                baseline_ftle = results["baseline"][r]["mean"]
                ours_ftle = results["ours"][r]["mean"]
                diff_pct = ((ours_ftle - baseline_ftle) / baseline_ftle) * 100
                
                print(f"\nr={r}:")
                print(f"  Baseline: λ = {baseline_ftle:.4f}")
                print(f"  NN:       λ = {ours_ftle:.4f}")
                print(f"  Change:   {diff_pct:+.1f}%")
                
                if diff_pct < -5:
                    print(f"  → NN reduces chaos (smoother dynamics)")
                elif diff_pct > 5:
                    print(f"  → NN increases chaos (more sensitive)")
                else:
                    print(f"  → NN preserves chaos structure")
