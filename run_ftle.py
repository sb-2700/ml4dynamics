#!/usr/bin/env python3
"""
Standalone FTLE analysis script for KS equation dynamics.

This script analyzes Finite-Time Lyapunov Exponents to quantify how chaotic
the KS equation dynamics are at different coarsening ratios, comparing
baseline and NN-corrected solvers.

Usage:
    python run_ftle.py -c ks -r 2 4 8 --n-ics 10
    python run_ftle.py -c ks -m ckpts/ks/model.pkl -r 4 --n-ics 5
"""

import argparse
import yaml
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml4dynamics.utils.ftle_analysis import run_ftle_analysis, print_ftle_interpretation


def main():
    parser = argparse.ArgumentParser(
        description='Run FTLE analysis for KS equation dynamics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -c ks -r 2 4 8 --n-ics 10
    Run baseline-only FTLE analysis for r=2,4,8 with 10 initial conditions

  %(prog)s -c ks -m ckpts/ks/model.pkl -r 4 --n-ics 5 --t-end 30
    Compare baseline vs NN model for r=4 with shorter simulation

  %(prog)s -c ks -m ckpts/ks/pbc_nu1.0_c0.0_n10_r4_box_s5_correction_ols_unet.pkl -r 4
    Use specific trained model for analysis
"""
    )
    
    parser.add_argument('-c', '--config', default='ks', 
                        help='Config file name (default: ks)')
    parser.add_argument('-m', '--model', default=None, 
                        help='Path to trained model file (e.g., ckpts/ks/model.pkl)')
    parser.add_argument('-r', '--ratios', nargs='+', type=int, default=[2, 4, 8],
                        help='Coarsening ratios to analyze (default: 2 4 8)')
    parser.add_argument('--n-ics', type=int, default=10,
                        help='Number of initial conditions (default: 10)')
    parser.add_argument('--t-end', type=float, default=50.0,
                        help='End time for FTLE analysis (default: 50.0)')
    parser.add_argument('--eps', type=float, default=1e-7,
                        help='Perturbation magnitude (default: 1e-7)')
    parser.add_argument('--early-frac', type=float, default=0.25,
                        help='Fraction of time for exponential fit (default: 0.25)')
    parser.add_argument('--baseline-only', action='store_true',
                        help='Only analyze baseline model (ignore --model)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output for model loading')
    
    args = parser.parse_args()
    
    # Load config
    config_file = f"config/{args.config}.yaml"
    if not os.path.exists(config_file):
        print(f"Error: Config file not found: {config_file}")
        print(f"Available configs: {[f.replace('.yaml', '') for f in os.listdir('config') if f.endswith('.yaml')]}")
        sys.exit(1)
    
    try:
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Validate config
    if config_dict.get('case') != 'ks':
        print(f"Warning: Config case is '{config_dict.get('case')}', expected 'ks'")
        print("FTLE analysis is designed for KS equation")
    
    # Handle model path
    model_path = None
    if not args.baseline_only and args.model:
        if os.path.exists(args.model):
            model_path = args.model
        else:
            print(f"Warning: Model file not found: {args.model}")
            print("Running baseline-only analysis")
    
    # Print analysis info
    print(f"FTLE Analysis Configuration:")
    print(f"  Config: {args.config}")
    print(f"  Coarsening ratios: {args.ratios}")
    print(f"  Initial conditions: {args.n_ics}")
    print(f"  Analysis time: {args.t_end}")
    print(f"  Perturbation: {args.eps}")
    print(f"  Model: {model_path if model_path else 'baseline only'}")
    
    # Validate ratios
    max_ratio = config_dict.get('sim', {}).get('n', 256) // 4  # Minimum 4 grid points
    invalid_ratios = [r for r in args.ratios if r > max_ratio or r < 1]
    if invalid_ratios:
        print(f"Warning: Invalid ratios {invalid_ratios} for grid size {config_dict.get('sim', {}).get('n', 256)}")
        args.ratios = [r for r in args.ratios if 1 <= r <= max_ratio]
        print(f"Using ratios: {args.ratios}")
    
    if not args.ratios:
        print("Error: No valid coarsening ratios")
        sys.exit(1)
    
    try:
        # Run FTLE analysis
        results = run_ftle_analysis(
            config_dict=config_dict,
            model_path=model_path,
            r_list=args.ratios,
            n_ics=args.n_ics,
            t_end=args.t_end,
            debug=args.debug
        )
        
        # Print interpretation
        print_ftle_interpretation(results, args.ratios)
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: results/ftle_analysis.pkl")
        
        # Summary statistics
        for model_type in results:
            valid_results = [r for r in args.ratios if results[model_type][r]['n_valid'] > 0]
            if valid_results:
                print(f"\n{model_type.upper()} MODEL SUMMARY:")
                ftles = [results[model_type][r]['mean'] for r in valid_results]
                print(f"  Ratios analyzed: {valid_results}")
                print(f"  FTLE range: {min(ftles):.4f} - {max(ftles):.4f}")
                print(f"  Trend: {'Increasing' if ftles[-1] > ftles[0] else 'Decreasing'} with coarsening")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during FTLE analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
