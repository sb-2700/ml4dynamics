import pickle
import os

def examine_aposteriori_all_pkl():
    """Examine the contents of aposteriori_all.pkl to see what's been tested"""
    
    aposteriori_all_path = "results/aposteriori_all.pkl"
    
    if not os.path.exists(aposteriori_all_path):
        print(f"File not found: {aposteriori_all_path}")
        return
    
    try:
        with open(aposteriori_all_path, "rb") as f:
            aposteriori_all = pickle.load(f)
        
        print(f"Found aposteriori_all.pkl with {len(aposteriori_all)} configurations:")
        print("="*60)
        
        # Group by coarsening ratio
        configs_by_ratio = {}
        
        for key in aposteriori_all.keys():
            print(f"\nKey: {key}")
            
            # Extract info from key (format: filter_bc_rx_sx)
            parts = key.split('_')
            if len(parts) >= 3:
                try:
                    # Find r and s values
                    r_part = [p for p in parts if p.startswith('r')]
                    s_part = [p for p in parts if p.startswith('s')]
                    
                    if r_part and s_part:
                        r_val = int(r_part[0][1:])  # Remove 'r' prefix
                        s_val = int(s_part[0][1:])  # Remove 's' prefix
                        
                        if r_val not in configs_by_ratio:
                            configs_by_ratio[r_val] = []
                        configs_by_ratio[r_val].append(s_val)
                        
                        print(f"  Coarsening ratio: {r_val}, Stencil size: {s_val}")
                except ValueError:
                    print(f"  Could not parse r/s values from key")
            
            # Check data structure
            data = aposteriori_all[key]
            if isinstance(data, dict):
                print(f"  Data keys: {list(data.keys())}")
                
                # Check number of samples
                if 'n_sample' in data:
                    print(f"  Number of samples: {data['n_sample']}")
                
                # Check length of arrays
                for metric_key in ['second_moment_baseline_all', 'second_moment_ours_all']:
                    if metric_key in data:
                        print(f"  {metric_key} length: {len(data[metric_key])}")
                        if len(data[metric_key]) > 0:
                            print(f"    Sample values: {data[metric_key][:10]}...")  # First 10 values

        print("\n" + "="*60)
        print("SUMMARY BY COARSENING RATIO:")
        print("="*60)
        
        for r_val in sorted(configs_by_ratio.keys()):
            stencil_sizes = sorted(set(configs_by_ratio[r_val]))
            print(f"r={r_val}: stencil sizes {stencil_sizes}")
        
        print(f"\nTotal configurations tested: {len(aposteriori_all)}")
        
        # Check if all configs have same number of ICs
        ic_counts = []
        for key, data in aposteriori_all.items():
            if isinstance(data, dict) and 'second_moment_baseline_all' in data:
                ic_counts.append(len(data['second_moment_baseline_all']))

        if ic_counts:
            unique_counts = set(ic_counts)
            if len(unique_counts) == 1:
                print(f"All configurations have {unique_counts.pop()} initial conditions ✓")
            else:
                print(f"WARNING: Inconsistent IC counts: {unique_counts}")
        
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    examine_aposteriori_all_pkl()