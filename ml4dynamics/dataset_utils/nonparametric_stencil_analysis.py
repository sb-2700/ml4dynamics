import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests
import pickle
import os
import argparse

def analyze_stencils_nonparametric(
    ours_signed,
    stencil_names=None,
    alpha=0.05,
    use_absolute=True,
):
    """
    Non-parametric comparison of stencil/filter choices using within-IC pairing.

    Parameters
    ----------
    ours_signed : array-like
        2D list/ndarray of 2nd-moment *signed* errors per (IC, stencil) or (stencil, IC).
        Function will convert to absolute values if use_absolute=True.
    stencil_names : list[str] or None
        Optional names for each stencil (length = k). Defaults to S1..Sk.
    alpha : float
        Familywise alpha for Holm-adjusted pairwise tests (default 0.05).
    use_absolute : bool
        If True (default), compare |error| magnitudes. If False, uses signed values.

    Returns
    -------
    results : dict with keys:
        - 'n_ic', 'k'
        - 'abs_err' : ndarray (n_ic × k) used in the analysis
        - 'friedman' : {'chi2', 'p', 'kendalls_W'}
        - 'rank_table' : DataFrame of mean ranks + robust summaries per stencil
        - 'pairwise' : DataFrame of Wilcoxon pairwise results (Holm-adjusted)
        - 'recommended' : list of stencil names with lowest mean rank that are not
                          significantly worse than any competitor (per Holm tests)
    """
    arr = np.array(ours_signed, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array-like (IC × stencil or stencil × IC).")

    # Auto-orient: assume axis-0 are ICs if that axis is larger (e.g., 10 ICs vs 5 stencils)
    if arr.shape[0] >= arr.shape[1]:
        data = arr
    else:
        data = arr.T

    # Use magnitudes if requested
    abs_err = np.abs(data) if use_absolute else data

    # Drop any IC rows with missing values across stencils (complete blocks for Friedman/Wilcoxon)
    mask = ~np.any(np.isnan(abs_err), axis=1)
    abs_err = abs_err[mask]
    n_ic, k = abs_err.shape
    if n_ic < 2 or k < 2:
        raise ValueError("Need at least 2 ICs and 2 stencils after NaN filtering.")

    if stencil_names is None:
        stencil_names = [f"S{j+1}" for j in range(k)]
    if len(stencil_names) != k:
        raise ValueError("stencil_names length must match number of stencils.")

    # ---------- Global test: Friedman on absolute errors ----------
    chi2, p_fried = friedmanchisquare(*[abs_err[:, j] for j in range(k)])
    kendalls_W = chi2 / (n_ic * (k - 1))  # standard relationship

    # ---------- Ranks & robust summaries ----------
    ranks = np.vstack([rankdata(row, method='average') for row in abs_err])  # per-IC ranks; 1=lowest error best
    mean_ranks = ranks.mean(axis=0)

    med = np.median(abs_err, axis=0)
    q1 = np.percentile(abs_err, 25, axis=0)
    q3 = np.percentile(abs_err, 75, axis=0)
    iqr = q3 - q1

    rank_table = pd.DataFrame({
        "stencil": stencil_names,
        "mean_rank": mean_ranks,
        "median_abs_err": med,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
    }).sort_values("mean_rank", kind="mergesort").reset_index(drop=True)

    # ---------- Pairwise Wilcoxon (paired by IC) with Holm correction ----------
    pairs = []
    pvals = []
    stats_W = []
    wins_i = []
    wins_j = []
    ties = []
    med_diff = []

    for i in range(k):
        for j in range(i+1, k):
            x = abs_err[:, i]
            y = abs_err[:, j]
            # Wilcoxon signed-rank on paired magnitudes; drop zero diffs via zero_method='wilcox'
            try:
                res = wilcoxon(x, y, alternative='two-sided', zero_method='wilcox', mode='auto')
                p = res.pvalue
                W = res.statistic
            except ValueError:
                # Happens if all differences are zero
                p, W = 1.0, np.nan

            pairs.append((stencil_names[i], stencil_names[j]))
            pvals.append(p)
            stats_W.append(W)

            # Simple interpretability: who wins per IC?
            d = x - y
            wins_i.append(np.sum(d < 0))  # i has smaller error
            wins_j.append(np.sum(d > 0))  # j has smaller error
            ties.append(np.sum(d == 0))
            med_diff.append(float(np.median(d)))  # <0 means i better

    reject, p_holm, _, _ = multipletests(pvals, alpha=alpha, method='holm')

    pairwise = pd.DataFrame({
        "stencil_i": [a for a, _ in pairs],
        "stencil_j": [b for _, b in pairs],
        "W_stat": stats_W,
        "p_raw": pvals,
        "p_holm": p_holm,
        "reject_at_alpha": reject,
        "wins_i": wins_i,
        "wins_j": wins_j,
        "ties": ties,
        "median_diff_i_minus_j": med_diff,  # negative => i has smaller errors
    }).sort_values("p_holm", kind="mergesort").reset_index(drop=True)

    # ---------- Recommendation: lowest mean rank not significantly worse than others ----------
    best_mean = rank_table["mean_rank"].min()
    candidates = rank_table.loc[rank_table["mean_rank"] == best_mean, "stencil"].tolist()

    def is_significantly_worse(a, b):
        """
        Returns True if Holm-adjusted Wilcoxon indicates a is worse than b (larger errors).
        We decide direction using median difference sign.
        """
        row = pairwise[((pairwise["stencil_i"] == a) & (pairwise["stencil_j"] == b)) |
                       ((pairwise["stencil_i"] == b) & (pairwise["stencil_j"] == a))]
        if row.empty:
            return False
        row = row.iloc[0]
        if not row["reject_at_alpha"]:
            return False
        # Determine sign of (a - b) from the stored median_diff
        if row["stencil_i"] == a:
            med_d = row["median_diff_i_minus_j"]
        else:
            med_d = -row["median_diff_i_minus_j"]
        # If median(a-b) > 0, a has larger errors => worse
        return med_d > 0

    recommended = []
    for a in candidates:
        worse = any(is_significantly_worse(a, b) for b in stencil_names if b != a)
        if not worse:
            recommended.append(a)

    results = {
        "n_ic": n_ic,
        "k": k,
        "abs_err": abs_err,
        "friedman": {"chi2": float(chi2), "p": float(p_fried), "kendalls_W": float(kendalls_W)},
        "rank_table": rank_table,
        "pairwise": pairwise,
        "recommended": recommended if recommended else candidates,  # fallback to min-mean-rank ties
    }

    # -------- Optional: nice console printout --------
    print(f"Friedman χ²({k-1}) = {chi2:.3f}, p = {p_fried:.3g}, Kendall's W = {kendalls_W:.3f}")
    print("\nMean ranks & robust summaries (lower median_abs_err is better):")
    print(rank_table.to_string(index=False))
    print("\nPairwise Wilcoxon (Holm-adjusted):")
    print(pairwise.to_string(index=False))
    print("\nRecommended stencil(s):", results["recommended"])

    return results


def load_stencil_comparison_data(coarsening_ratio=4, model_type='ours'):
    """
    Load and organize second moment error data for stencil comparison.
    
    Parameters
    ----------
    coarsening_ratio : int
        Coarsening ratio to analyze (2, 4, or 8)
    model_type : str
        'ours' for NN model, 'baseline' for baseline model
    
    Returns
    -------
    data_matrix : ndarray
        Shape (n_ic, n_stencils) - second moment errors for each IC and stencil
    stencil_names : list
        Names of stencils (e.g., ['stencil_5', 'stencil_7', ...])
    """
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
                stencil_size = int(s_part[0][1:])  # Remove 's' prefix
                configs[stencil_size] = key
    
    if not configs:
        raise ValueError(f"No data found for coarsening ratio {coarsening_ratio}")
    
    print(f"Found stencil sizes for r={coarsening_ratio}: {sorted(configs.keys())}")
    
    # Extract data for all stencils
    stencil_sizes = sorted(configs.keys())
    stencil_names = [f"stencil_{s}" for s in stencil_sizes]
    
    # Determine the data key
    if model_type == 'ours':
        data_key = 'second_moment_ours_all'
    elif model_type == 'baseline':
        data_key = 'second_moment_baseline_all'
    else:
        raise ValueError("model_type must be 'ours' or 'baseline'")
    
    # Check if all configurations have the same number of ICs
    ic_counts = []
    for stencil_size in stencil_sizes:
        key = configs[stencil_size]
        data = aposteriori_all[key]
        ic_counts.append(len(data[data_key]))
    
    min_ics = min(ic_counts)
    if len(set(ic_counts)) > 1:
        print(f"Warning: Different numbers of ICs across stencils: {dict(zip(stencil_sizes, ic_counts))}")
        print(f"Using first {min_ics} ICs from each stencil for fair comparison")
    
    # Build data matrix: rows = ICs, columns = stencils
    data_matrix = []
    for ic_idx in range(min_ics):
        ic_row = []
        for stencil_size in stencil_sizes:
            key = configs[stencil_size]
            data = aposteriori_all[key]
            error_val = data[data_key][ic_idx]
            ic_row.append(error_val)
        data_matrix.append(ic_row)
    
    data_matrix = np.array(data_matrix)
    
    print(f"\nData matrix shape: {data_matrix.shape} (ICs × stencils)")
    print(f"Model type: {model_type}")
    print(f"Stencil names: {stencil_names}")
    
    return data_matrix, stencil_names


def compare_stencils_and_models(coarsening_ratio=4, alpha=0.05):
    """
    Compare both stencil sizes and model types (NN vs baseline) using non-parametric tests.
    """
    print(f"{'='*80}")
    print(f"NON-PARAMETRIC STENCIL AND MODEL COMPARISON (r={coarsening_ratio})")
    print(f"{'='*80}")
    
    # Load data for both models
    try:
        ours_data, stencil_names = load_stencil_comparison_data(coarsening_ratio, 'ours')
        baseline_data, _ = load_stencil_comparison_data(coarsening_ratio, 'baseline')
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # 1. Compare stencil sizes for NN model
    print(f"\n{'='*60}")
    print("1. STENCIL SIZE COMPARISON - NEURAL NETWORK MODEL")
    print(f"{'='*60}")
    
    results_ours = analyze_stencils_nonparametric(
        ours_data,
        stencil_names=stencil_names,
        alpha=alpha,
        use_absolute=True
    )
    
    # 2. Compare stencil sizes for baseline model
    print(f"\n{'='*60}")
    print("2. STENCIL SIZE COMPARISON - BASELINE MODEL")
    print(f"{'='*60}")
    
    results_baseline = analyze_stencils_nonparametric(
        baseline_data,
        stencil_names=stencil_names,
        alpha=alpha,
        use_absolute=True
    )
    
    # 3. Compare NN vs baseline for each stencil size (pairwise Wilcoxon)
    print(f"\n{'='*60}")
    print("3. MODEL COMPARISON (NN vs BASELINE) PER STENCIL SIZE")
    print(f"{'='*60}")
    
    model_comparison_results = []
    p_values_model = []
    
    for i, stencil_name in enumerate(stencil_names):
        ours_errors = np.abs(ours_data[:, i])
        baseline_errors = np.abs(baseline_data[:, i])
        
        # Remove NaN pairs
        mask = ~(np.isnan(ours_errors) | np.isnan(baseline_errors))
        ours_clean = ours_errors[mask]
        baseline_clean = baseline_errors[mask]
        
        if len(ours_clean) < 2:
            print(f"{stencil_name}: Insufficient valid data")
            continue
        
        # Wilcoxon signed-rank test
        try:
            res = wilcoxon(ours_clean, baseline_clean, alternative='two-sided', zero_method='wilcox')
            p_val = res.pvalue
            W_stat = res.statistic
        except ValueError:
            p_val = 1.0
            W_stat = np.nan
        
        # Summary statistics
        median_ours = np.median(ours_clean)
        median_baseline = np.median(baseline_clean)
        median_diff = np.median(ours_clean - baseline_clean)  # negative = NN better
        
        # Who wins more often?
        nn_wins = np.sum(ours_clean < baseline_clean)
        baseline_wins = np.sum(ours_clean > baseline_clean)
        ties = np.sum(ours_clean == baseline_clean)
        
        result = {
            'stencil': stencil_name,
            'n_valid': len(ours_clean),
            'median_nn': median_ours,
            'median_baseline': median_baseline,
            'median_diff': median_diff,
            'nn_wins': nn_wins,
            'baseline_wins': baseline_wins,
            'ties': ties,
            'W_stat': W_stat,
            'p_raw': p_val
        }
        
        model_comparison_results.append(result)
        p_values_model.append(p_val)
    
    # Apply Holm correction to model comparison p-values
    if p_values_model:
        reject_model, p_holm_model, _, _ = multipletests(p_values_model, alpha=alpha, method='holm')
        
        for i, result in enumerate(model_comparison_results):
            result['p_holm'] = p_holm_model[i]
            result['significant'] = reject_model[i]
    
    # Print model comparison results
    model_df = pd.DataFrame(model_comparison_results)
    print("Model comparison results (Holm-adjusted):")
    print(model_df.to_string(index=False))
    
    # 4. Overall recommendations
    print(f"\n{'='*60}")
    print("4. OVERALL RECOMMENDATIONS")
    print(f"{'='*60}")
    
    print(f"Best NN stencil(s): {results_ours['recommended']}")
    print(f"Best baseline stencil(s): {results_baseline['recommended']}")
    
    if model_comparison_results:
        # Find stencils where NN significantly outperforms baseline
        nn_better = [r['stencil'] for r in model_comparison_results 
                     if r['significant'] and r['median_diff'] < 0]
        baseline_better = [r['stencil'] for r in model_comparison_results 
                          if r['significant'] and r['median_diff'] > 0]
        no_difference = [r['stencil'] for r in model_comparison_results 
                        if not r['significant']]
        
        print(f"\nStencils where NN significantly outperforms baseline: {nn_better}")
        print(f"Stencils where baseline significantly outperforms NN: {baseline_better}")
        print(f"Stencils with no significant difference: {no_difference}")
    
    return {
        'ours_results': results_ours,
        'baseline_results': results_baseline,
        'model_comparison': model_df if model_comparison_results else None
    }


def main():
    """Main function to run non-parametric stencil analysis"""
    parser = argparse.ArgumentParser(description='Non-parametric comparison of stencil sizes and models')
    parser.add_argument('-r', '--coarsening_ratio', type=int, default=4, choices=[2, 4, 8],
                        help='Coarsening ratio to analyze (2, 4, or 8). Default: 4')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level for tests. Default: 0.05')
    parser.add_argument('--model', type=str, default='both', choices=['ours', 'baseline', 'both'],
                        help='Which model(s) to analyze. Default: both')
    
    args = parser.parse_args()
    
    if args.model == 'both':
        compare_stencils_and_models(args.coarsening_ratio, args.alpha)
    else:
        # Analyze single model
        try:
            data_matrix, stencil_names = load_stencil_comparison_data(args.coarsening_ratio, args.model)
            print(f"\nAnalyzing {args.model.upper()} model:")
            analyze_stencils_nonparametric(
                data_matrix,
                stencil_names=stencil_names,
                alpha=args.alpha,
                use_absolute=True
            )
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
