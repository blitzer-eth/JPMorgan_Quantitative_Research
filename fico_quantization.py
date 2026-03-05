"""
FICO Score Quantization for Credit Risk Modeling
JPMorgan Chase - Mortgage Risk Team

This module implements optimal binning/quantization of FICO scores into discrete
buckets for categorical machine learning models. Two optimization approaches:

1. Mean Squared Error (MSE) Minimization
2. Log-Likelihood Maximization (using dynamic programming)

The goal is to find bucket boundaries that best summarize the credit risk profile
while preserving information about default probability.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


# DATA LOADING & EXPLORATION

def load_and_explore_data(filepath='Loan_Data.csv'):
    """Load loan data and analyze FICO score distribution."""
    
    df = pd.read_csv(filepath)
    
    print("═" * 80)
    print("FICO SCORE QUANTIZATION - DATA SUMMARY")
    print("═" * 80)
    print(f"\nTotal records: {len(df):,}")
    print(f"FICO score range: [{df['fico_score'].min()}, {df['fico_score'].max()}]")
    print(f"Mean FICO: {df['fico_score'].mean():.2f}")
    print(f"Median FICO: {df['fico_score'].median():.0f}")
    print(f"\nDefault rate: {df['default'].mean():.2%}")
    
    # Analyze default rate by FICO percentiles
    print(f"\n{'─' * 80}")
    print("Default Rate by FICO Percentiles")
    print(f"{'─' * 80}")
    
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        fico_cutoff = df['fico_score'].quantile(p/100)
        below_rate = df[df['fico_score'] <= fico_cutoff]['default'].mean()
        above_rate = df[df['fico_score'] > fico_cutoff]['default'].mean()
        print(f"{p}th percentile (FICO ≤ {fico_cutoff:.0f}): "
              f"{below_rate:.2%} default | "
              f"Above: {above_rate:.2%} default")
    
    return df


# APPROACH 1: MSE-BASED BINNING

def calculate_mse_for_bins(fico_scores: np.ndarray, boundaries: List[float]) -> float:
    """
    Calculate MSE when FICO scores are binned with given boundaries.
    
    Each score in a bin is approximated by the bin's mean value.
    MSE = (1/n) * Σ(actual_value - bin_mean)²
    """
    n = len(fico_scores)
    mse = 0.0
    
    # Add endpoints
    bounds = [fico_scores.min()] + sorted(boundaries) + [fico_scores.max() + 1]
    
    for i in range(len(bounds) - 1):
        lower, upper = bounds[i], bounds[i + 1]
        bin_values = fico_scores[(fico_scores >= lower) & (fico_scores < upper)]
        
        if len(bin_values) > 0:
            bin_mean = bin_values.mean()
            bin_mse = np.sum((bin_values - bin_mean) ** 2)
            mse += bin_mse
    
    return mse / n


def mse_binning(fico_scores: np.ndarray, n_bins: int) -> List[float]:
    """
    Find optimal bin boundaries using equal-frequency (quantile) binning.
    
    This minimizes MSE by ensuring bins have similar counts, which tends to
    minimize within-bin variance.
    
    Parameters
    ----------
    fico_scores : np.ndarray
        Array of FICO scores
    n_bins : int
        Number of bins to create
        
    Returns
    -------
    List[float]
        Bin boundaries (excluding min and max)
    """
    # Equal-frequency binning via quantiles
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    boundaries = [np.percentile(fico_scores, p) for p in percentiles]
    
    # Remove duplicates and sort
    boundaries = sorted(list(set(boundaries)))
    
    return boundaries


# APPROACH 2: LOG-LIKELIHOOD MAXIMIZATION (DYNAMIC PROGRAMMING)

def calculate_bucket_log_likelihood(fico_scores: np.ndarray, 
                                   defaults: np.ndarray,
                                   lower: float, 
                                   upper: float) -> float:
    """
    Calculate log-likelihood for a single bucket.
    
    LL = k*ln(p) + (n-k)*ln(1-p)
    
    where:
        n = number of records in bucket
        k = number of defaults in bucket
        p = k/n = probability of default
    """
    mask = (fico_scores >= lower) & (fico_scores < upper)
    n = mask.sum()
    k = defaults[mask].sum()
    
    if n == 0:
        return -np.inf
    
    p = k / n if n > 0 else 0
    
    # Handle edge cases
    if p == 0:
        # All non-defaults: LL = n*ln(1)
        return 0.0
    elif p == 1:
        # All defaults: LL = k*ln(1) = 0, but we want to penalize this
        return k * np.log(p)
    else:
        # Standard case
        ll = k * np.log(p) + (n - k) * np.log(1 - p)
        return ll


def log_likelihood_binning_dp(fico_scores: np.ndarray, 
                               defaults: np.ndarray, 
                               n_bins: int) -> List[float]:
    """
    Find optimal bin boundaries by maximizing log-likelihood using dynamic programming.
    
    This is the sophisticated approach that considers both:
    - Discretization roughness (fewer bins = smoother)
    - Default density in each bucket (pure buckets are better)
    
    Parameters
    ----------
    fico_scores : np.ndarray
        Array of FICO scores
    defaults : np.ndarray
        Array of default indicators (0 or 1)
    n_bins : int
        Number of bins to create
        
    Returns
    -------
    List[float]
        Optimal bin boundaries
    """
    # Sort data
    sorted_indices = np.argsort(fico_scores)
    sorted_fico = fico_scores[sorted_indices]
    sorted_defaults = defaults[sorted_indices]
    
    n = len(sorted_fico)
    unique_scores = np.unique(sorted_fico)
    
    # DP table: dp[i][j] = max log-likelihood using j bins for first i unique scores
    dp = np.full((len(unique_scores) + 1, n_bins + 1), -np.inf)
    splits = {}  # Track optimal split points
    
    # Base case: 0 bins for 0 scores
    dp[0][0] = 0
    
    # Fill DP table
    for i in range(1, len(unique_scores) + 1):
        for j in range(1, min(i, n_bins) + 1):
            # Try all possible positions for the (j-1)th split
            for k in range(j - 1, i):
                # Bucket from unique_scores[k] to unique_scores[i-1]
                lower = unique_scores[k] if k > 0 else sorted_fico.min()
                upper = unique_scores[i - 1] + 0.001  # Slightly above to include it
                
                bucket_ll = calculate_bucket_log_likelihood(
                    sorted_fico, sorted_defaults, lower, upper
                )
                
                total_ll = dp[k][j - 1] + bucket_ll
                
                if total_ll > dp[i][j]:
                    dp[i][j] = total_ll
                    splits[(i, j)] = k
    
    # Backtrack to find boundaries
    boundaries = []
    i, j = len(unique_scores), n_bins
    
    while j > 1:
        k = splits.get((i, j))
        if k is None:
            break
        boundaries.append(unique_scores[k])
        i, j = k, j - 1
    
    boundaries.reverse()
    
    # Ensure we have reasonable boundaries
    if len(boundaries) < n_bins - 1:
        # Fallback to quantile-based if DP fails
        print("  ⚠ DP optimization didn't converge well, using quantile fallback")
        percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
        boundaries = [np.percentile(sorted_fico, p) for p in percentiles]
    
    return sorted(list(set(boundaries)))


# BUCKET ANALYSIS

def analyze_buckets(fico_scores: np.ndarray, 
                    defaults: np.ndarray, 
                    boundaries: List[float],
                    method_name: str) -> pd.DataFrame:
    """Analyze the quality of binning."""
    
    bounds = [fico_scores.min()] + sorted(boundaries) + [fico_scores.max() + 1]
    
    bucket_stats = []
    
    for i in range(len(bounds) - 1):
        lower, upper = bounds[i], bounds[i + 1]
        mask = (fico_scores >= lower) & (fico_scores < upper)
        
        n = mask.sum()
        k = defaults[mask].sum()
        p = k / n if n > 0 else 0
        
        mean_fico = fico_scores[mask].mean() if n > 0 else 0
        
        bucket_stats.append({
            'rating': i + 1,  # Lower rating = better credit
            'fico_range': f"[{int(lower)}, {int(upper)})",
            'lower_bound': lower,
            'upper_bound': upper,
            'count': n,
            'defaults': k,
            'default_rate': p,
            'mean_fico': mean_fico
        })
    
    df_stats = pd.DataFrame(bucket_stats)
    
    print(f"\n{'═' * 80}")
    print(f"{method_name.upper()} - BUCKET ANALYSIS")
    print(f"{'═' * 80}")
    print(f"\n{df_stats.to_string(index=False)}")
    
    # Calculate overall metrics
    total_ll = 0
    for _, row in df_stats.iterrows():
        if row['count'] > 0 and 0 < row['default_rate'] < 1:
            k = row['defaults']
            n = row['count']
            p = row['default_rate']
            total_ll += k * np.log(p) + (n - k) * np.log(1 - p)
    
    mse = calculate_mse_for_bins(fico_scores, boundaries)
    
    print(f"\n{'─' * 80}")
    print(f"Overall Metrics:")
    print(f"  Log-Likelihood: {total_ll:.2f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  Number of buckets: {len(df_stats)}")
    print(f"{'─' * 80}")
    
    return df_stats


# RATING MAP CREATION

def create_rating_map(boundaries: List[float], 
                     fico_min: float, 
                     fico_max: float) -> Dict:
    """
    Create a rating map function that assigns FICO scores to buckets.
    
    Lower rating number = Better credit score (inverse of typical risk rating)
    """
    bounds = [fico_min] + sorted(boundaries) + [fico_max + 1]
    
    def assign_rating(fico_score: float) -> int:
        """Assign a rating (1 to n_bins) to a FICO score."""
        for i in range(len(bounds) - 1):
            if bounds[i] <= fico_score < bounds[i + 1]:
                # Higher FICO = lower rating number (better)
                # So we reverse: rating = total_bins - i
                return len(bounds) - 1 - i
        return 1  # Fallback
    
    rating_map = {
        'boundaries': bounds,
        'assign_rating': assign_rating,
        'n_bins': len(bounds) - 1
    }
    
    return rating_map


# VISUALIZATION

def visualize_binning_comparison(df: pd.DataFrame,
                                 mse_boundaries: List[float],
                                 ll_boundaries: List[float],
                                 mse_stats: pd.DataFrame,
                                 ll_stats: pd.DataFrame):
    """Create comprehensive visualization comparing both methods."""
    
    fig = plt.figure(figsize=(18, 14))
    
    # 1. FICO Score Distribution
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(df['fico_score'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.set_title('FICO Score Distribution', fontweight='bold', fontsize=11)
    ax1.set_xlabel('FICO Score')
    ax1.set_ylabel('Frequency')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Default Rate by FICO (smoothed)
    ax2 = plt.subplot(3, 3, 2)
    fico_bins = pd.cut(df['fico_score'], bins=30)
    default_by_fico = df.groupby(fico_bins)['default'].mean()
    bin_centers = [interval.mid for interval in default_by_fico.index]
    ax2.plot(bin_centers, default_by_fico.values, marker='o', linewidth=2, 
             color='#e74c3c', markersize=4)
    ax2.set_title('Default Rate vs FICO Score', fontweight='bold', fontsize=11)
    ax2.set_xlabel('FICO Score')
    ax2.set_ylabel('Default Rate')
    ax2.grid(alpha=0.3)
    
    # 3. MSE Binning - Boundaries
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(df['fico_score'], bins=50, color='lightgray', alpha=0.5, edgecolor='black')
    for boundary in mse_boundaries:
        ax3.axvline(boundary, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8)
    ax3.set_title('MSE Binning - Boundaries', fontweight='bold', fontsize=11)
    ax3.set_xlabel('FICO Score')
    ax3.set_ylabel('Frequency')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. MSE Binning - Bucket Statistics
    ax4 = plt.subplot(3, 3, 4)
    bars = ax4.bar(mse_stats['rating'], mse_stats['count'], 
                   color='#3498db', alpha=0.7, edgecolor='black')
    ax4.set_title('MSE - Bucket Sizes', fontweight='bold', fontsize=11)
    ax4.set_xlabel('Rating (Lower = Better Credit)')
    ax4.set_ylabel('Count')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. MSE Binning - Default Rates
    ax5 = plt.subplot(3, 3, 5)
    bars = ax5.bar(mse_stats['rating'], mse_stats['default_rate'] * 100,
                   color='#e74c3c', alpha=0.7, edgecolor='black')
    ax5.set_title('MSE - Default Rate by Rating', fontweight='bold', fontsize=11)
    ax5.set_xlabel('Rating (Lower = Better Credit)')
    ax5.set_ylabel('Default Rate (%)')
    ax5.grid(axis='y', alpha=0.3)
    
    # Annotate bars
    for i, (rating, rate) in enumerate(zip(mse_stats['rating'], mse_stats['default_rate'])):
        ax5.text(rating, rate * 100 + 1, f"{rate*100:.1f}%", 
                ha='center', va='bottom', fontsize=8)
    
    # 6. Log-Likelihood Binning - Boundaries
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(df['fico_score'], bins=50, color='lightgray', alpha=0.5, edgecolor='black')
    for boundary in ll_boundaries:
        ax6.axvline(boundary, color='#9b59b6', linestyle='--', linewidth=2, alpha=0.8)
    ax6.set_title('Log-Likelihood Binning - Boundaries', fontweight='bold', fontsize=11)
    ax6.set_xlabel('FICO Score')
    ax6.set_ylabel('Frequency')
    ax6.grid(axis='y', alpha=0.3)
    
    # 7. Log-Likelihood Binning - Bucket Sizes
    ax7 = plt.subplot(3, 3, 7)
    bars = ax7.bar(ll_stats['rating'], ll_stats['count'],
                   color='#9b59b6', alpha=0.7, edgecolor='black')
    ax7.set_title('Log-Likelihood - Bucket Sizes', fontweight='bold', fontsize=11)
    ax7.set_xlabel('Rating (Lower = Better Credit)')
    ax7.set_ylabel('Count')
    ax7.grid(axis='y', alpha=0.3)
    
    # 8. Log-Likelihood Binning - Default Rates
    ax8 = plt.subplot(3, 3, 8)
    bars = ax8.bar(ll_stats['rating'], ll_stats['default_rate'] * 100,
                   color='#e74c3c', alpha=0.7, edgecolor='black')
    ax8.set_title('Log-Likelihood - Default Rate by Rating', fontweight='bold', fontsize=11)
    ax8.set_xlabel('Rating (Lower = Better Credit)')
    ax8.set_ylabel('Default Rate (%)')
    ax8.grid(axis='y', alpha=0.3)
    
    # Annotate bars
    for i, (rating, rate) in enumerate(zip(ll_stats['rating'], ll_stats['default_rate'])):
        ax8.text(rating, rate * 100 + 1, f"{rate*100:.1f}%",
                ha='center', va='bottom', fontsize=8)
    
    # 9. Comparison of Default Rates
    ax9 = plt.subplot(3, 3, 9)
    x = np.arange(len(mse_stats))
    width = 0.35
    
    ax9.bar(x - width/2, mse_stats['default_rate'] * 100, width, 
            label='MSE', color='#3498db', alpha=0.7, edgecolor='black')
    ax9.bar(x + width/2, ll_stats['default_rate'] * 100, width,
            label='Log-Likelihood', color='#9b59b6', alpha=0.7, edgecolor='black')
    
    ax9.set_title('Method Comparison - Default Rates', fontweight='bold', fontsize=11)
    ax9.set_xlabel('Rating')
    ax9.set_ylabel('Default Rate (%)')
    ax9.set_xticks(x)
    ax9.set_xticklabels(mse_stats['rating'])
    ax9.legend()
    ax9.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fico_quantization_analysis.png', 
                dpi=150, bbox_inches='tight')
    print("\n✓ Visualizations saved to fico_quantization_analysis.png")


# MAIN EXECUTION

def main(n_bins: int = 5):
    """
    Main execution pipeline for FICO score quantization.
    
    Parameters
    ----------
    n_bins : int
        Number of bins to create (default: 5)
    """
    
    print("\n" + "█" * 80)
    print("FICO SCORE QUANTIZATION FOR CATEGORICAL ML MODELS")
    print("JPMorgan Chase - Mortgage Risk Team")
    print("█" * 80)
    
    # Load data
    df = load_and_explore_data()
    
    fico_scores = df['fico_score'].values
    defaults = df['default'].values
    
    print(f"\n{'═' * 80}")
    print(f"CREATING {n_bins} OPTIMAL BUCKETS")
    print(f"{'═' * 80}")
    
    # Method 1: MSE-based binning
    print(f"\n{'─' * 80}")
    print("Method 1: MSE Minimization (Quantile-based)")
    print(f"{'─' * 80}")
    mse_boundaries = mse_binning(fico_scores, n_bins)
    mse_stats = analyze_buckets(fico_scores, defaults, mse_boundaries, "MSE Method")
    
    # Method 2: Log-Likelihood maximization
    print(f"\n{'─' * 80}")
    print("Method 2: Log-Likelihood Maximization (Dynamic Programming)")
    print(f"{'─' * 80}")
    ll_boundaries = log_likelihood_binning_dp(fico_scores, defaults, n_bins)
    ll_stats = analyze_buckets(fico_scores, defaults, ll_boundaries, "Log-Likelihood Method")
    
    # Create rating maps
    mse_map = create_rating_map(mse_boundaries, fico_scores.min(), fico_scores.max())
    ll_map = create_rating_map(ll_boundaries, fico_scores.min(), fico_scores.max())
    
    # Visualize
    visualize_binning_comparison(df, mse_boundaries, ll_boundaries, mse_stats, ll_stats)
    
    # Comparison summary
    print(f"\n{'═' * 80}")
    print("RECOMMENDATION")
    print(f"{'═' * 80}")
    
    # Calculate metrics for comparison
    mse_ll = sum(
        row['defaults'] * np.log(row['default_rate']) + 
        (row['count'] - row['defaults']) * np.log(1 - row['default_rate'])
        for _, row in mse_stats.iterrows()
        if 0 < row['default_rate'] < 1
    )
    
    ll_ll = sum(
        row['defaults'] * np.log(row['default_rate']) + 
        (row['count'] - row['defaults']) * np.log(1 - row['default_rate'])
        for _, row in ll_stats.iterrows()
        if 0 < row['default_rate'] < 1
    )
    
    mse_mse = calculate_mse_for_bins(fico_scores, mse_boundaries)
    ll_mse = calculate_mse_for_bins(fico_scores, ll_boundaries)
    
    print(f"\nMetric Comparison:")
    print(f"{'Method':<25} {'Log-Likelihood':>15} {'MSE':>12}")
    print("─" * 80)
    print(f"{'MSE Method':<25} {mse_ll:>15.2f} {mse_mse:>12.2f}")
    print(f"{'Log-Likelihood Method':<25} {ll_ll:>15.2f} {ll_mse:>12.2f}")
    
    if ll_ll > mse_ll:
        print(f"\n✓ RECOMMENDED: Log-Likelihood Method")
        print(f"  Higher log-likelihood indicates better separation of default risk")
        recommended_boundaries = ll_boundaries
        recommended_stats = ll_stats
    else:
        print(f"\n✓ RECOMMENDED: MSE Method")
        print(f"  Better approximation of FICO score distribution")
        recommended_boundaries = mse_boundaries
        recommended_stats = mse_stats
    
    print(f"{'═' * 80}\n")
    
    return {
        'mse_boundaries': mse_boundaries,
        'll_boundaries': ll_boundaries,
        'mse_stats': mse_stats,
        'll_stats': ll_stats,
        'mse_map': mse_map,
        'll_map': ll_map,
        'recommended_boundaries': recommended_boundaries,
        'recommended_stats': recommended_stats
    }


if __name__ == "__main__":
    # Run with 5 buckets (standard credit rating categories)
    results = main(n_bins=5)
    
    # Test different bucket sizes
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT BUCKET SIZES")
    print("=" * 80)
    
    for n in [3, 5, 7, 10]:
        print(f"\n{n} buckets:")
        df = pd.read_csv('Loan_Data.csv')
        fico = df['fico_score'].values
        defaults = df['default'].values
        
        ll_bounds = log_likelihood_binning_dp(fico, defaults, n)
        ll_metric = calculate_mse_for_bins(fico, ll_bounds)
        
        mse_bounds = mse_binning(fico, n)
        mse_metric = calculate_mse_for_bins(fico, mse_bounds)
        
        print(f"  Log-Likelihood MSE: {ll_metric:.2f}")
        print(f"  Quantile MSE: {mse_metric:.2f}")
