"""
FICO Score Rating Assignment
JPMorgan Chase - Mortgage Risk Team

Simple module for assigning credit ratings to FICO scores using optimal boundaries.
"""

import numpy as np
from typing import Union, List
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMAL BOUNDARIES (from Log-Likelihood optimization)
# ══════════════════════════════════════════════════════════════════════════════

# These boundaries were determined by maximizing log-likelihood using dynamic programming
# on 10,000 historical mortgage loans with 18.51% default rate

OPTIMAL_BOUNDARIES_5_BINS = [521, 581, 641, 697]
OPTIMAL_BOUNDARIES_7_BINS = [497, 545, 587, 623, 653, 688]
OPTIMAL_BOUNDARIES_10_BINS = [472, 521, 557, 587, 617, 641, 665, 688, 722]

# Rating characteristics (for 5-bin model)
RATING_PROFILES = {
    1: {
        'name': 'Excellent',
        'fico_range': '[697, 850]',
        'default_rate': 0.046,  # 4.6%
        'description': 'Very low risk - Prime borrowers'
    },
    2: {
        'name': 'Good', 
        'fico_range': '[641, 697)',
        'default_rate': 0.105,  # 10.5%
        'description': 'Low risk - Above average credit'
    },
    3: {
        'name': 'Fair',
        'fico_range': '[581, 641)',
        'default_rate': 0.204,  # 20.4%
        'description': 'Moderate risk - Average credit'
    },
    4: {
        'name': 'Poor',
        'fico_range': '[521, 581)',
        'default_rate': 0.381,  # 38.1%
        'description': 'High risk - Below average credit'
    },
    5: {
        'name': 'Very Poor',
        'fico_range': '[408, 521)',
        'default_rate': 0.661,  # 66.1%
        'description': 'Very high risk - Subprime borrowers'
    }
}


# ══════════════════════════════════════════════════════════════════════════════
# RATING ASSIGNMENT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def assign_fico_rating(fico_score: Union[int, float], 
                       n_bins: int = 5) -> int:
    """
    Assign a credit rating to a FICO score.
    
    Ratings range from 1 (best) to n_bins (worst).
    Lower rating = Better credit score
    
    Parameters
    ----------
    fico_score : int or float
        FICO credit score (300-850)
    n_bins : int, optional
        Number of rating buckets (3, 5, 7, or 10). Default: 5
        
    Returns
    -------
    int
        Credit rating (1 = best, n_bins = worst)
        
    Examples
    --------
    >>> assign_fico_rating(750)
    1  # Excellent
    
    >>> assign_fico_rating(620)
    3  # Fair
    
    >>> assign_fico_rating(480)
    5  # Very Poor
    """
    
    # Select boundaries based on n_bins
    if n_bins == 3:
        boundaries = [581, 641]
    elif n_bins == 5:
        boundaries = OPTIMAL_BOUNDARIES_5_BINS
    elif n_bins == 7:
        boundaries = OPTIMAL_BOUNDARIES_7_BINS
    elif n_bins == 10:
        boundaries = OPTIMAL_BOUNDARIES_10_BINS
    else:
        raise ValueError(f"n_bins must be 3, 5, 7, or 10. Got: {n_bins}")
    
    # Assign rating (lower is better, so we reverse)
    for i, boundary in enumerate(sorted(boundaries, reverse=True)):
        if fico_score >= boundary:
            return i + 1
    
    # Lowest FICO scores get worst rating
    return n_bins


def assign_fico_rating_batch(fico_scores: Union[List, np.ndarray, pd.Series],
                             n_bins: int = 5) -> np.ndarray:
    """
    Assign credit ratings to multiple FICO scores at once.
    
    Parameters
    ----------
    fico_scores : list, array, or Series
        FICO credit scores to rate
    n_bins : int, optional
        Number of rating buckets. Default: 5
        
    Returns
    -------
    np.ndarray
        Array of credit ratings
        
    Examples
    --------
    >>> scores = [750, 650, 580, 500]
    >>> assign_fico_rating_batch(scores)
    array([1, 2, 4, 5])
    """
    
    if isinstance(fico_scores, pd.Series):
        fico_scores = fico_scores.values
    elif isinstance(fico_scores, list):
        fico_scores = np.array(fico_scores)
    
    return np.array([assign_fico_rating(score, n_bins) for score in fico_scores])


def get_rating_info(rating: int) -> dict:
    """
    Get detailed information about a credit rating.
    
    Parameters
    ----------
    rating : int
        Credit rating (1-5 for default 5-bin model)
        
    Returns
    -------
    dict
        Dictionary with rating details
        
    Examples
    --------
    >>> info = get_rating_info(3)
    >>> print(info['name'])
    'Fair'
    >>> print(f"{info['default_rate']:.1%}")
    '20.4%'
    """
    
    if rating not in RATING_PROFILES:
        raise ValueError(f"Rating must be 1-5. Got: {rating}")
    
    return RATING_PROFILES[rating]


def get_expected_default_rate(fico_score: Union[int, float]) -> float:
    """
    Get the expected default rate for a given FICO score.
    
    Based on historical data from 10,000 loans.
    
    Parameters
    ----------
    fico_score : int or float
        FICO credit score
        
    Returns
    -------
    float
        Expected default probability (0 to 1)
        
    Examples
    --------
    >>> get_expected_default_rate(750)
    0.046  # 4.6% default rate
    
    >>> get_expected_default_rate(550)
    0.381  # 38.1% default rate
    """
    
    rating = assign_fico_rating(fico_score)
    return RATING_PROFILES[rating]['default_rate']


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_fico_portfolio(fico_scores: Union[List, np.ndarray, pd.Series],
                           n_bins: int = 5) -> pd.DataFrame:
    """
    Analyze the credit quality distribution of a loan portfolio.
    
    Parameters
    ----------
    fico_scores : list, array, or Series
        FICO scores in the portfolio
    n_bins : int, optional
        Number of rating buckets. Default: 5
        
    Returns
    -------
    pd.DataFrame
        Summary statistics by rating
        
    Examples
    --------
    >>> import pandas as pd
    >>> portfolio = pd.read_csv('loans.csv')
    >>> summary = analyze_fico_portfolio(portfolio['fico_score'])
    >>> print(summary)
    """
    
    ratings = assign_fico_rating_batch(fico_scores, n_bins)
    
    if n_bins == 5:
        summary = []
        for rating in range(1, 6):
            count = np.sum(ratings == rating)
            pct = count / len(ratings) * 100
            profile = RATING_PROFILES[rating]
            
            summary.append({
                'Rating': rating,
                'Name': profile['name'],
                'FICO Range': profile['fico_range'],
                'Count': count,
                'Percentage': f"{pct:.1f}%",
                'Expected Default Rate': f"{profile['default_rate']:.1%}"
            })
        
        return pd.DataFrame(summary)
    else:
        # Generic summary for other bin sizes
        unique_ratings = np.unique(ratings)
        summary = []
        for rating in unique_ratings:
            count = np.sum(ratings == rating)
            pct = count / len(ratings) * 100
            summary.append({
                'Rating': rating,
                'Count': count,
                'Percentage': f"{pct:.1f}%"
            })
        return pd.DataFrame(summary)


# ══════════════════════════════════════════════════════════════════════════════
# DEMO / TESTING
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("FICO RATING ASSIGNMENT - DEMO")
    print("=" * 80)
    
    # Test individual scores
    test_scores = [800, 720, 650, 600, 550, 480]
    
    print("\n" + "-" * 80)
    print("Individual Score Ratings (5-bin model)")
    print("-" * 80)
    print(f"{'FICO Score':<12} {'Rating':<8} {'Name':<15} {'Default Rate':<15}")
    print("-" * 80)
    
    for score in test_scores:
        rating = assign_fico_rating(score)
        info = get_rating_info(rating)
        print(f"{score:<12} {rating:<8} {info['name']:<15} {info['default_rate']:<14.1%}")
    
    # Test portfolio analysis
    print("\n" + "-" * 80)
    print("Portfolio Analysis Example")
    print("-" * 80)
    
    # Load actual data
    import pandas as pd
    df = pd.read_csv('Loan_Data.csv')
    
    summary = analyze_fico_portfolio(df['fico_score'])
    print("\n" + summary.to_string(index=False))
    
    # Calculate portfolio risk
    print("\n" + "-" * 80)
    print("Portfolio Risk Metrics")
    print("-" * 80)
    
    ratings = assign_fico_rating_batch(df['fico_score'])
    expected_defaults = [RATING_PROFILES[r]['default_rate'] for r in ratings]
    expected_default_rate = np.mean(expected_defaults)
    actual_default_rate = df['default'].mean()
    
    print(f"Expected default rate: {expected_default_rate:.2%}")
    print(f"Actual default rate: {actual_default_rate:.2%}")
    print(f"Difference: {abs(expected_default_rate - actual_default_rate):.2%}")
    
    print("\n" + "=" * 80 + "\n")
