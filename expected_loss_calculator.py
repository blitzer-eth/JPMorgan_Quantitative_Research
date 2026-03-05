"""
Expected Loss Calculator - Simple Interface
JPMorgan Chase - Retail Banking Risk Team

This module provides a simple function interface for calculating expected loss
on personal loans. Load the pre-trained model and use the estimate_expected_loss
function to get predictions.

Usage:
    from expected_loss_calculator import estimate_expected_loss
    
    loan = {
        'credit_lines_outstanding': 2,
        'loan_amt_outstanding': 8000,
        'total_debt_outstanding': 15000,
        'income': 50000,
        'years_employed': 3,
        'fico_score': 650
    }
    
    result = estimate_expected_loss(loan)
    print(f"Expected Loss: ${result['expected_loss']:,.2f}")
"""

import pickle
import pandas as pd
import numpy as np
from typing import Dict, Union
import sys
import os

# Add the directory to path to import the class definition
sys.path.insert(0, '/home/claude')
sys.path.insert(0, '/mnt/user-data/outputs')

# Import the class definition so pickle can deserialize
try:
    from credit_risk_model import ExpectedLossCalculator
except ImportError:
    pass

# Load the pre-trained model and calculator
_CALCULATOR = None
try:
    with open('/mnt/user-data/outputs/expected_loss_calculator.pkl', 'rb') as f:
        _CALCULATOR = pickle.load(f)
    print("✓ Expected loss calculator loaded successfully")
except (FileNotFoundError, AttributeError) as e:
    print(f"⚠ Calculator not found or could not be loaded: {e}")
    print("Please run credit_risk_model.py first.")
    _CALCULATOR = None


def estimate_expected_loss(loan_data: Dict[str, Union[int, float]]) -> Dict:
    """
    Calculate expected loss for a personal loan.
    
    This is the main function that the risk team should use for calculating
    expected loss on individual loans or portfolios.
    
    Parameters
    ----------
    loan_data : dict
        Dictionary containing the following keys:
        - credit_lines_outstanding : int
            Number of other credit lines the borrower has
        - loan_amt_outstanding : float
            Amount of the loan in dollars
        - total_debt_outstanding : float
            Total debt including this loan in dollars
        - income : float
            Annual income in dollars
        - years_employed : int
            Number of years in current employment
        - fico_score : int
            FICO credit score (300-850)
    
    Returns
    -------
    dict
        Dictionary containing:
        - probability_of_default : float
            Estimated probability that borrower will default (0 to 1)
        - exposure_at_default : float
            Loan amount at risk (same as loan_amt_outstanding)
        - loss_given_default : float
            Percentage of loan lost in default (90% = 0.90)
        - expected_loss : float
            Expected loss in dollars
        - recovery_rate : float
            Assumed recovery rate (10% = 0.10)
    
    Examples
    --------
    >>> loan = {
    ...     'credit_lines_outstanding': 1,
    ...     'loan_amt_outstanding': 5000,
    ...     'total_debt_outstanding': 6000,
    ...     'income': 60000,
    ...     'years_employed': 4,
    ...     'fico_score': 680
    ... }
    >>> result = estimate_expected_loss(loan)
    >>> print(f"PD: {result['probability_of_default']:.2%}")
    >>> print(f"Expected Loss: ${result['expected_loss']:,.2f}")
    
    Notes
    -----
    The expected loss is calculated as:
        EL = PD × EAD × LGD
    
    where:
        PD = Probability of Default (from machine learning model)
        EAD = Exposure at Default (loan amount)
        LGD = Loss Given Default (1 - recovery rate = 90%)
    
    The model uses features including:
        - Credit utilization metrics
        - Debt-to-income ratios
        - Employment stability
        - FICO score
    """
    
    if _CALCULATOR is None:
        raise RuntimeError("Calculator not loaded. Run credit_risk_model.py first.")
    
    # Validate input
    required_fields = [
        'credit_lines_outstanding',
        'loan_amt_outstanding',
        'total_debt_outstanding',
        'income',
        'years_employed',
        'fico_score'
    ]
    
    missing_fields = [field for field in required_fields if field not in loan_data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Calculate expected loss
    return _CALCULATOR.calculate_expected_loss(loan_data)


def estimate_portfolio_loss(loans_df: pd.DataFrame) -> Dict:
    """
    Calculate expected loss for a portfolio of loans.
    
    Parameters
    ----------
    loans_df : pd.DataFrame
        DataFrame with columns matching loan_data requirements
        (see estimate_expected_loss for details)
    
    Returns
    -------
    dict
        Dictionary containing:
        - total_loans : int
            Number of loans in portfolio
        - total_exposure : float
            Sum of all loan amounts
        - total_expected_loss : float
            Sum of expected losses across all loans
        - expected_loss_rate : float
            Expected loss as percentage of total exposure
        - loan_details : list
            List of dictionaries with per-loan results
    
    Examples
    --------
    >>> import pandas as pd
    >>> loans = pd.DataFrame({
    ...     'credit_lines_outstanding': [0, 2, 5],
    ...     'loan_amt_outstanding': [5000, 8000, 10000],
    ...     'total_debt_outstanding': [5000, 15000, 25000],
    ...     'income': [80000, 50000, 35000],
    ...     'years_employed': [5, 3, 1],
    ...     'fico_score': [750, 650, 550]
    ... })
    >>> result = estimate_portfolio_loss(loans)
    >>> print(f"Total Expected Loss: ${result['total_expected_loss']:,.2f}")
    >>> print(f"Expected Loss Rate: {result['expected_loss_rate']:.2%}")
    """
    
    if _CALCULATOR is None:
        raise RuntimeError("Calculator not loaded. Run credit_risk_model.py first.")
    
    results = []
    total_el = 0
    total_exposure = 0
    
    for idx, row in loans_df.iterrows():
        loan_data = row.to_dict()
        result = estimate_expected_loss(loan_data)
        results.append(result)
        total_el += result['expected_loss']
        total_exposure += result['exposure_at_default']
    
    return {
        'total_loans': len(loans_df),
        'total_exposure': round(total_exposure, 2),
        'total_expected_loss': round(total_el, 2),
        'expected_loss_rate': round(total_el / total_exposure, 6) if total_exposure > 0 else 0,
        'loan_details': results
    }


# ══════════════════════════════════════════════════════════════════════════════
# DEMO / TESTING
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("EXPECTED LOSS CALCULATOR - DEMO")
    print("=" * 80)
    
    # Test case 1: Low risk borrower
    print("\n" + "-" * 80)
    print("Example 1: Low Risk Borrower")
    print("-" * 80)
    
    low_risk = {
        'credit_lines_outstanding': 0,
        'loan_amt_outstanding': 5000,
        'total_debt_outstanding': 5000,
        'income': 80000,
        'years_employed': 5,
        'fico_score': 750
    }
    
    result = estimate_expected_loss(low_risk)
    print(f"Loan Amount: ${low_risk['loan_amt_outstanding']:,.2f}")
    print(f"Income: ${low_risk['income']:,.2f}")
    print(f"FICO: {low_risk['fico_score']}")
    print(f"\n→ Probability of Default: {result['probability_of_default']:.2%}")
    print(f"→ Expected Loss: ${result['expected_loss']:,.2f}")
    
    # Test case 2: High risk borrower
    print("\n" + "-" * 80)
    print("Example 2: High Risk Borrower")
    print("-" * 80)
    
    high_risk = {
        'credit_lines_outstanding': 5,
        'loan_amt_outstanding': 10000,
        'total_debt_outstanding': 25000,
        'income': 35000,
        'years_employed': 1,
        'fico_score': 550
    }
    
    result = estimate_expected_loss(high_risk)
    print(f"Loan Amount: ${high_risk['loan_amt_outstanding']:,.2f}")
    print(f"Income: ${high_risk['income']:,.2f}")
    print(f"FICO: {high_risk['fico_score']}")
    print(f"\n→ Probability of Default: {result['probability_of_default']:.2%}")
    print(f"→ Expected Loss: ${result['expected_loss']:,.2f}")
    
    # Test case 3: Portfolio calculation
    print("\n" + "-" * 80)
    print("Example 3: Portfolio of 3 Loans")
    print("-" * 80)
    
    portfolio = pd.DataFrame([
        low_risk,
        {
            'credit_lines_outstanding': 2,
            'loan_amt_outstanding': 8000,
            'total_debt_outstanding': 15000,
            'income': 50000,
            'years_employed': 3,
            'fico_score': 650
        },
        high_risk
    ])
    
    portfolio_result = estimate_portfolio_loss(portfolio)
    print(f"Number of Loans: {portfolio_result['total_loans']}")
    print(f"Total Exposure: ${portfolio_result['total_exposure']:,.2f}")
    print(f"Total Expected Loss: ${portfolio_result['total_expected_loss']:,.2f}")
    print(f"Expected Loss Rate: {portfolio_result['expected_loss_rate']:.2%}")
    
    print("\n" + "=" * 80 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_risk_category(probability_of_default: float) -> str:
    """
    Categorize loan risk based on PD.
    
    Parameters
    ----------
    probability_of_default : float
        Probability of default (0 to 1)
    
    Returns
    -------
    str
        Risk category: 'Low', 'Medium', 'High', or 'Very High'
    """
    if probability_of_default < 0.05:
        return "Low"
    elif probability_of_default < 0.15:
        return "Medium"
    elif probability_of_default < 0.30:
        return "High"
    else:
        return "Very High"


def calculate_required_capital(expected_loss: float, confidence_level: float = 0.99) -> float:
    """
    Calculate required regulatory capital using a simple multiplier approach.
    
    Note: This is a simplified calculation. Production systems should use
    Basel III or other regulatory frameworks.
    
    Parameters
    ----------
    expected_loss : float
        Expected loss in dollars
    confidence_level : float
        Confidence level for capital adequacy (default: 99%)
    
    Returns
    -------
    float
        Required capital buffer
    """
    # Simple approach: capital = EL × multiplier
    # Multiplier depends on confidence level
    if confidence_level >= 0.99:
        multiplier = 3.0
    elif confidence_level >= 0.95:
        multiplier = 2.0
    else:
        multiplier = 1.5
    
    return expected_loss * multiplier
