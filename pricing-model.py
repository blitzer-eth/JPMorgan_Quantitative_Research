"""
Natural Gas Storage Contract Pricing Model
JPMorgan Chase Quantitative Research Task 2

This module provides a pricing function for natural gas storage contracts.
The contract allows a client to inject gas at certain dates, store it, and
withdraw it at later dates to profit from price differences.

Contract Value = Revenue from Sales - Cost of Purchases - All Associated Costs

Associated costs include:
  - Storage rental fees (monthly)
  - Injection fees (per MMBtu)
  - Withdrawal fees (per MMBtu)
  - Transportation costs (per injection/withdrawal event)
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import List, Union, Dict, Tuple
from scipy.optimize import curve_fit


# ── Price Estimation Model (from Task 1) ─────────────────────────────────────

def _load_price_model():
    """Load and fit the price estimation model from historical data."""
    import os
    
    # Try multiple possible locations for the CSV file
    possible_paths = [
        "Nat_Gas.csv",
        "/home/claude/Nat_Gas.csv",
        "/mnt/user-data/uploads/Nat_Gas.csv"
    ]
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if csv_path is None:
        raise FileNotFoundError("Nat_Gas.csv not found in expected locations")
    
    df = pd.read_csv(csv_path)
    df.columns = ["Date", "Price"]
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
    df = df.sort_values("Date").reset_index(drop=True)
    
    t0 = df["Date"].iloc[0]
    df["t"] = (df["Date"] - t0).dt.days / 365.25
    
    prices = df["Price"].values
    t = df["t"].values
    
    def model(t, a, b, A, phi):
        return a + b * t + A * np.sin(2 * np.pi * t + phi)
    
    p0 = [prices.mean(), 0.5, 0.5, 0]
    params, _ = curve_fit(model, t, prices, p0=p0, maxfev=10_000)
    
    return model, params, t0


# Global model parameters (loaded once)
_MODEL, _PARAMS, _T0 = _load_price_model()


def estimate_price(input_date: Union[str, datetime, date]) -> float:
    """
    Estimate natural gas price for any given date.
    
    Parameters
    ----------
    input_date : str | datetime | date
        Target date for price estimation.
        
    Returns
    -------
    float
        Estimated price in $/MMBtu.
    """
    if isinstance(input_date, str):
        input_date = pd.to_datetime(input_date)
    elif isinstance(input_date, date) and not isinstance(input_date, datetime):
        input_date = datetime.combine(input_date, datetime.min.time())
    
    t_query = (input_date - _T0).days / 365.25
    price = _MODEL(t_query, *_PARAMS)
    return float(price)


# ── Storage Contract Pricing ──────────────────────────────────────────────────

def price_storage_contract(
    injection_dates: List[Union[str, datetime]],
    withdrawal_dates: List[Union[str, datetime]],
    injection_rate: float,
    withdrawal_rate: float,
    max_storage_volume: float,
    storage_cost_per_month: float,
    injection_cost_per_mmbtu: float = 0.0,
    withdrawal_cost_per_mmbtu: float = 0.0,
    transport_cost_per_event: float = 0.0,
    verbose: bool = True
) -> Dict:
    """
    Price a natural gas storage contract.
    
    The contract allows purchasing gas at injection dates, storing it, and 
    selling at withdrawal dates. The value is calculated as the net cash flow
    from all transactions minus all costs.
    
    Parameters
    ----------
    injection_dates : List[str | datetime]
        Dates when gas is purchased and injected into storage.
    withdrawal_dates : List[str | datetime]
        Dates when gas is withdrawn from storage and sold.
    injection_rate : float
        Volume of gas injected per injection event (MMBtu).
    withdrawal_rate : float
        Volume of gas withdrawn per withdrawal event (MMBtu).
    max_storage_volume : float
        Maximum storage capacity (MMBtu). Enforces constraints.
    storage_cost_per_month : float
        Monthly fixed rental cost for storage facility ($).
    injection_cost_per_mmbtu : float, optional
        Variable cost per MMBtu for injection ($/MMBtu). Default: 0.
    withdrawal_cost_per_mmbtu : float, optional
        Variable cost per MMBtu for withdrawal ($/MMBtu). Default: 0.
    transport_cost_per_event : float, optional
        Fixed cost per transportation event ($/event). Default: 0.
        Applied to both injections and withdrawals.
    verbose : bool, optional
        If True, print detailed breakdown. Default: True.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'contract_value': Net present value of the contract ($)
        - 'total_revenue': Revenue from gas sales ($)
        - 'total_purchase_cost': Cost of gas purchases ($)
        - 'total_storage_cost': Storage rental fees ($)
        - 'total_injection_cost': Injection operation fees ($)
        - 'total_withdrawal_cost': Withdrawal operation fees ($)
        - 'total_transport_cost': Transportation costs ($)
        - 'storage_utilization': Peak storage volume / max capacity
        - 'injections': List of injection details
        - 'withdrawals': List of withdrawal details
        
    Raises
    ------
    ValueError
        If storage volume constraints are violated or dates are invalid.
        
    Examples
    --------
    >>> result = price_storage_contract(
    ...     injection_dates=['2024-10-31', '2024-11-30'],
    ...     withdrawal_dates=['2025-01-31', '2025-02-28'],
    ...     injection_rate=500_000,
    ...     withdrawal_rate=500_000,
    ...     max_storage_volume=1_000_000,
    ...     storage_cost_per_month=100_000
    ... )
    >>> print(f"Contract Value: ${result['contract_value']:,.2f}")
    """
    
    # Convert all dates to datetime
    injection_dates = [pd.to_datetime(d) for d in injection_dates]
    withdrawal_dates = [pd.to_datetime(d) for d in withdrawal_dates]
    
    # Validate inputs
    if not injection_dates or not withdrawal_dates:
        raise ValueError("Must provide at least one injection and one withdrawal date")
    
    if injection_rate <= 0 or withdrawal_rate <= 0:
        raise ValueError("Injection and withdrawal rates must be positive")
    
    if max_storage_volume <= 0:
        raise ValueError("Maximum storage volume must be positive")
    
    # Sort dates
    injection_dates = sorted(injection_dates)
    withdrawal_dates = sorted(withdrawal_dates)
    
    # Check basic feasibility
    total_injected = len(injection_dates) * injection_rate
    total_withdrawn = len(withdrawal_dates) * withdrawal_rate
    
    if total_injected != total_withdrawn:
        raise ValueError(
            f"Total injected volume ({total_injected:,.0f} MMBtu) must equal "
            f"total withdrawn volume ({total_withdrawn:,.0f} MMBtu)"
        )
    
    if total_injected > max_storage_volume:
        raise ValueError(
            f"Total volume ({total_injected:,.0f} MMBtu) exceeds "
            f"max storage capacity ({max_storage_volume:,.0f} MMBtu)"
        )
    
    # Track storage level over time
    events = []
    for d in injection_dates:
        events.append(("injection", d, injection_rate))
    for d in withdrawal_dates:
        events.append(("withdrawal", d, -withdrawal_rate))
    
    events.sort(key=lambda x: x[1])
    
    storage_level = 0
    peak_storage = 0
    for event_type, event_date, delta in events:
        storage_level += delta
        peak_storage = max(peak_storage, storage_level)
        
        if storage_level < 0:
            raise ValueError(
                f"Storage level goes negative on {event_date.strftime('%Y-%m-%d')}. "
                f"Ensure injections occur before withdrawals."
            )
        
        if storage_level > max_storage_volume:
            raise ValueError(
                f"Storage level ({storage_level:,.0f} MMBtu) exceeds capacity "
                f"({max_storage_volume:,.0f} MMBtu) on {event_date.strftime('%Y-%m-%d')}"
            )
    
    # Calculate cash flows
    injection_details = []
    total_purchase_cost = 0
    total_injection_fees = 0
    
    for inj_date in injection_dates:
        price = estimate_price(inj_date)
        volume = injection_rate
        purchase_cost = price * volume
        injection_fee = injection_cost_per_mmbtu * volume
        
        total_purchase_cost += purchase_cost
        total_injection_fees += injection_fee
        
        injection_details.append({
            'date': inj_date.strftime('%Y-%m-%d'),
            'volume_mmbtu': volume,
            'price_per_mmbtu': round(price, 4),
            'purchase_cost': round(purchase_cost, 2),
            'injection_fee': round(injection_fee, 2)
        })
    
    withdrawal_details = []
    total_revenue = 0
    total_withdrawal_fees = 0
    
    for with_date in withdrawal_dates:
        price = estimate_price(with_date)
        volume = withdrawal_rate
        revenue = price * volume
        withdrawal_fee = withdrawal_cost_per_mmbtu * volume
        
        total_revenue += revenue
        total_withdrawal_fees += withdrawal_fee
        
        withdrawal_details.append({
            'date': with_date.strftime('%Y-%m-%d'),
            'volume_mmbtu': volume,
            'price_per_mmbtu': round(price, 4),
            'revenue': round(revenue, 2),
            'withdrawal_fee': round(withdrawal_fee, 2)
        })
    
    # Calculate storage duration in months
    contract_start = min(injection_dates)
    contract_end = max(withdrawal_dates)
    storage_months = ((contract_end.year - contract_start.year) * 12 + 
                      contract_end.month - contract_start.month)
    
    # Round up partial months
    if contract_end.day > contract_start.day:
        storage_months += 1
    
    total_storage_cost = storage_cost_per_month * storage_months
    
    # Transportation costs
    num_transport_events = len(injection_dates) + len(withdrawal_dates)
    total_transport_cost = transport_cost_per_event * num_transport_events
    
    # Contract value
    contract_value = (
        total_revenue
        - total_purchase_cost
        - total_storage_cost
        - total_injection_fees
        - total_withdrawal_fees
        - total_transport_cost
    )
    
    # Prepare results
    results = {
        'contract_value': round(contract_value, 2),
        'total_revenue': round(total_revenue, 2),
        'total_purchase_cost': round(total_purchase_cost, 2),
        'total_storage_cost': round(total_storage_cost, 2),
        'total_injection_cost': round(total_injection_fees, 2),
        'total_withdrawal_cost': round(total_withdrawal_fees, 2),
        'total_transport_cost': round(total_transport_cost, 2),
        'storage_months': storage_months,
        'storage_utilization': round(peak_storage / max_storage_volume, 4),
        'injections': injection_details,
        'withdrawals': withdrawal_details
    }
    
    # Print summary if verbose
    if verbose:
        print("=" * 70)
        print("NATURAL GAS STORAGE CONTRACT VALUATION")
        print("=" * 70)
        print(f"\n{'INJECTIONS':-^70}")
        print(f"  Number of injections: {len(injection_dates)}")
        print(f"  Volume per injection: {injection_rate:,.0f} MMBtu")
        print(f"  Total volume injected: {total_injected:,.0f} MMBtu")
        
        print(f"\n{'WITHDRAWALS':-^70}")
        print(f"  Number of withdrawals: {len(withdrawal_dates)}")
        print(f"  Volume per withdrawal: {withdrawal_rate:,.0f} MMBtu")
        print(f"  Total volume withdrawn: {total_withdrawn:,.0f} MMBtu")
        
        print(f"\n{'STORAGE':-^70}")
        print(f"  Maximum capacity: {max_storage_volume:,.0f} MMBtu")
        print(f"  Peak utilization: {peak_storage:,.0f} MMBtu ({results['storage_utilization']*100:.1f}%)")
        print(f"  Storage period: {storage_months} months")
        print(f"  Contract start: {contract_start.strftime('%Y-%m-%d')}")
        print(f"  Contract end: {contract_end.strftime('%Y-%m-%d')}")
        
        print(f"\n{'CASH FLOWS':-^70}")
        print(f"  Revenue from sales:        ${total_revenue:>18,.2f}")
        print(f"  Cost of purchases:        -${total_purchase_cost:>18,.2f}")
        print(f"  Storage rental costs:     -${total_storage_cost:>18,.2f}")
        print(f"  Injection fees:           -${total_injection_fees:>18,.2f}")
        print(f"  Withdrawal fees:          -${total_withdrawal_fees:>18,.2f}")
        print(f"  Transportation costs:     -${total_transport_cost:>18,.2f}")
        print(f"  {'-' * 70}")
        print(f"  NET CONTRACT VALUE:        ${contract_value:>18,.2f}")
        print("=" * 70)
    
    return results


# ── Test Cases ────────────────────────────────────────────────────────────────

def run_test_cases():
    """Run sample test cases to demonstrate the pricing model."""
    
    print("\n\n" + "=" * 70)
    print("TEST CASE 1: Simple Summer → Winter Trade")
    print("=" * 70)
    print("Scenario: Buy 1M MMBtu in summer, sell in winter, minimal costs\n")
    
    test1 = price_storage_contract(
        injection_dates=['2024-06-30'],
        withdrawal_dates=['2024-12-31'],
        injection_rate=1_000_000,
        withdrawal_rate=1_000_000,
        max_storage_volume=1_500_000,
        storage_cost_per_month=100_000,
        injection_cost_per_mmbtu=0.01,
        withdrawal_cost_per_mmbtu=0.01,
        transport_cost_per_event=50_000
    )
    
    
    print("\n\n" + "=" * 70)
    print("TEST CASE 2: Multiple Injections & Withdrawals")
    print("=" * 70)
    print("Scenario: Gradual accumulation in fall, gradual sale in winter\n")
    
    test2 = price_storage_contract(
        injection_dates=['2024-09-30', '2024-10-31', '2024-11-30'],
        withdrawal_dates=['2025-01-31', '2025-02-28', '2025-03-31'],
        injection_rate=500_000,
        withdrawal_rate=500_000,
        max_storage_volume=2_000_000,
        storage_cost_per_month=150_000,
        injection_cost_per_mmbtu=0.02,
        withdrawal_cost_per_mmbtu=0.02,
        transport_cost_per_event=30_000
    )
    
    
    print("\n\n" + "=" * 70)
    print("TEST CASE 3: High Storage Costs Scenario")
    print("=" * 70)
    print("Scenario: Testing impact of expensive storage facilities\n")
    
    test3 = price_storage_contract(
        injection_dates=['2024-08-31', '2024-09-30'],
        withdrawal_dates=['2025-01-31', '2025-02-28'],
        injection_rate=750_000,
        withdrawal_rate=750_000,
        max_storage_volume=1_500_000,
        storage_cost_per_month=250_000,
        injection_cost_per_mmbtu=0.05,
        withdrawal_cost_per_mmbtu=0.05,
        transport_cost_per_event=100_000
    )
    
    
    print("\n\n" + "=" * 70)
    print("TEST CASE 4: Capacity Utilization Test")
    print("=" * 70)
    print("Scenario: Multiple small injections building to full capacity\n")
    
    test4 = price_storage_contract(
        injection_dates=['2024-07-31', '2024-08-31', '2024-09-30', '2024-10-31'],
        withdrawal_dates=['2025-01-31', '2025-02-28', '2025-03-31', '2025-04-30'],
        injection_rate=250_000,
        withdrawal_rate=250_000,
        max_storage_volume=1_000_000,
        storage_cost_per_month=80_000,
        injection_cost_per_mmbtu=0.015,
        withdrawal_cost_per_mmbtu=0.015,
        transport_cost_per_event=25_000
    )
    
    return [test1, test2, test3, test4]


if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("NATURAL GAS STORAGE CONTRACT PRICING MODEL")
    print("JPMorgan Chase - Quantitative Research")
    print("█" * 70)
    
    # Run all test cases
    results = run_test_cases()
    
    # Summary comparison
    print("\n\n" + "=" * 70)
    print("SUMMARY: Test Case Comparison")
    print("=" * 70)
    print(f"{'Test':<8} {'Contract Value':>18} {'ROI':>12} {'Utilization':>14}")
    print("-" * 70)
    
    for i, result in enumerate(results, 1):
        revenue = result['total_revenue']
        costs = (result['total_purchase_cost'] + 
                result['total_storage_cost'] + 
                result['total_injection_cost'] + 
                result['total_withdrawal_cost'] + 
                result['total_transport_cost'])
        roi = (result['contract_value'] / costs * 100) if costs > 0 else 0
        
        print(f"Case {i}   ${result['contract_value']:>16,.2f}   "
              f"{roi:>10.2f}%   "
              f"{result['storage_utilization']*100:>12.1f}%")
    
    print("=" * 70)