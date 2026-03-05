"""
Quick Usage Script for Trading Desk
Natural Gas Storage Contract Pricing

This script provides a simplified interface for pricing contracts.
Modify the parameters below and run to get instant valuations.
"""

from contract_pricing import price_storage_contract

# ════════════════════════════════════════════════════════════════════════════
# MODIFY THESE PARAMETERS FOR YOUR CONTRACT
# ════════════════════════════════════════════════════════════════════════════

# When do we buy gas and put it in storage?
INJECTION_DATES = [
    '2024-10-31',
    '2024-11-30',
    '2024-12-31'
]

# When do we take gas out and sell it?
WITHDRAWAL_DATES = [
    '2025-01-31',
    '2025-02-28',
    '2025-03-31'
]

# How much gas per injection/withdrawal? (MMBtu)
INJECTION_VOLUME = 500_000
WITHDRAWAL_VOLUME = 500_000

# What's the storage facility capacity? (MMBtu)
MAX_STORAGE = 2_000_000

# What does storage cost per month? ($)
MONTHLY_STORAGE_COST = 100_000

# What are the operational costs?
INJECTION_FEE = 0.01        # $/MMBtu
WITHDRAWAL_FEE = 0.01       # $/MMBtu
TRANSPORT_FEE = 50_000      # $ per event

# ════════════════════════════════════════════════════════════════════════════
# RUN PRICING MODEL
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PRICING YOUR CONTRACT...")
    print("="*70 + "\n")
    
    try:
        result = price_storage_contract(
            injection_dates=INJECTION_DATES,
            withdrawal_dates=WITHDRAWAL_DATES,
            injection_rate=INJECTION_VOLUME,
            withdrawal_rate=WITHDRAWAL_VOLUME,
            max_storage_volume=MAX_STORAGE,
            storage_cost_per_month=MONTHLY_STORAGE_COST,
            injection_cost_per_mmbtu=INJECTION_FEE,
            withdrawal_cost_per_mmbtu=WITHDRAWAL_FEE,
            transport_cost_per_event=TRANSPORT_FEE,
            verbose=True
        )
        
        # Quick decision guidance
        print("\n" + "="*70)
        print("RECOMMENDATION")
        print("="*70)
        
        value = result['contract_value']
        roi = (value / (result['total_purchase_cost'] + 
                       result['total_storage_cost'] + 
                       result['total_injection_cost'] + 
                       result['total_withdrawal_cost'] + 
                       result['total_transport_cost']) * 100)
        
        if value > 500_000:
            print("✓ HIGHLY FAVORABLE - Strong profit opportunity")
        elif value > 100_000:
            print("✓ FAVORABLE - Reasonable profit margin")
        elif value > 0:
            print("○ MARGINAL - Small profit, monitor costs closely")
        else:
            print("✗ UNFAVORABLE - Loss expected, do not proceed")
        
        print(f"\n  Expected Profit/Loss: ${value:,.2f}")
        print(f"  Return on Investment: {roi:.2f}%")
        print(f"  Storage Utilization: {result['storage_utilization']*100:.1f}%")
        print("="*70 + "\n")
        
    except ValueError as e:
        print(f"\n❌ ERROR: {e}\n")
        print("Check your parameters and try again.\n")


# ════════════════════════════════════════════════════════════════════════════
# QUICK REFERENCE: Common Scenarios
# ════════════════════════════════════════════════════════════════════════════

"""
SCENARIO 1: Summer → Winter (Classic Seasonal Arbitrage)
---------------------------------------------------------
Injection Dates: ['2024-06-30', '2024-07-31', '2024-08-31']
Withdrawal Dates: ['2024-12-31', '2025-01-31', '2025-02-28']
Expected: Positive value (winter prices higher)


SCENARIO 2: Fall Accumulation → Spring Sale
--------------------------------------------
Injection Dates: ['2024-09-30', '2024-10-31', '2024-11-30']
Withdrawal Dates: ['2025-03-31', '2025-04-30', '2025-05-31']
Expected: Lower value (smaller seasonal spread)


SCENARIO 3: Rapid Turnaround
-----------------------------
Injection Dates: ['2024-11-30']
Withdrawal Dates: ['2025-01-31']
Expected: Best price spread, lowest storage costs


SCENARIO 4: Long-term Hold
---------------------------
Injection Dates: ['2024-06-30']
Withdrawal Dates: ['2025-03-31']
Expected: High storage costs offset price gains
"""
