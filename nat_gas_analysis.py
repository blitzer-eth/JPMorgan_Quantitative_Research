import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from datetime import datetime, date


# ── 1. Load Data

df = pd.read_csv("Nat_Gas.csv")
df.columns = ["Date", "Price"]
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
df = df.sort_values("Date").reset_index(drop=True)

# Numeric "time since start" in years – convenient for fitting
t0 = df["Date"].iloc[0]
df["t"] = (df["Date"] - t0).dt.days / 365.25

prices = df["Price"].values
t      = df["t"].values


# ── 2. Model: linear trend + annual sinusoidal seasonality
#
#   price(t) = a + b*t + A*sin(2π*t + φ)
#
#   where t is in years (so period = 1 year)

def model(t, a, b, A, phi):
    return a + b * t + A * np.sin(2 * np.pi * t + phi)

p0 = [prices.mean(), 0.5, 0.5, 0]  # initial guesses
params, _ = curve_fit(model, t, prices, p0=p0, maxfev=10_000)
a, b, A, phi = params

print("=== Fitted Model Parameters ===")
print(f"  Intercept (a)  : {a:.4f}")
print(f"  Trend slope (b): {b:.4f}  ($/year)")
print(f"  Seasonal amp(A): {A:.4f}")
print(f"  Phase (φ)      : {phi:.4f} rad")
print()


# ── 3. Estimate Price Function

def estimate_price(input_date) -> float:
    """
    Return an estimated natural gas price for any given date.

    Parameters
    ----------
    input_date : str | datetime | date
        The target date.  Strings are parsed automatically (e.g. '2025-03-15').

    Returns
    -------
    float
        Estimated price in $/MMBtu (same units as source data).

    Examples
    --------
    >>> estimate_price('2025-06-30')
    11.43
    >>> estimate_price(datetime(2020, 11, 15))
    10.25
    """
    if isinstance(input_date, str):
        input_date = pd.to_datetime(input_date)
    elif isinstance(input_date, date) and not isinstance(input_date, datetime):
        input_date = datetime.combine(input_date, datetime.min.time())

    t_query = (input_date - t0).days / 365.25
    price   = model(t_query, *params)
    return round(float(price), 4)


# ── 4. Visualisation

fig, axes = plt.subplots(3, 1, figsize=(13, 15))
fig.suptitle("Natural Gas Price Analysis", fontsize=16, fontweight="bold", y=0.98)

# --- Panel A: Raw data + fitted curve + 1-year extrapolation ---
ax1 = axes[0]

# Extrapolation range: last data point → +12 months
last_date  = df["Date"].iloc[-1]
extrap_end = last_date + pd.DateOffset(months=12)
future_dates = pd.date_range(last_date, extrap_end, freq="D")
future_t     = (future_dates - t0).days / 365.25
future_price = model(future_t, *params)

# Smooth fitted curve over the historical window
fit_dates  = pd.date_range(df["Date"].iloc[0], last_date, freq="D")
fit_t      = (fit_dates - t0).days / 365.25
fit_price  = model(fit_t, *params)

ax1.scatter(df["Date"], df["Price"], color="#1f77b4", s=50, zorder=5, label="Observed monthly price")
ax1.plot(fit_dates, fit_price, color="#ff7f0e", linewidth=2, label="Fitted model (trend + seasonality)")
ax1.plot(future_dates, future_price, color="#d62728", linewidth=2,
         linestyle="--", label="1-year extrapolation")
ax1.axvline(last_date, color="gray", linestyle=":", linewidth=1.5, label="Last observed date")

ax1.set_title("A) Historical Prices, Model Fit & Extrapolation", fontsize=12)
ax1.set_ylabel("Price ($/MMBtu)")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# --- Panel B: Seasonal pattern by calendar month ---
ax2 = axes[1]
df["Month"] = df["Date"].dt.month
monthly_avg = df.groupby("Month")["Price"].mean()

month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
bars = ax2.bar(monthly_avg.index, monthly_avg.values,
               color=plt.cm.coolwarm(np.linspace(0.1, 0.9, 12)),
               edgecolor="white", linewidth=0.5)
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(month_names)
ax2.set_title("B) Average Price by Calendar Month (Seasonal Pattern)", fontsize=12)
ax2.set_ylabel("Average Price ($/MMBtu)")
ax2.grid(axis="y", alpha=0.3)

# Annotate bars
for bar, val in zip(bars, monthly_avg.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{val:.2f}", ha="center", va="bottom", fontsize=8)

# --- Panel C: Residuals ---
ax3 = axes[2]
residuals = prices - model(t, *params)
ax3.bar(df["Date"], residuals,
        color=np.where(residuals >= 0, "#2ca02c", "#d62728"),
        width=20, alpha=0.8)
ax3.axhline(0, color="black", linewidth=1)
ax3.set_title("C) Residuals (Observed − Fitted)", fontsize=12)
ax3.set_ylabel("Residual ($/MMBtu)")
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("nat_gas_analysis.png", dpi=150, bbox_inches="tight")
print("Chart saved to nat_gas_analysis.png")


# ── 5. Quick demo

demo_dates = [
    "2020-11-15",   # interpolation – early historical
    "2022-07-04",   # interpolation – mid range
    "2024-09-30",   # last observed date
    "2024-12-31",   # near-future extrapolation
    "2025-06-30",   # 9 months out
    "2025-09-30",   # 1-year extrapolation limit
]

print("=== Demo: estimate_price() ===")
print(f"{'Date':<15}  {'Estimated Price':>16}")
print("-" * 33)
for d in demo_dates:
    print(f"{d:<15}  ${estimate_price(d):>14.4f}")
