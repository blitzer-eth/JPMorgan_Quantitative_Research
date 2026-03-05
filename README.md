# JPMorgan Quantitative Research

This repository contains quantitative models and analysis tools developed during the Forage JPMorgan Chase Job Simulation. The project focuses on two primary financial domains: credit risk modeling for retail banking and commodity pricing for natural gas storage contracts.

## Project Modules

### 1. Credit Risk & Expected Loss Modeling

This module predicts the probability of default for personal loan borrowers to estimate potential financial losses.

* **Core Script**: `credit_risk_model.py` builds and evaluates multiple machine learning models, including Logistic Regression, Random Forest, and Gradient Boosting, to identify high risk profiles.
* **FICO Quantization**: `fico_quantization.py` implements optimal binning of FICO scores using Mean Squared Error (MSE) minimization and Log-Likelihood maximization through dynamic programming to maximize information about default probability.
* **Rating Assignment**: `fico_rating_assignment.py` provides a utility to map FICO scores to discrete credit ratings (1 to 5) and risk profiles, ranging from "Excellent" to "Very Poor" based on historical default rates.
* **Feature Engineering**: Includes custom logic for Debt-to-Income (DTI) ratios, FICO score binning, and employment stability indicators.
* **Calculator**: `expected_loss_calculator.py` provides a production-ready interface to calculate Expected Loss (EL) using the standard formula: `EL = PD x EAD x LGD`.
* **Analytics**: Generates comprehensive risk distributions, feature importance plots, and ROC curves to compare model performance.

### 2. Natural Gas Analysis & Storage Pricing

This module analyzes historical natural gas price trends and prices complex storage contracts using seasonal arbitrage strategies.

* **Price Forecasting**: `nat_gas_analysis.py` fits a seasonal model using linear trends and sinusoidal patterns to estimate prices for any future date.
* **Storage Valuation**: `pricing-model.py` calculates the value of storage contracts by modeling the revenue from sales minus costs for storage rental, injection, withdrawal, and transportation.
* **Trading Interface**: `quick_pricer.py` serves as a simplified script for a trading desk to evaluate specific contract scenarios and return a recommendation.

## Tech Stack

* **Language**: Python 3.x
* **Data Science**: Pandas, NumPy, Scikit-learn
* **Optimization**: SciPy (Curve Fitting)
* **Visualization**: Matplotlib, Seaborn

## Project Structure

* `credit_risk_model.py`: Main model training and evaluation script.
* `fico_quantization.py`: Module for finding optimal FICO score bucket boundaries.
* `fico_rating_assignment.py`: Helper script for assigning credit ratings to individual scores.
* `expected_loss_calculator.py`: Interface for individual and portfolio loss estimation.
* `nat_gas_analysis.py`: Historical price modeling and forecasting.
* `pricing-model.py`: Core logic for storage contract valuation.
* `quick_pricer.py`: Simplified tool for rapid contract pricing.

## Getting Started

1. Ensure `Loan_Data.csv` and `Nat_Gas.csv` are available in the project directory.
2. Run `credit_risk_model.py` to train the risk model and export the serialized calculator.
3. Run `fico_quantization.py` to determine optimal FICO score buckets for the portfolio.
4. Execute `nat_gas_analysis.py` to view price trends and seasonal extrapolations.
5. Use `quick_pricer.py` to value a sample natural gas storage contract with custom parameters.

## Visualizations

The models generate automated charts to provide financial insights:

* **Credit Analysis**: Default rates segmented by FICO and income levels.
<img width="2683" height="1783" alt="credit_risk_analysis" src="https://github.com/user-attachments/assets/0e4a3593-52f5-406b-be9c-cc77617a7d02" />

* **Quantization Analysis**: Comparison of MSE and Log-Likelihood binning methods for FICO scores.
<img width="2683" height="2083" alt="fico_quantization_analysis" src="https://github.com/user-attachments/assets/55b5b27b-20ad-46bc-8c8d-9540e22cbe39" />

* **Commodity Trends**: Historical prices versus fitted seasonal models with one year extrapolations.
<img width="1934" height="2212" alt="nat_gas_analysis" src="https://github.com/user-attachments/assets/0f3e83dc-8752-4ee7-87db-eb0d6aaeab54" />

---

*Developed by Alex Lin as part of the Forage JPMorgan Chase Quantitative Research program.*
