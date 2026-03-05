"""
Credit Risk Modeling & Expected Loss Prediction
JPMorgan Chase - Retail Banking Risk Team

This module builds predictive models to estimate the probability of default (PD)
for personal loan borrowers and calculates expected loss.

Expected Loss Formula:
    EL = PD × EAD × LGD
    
    Where:
    - PD = Probability of Default (from model)
    - EAD = Exposure at Default (loan amount outstanding)
    - LGD = Loss Given Default (1 - Recovery Rate = 0.90)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')


# GLOBAL CONSTANTS

RECOVERY_RATE = 0.10
LOSS_GIVEN_DEFAULT = 1 - RECOVERY_RATE  # 90% loss on default
RANDOM_STATE = 42


# DATA LOADING & EXPLORATION

def load_and_explore_data(filepath = r'C:\Users\alexl\OneDrive\Desktop\桌面\ICG\Forage-JPMorgan\Loan_Data.csv'):
    """Load loan data and perform initial exploration."""
    df = pd.read_csv(filepath)
    
    print("═" * 80)
    print("LOAN PORTFOLIO DATA SUMMARY")
    print("═" * 80)
    print(f"\nDataset shape: {df.shape[0]:,} loans × {df.shape[1]} features")
    print(f"\nFeatures: {list(df.columns)}")
    
    print(f"\n{'─' * 80}")
    print("Default Rate Analysis")
    print(f"{'─' * 80}")
    default_rate = df['default'].mean()
    print(f"Overall default rate: {default_rate:.2%}")
    print(f"Non-defaulters: {(df['default']==0).sum():,} ({(1-default_rate):.2%})")
    print(f"Defaulters: {(df['default']==1).sum():,} ({default_rate:.2%})")
    
    print(f"\n{'─' * 80}")
    print("Feature Statistics")
    print(f"{'─' * 80}")
    print(df.describe())
    
    print(f"\n{'─' * 80}")
    print("Missing Values")
    print(f"{'─' * 80}")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✓ No missing values detected")
    else:
        print(missing[missing > 0])
    
    return df


# FEATURE ENGINEERING

def engineer_features(df):
    """Create additional features to improve model performance."""
    
    df = df.copy()
    
    # Debt-to-Income Ratio (key risk indicator)
    df['debt_to_income'] = df['total_debt_outstanding'] / (df['income'] + 1)
    
    # Loan-to-Income Ratio
    df['loan_to_income'] = df['loan_amt_outstanding'] / (df['income'] + 1)
    
    # Credit Utilization (debt beyond current loan)
    df['other_debt'] = df['total_debt_outstanding'] - df['loan_amt_outstanding']
    df['other_debt_ratio'] = df['other_debt'] / (df['income'] + 1)
    
    # FICO score categories (binning)
    df['fico_category'] = pd.cut(
        df['fico_score'], 
        bins=[0, 580, 670, 740, 850],
        labels=['Poor', 'Fair', 'Good', 'Excellent']
    )
    
    # Employment stability
    df['employment_stable'] = (df['years_employed'] >= 3).astype(int)
    
    # High credit lines flag
    df['many_credit_lines'] = (df['credit_lines_outstanding'] >= 3).astype(int)
    
    print(f"\n{'─' * 80}")
    print("Engineered Features")
    print(f"{'─' * 80}")
    print("✓ debt_to_income: Total debt / Income")
    print("✓ loan_to_income: Loan amount / Income")
    print("✓ other_debt_ratio: Non-loan debt / Income")
    print("✓ fico_category: Categorical FICO bins")
    print("✓ employment_stable: 1 if employed ≥3 years")
    print("✓ many_credit_lines: 1 if ≥3 credit lines")
    
    return df


# MODEL TRAINING

def prepare_data(df):
    """Prepare features and target for modeling."""
    
    # Select features for modeling
    feature_cols = [
        'credit_lines_outstanding',
        'loan_amt_outstanding',
        'total_debt_outstanding',
        'income',
        'years_employed',
        'fico_score',
        'debt_to_income',
        'loan_to_income',
        'other_debt_ratio',
        'employment_stable',
        'many_credit_lines'
    ]
    
    X = df[feature_cols].copy()
    y = df['default'].copy()
    
    # Handle any remaining issues
    X = X.fillna(X.median())
    
    return X, y, feature_cols


def train_models(X, y):
    """Train multiple models and compare performance."""
    
    print(f"\n{'═' * 80}")
    print("MODEL TRAINING & EVALUATION")
    print(f"{'═' * 80}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale features (important for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, 
            random_state=RANDOM_STATE,
            class_weight='balanced'
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'─' * 80}")
        print(f"Training: {name}")
        print(f"{'─' * 80}")
        
        # Use scaled data for logistic regression, raw for tree-based
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))
        
        # Store results
        results[name] = {
            'model': model,
            'auc': auc,
            'avg_precision': avg_precision,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba,
            'X_test': X_test_scaled if name == 'Logistic Regression' else X_test
        }
    
    return results, scaler, X_train, X_test, y_train, y_test


# MODEL SELECTION & FINAL MODEL

def select_best_model(results):
    """Select the best performing model based on AUC."""
    
    print(f"\n{'═' * 80}")
    print("MODEL COMPARISON")
    print(f"{'═' * 80}")
    print(f"\n{'Model':<25} {'AUC-ROC':<12} {'Avg Precision':<15}")
    print("─" * 80)
    
    best_auc = 0
    best_model_name = None
    
    for name, result in results.items():
        print(f"{name:<25} {result['auc']:<12.4f} {result['avg_precision']:<15.4f}")
        if result['auc'] > best_auc:
            best_auc = result['auc']
            best_model_name = name
    
    print(f"\n{'🏆 Best Model:':<25} {best_model_name}")
    print(f"{'   AUC-ROC:':<25} {best_auc:.4f}")
    
    return best_model_name, results[best_model_name]


# VISUALIZATION

def create_visualizations(df, results, feature_cols, best_model_name):
    """Create comprehensive visualizations."""
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Default rate by FICO score
    ax1 = plt.subplot(3, 3, 1)
    fico_bins = pd.cut(df['fico_score'], bins=10)
    default_by_fico = df.groupby(fico_bins)['default'].mean()
    default_by_fico.plot(kind='bar', ax=ax1, color='#e74c3c')
    ax1.set_title('Default Rate by FICO Score', fontweight='bold', fontsize=11)
    ax1.set_xlabel('FICO Score Range')
    ax1.set_ylabel('Default Rate')
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Default rate by debt-to-income
    ax2 = plt.subplot(3, 3, 2)
    dti_bins = pd.cut(df['debt_to_income'], bins=10)
    default_by_dti = df.groupby(dti_bins)['default'].mean()
    default_by_dti.plot(kind='bar', ax=ax2, color='#3498db')
    ax2.set_title('Default Rate by Debt-to-Income Ratio', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Debt-to-Income Ratio')
    ax2.set_ylabel('Default Rate')
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Income distribution by default status
    ax3 = plt.subplot(3, 3, 3)
    df[df['default']==0]['income'].hist(bins=30, alpha=0.6, label='No Default', 
                                         color='#2ecc71', ax=ax3)
    df[df['default']==1]['income'].hist(bins=30, alpha=0.6, label='Default', 
                                         color='#e74c3c', ax=ax3)
    ax3.set_title('Income Distribution by Default Status', fontweight='bold', fontsize=11)
    ax3.set_xlabel('Income ($)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4-7. ROC Curves for all models
    for idx, (name, result) in enumerate(results.items(), start=4):
        ax = plt.subplot(3, 3, idx)
        fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {result['auc']:.3f}")
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax.set_title(f'{name} - ROC Curve', fontweight='bold', fontsize=10)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
    
    # 8. Feature Importance (for best model if tree-based)
    ax8 = plt.subplot(3, 3, 8)
    best_model = results[best_model_name]['model']
    
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        ax8.barh(range(len(indices)), importances[indices], color='#9b59b6')
        ax8.set_yticks(range(len(indices)))
        ax8.set_yticklabels([feature_cols[i] for i in indices], fontsize=9)
        ax8.set_xlabel('Feature Importance')
        ax8.set_title(f'{best_model_name} - Top 10 Features', fontweight='bold', fontsize=10)
        ax8.grid(axis='x', alpha=0.3)
    else:
        # For logistic regression, use absolute coefficients
        if hasattr(best_model, 'coef_'):
            coef = np.abs(best_model.coef_[0])
            indices = np.argsort(coef)[::-1][:10]
            
            ax8.barh(range(len(indices)), coef[indices], color='#9b59b6')
            ax8.set_yticks(range(len(indices)))
            ax8.set_yticklabels([feature_cols[i] for i in indices], fontsize=9)
            ax8.set_xlabel('|Coefficient|')
            ax8.set_title(f'{best_model_name} - Top 10 Features', fontweight='bold', fontsize=10)
            ax8.grid(axis='x', alpha=0.3)
    
    # 9. Predicted Probability Distribution
    ax9 = plt.subplot(3, 3, 9)
    best_result = results[best_model_name]
    y_test = best_result['y_test']
    y_pred_proba = best_result['y_pred_proba']
    
    ax9.hist(y_pred_proba[y_test==0], bins=30, alpha=0.6, 
             label='No Default', color='#2ecc71')
    ax9.hist(y_pred_proba[y_test==1], bins=30, alpha=0.6, 
             label='Default', color='#e74c3c')
    ax9.set_title('Predicted Default Probability Distribution', fontweight='bold', fontsize=10)
    ax9.set_xlabel('Predicted Probability')
    ax9.set_ylabel('Frequency')
    ax9.legend()
    ax9.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('credit_risk_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualizations saved to credit_risk_analysis.png")
    
    return fig


# EXPECTED LOSS CALCULATION

class ExpectedLossCalculator:
    """
    Production-ready calculator for expected loss on loans.
    
    Uses the trained model to predict probability of default (PD) and 
    calculates expected loss using the formula:
        
        Expected Loss = PD × EAD × LGD
    
    where:
        - PD = Probability of Default (from model)
        - EAD = Exposure at Default (loan_amt_outstanding)
        - LGD = Loss Given Default (1 - recovery_rate)
    """
    
    def __init__(self, model, scaler, feature_cols, recovery_rate=RECOVERY_RATE):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.recovery_rate = recovery_rate
        self.lgd = 1 - recovery_rate
        
    def predict_default_probability(self, loan_data):
        """
        Predict probability of default for a loan.
        
        Parameters
        ----------
        loan_data : dict
            Dictionary containing loan features:
            - credit_lines_outstanding
            - loan_amt_outstanding
            - total_debt_outstanding
            - income
            - years_employed
            - fico_score
            
        Returns
        -------
        float
            Probability of default (0 to 1)
        """
        # Engineer features
        loan_df = pd.DataFrame([loan_data])
        
        loan_df['debt_to_income'] = loan_df['total_debt_outstanding'] / (loan_df['income'] + 1)
        loan_df['loan_to_income'] = loan_df['loan_amt_outstanding'] / (loan_df['income'] + 1)
        loan_df['other_debt'] = loan_df['total_debt_outstanding'] - loan_df['loan_amt_outstanding']
        loan_df['other_debt_ratio'] = loan_df['other_debt'] / (loan_df['income'] + 1)
        loan_df['employment_stable'] = (loan_df['years_employed'] >= 3).astype(int)
        loan_df['many_credit_lines'] = (loan_df['credit_lines_outstanding'] >= 3).astype(int)
        
        # Extract features in correct order
        X = loan_df[self.feature_cols].values
        
        # Scale if using logistic regression
        if hasattr(self.model, 'coef_'):
            X = self.scaler.transform(X)
        
        # Predict
        pd_prob = self.model.predict_proba(X)[0, 1]
        return float(pd_prob)
    
    def calculate_expected_loss(self, loan_data):
        """
        Calculate expected loss for a loan.
        
        Parameters
        ----------
        loan_data : dict
            Dictionary containing loan features (see predict_default_probability)
            
        Returns
        -------
        dict
            Dictionary containing:
            - probability_of_default: PD estimate
            - exposure_at_default: Loan amount
            - loss_given_default: LGD (1 - recovery rate)
            - expected_loss: Expected loss in dollars
            - recovery_rate: Assumed recovery rate
        """
        pd_prob = self.predict_default_probability(loan_data)
        ead = loan_data['loan_amt_outstanding']
        expected_loss = pd_prob * ead * self.lgd
        
        return {
            'probability_of_default': round(pd_prob, 6),
            'exposure_at_default': round(ead, 2),
            'loss_given_default': self.lgd,
            'expected_loss': round(expected_loss, 2),
            'recovery_rate': self.recovery_rate
        }


# MAIN EXECUTION

def main():
    """Main execution pipeline."""
    
    print("\n" + "█" * 80)
    print("CREDIT RISK MODELING - PROBABILITY OF DEFAULT & EXPECTED LOSS")
    print("JPMorgan Chase - Retail Banking Risk Team")
    print("█" * 80)
    
    # 1. Load and explore data
    df = load_and_explore_data()
    
    # 2. Feature engineering
    df = engineer_features(df)
    
    # 3. Prepare data
    X, y, feature_cols = prepare_data(df)
    
    # 4. Train models
    results, scaler, X_train, X_test, y_train, y_test = train_models(X, y)
    
    # 5. Select best model
    best_model_name, best_result = select_best_model(results)
    
    # 6. Create visualizations
    create_visualizations(df, results, feature_cols, best_model_name)
    
    # 7. Create expected loss calculator
    calculator = ExpectedLossCalculator(
        model=best_result['model'],
        scaler=scaler,
        feature_cols=feature_cols
    )
    
    # 8. Test with sample loans
    print(f"\n{'═' * 80}")
    print("EXPECTED LOSS CALCULATION - SAMPLE LOANS")
    print(f"{'═' * 80}")
    
    test_loans = [
        {
            'name': 'Low Risk Profile',
            'credit_lines_outstanding': 0,
            'loan_amt_outstanding': 5000,
            'total_debt_outstanding': 5000,
            'income': 80000,
            'years_employed': 5,
            'fico_score': 750
        },
        {
            'name': 'Medium Risk Profile',
            'credit_lines_outstanding': 2,
            'loan_amt_outstanding': 8000,
            'total_debt_outstanding': 15000,
            'income': 50000,
            'years_employed': 3,
            'fico_score': 650
        },
        {
            'name': 'High Risk Profile',
            'credit_lines_outstanding': 5,
            'loan_amt_outstanding': 10000,
            'total_debt_outstanding': 25000,
            'income': 35000,
            'years_employed': 1,
            'fico_score': 550
        }
    ]
    
    for test_loan in test_loans:
        name = test_loan.pop('name')
        result = calculator.calculate_expected_loss(test_loan)
        
        print(f"\n{'─' * 80}")
        print(f"{name}")
        print(f"{'─' * 80}")
        print(f"  Loan Amount:              ${test_loan['loan_amt_outstanding']:>12,.2f}")
        print(f"  Income:                   ${test_loan['income']:>12,.2f}")
        print(f"  FICO Score:               {test_loan['fico_score']:>14}")
        print(f"  Total Debt:               ${test_loan['total_debt_outstanding']:>12,.2f}")
        print(f"  Years Employed:           {test_loan['years_employed']:>14}")
        print(f"  Credit Lines:             {test_loan['credit_lines_outstanding']:>14}")
        print(f"\n  Probability of Default:   {result['probability_of_default']:>13.2%}")
        print(f"  Loss Given Default:       {result['loss_given_default']:>13.0%}")
        print(f"  EXPECTED LOSS:            ${result['expected_loss']:>12,.2f}")
    
    print(f"\n{'═' * 80}")
    print("PORTFOLIO EXPECTED LOSS CALCULATION")
    print(f"{'═' * 80}")
    
    # Calculate expected loss for entire test set
    portfolio_el = 0
    for idx in range(len(X_test)):
        loan_data = {
            'credit_lines_outstanding': X_test.iloc[idx]['credit_lines_outstanding'],
            'loan_amt_outstanding': X_test.iloc[idx]['loan_amt_outstanding'],
            'total_debt_outstanding': X_test.iloc[idx]['total_debt_outstanding'],
            'income': X_test.iloc[idx]['income'],
            'years_employed': X_test.iloc[idx]['years_employed'],
            'fico_score': X_test.iloc[idx]['fico_score']
        }
        result = calculator.calculate_expected_loss(loan_data)
        portfolio_el += result['expected_loss']
    
    total_exposure = X_test['loan_amt_outstanding'].sum()
    
    print(f"\nTest Portfolio Statistics:")
    print(f"  Number of loans:          {len(X_test):>14,}")
    print(f"  Total exposure (EAD):     ${total_exposure:>12,.2f}")
    print(f"  Total expected loss:      ${portfolio_el:>12,.2f}")
    print(f"  Expected loss rate:       {portfolio_el/total_exposure:>13.2%}")
    print(f"  Actual default rate:      {y_test.mean():>13.2%}")
    
    print(f"\n{'═' * 80}\n")
    
    return calculator, results, df


# SAVE CALCULATOR

if __name__ == "__main__":
    calculator, results, df = main()
    
    # Save calculator for production use
    import pickle
    with open(r'expected_loss_calculator.pkl', 'wb') as f:
        pickle.dump(calculator, f)
    print("✓ Expected loss calculator saved to expected_loss_calculator.pkl\n")
