"""
data_cleaning.py
================
AI-Based Loan Risk Prediction and Financial Analytics System
------------------------------------------------------------
Step 1: Load the LoanApprovalPrediction.csv dataset, perform a thorough
        data quality audit, and produce a cleaned version ready for EDA
        and model training.

Checks performed:
  - Shape and column overview
  - Data types
  - Missing values (count + percentage)
  - Duplicate records
  - Unique / unexpected values in categorical columns
  - Outlier detection in numerical columns (IQR method)
  - Incorrect / inconsistent values
  - Target class distribution

Cleaning steps:
  - Drop the Loan_ID identifier column (not a feature)
  - Impute missing categoricals with mode
  - Impute missing numerics with median
  - Standardise the Dependents '3+' ordinal to integer 3
  - Encode the binary target Loan_Status → 1 (Y) / 0 (N)
  - Report every action taken
"""

import os
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
# Resolve paths relative to this file so the script works regardless of
# the working directory it is launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "LoanApprovalPrediction.csv")

df = pd.read_csv(CSV_PATH)

print("=" * 65)
print("  LOAN APPROVAL DATASET — DATA CLEANING REPORT")
print("=" * 65)

# ─────────────────────────────────────────────
# 2. BASIC SHAPE AND COLUMN OVERVIEW
# ─────────────────────────────────────────────
print("\n[1] SHAPE")
print(f"    Rows   : {df.shape[0]}")
print(f"    Columns: {df.shape[1]}")

print("\n[2] COLUMNS AND DTYPES")
print(f"    {'Column':<22} {'Dtype':<12} {'Non-Null':>8}")
print(f"    {'-'*44}")
for col in df.columns:
    print(f"    {col:<22} {str(df[col].dtype):<12} {df[col].count():>8}")

# ─────────────────────────────────────────────
# 3. MISSING VALUES
# ─────────────────────────────────────────────
print("\n[3] MISSING VALUES")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    "Missing Count": missing,
    "Missing %": missing_pct
})
missing_df = missing_df[missing_df["Missing Count"] > 0]

if missing_df.empty:
    print("    No missing values found.")
else:
    print(f"    {'Column':<22} {'Missing Count':>14} {'Missing %':>10}")
    print(f"    {'-'*48}")
    for col, row in missing_df.iterrows():
        print(f"    {col:<22} {int(row['Missing Count']):>14} {row['Missing %']:>9}%")

# ─────────────────────────────────────────────
# 4. DUPLICATE ROWS
# ─────────────────────────────────────────────
print("\n[4] DUPLICATE ROWS")
dup_count = df.duplicated().sum()
print(f"    Fully duplicate rows : {dup_count}")
dup_loan_id = df["Loan_ID"].duplicated().sum()
print(f"    Duplicate Loan_IDs   : {dup_loan_id}")

# ─────────────────────────────────────────────
# 5. CATEGORICAL COLUMN AUDIT
# ─────────────────────────────────────────────
cat_cols = ["Gender", "Married", "Dependents", "Education",
            "Self_Employed", "Credit_History", "Property_Area", "Loan_Status"]

print("\n[5] CATEGORICAL UNIQUE VALUES")
expected = {
    "Gender"        : {"Male", "Female"},
    "Married"       : {"Yes", "No"},
    "Dependents"    : {"0", "1", "2", "3+"},
    "Education"     : {"Graduate", "Not Graduate"},
    "Self_Employed" : {"Yes", "No"},
    "Credit_History": {"0", "1", "0.0", "1.0"},   # may be read as float
    "Property_Area" : {"Urban", "Rural", "Semiurban"},
    "Loan_Status"   : {"Y", "N"},
}

for col in cat_cols:
    actual = set(df[col].dropna().astype(str).unique())
    unexpected = actual - expected.get(col, actual)
    status = "OK" if not unexpected else f"UNEXPECTED: {unexpected}"
    print(f"    {col:<22} values={sorted(actual)}")
    if unexpected:
        print(f"    {'':22} *** {status}")

# ─────────────────────────────────────────────
# 6. NUMERICAL COLUMN STATISTICS + OUTLIERS
# ─────────────────────────────────────────────
num_cols = ["ApplicantIncome", "CoapplicantIncome",
            "LoanAmount", "Loan_Amount_Term", "Credit_History"]

print("\n[6] NUMERICAL STATISTICS & OUTLIER CHECK (IQR method)")
print(f"    {'Column':<22} {'Min':>9} {'Max':>9} {'Mean':>9} {'Median':>9} {'Outliers':>9}")
print(f"    {'-'*68}")

for col in num_cols:
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = ((series < lower) | (series > upper)).sum()
    print(f"    {col:<22} {series.min():>9.1f} {series.max():>9.1f} "
          f"{series.mean():>9.1f} {series.median():>9.1f} {outliers:>9}")

# ─────────────────────────────────────────────
# 7. SPECIFIC INCORRECT VALUE CHECKS
# ─────────────────────────────────────────────
print("\n[7] SPECIFIC INCORRECT VALUE CHECKS")

# LoanAmount should be positive
df["LoanAmount"] = pd.to_numeric(df["LoanAmount"], errors="coerce")
neg_loan = (df["LoanAmount"] <= 0).sum()
print(f"    LoanAmount <= 0      : {neg_loan}")

# ApplicantIncome should be positive
df["ApplicantIncome"] = pd.to_numeric(df["ApplicantIncome"], errors="coerce")
neg_income = (df["ApplicantIncome"] <= 0).sum()
print(f"    ApplicantIncome <= 0 : {neg_income}")

# Loan_Amount_Term — standard terms (months)
df["Loan_Amount_Term"] = pd.to_numeric(df["Loan_Amount_Term"], errors="coerce")
valid_terms = {12, 36, 60, 84, 120, 180, 240, 300, 360, 480}
actual_terms = set(df["Loan_Amount_Term"].dropna().unique())
non_std = actual_terms - valid_terms
print(f"    Non-standard terms   : {sorted(non_std) if non_std else 'None'}")
print(f"    Term value counts    : {dict(df['Loan_Amount_Term'].value_counts().sort_index())}")

# Credit_History — should only be 0 or 1
df["Credit_History"] = pd.to_numeric(df["Credit_History"], errors="coerce")
invalid_ch = df["Credit_History"].dropna()
invalid_ch = invalid_ch[~invalid_ch.isin([0, 1])].count()
print(f"    Invalid Credit_History values: {invalid_ch}")

# ─────────────────────────────────────────────
# 8. TARGET CLASS DISTRIBUTION
# ─────────────────────────────────────────────
print("\n[8] TARGET DISTRIBUTION — Loan_Status")
dist = df["Loan_Status"].value_counts()
for label, count in dist.items():
    bar = "█" * int(count / 5)
    print(f"    {label}  : {count:>4} ({count/len(df)*100:.1f}%)  {bar}")

# ─────────────────────────────────────────────
# 9. CLEANING STEPS
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  APPLYING CLEANING STEPS")
print("=" * 65)

df_clean = df.copy()

# Step 9a: Drop Loan_ID (identifier, not a feature)
df_clean.drop(columns=["Loan_ID"], inplace=True)
print("\n[9a] Dropped 'Loan_ID' — identifier column, not a predictive feature.")

# Step 9b: Standardise Dependents '3+' → 3 (ordinal integer)
df_clean["Dependents"] = df_clean["Dependents"].replace("3+", "3")
print("[9b] Replaced Dependents '3+' with '3' for ordinal consistency.")

# Step 9c: Impute missing categorical columns with mode
cat_fill_cols = ["Gender", "Married", "Dependents", "Self_Employed"]
for col in cat_fill_cols:
    mode_val = df_clean[col].mode()[0]
    n_filled = df_clean[col].isnull().sum()
    df_clean[col].fillna(mode_val, inplace=True)
    print(f"[9c] '{col}': filled {n_filled} missing values with mode='{mode_val}'.")

# Step 9d: Impute missing numerical columns with median
num_fill_cols = ["LoanAmount", "Loan_Amount_Term", "Credit_History"]
for col in num_fill_cols:
    median_val = df_clean[col].median()
    n_filled = df_clean[col].isnull().sum()
    df_clean[col].fillna(median_val, inplace=True)
    print(f"[9d] '{col}': filled {n_filled} missing values with median={median_val}.")

# Step 9e: Encode target Loan_Status → 1 (Y) / 0 (N)
df_clean["Loan_Status"] = df_clean["Loan_Status"].map({"Y": 1, "N": 0})
print("[9e] Encoded Loan_Status: Y→1, N→0.")

# Step 9f: Convert Dependents to integer
df_clean["Dependents"] = pd.to_numeric(df_clean["Dependents"], errors="coerce").astype(int)
print("[9f] Converted Dependents to integer dtype.")

# ─────────────────────────────────────────────
# 10. POST-CLEANING VALIDATION
# ─────────────────────────────────────────────
print("\n[10] POST-CLEANING VALIDATION")
remaining_missing = df_clean.isnull().sum().sum()
print(f"     Total remaining missing values : {remaining_missing}")
print(f"     Final shape                    : {df_clean.shape}")
print(f"     Dtypes after cleaning:")
for col in df_clean.columns:
    print(f"       {col:<22} {str(df_clean[col].dtype)}")

# ─────────────────────────────────────────────
# 11. SAVE CLEANED DATA
# ─────────────────────────────────────────────
df_clean.to_csv(os.path.join(_HERE, "loan_cleaned.csv"), index=False)
print("\n[11] Cleaned dataset saved to 'loan_cleaned.csv'.")

print("\n" + "=" * 65)
print("  DATA CLEANING COMPLETE")
print("=" * 65)
