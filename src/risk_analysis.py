# ============================================================
# Traffic Risk Prediction Project
# Risk Analysis & Statistical Inference
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path("E:/traffic_risk_project")
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

EDA_DIR = RESULTS_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "traffic_clean.parquet"

# ============================================================
# Load data
# ============================================================

print("Loading processed dataset...")
df = pd.read_parquet(DATA_PATH)

print("Dataset loaded:", df.shape)

# Keep only required columns (memory safety)
REQ_COLS = [
    "high_risk",
    "visibility",
    "wind_speed",
    "precipitation",
    "Traffic_Signal",
    "Junction",
    "Crossing",
    "hour",
    "dayofweek",
    "month"
]

df = df[REQ_COLS].dropna()

# ============================================================
# 1️⃣ Infrastructure Risk (Odds Ratios)
# ============================================================

import statsmodels.api as sm

infra_cols = ["Traffic_Signal", "Junction", "Crossing"]

X = df[infra_cols].astype(int)
X = sm.add_constant(X)
y = df["high_risk"]

logit_model = sm.Logit(y, X).fit(disp=False)

odds_ratios = np.exp(logit_model.params)
p_values = logit_model.pvalues

infra_risk = pd.DataFrame({
    "feature": odds_ratios.index,
    "odds_ratio": odds_ratios.values,
    "p_value": p_values.values
}).query("feature != 'const'")

infra_risk = infra_risk.sort_values("odds_ratio", ascending=False)

print("\nInfrastructure Odds Ratios:")
print(infra_risk)

infra_risk.to_csv(
    RESULTS_DIR / "infra_odds_ratios.csv",
    index=False
)

# ============================================================
# 2️⃣ Visibility vs Risk (FIXED)
# ============================================================

# Bin visibility
df["visibility_bin"] = pd.cut(
    df["visibility"],
    bins=[0, 1, 2, 5, 10, 20, 50],
    include_lowest=True
)

# Aggregate safely
vis_risk = (
    df.groupby("visibility_bin", observed=True)["high_risk"]
    .mean()
    .reset_index(name="risk_rate")
)

# Convert interval bins → numeric midpoints
vis_risk["visibility_mid"] = vis_risk["visibility_bin"].apply(
    lambda x: x.mid
)

# Plot
plt.figure(figsize=(8, 5))
sns.lineplot(
    data=vis_risk,
    x="visibility_mid",
    y="risk_rate",
    marker="o"
)

plt.xlabel("Visibility (miles)")
plt.ylabel("High-Risk Accident Probability")
plt.title("Accident Risk vs Visibility")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(EDA_DIR / "risk_vs_visibility.png", dpi=300)
plt.close()

# ============================================================
# 3️⃣ Temporal Risk Patterns
# ============================================================

# Hourly risk
hourly_risk = (
    df.groupby("hour", observed=True)["high_risk"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(9, 5))
sns.lineplot(data=hourly_risk, x="hour", y="high_risk")
plt.xlabel("Hour of Day")
plt.ylabel("High-Risk Probability")
plt.title("Hourly Accident Risk Pattern")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(EDA_DIR / "risk_by_hour.png", dpi=300)
plt.close()

# Day of week risk
dow_risk = (
    df.groupby("dayofweek", observed=True)["high_risk"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(8, 5))
sns.barplot(data=dow_risk, x="dayofweek", y="high_risk")
plt.xlabel("Day of Week (0=Mon)")
plt.ylabel("High-Risk Probability")
plt.title("Accident Risk by Day of Week")

plt.tight_layout()
plt.savefig(EDA_DIR / "risk_by_dayofweek.png", dpi=300)
plt.close()

# ============================================================
# 4️⃣ Weather Impact Summary
# ============================================================

weather_summary = df[[
    "visibility",
    "wind_speed",
    "precipitation",
    "high_risk"
]].groupby("high_risk").mean()

weather_summary.to_csv(
    RESULTS_DIR / "weather_risk_summary.csv"
)

# ============================================================
# Completion
# ============================================================

print("\n✅ Risk analysis completed successfully.")
print("Outputs saved to:")
print(f"- {EDA_DIR}")
print(f"- {RESULTS_DIR}")
