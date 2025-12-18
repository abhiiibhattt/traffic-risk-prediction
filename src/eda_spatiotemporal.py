# ============================================================
# Traffic Risk Prediction Project
# Exploratory Data Analysis (Spatial + Temporal)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path("E:/traffic_risk_project")
PROC_DIR = BASE_DIR / "data" / "processed"
EDA_DIR  = BASE_DIR / "results" / "eda"

EDA_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = PROC_DIR / "traffic_clean.parquet"

# ============================================================
# Load data
# ============================================================

print("Loading processed dataset...")
df = pd.read_parquet(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())

# ============================================================
# Basic sanity checks
# ============================================================

print("\nHigh-risk ratio:")
print(df["high_risk"].value_counts(normalize=True))

# ============================================================
# 1. TEMPORAL ANALYSIS
# ============================================================

# ---- Accidents by hour ----
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="hour", hue="high_risk", palette="Set2")
plt.title("Accidents by Hour (Low vs High Risk)")
plt.xlabel("Hour of Day")
plt.ylabel("Count")
plt.legend(title="High Risk")
plt.tight_layout()
plt.savefig(EDA_DIR / "accidents_by_hour.png", dpi=300)
plt.close()

# ---- Accidents by day of week ----
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="dayofweek", hue="high_risk", palette="Set1")
plt.title("Accidents by Day of Week")
plt.xlabel("Day (0=Mon)")
plt.ylabel("Count")
plt.legend(title="High Risk")
plt.tight_layout()
plt.savefig(EDA_DIR / "accidents_by_dayofweek.png", dpi=300)
plt.close()

# ---- Accidents by month ----
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="month", hue="high_risk", palette="Set3")
plt.title("Accidents by Month")
plt.xlabel("Month")
plt.ylabel("Count")
plt.legend(title="High Risk")
plt.tight_layout()
plt.savefig(EDA_DIR / "accidents_by_month.png", dpi=300)
plt.close()

# ============================================================
# 2. WEATHER IMPACT ANALYSIS
# ============================================================

top_weather = (
    df["Weather_Condition"]
    .value_counts()
    .head(10)
    .index
)

weather_df = df[df["Weather_Condition"].isin(top_weather)]

plt.figure(figsize=(10, 5))
sns.barplot(
    data=weather_df,
    x="Weather_Condition",
    y="high_risk",
    estimator=np.mean,
    errorbar=None
)
plt.xticks(rotation=45, ha="right")
plt.title("Probability of High-Risk Accidents by Weather Condition")
plt.ylabel("High Risk Probability")
plt.xlabel("Weather Condition")
plt.tight_layout()
plt.savefig(EDA_DIR / "weather_vs_risk.png", dpi=300)
plt.close()

# ============================================================
# 3. SPATIAL ANALYSIS (Heatmap-style scatter)
# ============================================================

# Downsample for plotting performance
sample_df = df.sample(n=min(50_000, len(df)), random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(
    sample_df["longitude"],
    sample_df["latitude"],
    c=sample_df["high_risk"],
    cmap="coolwarm",
    s=1,
    alpha=0.4
)
plt.colorbar(label="High Risk")
plt.title("Spatial Distribution of Accident Risk")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig(EDA_DIR / "spatial_risk_scatter.png", dpi=300)
plt.close()

# ============================================================
# 4. INFRASTRUCTURE FEATURES
# ============================================================

infra_cols = ["Traffic_Signal", "Junction", "Crossing"]

infra_risk = (
    df.groupby(infra_cols)["high_risk"]
    .mean()
    .reset_index()
)

infra_risk.to_csv(EDA_DIR / "infrastructure_risk_table.csv", index=False)

print("\nSaved infrastructure risk table")

# ============================================================
# 5. CORRELATION SNAPSHOT
# ============================================================

num_cols = [
    "visibility",
    "wind_speed",
    "precipitation",
    "hour",
    "dayofweek",
    "month",
    "high_risk"
]

corr = df[num_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(EDA_DIR / "correlation_matrix.png", dpi=300)
plt.close()

# ============================================================
# Done
# ============================================================

print("\n✅ EDA completed.")
print(f"Plots and tables saved to:\n{EDA_DIR}")

# ============================================================
# Advanced Spatial EDA
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ADV_DIR = EDA_DIR / "spatial_advanced"
ADV_DIR.mkdir(parents=True, exist_ok=True)

# Reduce plotting size to avoid memory spikes
SAMPLE_SIZE = 300_000
df_spatial = df.sample(
    n=min(SAMPLE_SIZE, len(df)),
    random_state=42
)

print(f"Using {len(df_spatial):,} points for spatial EDA")

# ============================================================
# 1️⃣ Hexbin Risk Density Map
# ============================================================

plt.figure(figsize=(10, 8))
plt.hexbin(
    df_spatial["longitude"],
    df_spatial["latitude"],
    C=df_spatial["high_risk"],
    gridsize=120,
    reduce_C_function=np.mean,
    cmap="inferno"
)
plt.colorbar(label="High-Risk Probability")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Hexbin Map: Spatial Risk Hotspots")

plt.savefig(ADV_DIR / "hexbin_risk_hotspots.png", dpi=300)
plt.close()

# ============================================================
# 2️⃣ KDE Heatmap (High-Risk Only)
# ============================================================

high_risk_df = df_spatial[df_spatial["high_risk"] == 1]

plt.figure(figsize=(10, 8))
sns.kdeplot(
    x=high_risk_df["longitude"],
    y=high_risk_df["latitude"],
    fill=True,
    cmap="Reds",
    bw_adjust=0.7,
    levels=50,
    thresh=0.05
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("KDE Heatmap: High-Risk Accident Concentration")

plt.savefig(ADV_DIR / "kde_high_risk_heatmap.png", dpi=300)
plt.close()

# ============================================================
# 3️⃣ High vs Low Risk Spatial Contrast
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

axes[0].scatter(
    df_spatial.loc[df_spatial["high_risk"] == 0, "longitude"],
    df_spatial.loc[df_spatial["high_risk"] == 0, "latitude"],
    s=1, alpha=0.3
)
axes[0].set_title("Low-Risk Accidents")

axes[1].scatter(
    df_spatial.loc[df_spatial["high_risk"] == 1, "longitude"],
    df_spatial.loc[df_spatial["high_risk"] == 1, "latitude"],
    s=1, alpha=0.3, color="red"
)
axes[1].set_title("High-Risk Accidents")

for ax in axes:
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

plt.suptitle("Spatial Distribution: Low vs High Risk", fontsize=14)
plt.tight_layout()
plt.savefig(ADV_DIR / "spatial_risk_comparison.png", dpi=300)
plt.close()

# ============================================================
# 4️⃣ City-Level Risk Aggregation
# ============================================================

if "City" in df.columns:
    city_risk = (
        df.groupby("City", observed=True)
        .agg(
            total_accidents=("high_risk", "count"),
            high_risk_rate=("high_risk", "mean")
        )
        .query("total_accidents >= 1000")
        .sort_values("high_risk_rate", ascending=False)
    )

    city_risk.to_csv(ADV_DIR / "city_level_risk.csv")

    # Plot top 15 risky cities
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=city_risk.head(15)["high_risk_rate"],
        y=city_risk.head(15).index,
        errorbar=None
    )
    plt.xlabel("High-Risk Proportion")
    plt.title("Top 15 Cities by Accident Risk")

    plt.savefig(ADV_DIR / "top_risky_cities.png", dpi=300)
    plt.close()

print("\n✅ Advanced spatial EDA completed.")
print(f"Outputs saved to:\n{ADV_DIR}")

# ============================================================
# ADVANCED SPATIAL EDA (GIS-Lite)
# ============================================================

import matplotlib.pyplot as plt

print("\n===== RUNNING SPATIAL EDA =====")

EDA_DIR = BASE_DIR / "results" / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1️⃣ High-risk accident HEXBIN heatmap
# ------------------------------------------------------------

print("Generating high-risk hexbin map...")

high_risk = df[df["high_risk"] == 1]

plt.figure(figsize=(10, 8))
plt.hexbin(
    high_risk["longitude"],
    high_risk["latitude"],
    gridsize=200,
    cmap="inferno",
    bins="log",
    mincnt=5
)
plt.colorbar(label="Log accident density")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Spatial Density of High-Risk Traffic Accidents")

hexbin_path = EDA_DIR / "spatial_hexbin_high_risk.png"
plt.savefig(hexbin_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {hexbin_path}")

# ------------------------------------------------------------
# 2️⃣ Spatial grid aggregation (risk intensity)
# ------------------------------------------------------------

print("Computing spatial risk grid...")

GRID_SIZE = 0.05  # ~5km grid (safe + interpretable)

df["lat_bin"] = (df["latitude"] // GRID_SIZE) * GRID_SIZE
df["lon_bin"] = (df["longitude"] // GRID_SIZE) * GRID_SIZE

grid_risk = (
    df.groupby(["lat_bin", "lon_bin"])["high_risk"]
      .agg(
          total_accidents="count",
          high_risk_rate="mean"
      )
      .reset_index()
)

# Save table (VERY useful for portfolio)
grid_table_path = EDA_DIR / "spatial_grid_risk_table.csv"
grid_risk.to_csv(grid_table_path, index=False)

print(f"Saved spatial grid table: {grid_table_path}")

# ------------------------------------------------------------
# 3️⃣ Grid risk visualization
# ------------------------------------------------------------

plt.figure(figsize=(10, 8))
sc = plt.scatter(
    grid_risk["lon_bin"],
    grid_risk["lat_bin"],
    c=grid_risk["high_risk_rate"],
    s=grid_risk["total_accidents"] * 0.02,
    cmap="viridis",
    alpha=0.75
)
plt.colorbar(sc, label="High-risk accident proportion")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Spatial Risk Intensity (Grid Aggregation)")

grid_plot_path = EDA_DIR / "spatial_risk_grid.png"
plt.savefig(grid_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {grid_plot_path}")

print("\n✅ Spatial EDA completed successfully.")
