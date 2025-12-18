# ============================================================
# Traffic Risk Prediction Project
# Model Training Pipeline
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV

import matplotlib.pyplot as plt

# ============================================================
# Paths (DO NOT CHANGE)
# ============================================================

BASE_DIR = Path("E:/traffic_risk_project")

DATA_PATH = BASE_DIR / "data" / "processed" / "traffic_clean.parquet"

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
FIG_DIR     = RESULTS_DIR / "figures"

for d in [MODELS_DIR, METRICS_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load dataset
# ============================================================

df = pd.read_parquet(DATA_PATH)
print("Dataset loaded:", df.shape)

FEATURE_COLS = [
    "latitude",
    "longitude",
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

TARGET = "high_risk"

X = df[FEATURE_COLS]
y = df[TARGET]

# ============================================================
# Train / Test split (stratified)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape)
print("Test size :", X_test.shape)

# ============================================================
# Models (memory-safe + production-ready)
# ============================================================

models = {
    "LogisticRegression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ]),

    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.1,
        max_iter=200,
        random_state=42
    )
}

# ============================================================
# Train, evaluate, save
# ============================================================

results = []
best_auc = -np.inf
best_model_name = None
best_model = None

for name, base_model in models.items():
    print(f"\nTraining {name}")

    # Calibrate probabilities
    model = CalibratedClassifierCV(
        base_model,
        method="sigmoid",
        cv=3
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)

    print(f"{name} AUC: {auc:.4f}")
    print(f"{name} ACC: {acc:.4f}")

    # Save model
    model_path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, model_path)

    results.append({
        "model": name,
        "auc": auc,
        "accuracy": acc,
        "model_path": str(model_path)
    })

    # Track best model
    if auc > best_auc:
        best_auc = auc
        best_model_name = name
        best_model = model

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"roc_{name}.png", dpi=300)
    plt.close()

# ============================================================
# Save BEST MODEL (CRITICAL FIX)
# ============================================================

best_model_path = MODELS_DIR / "best_model.joblib"
joblib.dump(best_model, best_model_path)

print("\n🏆 Best model selected:", best_model_name)
print("Saved best model to:", best_model_path)

# ============================================================
# Save metrics
# ============================================================

metrics_df = pd.DataFrame(results).sort_values("auc", ascending=False)
metrics_df.to_csv(METRICS_DIR / "model_performance.csv", index=False)

print("\nSaved metrics to:", METRICS_DIR)
print("Saved figures to:", FIG_DIR)
print("Saved models to:", MODELS_DIR)

print("\n✅ MODEL TRAINING COMPLETED SUCCESSFULLY")
