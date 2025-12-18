# ============================================================
# Traffic Risk Prediction Project
# Inference Pipeline
# ============================================================

import joblib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path("E:/traffic_risk_project")

MODEL_DIR = BASE_DIR / "results" / "models"
DATA_DIR  = BASE_DIR / "data" / "processed"
FIG_DIR   = BASE_DIR / "results" / "figures"

MODEL_PATH = MODEL_DIR / "best_model.joblib"

FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Feature configuration (MUST match training)
# ============================================================

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

# ============================================================
# Load trained model
# ============================================================

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"❌ Model not found at {MODEL_PATH}\n"
            "Run train_models.py first."
        )
    return joblib.load(MODEL_PATH)

# ============================================================
# Core inference function
# ============================================================

def predict_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict traffic risk for given input data
    """

    model = load_model()

    # Check required features
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURE_COLS].copy()

    probs = model.predict_proba(X)[:, 1]
    labels = (probs >= 0.5).astype("int8")

    out = df.copy()
    out["risk_probability"] = probs
    out["risk_label"] = labels

    return out

# ============================================================
# Batch inference
# ============================================================

def run_batch_inference(input_csv: Path, output_csv: Path):
    print(f"📥 Loading data: {input_csv}")
    df = pd.read_csv(input_csv)

    preds = predict_risk(df)

    preds.to_csv(output_csv, index=False)
    print(f"✅ Predictions saved to: {output_csv}")

    return preds

# ============================================================
# Risk heatmap
# ============================================================

def plot_risk_heatmap(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        df["longitude"],
        df["latitude"],
        c=df["risk_probability"],
        cmap="hot",
        s=15,
        alpha=0.7
    )
    plt.colorbar(sc, label="Risk Probability")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Traffic Risk Heatmap (Batch Inference)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"🗺️ Risk heatmap saved to: {out_path}")

# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":

    print("🚀 Running inference example...\n")

    # ---- Single example ----
    sample = pd.DataFrame([{
        "latitude": 37.7749,
        "longitude": -122.4194,
        "visibility": 5.0,
        "wind_speed": 12.0,
        "precipitation": 0.1,
        "Traffic_Signal": True,
        "Junction": False,
        "Crossing": True,
        "hour": 18,
        "dayofweek": 4,
        "month": 10
    }])

    single_out = predict_risk(sample)
    print(single_out[["risk_probability", "risk_label"]])

    # ---- Batch inference ----
    batch_out = run_batch_inference(
        DATA_DIR / "new_accidents.csv",
        DATA_DIR / "predicted_risk.csv"
    )

    # ---- Batch summary ----
    print("\n📊 Batch summary:")
    print("Mean risk:", batch_out["risk_probability"].mean())
    print("High-risk %:", (batch_out["risk_label"] == 1).mean() * 100)

    # ---- Heatmap ----
    plot_risk_heatmap(
        batch_out,
        FIG_DIR / "risk_heatmap.png"
    )
