# ============================================================
# Traffic Risk Prediction Project
# Data Preprocessing Pipeline (US Accidents Dataset)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path("E:/traffic_risk_project")
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"

csv_path = RAW_DIR / "US_accidents.csv"   # <-- your dataset
out_path = PROC_DIR / "traffic_clean.parquet"

PROC_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Columns to load (ACTUAL dataset columns)
# ============================================================

USECOLS = [
    "Severity",
    "Start_Time",
    "Start_Lat",
    "Start_Lng",
    "Weather_Condition",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Traffic_Signal",
    "Junction",
    "Crossing"
]

DTYPES = {
    "Severity": "int8",
    "Start_Lat": "float32",
    "Start_Lng": "float32",
    "Visibility(mi)": "float32",
    "Wind_Speed(mph)": "float32",
    "Precipitation(in)": "float32",
    "Traffic_Signal": "bool",
    "Junction": "bool",
    "Crossing": "bool"
}

CHUNK_SIZE = 200_000

# ============================================================
# Chunk preprocessing
# ============================================================

def preprocess_chunk(chunk):
    # Rename columns to ML-friendly names
    chunk = chunk.rename(columns={
        "Start_Lat": "latitude",
        "Start_Lng": "longitude",
        "Visibility(mi)": "visibility",
        "Wind_Speed(mph)": "wind_speed",
        "Precipitation(in)": "precipitation"
    })

    # Drop invalid coords
    chunk = chunk.dropna(subset=["latitude", "longitude"])
    chunk = chunk[
        chunk["latitude"].between(-90, 90) &
        chunk["longitude"].between(-180, 180)
    ]

    # Time features
    chunk["Start_Time"] = pd.to_datetime(chunk["Start_Time"], errors="coerce")
    chunk["hour"] = chunk["Start_Time"].dt.hour
    chunk["dayofweek"] = chunk["Start_Time"].dt.dayofweek
    chunk["month"] = chunk["Start_Time"].dt.month

    # Target (binary risk)
    # Severity 1–2 → Low risk, 3–4 → High risk
    chunk["high_risk"] = (chunk["Severity"] >= 3).astype("int8")

    # Weather cleanup
    chunk["Weather_Condition"] = (
        chunk["Weather_Condition"]
        .str.lower()
        .fillna("unknown")
    )

    return chunk


# ============================================================
# Load + process safely
# ============================================================

def load_and_preprocess():
    print("Loading CSV in chunks...")

    processed_chunks = []

    reader = pd.read_csv(
        csv_path,
        usecols=USECOLS,
        dtype=DTYPES,
        chunksize=CHUNK_SIZE,
        low_memory=False
    )

    for i, chunk in enumerate(reader, 1):
        print(f"Processing chunk {i}")
        chunk = preprocess_chunk(chunk)
        processed_chunks.append(chunk)

    return pd.concat(processed_chunks, ignore_index=True)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    df = load_and_preprocess()

    print("\nFinal dataset shape:", df.shape)
    print(df.head())
    print(df.info(memory_usage="deep"))

    df.to_parquet(out_path, index=False)
    print(f"\n✅ Cleaned data saved to:\n{out_path}")
