# 🚦 Traffic Risk Prediction System

An end-to-end machine learning system for predicting **high-risk traffic accidents**
using spatio-temporal, weather, and road infrastructure features from the
**US Accidents dataset**.

The system supports large-scale data processing, exploratory analysis,
model training, and real-time or batch inference.

---

## 🔍 Problem Overview

Traffic accident severity is influenced by multiple factors such as:
- Time of day and seasonality
- Weather conditions
- Road infrastructure (junctions, crossings, traffic signals)
- Geographic location

The objective of this project is to:
- Predict whether an accident is **high risk** (severe)
- Estimate a **risk probability** that can be used for analysis or decision support

---

## 📊 Dataset

**Source:** US Accidents Dataset  
**Scale:** ~7.7 million accident records across the United States

### Target Definition
- `high_risk = 1` → Severity ≥ 3  
- `high_risk = 0` → Severity ≤ 2  

---

## 🔗 Dataset Access

This repository does **not** include the raw dataset due to its large size.

You can download the **US Accidents Dataset** from the official source:

🔗 https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

After downloading:

1. Extract the CSV file
2. Rename it to: US_accidents.csv

3. Place it in: data/raw/US_accidents.csv

The preprocessing pipeline will automatically handle the dataset in chunks.

## 🏗 Project Structure
```text
📦 traffic_risk_project
│
├── 📂 data
│   ├── 📂 raw
│   │   └── 🧾 US_accidents.csv
│   │
│   └── 📂 processed
│       ├── 🧾 traffic_clean.parquet
│       ├── 🧾 new_accidents.csv
│       └── 🧾 predicted_risk.csv
│
├── 📂 src
│   ├── 🧠 data_preprocessing.py      # Chunked preprocessing & feature creation
│   ├── 📊 eda_spatiotemporal.py       # Temporal & spatial EDA
│   ├── 📈 risk_analysis.py            # Statistical risk factor analysis
│   ├── 🤖 train_models.py             # Model training & evaluation
│   └── 🚦 inference_pipeline.py       # Real-time & batch inference
│
├── 📂 results
│   ├── 📂 figures                     # ROC curves, EDA plots, heatmaps
│   ├── 📂 metrics                     # AUC, accuracy, summary tables
│   └── 📂 models
│       └── 🧠 best_model.joblib        # Final trained model
│
├── 📘 README.md
└── 📜 requirements.txt
```
---
> ⚠️ **Note:** The raw dataset is excluded from the repository due to size.
> See the Dataset section above for download instructions.

---

## 🔧 Feature Engineering

**Spatial**
- Latitude
- Longitude

**Temporal**
- Hour of day
- Day of week
- Month

**Weather**
- Visibility
- Wind speed
- Precipitation

**Infrastructure**
- Traffic signals
- Junctions
- Crossings

---

## 📈 Exploratory Data Analysis

Performed analyses include:
- Temporal risk trends (hour, weekday, month)
- Weather vs accident severity analysis
- Infrastructure risk comparison
- Advanced spatial risk visualization (scatter, grid, hexbin, KDE)

EDA outputs are saved to:

results/eda/

results/eda/spatial_advanced/


---

## 🤖 Modeling Approach

Models trained and evaluated:

- **Logistic Regression**
  - Interpretable baseline
- **HistGradientBoostingClassifier**
  - Handles missing values natively
  - Scales efficiently to millions of rows
  - Captures non-linear patterns

The best-performing model is automatically saved as:

results/models/best_model.joblib


---

## 📊 Model Performance

| Model | ROC-AUC | Accuracy |
|------|--------:|---------:|
| Logistic Regression | ~0.62 | ~0.54 |
| HistGradientBoosting | ~0.81 | ~0.82 |

Evaluation artifacts:
- ROC curves
- Confusion matrices
- Metrics tables

Saved under:

results/figures/

results/metrics/


---

## 🚦 Inference Pipeline

Supports:

### ✅ Single-record inference
```python
from inference_pipeline import predict_risk
