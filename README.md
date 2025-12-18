# 🚦 Traffic Risk Prediction System

An end-to-end machine learning project that predicts **high-risk traffic accidents** using spatio-temporal, weather, and infrastructure features from the **US Accidents dataset**.

This project focuses on **real-world scalability**, **risk modeling**, and **interpretable ML**, and is designed as a strong **portfolio project** rather than a research paper.

---

## 🔍 Problem Statement

Traffic accidents are influenced by:
- Time of day & seasonality
- Weather conditions
- Road infrastructure (junctions, crossings, signals)
- Geographic location

The goal is to **predict whether an accident is high-risk** (severe) and estimate a **risk probability** that can be used for:
- Traffic safety analysis
- Risk-aware routing
- Decision support systems

---

## 📊 Dataset

**Source:** US Accidents Dataset  
**Size:** ~7.7 million accident records  

### Target Definition
- `high_risk = 1` → Severity ≥ 3  
- `high_risk = 0` → Severity ≤ 2  

---

## 🏗 Project Structure

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

---

## 🔧 Feature Engineering

Key features used:

- **Spatial**
  - Latitude, Longitude
- **Temporal**
  - Hour of day
  - Day of week
  - Month
- **Weather**
  - Visibility
  - Wind speed
  - Precipitation
- **Infrastructure**
  - Traffic signals
  - Junctions
  - Crossings

---

## 📈 Exploratory Data Analysis

Performed:
- Temporal risk trends (hour, weekday, month)
- Weather vs accident severity analysis
- Infrastructure risk comparison
- Spatial risk density visualization

Outputs saved to:
