# Electricity Usage Pattern Predictor ⚡

A machine learning project that predicts household electricity consumption using time-based features and a Random Forest regression model.

---

## 🔹 Overview
This project uses the *Individual Household Electric Power Consumption* dataset to model usage patterns and forecast electricity demand. It focuses on feature engineering from date-time data and evaluates prediction accuracy using standard regression metrics.

---

## 🧠 Model
- Algorithm: `RandomForestRegressor`
- Input features:
  - Hour, Day of week, Weekend flag
  - Voltage, Intensity, Sub-metering values
  - Lag features (`t-1`, `t-2`)
- Target: `Global_active_power`
- Train/Test split: 80% / 20% (time-based)

---

## 📊 Outputs
- R² Score and MSE
- Visualizations:
  - Actual vs Predicted usage
  - Feature importance
  - Residual analysis
  - Daily usage trends
- Saved files:
  - `rf_model_colab.pkl` (trained model)
  - `test_outputs_colab.pkl` (predictions & test data)

---

## 📂 Files
```
review3.ipynb   # Notebook version
review3.py      # Script version
README.md
```

---

## ⚙️ Setup

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

Place dataset:

- Colab: `/content/household_power_consumption.txt`
- Local: Update `dataset_path` in `review3.py`

---

## ▶️ Run

### Google Colab
Upload the notebook and dataset, then run all cells.

### Local
```bash
python review3.py
```

---

## 🚀 Future Scope
- Add deep learning models (LSTM / GRU)
- Improve feature engineering
- Deploy as API (Flask / FastAPI)
- Build dashboard (Streamlit / Dash)

---

## 📌 License
MIT License


