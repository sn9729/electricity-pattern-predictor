# Electricity Usage Pattern Predictor ⚡

A professional machine learning project that predicts household electricity consumption using time-based features and a Random Forest regression model. This project focuses on a modular training pipeline and comprehensive model evaluation.

## 🔹 Overview
This project uses the **Individual Household Electric Power Consumption** dataset (UCI Machine Learning Repository) to model usage patterns and forecast electricity demand. It demonstrates:
- **Modular Code Structure**: Clean separation of data loading, modeling, and training logic.
- **Feature Engineering**: Creation of lag features and datetime-based predictors.
- **Model Evaluation**: Detailed performance metrics and visualization of predictions.

## 📂 Project Structure
```
electricity-pattern-predictor/
├── data/                   # Dataset storage (auto-downloaded)
├── models/                 # Saved trained models and plots
├── notebooks/              # Jupyter notebooks for experimentation
├── src/                    # Source code
│   ├── data_loader.py      # Data ingestion and preprocessing
│   ├── model.py            # Random Forest model definition
│   ├── train.py            # Main training pipeline
│   └── utils.py            # Evaluation and plotting utilities
├── requirements.txt        # Project dependencies
└── README.md               # Documentation
```

## ⚙️ Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sn9729/electricity-pattern-predictor.git
   cd electricity-pattern-predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage

### Train and Evaluate
Run the training pipeline to download data, train the model, save artifacts, and generate performance plots:
```bash
python -m src.train
```
*Note: The dataset (~20MB) will be downloaded automatically to the `data/` folder on the first run.*

## 📊 Outputs
After running the script, check the `models/` directory for:
- **`rf_model.pkl`**: The trained Random Forest model.
- **`actual_vs_predicted.png`**: A plot comparing actual vs. predicted usage.
- **`feature_importance.png`**: A chart showing the most influential features.

## 📊 Model Performance
- **Algorithm**: Random Forest Regressor
- **Metrics**:
  - R² Score: ~0.97 (Realistic fit)
  - MSE: Very low error on test set
- **Key Features**: Voltage, Global Intensity, Sub-metering, Time of Day, Lag Features.

## 📌 License
MIT License
