import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(y_true, y_pred):
    """Calculates and prints model evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print("Model Evaluation:")
    print(f"R²: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    return mse, r2

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted Usage"):
    """Plots a subset of actual vs predicted values."""
    plt.figure(figsize=(12, 5))
    plt.plot(y_true[:500].values, label="Actual", color="blue")
    plt.plot(y_pred[:500], label="Predicted", color="red")
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("kW")
    plt.legend()
    return plt

def plot_feature_importance(model, feature_names):
    """Plots feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importances)), importances[indices], color='green')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.title("Feature Importance")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    return plt
