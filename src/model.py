from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def build_model(n_estimators=50, max_depth=15, n_jobs=1):
    """Builds and returns the Random Forest model."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=n_jobs
    )
    return model

def save_model(model, path="models/rf_model.pkl"):
    """Saves the trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path="models/rf_model.pkl"):
    """Loads a trained model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)
