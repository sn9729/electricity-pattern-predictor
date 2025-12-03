from src.data_loader import download_data, load_and_preprocess_data, create_features, get_train_test_split
from src.model import build_model, save_model
from src.utils import evaluate_model, plot_actual_vs_predicted, plot_feature_importance
import matplotlib.pyplot as plt
import os

def main():
    # 1. Data Loading
    print("--- Step 1: Data Loading ---")
    data_path = download_data()
    df = load_and_preprocess_data(data_path)
    
    # 2. Feature Engineering
    print("\n--- Step 2: Feature Engineering ---")
    df = create_features(df)
    print(f"Data shape after preprocessing: {df.shape}")
    
    # 3. Split Data
    print("\n--- Step 3: Splitting Data ---")
    X_train, y_train, X_test, y_test = get_train_test_split(df)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # 4. Model Training
    print("\n--- Step 4: Model Training ---")
    model = build_model()
    model.fit(X_train, y_train)
    print("Training complete.")
    
    # 5. Evaluation
    print("\n--- Step 5: Evaluation ---")
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred)
    
    # 6. Save Model
    print("\n--- Step 6: Saving Model ---")
    save_model(model)
    
    # 7. Visualizations (Optional save)
    print("\n--- Step 7: Generating Plots ---")
    plot_actual_vs_predicted(y_test, y_pred)
    plt.savefig("models/actual_vs_predicted.png")
    print("Saved actual_vs_predicted.png")
    
    plot_feature_importance(model, X_train.columns)
    plt.savefig("models/feature_importance.png")
    print("Saved feature_importance.png")

if __name__ == "__main__":
    main()
