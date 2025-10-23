import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime


def load_latest_processed_data():
    """Load the most recent processed data file"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    # Find the most recent processed data file
    processed_files = [
        f
        for f in os.listdir(data_dir)
        if f.startswith("processed_nfl_data_") and f.endswith(".pkl")
    ]

    if not processed_files:
        print(
            "No processed data files found. Run train.py first to create processed data."
        )
        return None

    # Get the most recent file
    latest_file = sorted(processed_files)[-1]
    file_path = os.path.join(data_dir, latest_file)

    print(f"Loading processed data from: {latest_file}")
    return joblib.load(file_path)


def train_from_processed():
    """Train model using pre-processed data"""

    # Load processed data
    data = load_latest_processed_data()
    if data is None:
        return

    X = data["X"]
    Y = data["Y"]
    feature_columns = data["feature_columns"]

    print(f"Loaded data: {len(X)} games, {len(feature_columns)} features")
    print(f"Years: {data['years_processed']}")
    print(f"Data timestamp: {data['timestamp']}")

    # Split and train (much faster now!)
    print("\nTraining model...")
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(Xtrain, ytrain)

    # Evaluate
    y_pred = model.predict(Xtest)
    accuracy = accuracy_score(ytest, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Save model
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    accuracy_str = f"{accuracy:.4f}".replace(".", "")
    model_filename = f"nfl_spread_model_{timestamp}_acc{accuracy_str}.pkl"
    model_path = os.path.join(models_dir, model_filename)

    # Save the model and metadata
    model_data = {
        "model": model,
        "feature_columns": feature_columns,
        "label_encoders": data["label_encoders"],
        "accuracy": accuracy,
        "training_samples": len(Xtrain),
        "test_samples": len(Xtest),
        "years_trained": data["years_processed"],
        "timestamp": timestamp,
    }

    joblib.dump(model_data, model_path)
    print(f"\nModel saved to: {model_path}")
    print(f"Training samples: {len(Xtrain)}")
    print(f"Test samples: {len(Xtest)}")


if __name__ == "__main__":
    train_from_processed()
