import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import preprocess_pipeline


MODEL_PATH = Path("models/aqi_random_forest_model.pkl")


def train_model(X_train, y_train):
    """Train AQI prediction model."""
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    predictions = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "mse": mean_squared_error(y_test, predictions),
        "r2_score": r2_score(y_test, predictions)
    }

    return metrics


def save_model(model, model_path: Path = MODEL_PATH):
    """Save trained model as pickle file."""
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as file:
        pickle.dump(model, file)


def main():
    data_path = "data/AQI_Data.csv"

    X_train, X_test, y_train, y_test = preprocess_pipeline(data_path)

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    save_model(model)

    print("Model training completed successfully")
    print(metrics)


if __name__ == "__main__":
    main()
