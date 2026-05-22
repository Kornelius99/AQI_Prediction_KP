import pickle
from pathlib import Path
import pandas as pd


MODEL_PATH = Path("models/aqi_random_forest_model.pkl")


def load_model(model_path: Path = MODEL_PATH):
    """Load trained AQI prediction model."""
    with open(model_path, "rb") as file:
        return pickle.load(file)


def predict_aqi(model, input_data: pd.DataFrame):
    """Generate AQI predictions."""
    return model.predict(input_data)


def main():
    model = load_model()

    sample_input = pd.DataFrame({
        "pm25": [118],
        "pm10": [199],
        "o3": [20],
        "no2": [45],
        "so2": [4],
        "co": [15]
    })

    prediction = predict_aqi(model, sample_input)

    print(f"Predicted AQI: {prediction[0]:.2f}")


if __name__ == "__main__":
    main()
