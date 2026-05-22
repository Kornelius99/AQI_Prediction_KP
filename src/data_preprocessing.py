
import pandas as pd
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = ["pm25", "pm10", "o3", "no2", "so2", "co"]
TARGET_COLUMN = "AQI"


def load_data(file_path: str) -> pd.DataFrame:
    """Load AQI dataset from CSV."""
    return pd.read_csv(file_path, skipinitialspace=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean AQI dataset by removing duplicates and invalid records."""
    df = df.copy()

    df = df.drop_duplicates()
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    df = df[df[TARGET_COLUMN] >= 0]

    return df


def prepare_features(df: pd.DataFrame):
    """Split dataset into feature matrix and target variable."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Create train and test datasets."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )


def preprocess_pipeline(file_path: str):
    """Full preprocessing pipeline."""
    raw_df = load_data(file_path)
    clean_df = clean_data(raw_df)
    X, y = prepare_features(clean_df)
    return split_data(X, y)
