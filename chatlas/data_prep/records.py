import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
DEFAULT_RECORDS_PATH = Path(".../data/sample/location_history/Records.json")
DEFAULT_OUTPUT_PATH = Path(".../data/sample/processed/records.pkl")


def load_data(file_path: Path) -> Dict:
    """Load data from a JSON file."""
    logging.info(f"Loading data from {file_path}...")
    with file_path.open("r") as f:
        data = json.load(f)
    logging.info("Data loaded successfully.")
    return data


def preprocess_data(df: pd.DataFrame, sample_n: Optional[int] = None) -> pd.DataFrame:
    """Preprocess DataFrame."""
    logging.info("Preprocessing data...")
    if sample_n:
        df = df.sample(sample_n)
        logging.info(f"Sampled {sample_n} rows from the DataFrame.")

    # Extract the top activity and its confidence
    df["top_activity"] = df["activity"].apply(get_top_activity)
    df["confidence"] = df["top_activity"].apply(lambda x: x["confidence"] if x else None)
    df["top_activity"] = df["top_activity"].apply(lambda x: x["type"] if x else None)

    # Drop redundant columns
    df.drop(columns=["activity"], inplace=True)

    # Convert all string data to lowercase
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    logging.info("Preprocessing completed.")
    return df


def get_top_activity(activities: Union[List[Dict], float]) -> Optional[Dict]:
    """Extract the activity with the highest confidence."""
    if isinstance(activities, list):
        if not activities:
            return None
        top_activities = [max(act["activity"], key=lambda x: x["confidence"]) for act in activities]
        return max(top_activities, key=lambda x: x["confidence"])
    elif pd.isna(activities):
        return None


def save_data(df: pd.DataFrame, output_file: Path) -> None:
    """Save DataFrame to a pickle file."""
    logging.info("Saving processed data...")
    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(output_file)
    logging.info(f"Saved DataFrame of shape {df.shape} to: {output_file}")


if __name__ == "__main__":
    # Load raw data
    raw_data = load_data(DEFAULT_RECORDS_PATH)

    # Convert JSON to DataFrame and preprocess
    df_loc = pd.json_normalize(raw_data["locations"])
    df_loc["timestamp"] = pd.to_datetime(df_loc["timestamp"])
    df = preprocess_data(df_loc, sample_n=10_000)

    # Save the preprocessed data
    save_data(df, DEFAULT_OUTPUT_PATH)
