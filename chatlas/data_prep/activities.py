import datetime
import json
import os
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
SEMANTIC_DIR = "./data/location_history/semantic/2023"
PROCESSED_DIR = "./data/processed"
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "semantic.pkl")


def parse_datetime(dt_str: str) -> datetime.datetime:
    try:
        return datetime.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        return datetime.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ")


def load_data_from_files(directory: str) -> list:
    logging.info("Loading data from files...")
    data_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".json")]
    data = []

    for file in data_files:
        with open(file, "r") as f:
            data.append(json.load(f))

    return data


def process_data(data: list) -> pd.DataFrame:
    logging.info("Processing data...")

    # Extract segments
    activity_segments = [i["activitySegment"] for i in data[0]["timelineObjects"] if "activitySegment" in i]
    df_acts = pd.json_normalize(activity_segments)
    df_acts["duration.startTimestamp"] = df_acts["duration.startTimestamp"].apply(parse_datetime)
    df_acts["duration.endTimestamp"] = df_acts["duration.endTimestamp"].apply(parse_datetime)

    new_rows = []
    for _, row in df_acts.iterrows():
        start_time = row["duration.startTimestamp"]
        end_time = row["duration.endTimestamp"]
        current_time = start_time

        while current_time < end_time:
            next_interval_end = current_time + pd.Timedelta(minutes=5)
            new_row = row.copy()
            new_row["duration.startTimestamp"] = current_time
            new_row["duration.endTimestamp"] = min(next_interval_end, end_time)
            new_rows.append(new_row)
            current_time = next_interval_end

    granular_df = pd.DataFrame(new_rows)

    # Create interval df
    start_date = df_acts["duration.startTimestamp"].min().replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = df_acts["duration.endTimestamp"].max().replace(hour=23, minute=59, second=59, microsecond=0)
    date_range = pd.date_range(start_date, end_date, freq="5T")
    all_intervals_df = pd.DataFrame(index=date_range)
    all_intervals_df = all_intervals_df.reset_index().rename(columns={"index": "interval_start"})

    # Merge
    merged_df = pd.merge_asof(
        all_intervals_df.sort_values(by="interval_start"),
        granular_df.sort_values(by="duration.startTimestamp"),
        left_on="interval_start",
        right_on="duration.startTimestamp",
        direction="backward",
        suffixes=("", "_y"),
    )

    return merged_df


def save_data(df: pd.DataFrame, filepath: str) -> None:
    logging.info("Saving processed data...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_pickle(filepath)


def main():
    logging.info("Starting data processing...")
    data = load_data_from_files(SEMANTIC_DIR)
    processed_df = process_data(data)
    save_data(processed_df, OUTPUT_FILE)
    logging.info("Data processing complete!")


if __name__ == "__main__":
    main()
