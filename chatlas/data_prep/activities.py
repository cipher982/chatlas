import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_datetime(dt_str: str) -> datetime:
    """Parse datetime strings into datetime objects."""
    formats = ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"]
    for fmt in formats:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unknown datetime format: {dt_str}")


def load_data_from_files(directory: Path) -> list:
    """Load JSON files from a directory."""
    logging.info("Loading data from files...")
    data_files = [file for file in directory.glob("*.json")]
    data = []

    for file in data_files:
        with file.open("r") as f:
            data.append(json.load(f))

    logging.info(f"Loaded {len(data)} files from {directory}")
    return data


def process_data(data: list) -> pd.DataFrame:
    logging.info("Processing data...")

    # Extract segments
    activity_segments = [i["activitySegment"] for i in data[0]["timelineObjects"] if "activitySegment" in i]
    df_acts = pd.json_normalize(activity_segments)
    df_acts["duration.startTimestamp"] = df_acts["duration.startTimestamp"].apply(parse_datetime)
    df_acts["duration.endTimestamp"] = df_acts["duration.endTimestamp"].apply(parse_datetime)
    logging.info(f"Number of activity segments: {len(df_acts)}")

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
    logging.info(f"Number of 5-minute intervals: {len(all_intervals_df)}")
    logging.info(f"Minimum date: {start_date}")
    logging.info(f"Maximum date: {end_date}")

    # Merge
    merged_df = pd.merge_asof(
        all_intervals_df.sort_values(by="interval_start"),
        granular_df.sort_values(by="duration.startTimestamp"),
        left_on="interval_start",
        right_on="duration.startTimestamp",
        direction="backward",
        suffixes=("", "_y"),
    )
    logging.info(f"Final processed dataframe shape: {merged_df.shape}")

    return merged_df


def save_data(df: pd.DataFrame, output_dir: Path) -> None:
    """Save DataFrame to a pickle file."""
    logging.info("Saving processed data...")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "semantic.pkl"
    df.to_pickle(output_file)
    logging.info(f"Saved DataFrame of shape {df.shape} to: {output_file}")
