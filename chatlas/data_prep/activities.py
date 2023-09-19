import datetime
import json
import os

import pandas as pd


def parse_datetime(dt_str):
    try:
        # Try the format with milliseconds
        return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        # Fall back to the format without milliseconds
        return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ")


SEMANTIC_DIR = "./data/location_history/semantic/2023"

# Load all the files in the directory
data_files = [os.path.join(SEMANTIC_DIR, file) for file in os.listdir(SEMANTIC_DIR) if file.endswith(".json")]

# Load the data from each file into a list and parse json
data = []
for file in data_files:
    with open(file, "r") as f:
        data.append(json.load(f))

# Convert to a DataFrame
df = pd.json_normalize(data)

# Extract the activity segments and place visits
activity_segments = []
place_visits = []
for i in data[0]["timelineObjects"]:
    if "activitySegment" in i.keys():
        activity_segments.append(i["activitySegment"])
    if "placeVisit" in i.keys():
        place_visits.append(i["placeVisit"])

# Convert to DataFrames
df_acts = pd.json_normalize(activity_segments)
df_visits = pd.json_normalize(place_visits)

# Convert the start and end times to datetime objects
df_acts["duration.startTimestamp"] = df_acts["duration.startTimestamp"].apply(parse_datetime)
df_acts["duration.endTimestamp"] = df_acts["duration.endTimestamp"].apply(parse_datetime)


# Create a new DataFrame with one row per 5-minute interval
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


# Create an empty DataFrame that covers all 5-minute intervals for the month
start_date = df_acts["duration.startTimestamp"].min().replace(hour=0, minute=0, second=0, microsecond=0)
end_date = df_acts["duration.endTimestamp"].max().replace(hour=23, minute=59, second=59, microsecond=0)
date_range = pd.date_range(start_date, end_date, freq="5T")
all_intervals_df = pd.DataFrame(index=date_range)
all_intervals_df = all_intervals_df.reset_index().rename(columns={"index": "interval_start"})

# Merge the base DataFrame with the granular_df
merged_df = pd.merge_asof(
    all_intervals_df.sort_values(by="interval_start"),
    granular_df.sort_values(by="duration.startTimestamp"),
    left_on="interval_start",
    right_on="duration.startTimestamp",
    direction="backward",
    suffixes=("", "_y"),
)

# Drop unwanted columns and clean up if necessary
# merged_df.drop(columns=['duration.startTimestamp_y', 'duration.endTimestamp_y'], inplace=True)


# Save processed merged df as pickle, ensuring directory exists
os.makedirs("./data/processed", exist_ok=True)
merged_df.to_pickle("./data/processed/semantic.pkl")
