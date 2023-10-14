import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from chatlas.utils import get_nested_value

# logging configuration
LOG = logging.getLogger(__name__)

# Base path
BASE_PATH = Path("./data/sample")

# Sub-paths
SEMANTIC_PATH = BASE_PATH / "location_history/semantic"
PROCESSED_PATH = BASE_PATH / "processed"

# Constants
DEFAULT_SEMANTIC_PATH = SEMANTIC_PATH
DEFAULT_PLACES_OUTPUT_PATH = PROCESSED_PATH / "semantic_places.pkl"
DEFAULT_ACTIVITIES_OUTPUT_PATH = PROCESSED_PATH / "semantic_activities.pkl"
SQL_DB_PATH = PROCESSED_PATH / "chatlas.db"


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
    LOG.info("Loading data from files...")
    data_files = [file for file in directory.glob("*.json")]
    data = []

    for file in data_files:
        with file.open("r") as f:
            data.append(json.load(f))

    LOG.info(f"Loaded {len(data)} json files from {directory}")
    return data


def extract_single_year(data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    places = []
    activities = []

    for month in data:
        for i in month["timelineObjects"]:
            timeline_keys = list(i.keys())

            # Check to make sure data is in the expected format
            if len(timeline_keys) == 1:
                key = timeline_keys[0]
            else:
                raise ValueError(f"Expected 1 key, got {len(timeline_keys)} keys.")

            if key == "placeVisit":
                place_visit = {}

                visit = i["placeVisit"]

                # Location
                place_visit["name"] = get_nested_value(visit, ["location", "name"], None)
                place_visit["address"] = get_nested_value(visit, ["location", "address"], None)
                place_visit["lat"] = get_nested_value(visit, ["location", "latitudeE7"], None)
                place_visit["lon"] = get_nested_value(visit, ["location", "longitudeE7"], None)
                place_visit["place_id"] = get_nested_value(visit, ["location", "placeId"], None)

                # Confidence
                place_visit["confidence_visit"] = get_nested_value(visit, ["visitConfidence"], None)
                place_visit["confidence_location"] = get_nested_value(visit, ["locationConfidence"], None)

                # Times
                place_visit["start_time"] = get_nested_value(visit, ["duration", "startTimestamp"], None)
                place_visit["end_time"] = get_nested_value(visit, ["duration", "endTimestamp"], None)

                # Other
                place_visit["visit_type"] = get_nested_value(visit, ["placeVisitType"], None)
                place_visit["visit_importance"] = get_nested_value(visit, ["placeVisitImportance"], None)

                places.append(place_visit)

            elif key == "activitySegment":
                activity = {}

                segment = i["activitySegment"]

                # Locations
                activity["startLocation_lat"] = get_nested_value(segment, ["startLocation", "latitudeE7"], None)
                activity["startLocation_lon"] = get_nested_value(segment, ["startLocation", "longitudeE7"], None)
                activity["endLocation_lat"] = get_nested_value(segment, ["endLocation", "latitudeE7"], None)
                activity["endLocation_lon"] = get_nested_value(segment, ["endLocation", "longitudeE7"], None)

                # Times
                activity["start_time"] = get_nested_value(segment, ["duration", "startTimestamp"], None)
                activity["end_time"] = get_nested_value(segment, ["duration", "endTimestamp"], None)

                # Distance
                activity["distance"] = get_nested_value(segment, ["distance"], None)

                # Activity Type
                activity["activity_type"] = get_nested_value(segment, ["activityType"], None)

                # Confidence
                activity["confidence"] = get_nested_value(segment, ["activities", 0, "probability"], None)

                activities.append(activity)

            else:
                raise ValueError(f"Unexpected key {key}")

    LOG.info(f"Extracted {len(places)} places and {len(activities)} activities.")

    return places, activities


def extract_all_semantic(semantic_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes data for all years present in the given semantic directory.

    Parameters:
        semantic_dir (Path): The path to the semantic directory containing subdirectories for each year.

    Returns:
        all_places (pd.DataFrame): DataFrame containing all places.
        all_activities (pd.DataFrame): DataFrame containing all activities.
    """
    all_places = []
    all_activities = []

    for year_dir in semantic_dir.iterdir():
        if year_dir.is_dir():
            LOG.info(f"=== Processing {year_dir.name}... ===")
            data = load_data_from_files(year_dir)
            places, activities = extract_single_year(data)
            all_places.extend(places)
            all_activities.extend(activities)
        else:
            LOG.info(f"Skipping {year_dir} as it is not a directory.")

    # Convert to DataFrames
    all_places = pd.DataFrame(all_places)
    all_activities = pd.DataFrame(all_activities)

    LOG.info(f"Extracted {len(all_places)} places and {len(all_activities)} activities in total.")

    return all_places, all_activities


def process_places(places: pd.DataFrame) -> pd.DataFrame:
    # Convert timestamp to datetime format
    places["start_time"] = pd.to_datetime(places["start_time"], format="ISO8601")
    places["end_time"] = pd.to_datetime(places["end_time"], format="ISO8601")

    # Extract address components
    addr_components = places["address"].apply(extract_address_components)
    places[["addr_name", "street", "city", "state", "country"]] = addr_components

    # Convert lat/lon to float
    places["lat"] = (places["lat"] / 1e7).astype("float64")
    places["lon"] = (places["lon"] / 1e7).astype("float64")

    # Convert confidence to int
    places["confidence_visit"] = places["confidence_visit"].round().astype("Int64")
    places["confidence_location"] = places["confidence_location"].round().astype("Int64")

    return places


def process_activities(activities: pd.DataFrame) -> pd.DataFrame:
    # Convert timestamp to datetime format
    activities["start_time"] = pd.to_datetime(activities["start_time"], format="ISO8601")
    activities["end_time"] = pd.to_datetime(activities["end_time"], format="ISO8601")

    # Convert lat/lon to float
    activities["startLocation_lat"] = (activities["startLocation_lat"] / 1e7).astype("float64")
    activities["startLocation_lon"] = (activities["startLocation_lon"] / 1e7).astype("float64")
    activities["endLocation_lat"] = (activities["endLocation_lat"] / 1e7).astype("float64")
    activities["endLocation_lon"] = (activities["endLocation_lon"] / 1e7).astype("float64")

    # Convert distance to int
    activities["distance"] = activities["distance"].round().astype("Int64")

    # Convert confidence to int
    activities["confidence"] = activities["confidence"].round().astype("Int64")

    # Drop rows without location data
    loc_cols = ["startLocation_lat", "startLocation_lon", "endLocation_lat", "endLocation_lon"]
    activities = activities.dropna(subset=loc_cols)

    # Drop rows without activity type
    activities = activities.dropna(subset=["activity_type"])

    return activities


def extract_address_components(address):
    components = address.split(",") if address else []
    street, city, state, country = None, None, None, None

    if len(components) == 3:
        street, city, country = components
        addr_name = "missing"
        state = "missing"
    elif len(components) == 4:
        street, city, state, country = components
        addr_name = "missing"
    elif len(components) == 5:
        addr_name, street, city, state, country = components
    else:
        LOG.debug(f"Unexpected address format: {address}")
        addr_name, street, city, state, country = "missing", "missing", "missing", "missing", "missing"

    return pd.Series(
        [
            addr_name.strip(),
            street.strip(),
            city.strip(),
            state.strip(),
            country.strip(),
        ]
    )


def save_data(df: pd.DataFrame, output_file: Path) -> None:
    """Save DataFrame to a pickle file."""
    LOG.info("Saving processed data...")
    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(output_file)
    LOG.info(f"Saved DataFrame of shape {df.shape} to: {output_file}")


def main(load_sql: bool = False):
    places, activities = extract_all_semantic(DEFAULT_SEMANTIC_PATH)
    places = process_places(places)
    activities = process_activities(activities)
    save_data(places, DEFAULT_PLACES_OUTPUT_PATH)
    save_data(activities, DEFAULT_ACTIVITIES_OUTPUT_PATH)

    if load_sql:
        LOG.info("Loading data into sqlite3 database...")
        import sqlite3

        conn = sqlite3.connect(SQL_DB_PATH)
        places.to_sql("places", conn, if_exists="replace", index=False)
        activities.to_sql("activities", conn, if_exists="replace", index=False)
        conn.close()
        LOG.info("Done loading data into sqlite3 database.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(load_sql=True)
