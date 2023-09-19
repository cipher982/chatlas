import os
import pandas as pd
from chatlas.data_prep.activities import parse_datetime, load_data_from_files, process_data, save_data


def test_parse_datetime():
    assert parse_datetime("2023-01-01T12:00:00.000Z") == pd.Timestamp("2023-01-01 12:00:00")
    assert parse_datetime("2023-01-01T12:00:00Z") == pd.Timestamp("2023-01-01 12:00:00")


def test_load_data_from_files():
    data = load_data_from_files("./data/location_history/semantic/2023_sample/")
    assert type(data) == list
    assert len(data) > 0


def test_process_data():
    sample_data = [{"timelineObjects": [{"activitySegment": {...}, "placeVisit": {...}}]}]  # fill in sample objects
    df = process_data(sample_data)
    assert "interval_start" in df.columns


def test_save_data():
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    save_data(df, "./data/processed/test.pkl")
    assert os.path.exists("./data/processed/test.pkl")
    os.remove("./data/processed/test.pkl")
