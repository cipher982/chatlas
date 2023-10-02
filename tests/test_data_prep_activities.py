from pathlib import Path
import time

import pandas as pd

from chatlas.data_prep.activities import load_data_from_files, parse_datetime, process_data, save_data


def test_parse_datetime():
    assert parse_datetime("2023-01-01T12:00:00.000Z") == pd.Timestamp("2023-01-01 12:00:00")
    assert parse_datetime("2023-01-01T12:00:00Z") == pd.Timestamp("2023-01-01 12:00:00")


def test_load_data_from_files():
    data = load_data_from_files(Path("./data/sample/location_history/semantic/2023"))
    assert type(data) == list
    assert len(data) > 0


def test_process_data():
    # sample_data = [{"timelineObjects": [{"activitySegment": {...}, "placeVisit": {...}}]}]
    data = load_data_from_files(Path("./data/sample/location_history/semantic/2023"))
    df = process_data(data)
    assert "interval_start" in df.columns


def test_save_data():
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    dir_path = Path("./data/processed/")
    save_data(df, dir_path)
    file_path = dir_path / "semantic.pkl"
    assert file_path.exists()
    time.sleep(1)  # delay to ensure the file can be deleted
    file_path.unlink()  # delete the file
