from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from chatlas.data_prep.semantic import load_data_from_files, parse_datetime, process_data, save_data

# Constants at the top of the file
SAMPLE_DATA_PATH = Path("./data/sample/location_history/semantic/2023")


# Define the fixture to load sample data for multiple test functions
@pytest.fixture(scope="module")
def loaded_data():
    return load_data_from_files(SAMPLE_DATA_PATH)


# Define the fixture to process the loaded data
@pytest.fixture(scope="module")
def processed_data(loaded_data):
    return process_data(loaded_data)


def test_parse_datetime():
    assert parse_datetime("2023-01-01T12:00:00.000Z") == pd.Timestamp("2023-01-01 12:00:00")
    assert parse_datetime("2023-01-01T12:00:00Z") == pd.Timestamp("2023-01-01 12:00:00")


def test_load_data_from_files(loaded_data):
    assert isinstance(loaded_data, list)
    assert len(loaded_data) > 0


def test_process_data(processed_data):
    assert "interval_start" in processed_data.columns
    # Add more asserts for robustness if needed, for example:
    assert not processed_data.empty
    assert processed_data["interval_start"].dtype == "datetime64[ns]"


def test_save_data(tmp_path):
    """Test the save_data function."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    output_file = tmp_path / "output.pkl"
    save_data(df, output_file)
    assert output_file.exists()

    loaded_df = pd.read_pickle(output_file)
    pd.testing.assert_frame_equal(df, loaded_df)