from pathlib import Path

import pandas as pd
import pytest
from chatlas.data_prep.records import get_top_activity, load_data, preprocess_data, save_data


# Update this to your actual path relative to the test file
DEFAULT_RECORDS_PATH = Path("./data/sample/location_history/Records.json")


@pytest.fixture(scope="function")
def sample_data():
    """Load sample data from JSON file."""
    return load_data(DEFAULT_RECORDS_PATH, 1_000)


def test_load_data(sample_data):
    """Test the load_data function."""
    assert isinstance(sample_data, pd.DataFrame), "Loaded data should be a DataFrame."


def test_preprocess_data(sample_data):
    """Test the preprocess_data function."""
    preprocessed_df = preprocess_data(sample_data)

    assert "top_activity" in preprocessed_df.columns
    assert "confidence" in preprocessed_df.columns


def test_get_top_activity(sample_data):
    """Test the get_top_activity function."""
    top_activities = sample_data["activity"].apply(get_top_activity)
    percent_populated = top_activities.isna().sum() / len(sample_data)
    assert isinstance(top_activities, pd.Series), "Function should return a Series."
    assert percent_populated < 0.9, "More than 90% of the activities appear to be missing."


def test_save_data(tmp_path):
    """Test the save_data function."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    output_file = tmp_path / "output.pkl"
    save_data(df, output_file)
    assert output_file.exists()

    loaded_df = pd.read_pickle(output_file)
    pd.testing.assert_frame_equal(df, loaded_df)
