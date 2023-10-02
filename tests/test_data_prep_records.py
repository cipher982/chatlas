import pandas as pd
import pytest
from pathlib import Path
from your_module import load_data, preprocess_data, get_top_activity, save_data

# Update this to your actual path relative to the test file
DEFAULT_RECORDS_PATH = Path("./data/sample/location_history/Records.json")


@pytest.fixture(scope="function")
def sample_data():
    """Load sample data from JSON file."""
    return load_data(DEFAULT_RECORDS_PATH)


def test_load_data(sample_data):
    """Test the load_data function."""
    assert isinstance(sample_data, dict), "Loaded data should be a dictionary."


def test_preprocess_data(sample_data):
    """Test the preprocess_data function."""
    df = pd.json_normalize(sample_data["locations"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    preprocessed_df = preprocess_data(df)

    assert "top_activity" in preprocessed_df.columns
    assert "confidence" in preprocessed_df.columns


def test_get_top_activity(sample_data):
    """Test the get_top_activity function."""
    activities = sample_data["locations"][0]["activity"]
    result = get_top_activity(activities)
    assert isinstance(result, dict), "Top activity should be a dictionary."
    assert "type" in result
    assert "confidence" in result


def test_save_data(sample_data):
    """Test the save_data function."""
    df = pd.json_normalize(sample_data["locations"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    preprocessed_df = preprocess_data(df)

    with TemporaryDirectory() as tmpdirname:
        temp_path = Path(tmpdirname) / "temp.pkl"
        save_data(preprocessed_df, temp_path)
        assert temp_path.exists(), "Pickle file was not saved."
