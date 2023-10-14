from pathlib import Path

import pandas as pd
import pytest
from unittest.mock import patch


from chatlas.data_prep.semantic import parse_datetime
from chatlas.data_prep.semantic import load_data_from_files
from chatlas.data_prep.semantic import extract_single_year
from chatlas.data_prep.semantic import extract_all_semantic
from chatlas.data_prep.semantic import extract_address_components
from chatlas.data_prep.semantic import process_places
from chatlas.data_prep.semantic import process_activities
from chatlas.data_prep.semantic import save_data
from chatlas.data_prep.semantic import main
from chatlas.data_prep.semantic import DEFAULT_SEMANTIC_PATH


# Provide a fixture to load the sample data
@pytest.fixture(scope="module")
def sample_data():
    return load_data_from_files(".." / DEFAULT_SEMANTIC_PATH)


######################
##### UNIT TESTS #####
######################


def test_parse_datetime():
    assert parse_datetime("2023-01-01T12:00:00.000Z") == pd.Timestamp("2023-01-01 12:00:00")
    assert parse_datetime("2023-01-01T12:00:00Z") == pd.Timestamp("2023-01-01 12:00:00")


def test_load_data_from_files(sample_data):
    assert isinstance(sample_data, list)
    assert len(sample_data) > 0


def test_extract_single_year(sample_data):
    places, activities = extract_single_year(sample_data)
    assert len(places) > 0
    assert len(activities) > 0


def test_extract_all_semantic(sample_data):
    pass


def test_process_places(sample_data):
    pass


def test_process_activities(sample_data):
    pass


def test_extract_address_components(sample_data):
    pass


def test_extract_address_components():
    # Test with all five components
    assert extract_address_components("Name, Street, City, State, Country").equals(
        pd.Series(["Name", "Street", "City", "State", "Country"])
    )

    # Test with four components
    assert extract_address_components("Street, City, State, Country").equals(
        pd.Series(["missing", "Street", "City", "State", "Country"])
    )

    # Test with three components
    assert extract_address_components("Street, City, Country").equals(
        pd.Series(["missing", "Street", "City", "missing", "Country"])
    )

    # Test with invalid input
    assert extract_address_components("Street, City").equals(
        pd.Series(["missing", "missing", "missing", "missing", "missing"])
    )

    # Test with empty input
    assert extract_address_components("").equals(pd.Series(["missing", "missing", "missing", "missing", "missing"]))

    # Test with None input
    assert extract_address_components(None).equals(pd.Series(["missing", "missing", "missing", "missing", "missing"]))


def test_save_data(tmp_path):
    """Test the save_data function."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    output_file = tmp_path / "output.pkl"
    save_data(df, output_file)
    assert output_file.exists()

    loaded_df = pd.read_pickle(output_file)
    pd.testing.assert_frame_equal(df, loaded_df)


#############################
##### INTEGRATION TESTS #####
#############################


def test_full_pipeline(sample_data):
    # Extract data
    places, activities = extract_all_semantic(sample_data)

    # Process data
    processed_places = process_places(places)
    processed_activities = process_activities(activities)

    # Validate processed_places
    assert "start_time" in processed_places.columns
    assert "end_time" in processed_places.columns
    assert "lat" in processed_places.columns
    assert "lon" in processed_places.columns

    # Validate processed_activities
    assert "start_time" in processed_activities.columns
    assert "end_time" in processed_activities.columns
    assert "startLocation_lat" in processed_activities.columns
    assert "startLocation_lon" in processed_activities.columns


@patch("your_project.chatlas.data_prep.semantic.save_data")  # Replace with your actual import path
def test_main(mock_save, sample_data):
    # Mock the save_data function to avoid disk writes
    mock_save.return_value = True

    # Run main function
    main(load_sql=False)

    # Check if the save function was called twice (for places and activities)
    assert mock_save.call_count == 2
