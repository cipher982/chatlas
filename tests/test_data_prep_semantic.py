import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest
import sqlite3

from chatlas.data_prep.semantic import (
    DEFAULT_SEMANTIC_PATH,
    extract_address_components,
    extract_all_semantic,
    extract_single_year,
    load_data_from_files,
    main,
    parse_datetime,
    write_to_df,
)


# Provide a fixture to load the sample data
@pytest.fixture(scope="module")
def sample_data():
    loaded_data = load_data_from_files(DEFAULT_SEMANTIC_PATH / "2023")
    return loaded_data


# Provide a fixture to not persst the data to disk
@pytest.fixture
def mock_save(monkeypatch):
    mock = Mock()
    monkeypatch.setattr("chatlas.data_prep.semantic.write_to_df", mock)
    return mock


@pytest.fixture
def mock_sqlite(monkeypatch):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = []
    monkeypatch.setattr("sqlite3.connect", Mock(return_value=mock_conn))
    return mock_conn


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


def test_extract_all_semantic():
    places, activities = extract_all_semantic(DEFAULT_SEMANTIC_PATH)
    assert len(places) > 0
    assert len(activities) > 0


def test_process_places(sample_data):
    pass


def test_process_activities(sample_data):
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


def test_write_to_df(tmp_path):
    """Test the write_to_df function."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    output_file = tmp_path / "output.pkl"
    write_to_df(df, output_file)
    assert output_file.exists()

    loaded_df = pd.read_pickle(output_file)
    pd.testing.assert_frame_equal(df, loaded_df)


#############################
##### INTEGRATION TESTS #####
#############################


# def test_full_pipeline(sample_data):
#     # Extract data
#     places, activities = extract_single_year(sample_data)

#     # Process data
#     processed_places = process_places(places)
#     processed_activities = process_activities(activities)

#     # Validate processed_places
#     assert "start_time" in processed_places.columns
#     assert "end_time" in processed_places.columns
#     assert "lat" in processed_places.columns
#     assert "lon" in processed_places.columns

#     # Validate processed_activities
#     assert "start_time" in processed_activities.columns
#     assert "end_time" in processed_activities.columns
#     assert "startLocation_lat" in processed_activities.columns
#     assert "startLocation_lon" in processed_activities.columns


def test_main_write_df():
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        temp_places_output_path = os.path.join(temp_dir, "places.pkl")
        temp_activities_output_path = os.path.join(temp_dir, "activities.pkl")

        # Run your main function, saving the DataFrames to the temp directory
        main(
            write_df=True,
            write_sql=False,
            places_output_path=Path(temp_places_output_path),
            activities_output_path=Path(temp_activities_output_path),
        )

        # Read back the DataFrames from the pickle files
        places_df_read = pd.read_pickle(temp_places_output_path)
        activities_df_read = pd.read_pickle(temp_activities_output_path)

        # Validate the DataFrames (replace these with your own checks)
        places_cols = [
            "name",
            "address",
            "lat",
            "lon",
            "place_id",
            "confidence_visit",
            "confidence_location",
            "start_time",
            "end_time",
            "visit_type",
            "visit_importance",
            "addr_name",
            "street",
            "city",
            "state",
            "country",
        ]
        assert places_df_read.columns.tolist() == places_cols

        activities_cols = [
            "startLocation_lat",
            "startLocation_lon",
            "endLocation_lat",
            "endLocation_lon",
            "start_time",
            "end_time",
            "distance",
            "activity_type",
            "confidence",
        ]
        assert activities_df_read.columns.tolist() == activities_cols

        assert len(places_df_read) > 10
        assert len(activities_df_read) > 10

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)


def test_main_write_sql(mock_save):
    temp_db_path = "temp.db"

    main(write_df=False, write_sql=True, sql_db_path=temp_db_path)

    conn = sqlite3.connect(temp_db_path)

    # Validate data
    for table_name in ["places", "activities"]:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        assert len(df) > 10

    os.remove(temp_db_path)
