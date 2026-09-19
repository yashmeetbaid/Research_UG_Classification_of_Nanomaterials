import pandas as pd
import numpy as np


def clean_dataframe(df):
    """
    Basic preprocessing function used for testing.

    - Removes duplicate rows
    - Handles missing numerical values
    - Handles missing categorical values
    """
    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Fill numerical missing values with median
    numerical_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].median())

    # Fill categorical missing values
    categorical_columns = df.select_dtypes(
        exclude=[np.number]
    ).columns

    for column in categorical_columns:
        if df[column].isna().any():
            df[column] = df[column].fillna("Unknown")

    return df


def test_duplicate_rows_are_removed():
    """Verify that duplicate observations are removed."""

    data = pd.DataFrame({
        "size": [10, 20, 20, 30],
        "surface_area": [100, 200, 200, 300],
        "material": ["A", "B", "B", "C"]
    })

    cleaned = clean_dataframe(data)

    assert len(cleaned) == 3
    assert not cleaned.duplicated().any()


def test_missing_numerical_values_are_handled():
    """Verify that numerical missing values are replaced."""

    data = pd.DataFrame({
        "size": [10, np.nan, 30],
        "surface_area": [100, 200, np.nan]
    })

    cleaned = clean_dataframe(data)

    assert not cleaned["size"].isna().any()
    assert not cleaned["surface_area"].isna().any()


def test_missing_categorical_values_are_handled():
    """Verify that missing categorical values are replaced."""

    data = pd.DataFrame({
        "material": ["Gold", None, "Silver"],
        "size": [10, 20, 30]
    })

    cleaned = clean_dataframe(data)

    assert not cleaned["material"].isna().any()
    assert "Unknown" in cleaned["material"].values


def test_preprocessing_preserves_columns():
    """Verify that preprocessing does not unexpectedly remove columns."""

    data = pd.DataFrame({
        "size": [10, 20, 30],
        "surface_area": [100, 200, 300],
        "material": ["A", "B", "C"]
    })

    cleaned = clean_dataframe(data)

    assert list(cleaned.columns) == list(data.columns)
