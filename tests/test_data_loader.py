import pytest
import pandas as pd
from src.data.loader import load_german_data

def test_load_german_data_returns_dataframe():
    """Test if the loader returns a tuple of pandas DataFrames (X, y)."""
    try:
        X, y = load_german_data()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)
        assert not X.empty
        assert not y.empty
    except Exception as e:
        pytest.fail(f"Data loading failed with error: {e}")

def test_german_data_shape():
    """Test if the dataset has the expected shape (approx 1000 rows)."""
    X, y = load_german_data()
    assert len(X) == 1000
    # 20 attributes + target is standard, but ucimlrepo might differ slightly in split
    # We expect at least 20 features
    assert X.shape[1] >= 20

def test_target_values():
    """Test if target values are within expected range (1=Good, 2=Bad usually)."""
    X, y = load_german_data()
    # y is a DataFrame, so we need to check the unique values of its first column
    unique_values = y.iloc[:, 0].unique()
    # German credit data target is usually 1 (Good) and 2 (Bad)
    assert 1 in unique_values
    assert 2 in unique_values
