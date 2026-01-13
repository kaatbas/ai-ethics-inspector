import pytest
import pandas as pd
import numpy as np
from src.ethics.fairness import calculate_fairness_metrics

def test_fairness_metrics_structure():
    """Test if the function returns a dictionary with expected keys."""
    # Synthetic data
    y_true = pd.Series([1, 1, 1, 0, 0, 1, 0, 0, 1, 0])
    y_pred = pd.Series([1, 1, 0, 0, 0, 1, 0, 1, 1, 0])
    sensitive_features = pd.Series(['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'])
    
    metrics = calculate_fairness_metrics(y_true, y_pred, sensitive_features, unique_privileged_group='Male')
    
    assert isinstance(metrics, dict)
    assert 'statistical_parity_difference' in metrics
    assert 'disparate_impact' in metrics
    assert 'accuracy' in metrics
    assert 'demographic_parity_ratio' in metrics

def test_perfect_fairness():
    """Test scenario where groups are treated exactly equally."""
    # Both groups have 50% acceptance rate
    y_true = pd.Series([1, 0, 1, 0])
    y_pred = pd.Series([1, 0, 1, 0]) # 50% positive rate for both if aligned
    sensitive_features = pd.Series(['A', 'A', 'B', 'B'])
    
    metrics = calculate_fairness_metrics(y_true, y_pred, sensitive_features, unique_privileged_group='A')
    
    # Statistical parity diff should be 0
    assert abs(metrics['statistical_parity_difference']) < 0.01

def test_extreme_bias():
    """Test scenario with extreme bias (Group A gets all 1s, Group B gets all 0s)."""
    y_true = pd.Series([1, 1, 0, 0]) # Irrelevant for selection rate, but needed for API
    y_pred = pd.Series([1, 1, 0, 0]) # Group A accepted, Group B rejected
    sensitive_features = pd.Series(['A', 'A', 'B', 'B'])
    
    metrics = calculate_fairness_metrics(y_true, y_pred, sensitive_features, unique_privileged_group='A')
    
    # Group A selection rate = 1.0
    # Group B selection rate = 0.0
    # Stat Parity Diff = 0.0 - 1.0 = -1.0 (or 1.0 depending on direction)
    assert abs(metrics['statistical_parity_difference']) > 0.9
