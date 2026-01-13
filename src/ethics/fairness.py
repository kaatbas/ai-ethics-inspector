import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    demographic_parity_ratio
)
from sklearn.metrics import accuracy_score

def calculate_fairness_metrics(y_true, y_pred, sensitive_features, unique_privileged_group):
    """
    Calculates fairness metrics for a given model prediction.
    
    Args:
        y_true (pd.Series): True targets.
        y_pred (pd.Series): Predicted targets.
        sensitive_features (pd.Series): Sensitive attribute column (e.g. Sex).
        unique_privileged_group (str/int): The value indicating the privileged group (e.g. 'Male').
        
    Returns:
        dict: Dictionary containing fairness and performance metrics.
    """
    
    # Selection Rate (Overall)
    sr = selection_rate(y_true, y_pred)
    
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # Statistical Parity Difference (Demographic Parity Difference)
    # The difference between selection rates of the groups.
    spd = demographic_parity_difference(
        y_true,
        y_pred,
        sensitive_features=sensitive_features
    )
    
    # Disparate Impact (Demographic Parity Ratio)
    # Ratio of selection rates.
    dpr = demographic_parity_ratio(
        y_true,
        y_pred,
        sensitive_features=sensitive_features
    )
    
    # Fairness Metric Frame (Detailed view per group) using MetricFrame
    # For now, we return summary scalars, but MetricFrame is powerful for detailed reports.
    
    return {
        "accuracy": acc,
        "selection_rate": sr,
        "statistical_parity_difference": spd,
        "disparate_impact": dpr,
        "demographic_parity_ratio": dpr # Alias for clarity
    }
