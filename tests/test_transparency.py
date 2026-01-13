import pytest
import pandas as pd
import numpy as np
from src.ethics.transparency import generate_explanations
from sklearn.ensemble import RandomForestClassifier

def test_shap_explanation_structure():
    """Test if explanation returns expected shapes/types."""
    # Synthetic Data
    X = pd.DataFrame(np.random.rand(100, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
    y = np.random.randint(0, 2, 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Generate explanations
    try:
        explanations = generate_explanations(model, X)
    except Exception as e:
        pytest.fail(f"Explanation generation failed: {e}")
        
    assert isinstance(explanations, dict)
    
    # Updated: We switched to Native Feature Importance
    assert 'feature_importance' in explanations
    
    fi_df = explanations['feature_importance']
    assert isinstance(fi_df, pd.DataFrame)
    assert not fi_df.empty
    assert 'feature' in fi_df.columns
    assert 'importance' in fi_df.columns

def test_global_importance_sorting():
    """Test if global feature importance is returned and likely sorted."""
    X = pd.DataFrame(np.random.rand(50, 3), columns=['A', 'B', 'C'])
    y = np.random.randint(0, 2, 50)
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    
    explanations = generate_explanations(model, X)
    global_imp = explanations['feature_importance']
    
    assert isinstance(global_imp, pd.DataFrame)
    assert 'feature' in global_imp.columns
    assert 'importance' in global_imp.columns
    # Check if sorted descending
    assert global_imp['importance'].iloc[0] >= global_imp['importance'].iloc[-1]
