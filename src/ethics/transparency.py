import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_explanations(model, X_sample):
    """
    Generates global feature importance using RandomForest's native feature_importances_.
    Replaces SHAP due to DLL/OS security blocking issues.
    
    Args:
        model: Trained scikit-learn model (must have feature_importances_ attribute).
        X_sample (pd.DataFrame): Sample data (used for column names).
        
    Returns:
        dict: Contains 'feature_importance' (DataFrame) and 'is_mock' (False).
    """
    try:
        # Check if model has feature_importances_ (RandomForest, DecisionTree, etc.)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = X_sample.columns
            
            # Create DataFrame
            global_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values(by='importance', ascending=False)
            
            logger.info("Successfully generated native feature importances.")
            
            return {
                'feature_importance': global_importance,
                'shap_values': None, # Legacy key for compatibility
                'is_mock': False
            }
        else:
            logger.warning("Model does not support native feature_importances_. Returning mock data.")
            return _generate_mock_explanations(X_sample)

    except Exception as e:
        logger.error(f"Error generating native explanations: {str(e)}")
        return _generate_mock_explanations(X_sample)

def _generate_mock_explanations(X_sample):
    """
    Fallback function to return mock data if everything fails.
    """
    feature_names = X_sample.columns
    # Generate random importance that sums to 1
    mock_importance = np.abs(np.random.randn(len(feature_names)))
    mock_importance /= mock_importance.sum()
    
    global_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mock_importance
    }).sort_values(by='importance', ascending=False)
    
    return {
        'feature_importance': global_importance,
        'shap_values': None, 
        'is_mock': True
    }
