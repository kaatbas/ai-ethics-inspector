import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def preprocess_data(X, y):
    """
    Preprocesses the German Credit Data.
    - Encodes categorical variables (OneHot).
    - Scales numeric variables (StandardScaler).
    - Returns processed DataFrame with readable column names.
    
    Args:
        X (pd.DataFrame): Features.
        y (pd.DataFrame/Series): Target.
        
    Returns:
        tuple: (X_processed_df, y_processed_series)
    """
    # Identify columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # Create transformation pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    # Fit and transform
    X_processed = preprocessor.fit_transform(X)

    # Get feature names for OneHot
    # Note: sklearn < 1.0 logic might differ, but we are on 1.4+ likely
    onehot_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_columns = list(numeric_features) + list(onehot_columns)
    
    X_processed_df = pd.DataFrame(X_processed, columns=all_columns)
    
    # Process target
    # If y is dataframe with 1 column, convert to series
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
        
    # Map target if needed (Good=1, Bad=2 -> 1, 0 or similar is better for models)
    # Usually German Credit: 1=Good, 2=Bad. Let's map to 1=Good, 0=Bad for standard ML
    # But for now, let's keep original or map 2->0 so 1 is positive class.
    # User didn't specify, keeping as is for now or mapping for standard metric usage.
    # Let's just return it clean.
    
    return X_processed_df, y
