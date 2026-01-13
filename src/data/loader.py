from ucimlrepo import fetch_ucirepo
import pandas as pd

def load_german_data():
    """
    Fetches the German Credit Data from UCI Repository.
    
    Returns:
        tuple: (X, y) where X is the features DataFrame and y is the target Series/DataFrame.
    """
    try:
        # fetch dataset 
        german_credit_data = fetch_ucirepo(id=144) 
          
        # data (as pandas dataframes) 
        X = german_credit_data.data.features 
        y = german_credit_data.data.targets 
        
        # Rename columns to human readable format
        column_map = {
            'Attribute1': 'checking_status',
            'Attribute2': 'duration',
            'Attribute3': 'credit_history',
            'Attribute4': 'purpose',
            'Attribute5': 'credit_amount',
            'Attribute6': 'savings_status',
            'Attribute7': 'employment',
            'Attribute8': 'installment_rate',
            'Attribute9': 'personal_status_sex',
            'Attribute10': 'other_debtors',
            'Attribute11': 'residence_since',
            'Attribute12': 'property',
            'Attribute13': 'age',
            'Attribute14': 'other_payment_plans',
            'Attribute15': 'housing',
            'Attribute16': 'existing_credits',
            'Attribute17': 'job',
            'Attribute18': 'num_dependents',
            'Attribute19': 'telephone',
            'Attribute20': 'foreign_worker'
        }
        
        # Check if columns are indeed Attribute1 etc before renaming to avoid double renaming or errors
        # UCIMLRepo might sometimes return names if updated
        if 'Attribute1' in X.columns:
            X = X.rename(columns=column_map)
          
        return X, y
    except Exception as e:
        raise RuntimeError(f"Failed to fetch German Credit Data: {e}")
