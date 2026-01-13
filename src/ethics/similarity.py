import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class SimilarityAnalyzer:
    """
    Analyzes neighborhood similarity to detect individual discrimination.
    """
    
    def __init__(self):
        self.knn_model = None
        self.X_masked = None
        self.scaler = StandardScaler()
        
    def train(self, X_processed, sensitive_columns_masked):
        """
        Trains KNN on the dataset EXCLUDING the sensitive columns.
        
        Args:
            X_processed (pd.DataFrame): The fully processed (numeric) dataset.
            sensitive_columns_masked (list): List of column names in X_processed 
                                             that should be ignored for similarity calculation.
        """
        # Drop sensitive columns from the features used for distance calculation
        # We need to ensure we drop all OHE columns related to the sensitive feature if they exist
        cols_to_drop = [c for c in X_processed.columns if c in sensitive_columns_masked]
        
        self.X_masked = X_processed.drop(columns=cols_to_drop)
        
        # Scale the data for KNN (Euclidean distance requires scaling)
        # Note: X_processed might already be scaled, but safe to ensure or just use as is if known scaled.
        # Given our pipeline in preprocessing.py scales numeric but OHE is 0/1.
        # It's better to rescale everything to 0-1 or similar range so OHE doesn't dominate or vanish.
        # For simplicity here, we assume X_processed is already reasonably scaled or we trust the workflow.
        
        self.knn_model = NearestNeighbors(n_neighbors=10, algorithm='auto')
        self.knn_model.fit(self.X_masked)
        
    def find_neighbors(self, target_idx, n_neighbors=5):
        """
        Finds neighbors for a specific row index.
        
        Args:
            target_idx (int): Index of the target person in the dataset.
            n_neighbors (int): Number of neighbors to find.
            
        Returns:
            tuple: (distances, indices)
        """
        if self.knn_model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Get target feature vector
        target_features = self.X_masked.iloc[target_idx].values.reshape(1, -1)
        
        distances, indices = self.knn_model.kneighbors(target_features, n_neighbors=n_neighbors)
        return distances[0], indices[0]

    def find_all_similar_pairs(self, n_neighbors=2, distance_threshold=0.5):
        """
        Finds all pairs/groups of individuals who are highly similar.
        Returns unique pairs (i, j) where distance(i, j) < threshold.
        
        Args:
            n_neighbors (int): How many neighbors to check per point.
            distance_threshold (float): Similarity cutoff. Lower = Strict identity.
            
        Returns:
            pd.DataFrame: Columns ['Person A', 'Person B', 'Distance']
        """
        if self.knn_model is None:
            raise ValueError("Model not trained.")
            
        # Find neighbors for everyone
        # This returns (n_samples, n_neighbors)
        distances, indices = self.knn_model.kneighbors(self.X_masked, n_neighbors=n_neighbors)
        
        pairs = []
        
        # indices[:, 0] is the point itself (dist 0), so look at others
        for i in range(len(self.X_masked)):
            for k in range(1, n_neighbors): # Start from 1 to skip self
                neighbor_idx = indices[i, k]
                dist = distances[i, k]
                
                if dist < distance_threshold:
                    # Avoid duplicates (pair 2-5 is same as 5-2)
                    p1, p2 = sorted((i, neighbor_idx))
                    pairs.append((p1, p2, dist))
                    
        # Dedup list
        pairs = sorted(list(set(pairs)))
        
        return pd.DataFrame(pairs, columns=['Person A', 'Person B', 'Distance'])

    def analyze_neighborhood_bias(self, target_idx, neighbor_indices, y_pred, sensitive_values):
        """
        Analyzes the outcomes within the neighborhood grouped by sensitive value.
        
        Args:
            target_idx (int): The target person's index.
            neighbor_indices (array): Indices of the neighbors.
            y_pred (pd.Series): Predictions for the whole dataset.
            sensitive_values (pd.Series): The sensitive attribute (e.g., 'Male', 'Female') for the dataset.
            
        Returns:
            dict: Analysis results containing local approval rates by group.
        """
        # Combine target and neighbors
        # We verify if target is in neighbors (KNN usually implies yes if k>=1, but distance 0)
        # If target is not in neighbor_indices (e.g. user queried separately), add them.
        all_indices = np.unique(np.append(neighbor_indices, target_idx))
        
        # Extract data for this group
        group_preds = y_pred.iloc[all_indices]
        group_sensitive = sensitive_values.iloc[all_indices]
        
        # Create a small frame for aggregation
        analysis_df = pd.DataFrame({
            'Prediction': group_preds,
            'Sensitive': group_sensitive
        })
        
        # Calculate approval rate by group
        # e.g. Male: 0.8, Female: 0.2
        stats = analysis_df.groupby('Sensitive')['Prediction'].mean().to_dict()
        counts = analysis_df['Sensitive'].value_counts().to_dict()
        
        return {
            'stats': stats,
            'counts': counts,
            'consistency_score': 1.0 - group_preds.std(), # 0 variance means high consistency (1.0)
            'target_prediction': y_pred.iloc[target_idx],
            'target_sensitive': sensitive_values.iloc[target_idx]
        }
