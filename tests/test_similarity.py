import pytest
import pandas as pd
import numpy as np
from src.ethics.similarity import SimilarityAnalyzer

def test_similarity_masking():
    """Test if the analyzer correctly ignores masking columns."""
    # Data: 3 people
    # P1: Skill=10, Gender=0 (Target)
    # P2: Skill=10, Gender=1 (Ideally identical to P1 if Gender masked)
    # P3: Skill=0,  Gender=0 (Different)
    
    df = pd.DataFrame({
        'Skill': [10, 10, 0],
        'Gender': [0, 1, 0]
    })
    
    analyzer = SimilarityAnalyzer()
    
    # Train masking "Gender"
    # We expect P1 to be closest to P2 (dist 0), then P3 (dist 10)
    analyzer.train(df, sensitive_columns_masked=['Gender'])
    
    dist, indices = analyzer.find_neighbors(target_idx=0, n_neighbors=2)
    
    # Indices should be [0, 1] (Self and P2)
    assert 1 in indices
    assert 2 not in indices # P3 is far away
    
    # Distance to P2 should be 0 (since Skill is same, Gender ignored)
    p2_idx_in_subset = list(indices).index(1)
    assert dist[p2_idx_in_subset] == 0.0

def test_neighborhood_bias_analysis():
    """Test bias stats calculation."""
    analyzer = SimilarityAnalyzer()
    
    # Mock data references
    y_pred = pd.Series([1, 1, 0, 0, 1]) # 5 people
    sensitive = pd.Series(['M', 'M', 'F', 'F', 'M'])
    
    # Say we found neighbors [0, 1, 2, 3] for target 0
    # Group:
    # 0: M, 1 (Target)
    # 1: M, 1
    # 2: F, 0
    # 3: F, 0
    
    # Males: 2/2 = 1.0 acceptance
    # Females: 0/2 = 0.0 acceptance
    # High bias!
    
    stats = analyzer.analyze_neighborhood_bias(0, np.array([1, 2, 3]), y_pred, sensitive)
    
    assert stats['stats']['M'] == 1.0
    assert stats['stats']['F'] == 0.0
    assert stats['counts']['M'] == 2 # Target(0) + neighbor(1)
    assert stats['counts']['F'] == 2
