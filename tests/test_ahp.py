import pytest
import numpy as np
from src.scoring.ahp import AHPScorer

def test_ahp_initialization():
    scorer = AHPScorer(criteria=['A', 'B', 'C'])
    assert scorer.num_criteria == 3
    assert scorer.weights is None

def test_weight_calculation_equal_importance():
    # If all criteria are equally important
    scorer = AHPScorer(criteria=['A', 'B', 'C'])
    matrix = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    scorer.set_comparison_matrix(matrix)
    weights = scorer.calculate_weights()
    
    assert weights['A'] == pytest.approx(1/3)
    assert weights['B'] == pytest.approx(1/3)
    assert weights['C'] == pytest.approx(1/3)
    assert scorer.get_consistency_ratio() < 0.1

def test_weight_calculation_known_scenario():
    # A is much more important than B (3) and C (5). B is more than C (2).
    # Common AHP example
    #      A    B    C
    # A    1    3    5
    # B   1/3   1    2
    # C   1/5  1/2   1
    
    scorer = AHPScorer(criteria=['A', 'B', 'C'])
    matrix = [
        [1,    3,    5],
        [1/3,  1,    2],
        [1/5, 1/2,   1]
    ]
    scorer.set_comparison_matrix(matrix)
    weights = scorer.calculate_weights()
    
    # Expected: A > B > C
    assert weights['A'] > weights['B']
    assert weights['B'] > weights['C']
    
    # Sum should be 1
    total = sum(weights.values())
    assert total == pytest.approx(1.0)
    
    # Consistency check logic (approximation)
    # This matrix is reasonably consistent
    assert scorer.get_consistency_ratio() < 0.1

def test_invalid_matrix_shape():
    scorer = AHPScorer(criteria=['A', 'B'])
    matrix = [[1, 1, 1], [1, 1, 1]] # 2x3 instead of 2x2
    
    with pytest.raises(ValueError):
        scorer.set_comparison_matrix(matrix)
