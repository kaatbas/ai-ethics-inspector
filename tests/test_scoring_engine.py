import pytest
from src.scoring.ahp import AHPScorer
from src.scoring.engine import EthicsScoringEngine

def test_scoring_engine_flow():
    # Setup AHP
    ahp = AHPScorer() # Default weights (equal if matrix not set, actually AHPScorer init is ones so equal)
    ahp.calculate_weights()
    
    engine = EthicsScoringEngine(ahp)
    
    # Mock Metrics
    import pandas as pd
    fairness_metrics = {'statistical_parity_difference': 0.05} # Small bias -> high score (100 - 25 = 75)
    transparency_metrics = {'is_mock': False, 'global_importance': pd.DataFrame({'feature': ['a'], 'importance': [0.1]})} # Real -> 100
    
    raw_scores = engine.calculate_raw_score(fairness_metrics, transparency_metrics)
    
    assert raw_scores['Transparency'] == 100.0
    assert raw_scores['Fairness'] == 75.0 # 100 - (0.05 * 500) = 75
    
    final_score = engine.calculate_final_score()
    
    # Weights are equal (0.25 each)
    # Score = (75 + 100 + 75 + 75) / 4 = 325 / 4 = 81.25
    # Final (1-5) = 1 + (81.25 / 25) = 1 + 3.25 = 4.25
    assert 4.0 < final_score < 4.5

def test_scoring_engine_mock_transparency():
    ahp = AHPScorer()
    ahp.calculate_weights()
    engine = EthicsScoringEngine(ahp)
    
    fairness_metrics = {'statistical_parity_difference': 0.0} # Perfect -> 100
    transparency_metrics = {'is_mock': True} # Mock -> 50
    
    engine.calculate_raw_score(fairness_metrics, transparency_metrics)
    final_score = engine.calculate_final_score()
    
    # Scores: F=100, T=50, P=75, A=75
    # Avg = 300 / 4 = 75
    # Rating = 1 + (75/25) = 4.0
    assert final_score == pytest.approx(4.0)
