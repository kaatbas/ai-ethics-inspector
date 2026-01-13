import pandas as pd
import numpy as np
from src.scoring.ahp import AHPScorer

class EthicsScoringEngine:
    """
    Main engine to calculate the final Ethical Compliance Score.
    Integrates Fairness, Transparency, Privacy, and Accountability scores based on AHP weights.
    """
    
    def __init__(self, ahp_scorer: AHPScorer):
        self.ahp_scorer = ahp_scorer
        self.scores = {}
        
    def calculate_raw_score(self, fairness_metrics, transparency_metrics):
        """
        Normalizes individual module outputs into 0-100 scales.
        
        Args:
            fairness_metrics (dict): Output from src.ethics.fairness
            transparency_metrics (dict): Output from src.ethics.transparency
            
        Returns:
            dict: {criterion: raw_score (0-100)}
        """
        # --- Fairness Score Calculation ---
        # Ideal Statistical Parity Difference is 0. 
        # We penalize deviation. Let's map |SPD| > 0.2 to 0 score, 0.0 to 100.
        spd = abs(fairness_metrics.get('statistical_parity_difference', 1.0))
        fairness_score = max(0, 100 - (spd * 500)) # Simple linear penalty, strict.
        
        # --- Transparency Score Calculation ---
        # If we have SHAP values (not mock), we give high score.
        # If global importance exists and is not empty, good.
        # We can detect if it's mock or real.
        is_mock = transparency_metrics.get('is_mock', False)
        
        if is_mock:
            transparency_score = 50.0 # Penalty for not having real explanations
        else:
            # Check if we have meaningful feature importance
            imp = transparency_metrics.get('global_importance')
            if imp is not None and not imp.empty:
                transparency_score = 100.0 
            else:
                transparency_score = 0.0
                
        # --- Placeholders for Privacy & Accountability (Pending Implementation) ---
        privacy_score = 75.0 # Placeholder
        accountability_score = 75.0 # Placeholder
        
        self.scores = {
            "Fairness": fairness_score,
            "Transparency": transparency_score,
            "Privacy": privacy_score,
            "Accountability": accountability_score
        }
        return self.scores

    def calculate_final_score(self):
        """
        Calculates the final weighted score (1-5 scale).
        """
        if self.ahp_scorer.weights is None:
            self.ahp_scorer.calculate_weights()
            
        weights = self.ahp_scorer.weights
        
        total_score = 0
        
        for criterion, weight in zip(self.ahp_scorer.criteria, weights):
            raw = self.scores.get(criterion, 0)
            total_score += raw * weight
            
        # Convert 0-100 to 1-5
        # 0 -> 1
        # 100 -> 5
        # Formula: 1 + (Score / 25) ? No.
        # 0 -> 1, 25 -> 2, 50 -> 3, 75 -> 4, 100 -> 5
        final_rating = 1 + (total_score / 25)
        
        return min(5.0, max(1.0, final_rating))
