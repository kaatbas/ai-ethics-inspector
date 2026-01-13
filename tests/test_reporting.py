import pytest
import os
import pandas as pd
from src.reporting.generator import EthicsReportPDF

def test_pdf_creation():
    """Test if PDF report is generated successfully."""
    
    # Mock Data matching app.py structure
    metrics = {
        'fairness': {'statistical_parity_difference': 0.1},
        'transparency': {'feature_importance': pd.DataFrame({'feature': ['A', 'B'], 'importance': [0.5, 0.3]})},
        'similarity_score': 85.0, # Top level key
        'sim_bias_detected': False
    }
    weights = {'Fairness': 5, 'Transparency': 3, 'Similarity': 7}
    final_score = 4.2
    
    config = {
        'sensitive_attributes': ['Sex'],
        'model_name': 'TestModel'
    }
    
    pdf = EthicsReportPDF(lang='en')
    
    # Should not raise error
    try:
        pdf_bytes = pdf.generate(metrics, weights, final_score, config)
    except Exception as e:
        pytest.fail(f"PDF Generation failed: {e}")
        
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    # Check PDF Header Signature (starts with %PDF)
    assert pdf_bytes.startswith(b'%PDF')

def test_turkish_character_sanitization():
    """Test if Turkish characters are handled without error."""
    pdf = EthicsReportPDF(lang='tr')
    
    # Identify a string with special chars
    test_str = "Şeffaflık ve Adillik"
    
    # Internal sanitize check
    cleaned = pdf.sanitize(test_str)
    assert "Seffaflik" in cleaned
    assert "Ş" not in cleaned 
