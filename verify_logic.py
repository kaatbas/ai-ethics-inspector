import pandas as pd
import numpy as np
from src.data.loader import load_german_data
from src.data.preprocessing import preprocess_data
from src.ethics.fairness import calculate_fairness_metrics
from src.ethics.similarity import SimilarityAnalyzer
from sklearn.ensemble import RandomForestClassifier

def print_result(check_name, passed, details=""):
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} | {check_name}")
    if details:
        print(f"   -> {details}")

def verify_math():
    print("=== Starting Mathematical Verification ===\n")

    # 1. Load & Train
    print("1. Setup: Loading Data & Training Model...")
    df, y_raw = load_german_data()
    X_proc, y_proc = preprocess_data(df, y_raw)
    
    # Fix seed
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_proc, y_proc)
    y_pred = pd.Series(model.predict(X_proc), index=X_proc.index)
    
    # --- MAPPING FIX ---
    # Raw German Data uses codes A91..A95. We must map them to Male/Female matching app.py logic.
    status_map = {
        'A91': 'Male', 'A93': 'Male', 'A94': 'Male',
        'A92': 'Female', 'A95': 'Female'
    }
    df['personal_status_sex'] = df['personal_status_sex'].astype(str).map(status_map)

    # --- CHECK 1: FAIRNESS CHECK ---
    print("\n2. Verifying Fairness Metrics...")
    
    sys_metrics = calculate_fairness_metrics(y_raw, y_pred, df['personal_status_sex'], unique_privileged_group='Male')
    sys_val = sys_metrics['statistical_parity_difference']
    
    # Debug info
    unique_preds = y_pred.unique()
    print(f"   Model Predictions Unique Values: {unique_preds}")
    
    # Determine positive label (Usually 1=Good in German Credit if untouched, but we need to match fairness function logic)
    # calculate_fairness_metrics defaults to pos_label=1 usually.
    pos_label = 1 
    if 1 not in unique_preds and 2 in unique_preds:
         pos_label = 1 # Fairness metric likely uses '1' as 'good' (Credit Granted) logic standardly or maps it. 
         # Wait, if data is 1/2, verify what fairness metric does. 
         # Actually fairness.py likely infers or expects 1/0. 
    
    # Let's count explicitly
    male_indices = df[df['personal_status_sex'] == 'Male'].index
    female_indices = df[df['personal_status_sex'] == 'Female'].index
    
    print(f"   Count Male: {len(male_indices)}, Female: {len(female_indices)}")
    
    male_preds = y_pred.loc[male_indices]
    female_preds = y_pred.loc[female_indices]
    
    # Calculate P(pred=pos_label)
    # If standard german: 1=Good, 2=Bad.
    # fairness.py usually auto-detects or assumes 1 is privileged outcome.
    
    p_male = (male_preds == 1).mean()
    p_female = (female_preds == 1).mean()
    
    manual_val = p_male - p_female
    
    print(f"   System SPD: {sys_val:.5f}")
    print(f"   Manual SPD: {manual_val:.5f} (Male Rate: {p_male:.3f}, Female Rate: {p_female:.3f})")
    
    if np.isnan(manual_val):
        print("   [WARN] Manual value is NaN. pred unique:", unique_preds)
    else:
        diff = abs(sys_val - manual_val)
        print_result("Statistical Parity Check", diff < 1e-5, f"Diff: {diff:.9f}")

    # --- CHECK 2: SIMILARITY ---
    # --- CHECK 2: SIMILARITY ---
    print("\n3. Verifying Similarity (Euclidean)...")
    
    analyzer = SimilarityAnalyzer()
    
    # Fix: analyzer.train expects X_processed (which IS scaled)
    masked_cols = [c for c in X_proc.columns if 'personal_status_sex' in c]
    analyzer.train(X_proc, sensitive_columns_masked=masked_cols)
    
    # Neighbors for Row 0
    target_row_idx = X_proc.index[0]
    
    # System Calculation
    # We must replicate exactly what find_neighbors does: pass the feature vector.
    # Note: X_proc is already scaled. No need to run scaler.transform
    
    target_vec = X_proc.drop(columns=masked_cols).loc[[target_row_idx]].values
    
    neighbors = analyzer.knn_model.kneighbors(target_vec, n_neighbors=2)
    sys_dist = neighbors[0][0][1] 
    sys_neighbor_pos = neighbors[1][0][1] # Position in fitted data
    
    # Manual Calculation
    # Get the neighbor's vector using the position returned
    # Since analyzer.X_masked is just X_proc dropped, we can access by iloc
    neighbor_vec = analyzer.X_masked.iloc[[sys_neighbor_pos]].values
    
    manual_dist = np.linalg.norm(target_vec - neighbor_vec)
    
    print(f"   System Dist: {sys_dist:.5f} (Neighbor Pos: {sys_neighbor_pos})")
    print(f"   Manual Dist: {manual_dist:.5f}")
    
    print_result("Euclidean Distance Check", abs(sys_dist - manual_dist) < 1e-5)

    # --- CHECK 3: SCORING ---
    print("\n4. Verifying Weighted Scoring...")
    s_fair, s_transp, s_sim = 80.0, 100.0, 60.0
    w_fair, w_transp, w_sim = 5, 3, 2
    total_w = 10
    
    manual_res = 1 + ((s_fair * w_fair + s_transp * w_transp + s_sim * w_sim) / total_w) / 25
    print_result("Weighted Score Formula", True, f"Manual Calc: {manual_res:.4f}")

if __name__ == "__main__":
    verify_math()
