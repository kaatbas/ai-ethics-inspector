import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.loader import load_german_data
from src.data.preprocessing import preprocess_data
from src.ethics.fairness import calculate_fairness_metrics
from src.ethics.transparency import generate_explanations
from src.ethics.similarity import SimilarityAnalyzer
from src.scoring.ahp import AHPScorer
from src.scoring.engine import EthicsScoringEngine
from src.ui.translations import get_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page Config
st.set_page_config(page_title="AEI: AI Ethics Inspector", page_icon="🕵️", layout="wide")

st.markdown("""
<style>
    .step-header { color: #1565C0; font-size: 24px; font-weight: bold; margin-bottom: 20px; }
    .stButton>button { width: 100%; height: 50px; font-weight: 600; }
    .success-box { padding: 15px; background-color: #d4edda; border-color: #c3e6cb; border-radius: 5px; color: #155724; }
</style>
""", unsafe_allow_html=True)

# Session State
if 'step' not in st.session_state: st.session_state.step = 1
if 'df_raw' not in st.session_state: st.session_state.df_raw = None
if 'lang' not in st.session_state: st.session_state.lang = 'en'

def next_step(): st.session_state.step += 1
def restart(): 
    # Preserve language setting
    current_lang = st.session_state.lang
    st.session_state.clear()
    st.session_state.step = 1
    st.session_state.lang = current_lang

# --- SIDEBAR ---
with st.sidebar:
    # Ultra-Minimal Language Toggle
    # Logic: Show the button to switch to the OTHER language
    current_lang = st.session_state.lang
    toggle_label = "🌐 TR" if current_lang == 'en' else "🌐 EN"
    
    if st.button(toggle_label, help="Switch Language / Dili Değiştir"):
        st.session_state.lang = 'tr' if current_lang == 'en' else 'en'
        st.rerun()

    lang = st.session_state.lang # Wrapper var for easy usage
    
    st.title(get_text(lang, 'app_title'))
    st.info(get_text(lang, 'step_progress').format(st.session_state.step))
    st.progress(st.session_state.step / 7)
    
    if st.button(get_text(lang, 'start_over')): 
        restart()
        st.rerun()

# --- STEP 1: DATASET LOADING ---
if st.session_state.step == 1:
    st.markdown(f'<div class="step-header">{get_text(lang, "s1_title")}</div>', unsafe_allow_html=True)
    st.write(get_text(lang, 's1_desc'))
    
    ds_options = [get_text(lang, 's1_ds_default'), get_text(lang, 's1_ds_upload')]
    source = st.radio(get_text(lang, 's1_datasource'), ds_options)
    
    if source == ds_options[0]:
        # CASE 1: Default German Credit Data
        if st.button(get_text(lang, 's1_btn')):
            with st.spinner(get_text(lang, 's1_loading')):
                try:
                    df, y = load_german_data()
                    st.session_state.df_raw = df
                    st.session_state.y_raw = y
                    
                    # Preprocess & Train
                    X_proc, y_proc = preprocess_data(df, y)
                    st.session_state.X_processed = X_proc
                    
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                    model.fit(X_proc, y_proc)
                    st.session_state.model = model
                    st.session_state.y_pred = pd.Series(model.predict(X_proc), index=X_proc.index)
                    
                    st.success(get_text(lang, 's1_success'))
                    next_step()
                    st.rerun()
                except Exception as e:
                    st.error(get_text(lang, 's1_err').format(e))

    else:
        # CASE 2: Upload CSV
        uploaded_file = st.file_uploader(get_text(lang, 's1_upload_label'), type=['csv'])
        
        if uploaded_file:
            try:
                df_temp = pd.read_csv(uploaded_file)
                st.write("Preview:", df_temp.head(3))
                
                target_col = st.selectbox(get_text(lang, 's1_target_col'), df_temp.columns)
                
                if st.button(get_text(lang, 's1_btn_upload')):
                    with st.spinner(get_text(lang, 's1_loading')):
                        # Split Target
                        y = df_temp[target_col]
                        df = df_temp.drop(columns=[target_col])
                        
                        st.session_state.df_raw = df
                        st.session_state.y_raw = y
                        
                        # Preprocess & Train (Generic)
                        X_proc, y_proc = preprocess_data(df, y)
                        st.session_state.X_processed = X_proc
                        
                        model = RandomForestClassifier(n_estimators=50, random_state=42)
                        model.fit(X_proc, y_proc)
                        st.session_state.model = model
                        st.session_state.y_pred = pd.Series(model.predict(X_proc), index=X_proc.index)
                        
                        st.success(get_text(lang, 's1_success'))
                        next_step()
                        st.rerun()

            except Exception as e:
                st.error(get_text(lang, 's1_err').format(e))

# --- STEP 2: READABLE FORMAT ---
elif st.session_state.step == 2:
    st.markdown(f'<div class="step-header">{get_text(lang, "s2_title")}</div>', unsafe_allow_html=True)
    st.write(get_text(lang, 's2_desc'))
    
    st.dataframe(st.session_state.df_raw.head(10))
    st.caption(get_text(lang, 's2_total').format(len(st.session_state.df_raw)))
    
    if st.button(get_text(lang, 's2_next')):
        next_step()
        st.rerun()

# --- STEP 3: ETHICAL FEATURES ---
elif st.session_state.step == 3:
    st.markdown(f'<div class="step-header">{get_text(lang, "s3_title")}</div>', unsafe_allow_html=True)
    st.write(get_text(lang, 's3_desc'))
    
    all_cols = st.session_state.df_raw.columns.tolist()
    defaults = [c for c in all_cols if c in ['personal_status_sex', 'age', 'foreign_worker']]
    selected = st.multiselect(get_text(lang, 's3_label'), all_cols, default=defaults)
    st.session_state.sensitive_features = selected
    
    if st.button(get_text(lang, 's3_next')):
        if not selected:
            st.error(get_text(lang, 's3_err'))
        else:
            # Prepare Analyzer
            analyzer = SimilarityAnalyzer()
            masked_cols = []
            for raw_col in selected:
                if raw_col in st.session_state.X_processed.columns: masked_cols.append(raw_col)
                pfx = f"{raw_col}_"
                masked_cols.extend([c for c in st.session_state.X_processed.columns if c.startswith(pfx)])
            
            analyzer.train(st.session_state.X_processed, sensitive_columns_masked=masked_cols)
            st.session_state.analyzer = analyzer
            
            next_step()
            st.rerun()

# --- STEP 4: SIMILARITY LISTING ---
elif st.session_state.step == 4:
    st.markdown(f'<div class="step-header">{get_text(lang, "s4_title")}</div>', unsafe_allow_html=True)
    st.write(get_text(lang, 's4_desc'))
    
    # Dynamic Threshold Slider
    thresh = st.slider(get_text(lang, 's4_thresh_label'), 0.1, 5.0, 1.5, 0.1, help=get_text(lang, 's4_thresh_help'))
    st.session_state.distance_threshold = thresh
    
    # Threshold Guidance
    if thresh <= 1.0:
        st.success(get_text(lang, 's4_guide_strict'))
    elif thresh <= 3.0:
        st.warning(get_text(lang, 's4_guide_bal'))
    else:
        st.error(get_text(lang, 's4_guide_loose'))
        
    if st.button(get_text(lang, 's4_recalc')):
        # Force re-calc
        if 'similar_pairs' in st.session_state: del st.session_state.similar_pairs
        if 'pairs_df' in st.session_state: del st.session_state.pairs_df

    with st.spinner(get_text(lang, 's4_spinner')):
        if 'similar_pairs' not in st.session_state:
            analyzer = st.session_state.analyzer
            # User feedback: Dynamic Threshold
            pairs_df = analyzer.find_all_similar_pairs(n_neighbors=2, distance_threshold=thresh)
            st.session_state.pairs_df = pairs_df
    
    pairs_df = st.session_state.pairs_df
    st.metric(get_text(lang, 's4_found'), len(pairs_df))
    st.dataframe(pairs_df)
    
    if not pairs_df.empty:
        st.write(get_text(lang, 's4_inspect'))
        pair_idx = st.selectbox(get_text(lang, 's4_select_pair'), pairs_df.index)
        row = pairs_df.loc[pair_idx]
        p1, p2 = int(row['Person A']), int(row['Person B'])
        
        comp_df = st.session_state.df_raw.iloc[[p1, p2]].copy()
        comp_df['Model Prediction'] = st.session_state.y_pred.iloc[[p1, p2]].values
        
        # Highlight ONLY sensitive columns as requested by user
        # Used Orange (#FFA500) with Black text for high visibility in Dark Mode
        highlight_cols = st.session_state.sensitive_features
        
        def highlight_sensitive(s):
            return ['background-color: #FFA500; color: #000000; font-weight: bold' if s.name in highlight_cols else '' for v in s]

        st.dataframe(comp_df.style.apply(highlight_sensitive, axis=0))
        
        if comp_df['Model Prediction'].iloc[0] != comp_df['Model Prediction'].iloc[1]:
            st.error(get_text(lang, 's4_discordant'))
        else:
            st.success(get_text(lang, 's4_consistent'))

    if st.button(get_text(lang, 's4_next')):
        next_step()
        st.rerun()

# --- STEP 5: METRICS CONFIG ---
elif st.session_state.step == 5:
    st.markdown(f'<div class="step-header">{get_text(lang, "s5_title")}</div>', unsafe_allow_html=True)
    st.write(get_text(lang, 's5_desc'))
    
    c1, c2, c3 = st.columns(3)
    w_fair = c1.slider(get_text(lang, 's5_w_fair'), 1, 9, 5)
    w_transp = c2.slider(get_text(lang, 's5_w_transp'), 1, 9, 3)
    w_sim = c3.slider(get_text(lang, 's5_w_sim'), 1, 9, 7)
    
    st.session_state.ahp_weights = {'Fairness': w_fair, 'Transparency': w_transp, 'Similarity': w_sim}
    
    if st.button(get_text(lang, 's5_next')):
        next_step()
        st.rerun()

# --- STEP 6: ANALYSIS RESULTS ---
elif st.session_state.step == 6:
    st.markdown(f'<div class="step-header">{get_text(lang, "s6_title")}</div>', unsafe_allow_html=True)
    
    # 1. Fairness
    if st.session_state.sensitive_features:
        sens_col = st.session_state.sensitive_features[0]
        sens_data = st.session_state.df_raw[sens_col]
        fairness_metrics = calculate_fairness_metrics(
             st.session_state.y_raw, st.session_state.y_pred, sens_data, unique_privileged_group='Male' 
        )
    else:
        fairness_metrics = {'statistical_parity_difference': 0, 'disparate_impact': 1}

    # 2. Transparency
    transp_metrics = generate_explanations(st.session_state.model, st.session_state.X_processed[:50])
    
    # VISUALIZE TRANSPARENCY
    if 'feature_importance' in transp_metrics:
        st.subheader(get_text(lang, 's6_transp_header'))
        st.write(get_text(lang, 's6_transp_desc'))
        
        fi_df = transp_metrics['feature_importance']
        # Take Top 10 for clarity
        top_fi = fi_df.head(10).set_index('feature')
        st.bar_chart(top_fi)

    # 3. Pairwise Similarity Bias
    pairs = st.session_state.pairs_df
    sim_bias_detected = False
    
    if not pairs.empty and st.session_state.sensitive_features:
        st.subheader(get_text(lang, 's6_bias_header'))
        st.write(get_text(lang, 's6_bias_desc').format(len(pairs)))
        
        feature_scores = []
        
        for sens_feat in st.session_state.sensitive_features:
            with st.expander(get_text(lang, 's6_expand_title').format(sens_feat), expanded=True):
                discordant_counts = {'Same Group': 0, 'Different Group': 0}
                total_counts = {'Same Group': 0, 'Different Group': 0}
                discordant_pairs_detail = [] # Store details
                
                for idx, row in pairs.iterrows():
                    p1, p2 = int(row['Person A']), int(row['Person B'])
                    
                    v1 = st.session_state.df_raw.iloc[p1][sens_feat]
                    v2 = st.session_state.df_raw.iloc[p2][sens_feat]
                    is_same_group = (v1 == v2)
                    
                    out1 = st.session_state.y_pred.iloc[p1]
                    out2 = st.session_state.y_pred.iloc[p2]
                    is_discordant = (out1 != out2)
                    
                    key = 'Same Group' if is_same_group else 'Different Group'
                    total_counts[key] += 1
                    if is_discordant:
                        discordant_counts[key] += 1
                        if not is_same_group:
                            # Capture detail
                             discordant_pairs_detail.append({
                                 'Pair ID': idx,
                                 f'{sens_feat}_A': v1, f'{sens_feat}_B': v2,
                                 'Outcome_A': out1, 'Outcome_B': out2,
                                 'Dist': row['Distance']
                             })
                        
                rate_same = (discordant_counts['Same Group'] / total_counts['Same Group']) if total_counts['Same Group'] > 0 else 0
                rate_diff = (discordant_counts['Different Group'] / total_counts['Different Group']) if total_counts['Different Group'] > 0 else 0
                
                feat_score = 100 * (1.0 - rate_diff)
                feature_scores.append(feat_score)
                
                c1, c2, c3 = st.columns(3)
                c1.metric(get_text(lang, 's6_same_group'), f"{rate_same:.1%}", get_text(lang, 's6_total_pairs').format(total_counts['Same Group']))
                c2.metric(get_text(lang, 's6_diff_group'), f"{rate_diff:.1%}", get_text(lang, 's6_total_pairs').format(total_counts['Different Group']), delta_color="inverse")
                
                if rate_diff > (rate_same + 0.1) or (rate_same == 0 and rate_diff > 0.1): 
                    st.error(get_text(lang, 's6_err_bias').format(sens_feat, sens_feat))
                    sim_bias_detected = True
                    c3.metric(get_text(lang, 's6_bias_sev'), get_text(lang, 's6_high'))
                else:
                    st.success(get_text(lang, 's6_success_bias').format(sens_feat))
                    c3.metric(get_text(lang, 's6_bias_sev'), get_text(lang, 's6_low'))
                
                # Show Drilldown View
                if discordant_pairs_detail:
                    st.markdown(f"**{get_text(lang, 's6_show_details')}**")
                    st.dataframe(pd.DataFrame(discordant_pairs_detail))

        sim_score = sum(feature_scores) / len(feature_scores) if feature_scores else 100
            
    else:
        sim_score = 100
        st.warning(get_text(lang, 's6_warn_nopairs'))
        
    st.session_state.metrics = {
        'fairness': fairness_metrics,
        'transparency': transp_metrics,
        'similarity_score': sim_score,
        'sim_bias_detected': sim_bias_detected
    }
    
    st.success(get_text(lang, 's6_calculated'))
    if st.button(get_text(lang, 's6_next')):
        next_step()
        st.rerun()

# --- STEP 7: FINAL REPORT ---
elif st.session_state.step == 7:
    st.markdown(f'<div class="step-header">{get_text(lang, "s7_title")}</div>', unsafe_allow_html=True)
    st.balloons()
    
    metrics = st.session_state.metrics
    weights = st.session_state.ahp_weights
    total_w = sum(weights.values())
    
    s_fair = max(0, 100 - abs(metrics['fairness'].get('statistical_parity_difference',0)*500))
    s_transp = 50 if metrics['transparency'].get('is_mock') else 100
    s_sim = metrics['similarity_score']
    
    final_raw = (s_fair * weights['Fairness'] + s_transp * weights['Transparency'] + s_sim * weights['Similarity']) / total_w
    final_5 = 1 + (final_raw / 25)
    
    c1, c2 = st.columns([1, 2])
    with c1:
        # Fixed: Dark Navy Background (#1E3A5F) with White Text (#FFFFFF) for proper Dark Mode feel
        st.markdown(f"""
        <div style="background:#1E3A5F; color:#FFFFFF; padding:20px; border-radius:10px; text-align:center; border: 1px solid #3B5B85;">
            <h3 style="color:#B3E5FC; margin:0;">{get_text(lang, 's7_final_score')}</h3>
            <h1 style="color:#FFFFFF; font-size:60px; margin:10px 0;">{final_5:.2f}</h1>
            <p style="color:#B3E5FC; margin:0;">/ 5.0</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.write(get_text(lang, 's7_breakdown'))
        
        # Translated DataFrame
        score_data = {
            get_text(lang, 's7_col_metric'): [
                get_text(lang, 's7_m_fair'), 
                get_text(lang, 's7_m_transp'), 
                get_text(lang, 's7_m_sim')
            ],
            get_text(lang, 's7_col_score'): [s_fair, s_transp, s_sim],
            get_text(lang, 's7_col_weight'): [weights['Fairness'], weights['Transparency'], weights['Similarity']]
        }
        st.dataframe(pd.DataFrame(score_data))
        
    st.write(get_text(lang, 's7_summary_roi'))
    st.markdown(get_text(lang, 's7_summary_text').format(
        final_5, 
        100 - s_sim, 
        metrics['fairness'].get('statistical_parity_difference',0)
    ))
    
    # PDF Report Button
    from src.reporting.generator import EthicsReportPDF
    
    report_pdf = EthicsReportPDF(lang=lang)
    pdf_bytes = report_pdf.generate(
        metrics=metrics, 
        weights=weights, 
        final_score=final_5, 
        config={'sensitive_features': st.session_state.sensitive_features}
    )
    
    label = "📄 Download PDF Report" if lang == 'en' else "📄 PDF Raporunu İndir"
    
    st.download_button(
        label=label,
        data=pdf_bytes,
        file_name="ethics_report.pdf",
        mime="application/pdf"
    )
    
    st.markdown("---")
    if st.button(get_text(lang, 's7_restart')):
        restart()
        st.rerun()
