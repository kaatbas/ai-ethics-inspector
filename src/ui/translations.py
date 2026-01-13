
TRANSLATIONS = {
    'en': {
        # Sidebar
        'app_title': "🕵️ AI Ethics Inspector",
        'step_progress': "Step {} / 7",
        'start_over': "🔄 Start Over",
        'language': "Language / Dil",
        
        # Step 1
        's1_title': "Step 1: Load Dataset",
        's1_desc': "Welcome to the AI Ethics Inspector. Please load the dataset to begin.",
        's1_btn': "📂 Load German Credit Dataset",
        's1_loading': "Fetching...",
        's1_success': "Data Loaded Successfully!",
        's1_success': "Data Loaded Successfully!",
        's1_err': "Error loading data: {}",
        's1_datasource': "Data Source",
        's1_ds_default': "Sample Data (German Credit)",
        's1_ds_upload': "Upload Your Own (CSV)",
        's1_upload_label': "Upload CSV File",
        's1_target_col': "Target Column (Label)",
        's1_btn_upload': "Process & Train",
        
        # Step 2
        's2_title': "Step 2: Data Inspection",
        's2_desc': "The dataset has been converted to a human-readable format.",
        's2_total': "Total Rows: {}",
        's2_next': "Confirm & Next ➡️",
        
        # Step 3
        's3_title': "Step 3: Select Ethical Features",
        's3_desc': "Select the attributes you want to protect / monitor (Sensitive Attributes).",
        's3_label': "Sensitive Attributes:",
        's3_err': "Please select at least one feature.",
        's3_next': "Next: Run Similarity Search ➡️",
        
        # Step 4
        's4_title': "Step 4: Global Similarity Listing",
        's4_desc': "Listing all pairs of individuals who are **statistically identical** in non-sensitive attributes.",
        's4_thresh_label': "Similarity Threshold (Distance)",
        's4_thresh_help': "Low (0.1-1.0) = Strict 'Twins'. High (2.0+) = 'Similar Peers'.",
        's4_guide_strict': "🟢 Strict (Twins Only)",
        's4_guide_bal': "🟡 Balanced (Similar Peers)",
        's4_guide_loose': "🔴 Loose (Broad Neighborhood)",
        's4_recalc': "🔄 Apply Threshold & Search",
        's4_spinner': "Finding twins...",
        's4_found': "Identical Pairs Found",
        's4_inspect': "### Inspect a Pair",
        's4_select_pair': "Select a Pair to Inspect:",
        's4_discordant': "🚨 Discordant Outcome! These twins got different results.",
        's4_consistent': "✅ Consistent Outcome.",
        's4_next': "Next: Configure Scoring ➡️",

        # Step 5
        's5_title': "Step 5: Scoring Configuration",
        's5_desc': "Configure the weights for the final AHP Ethics Score.",
        's5_w_fair': "Fairness Weight",
        's5_w_transp': "Transparency Weight",
        's5_w_sim': "Similarity Consistency Weight",
        's5_next': "Calculate Scores ➡️",

        # Step 6
        's6_title': "Step 6: Analysis Results",
        's6_bias_header': "👥 Neighborhood Bias Analysis",
        's6_transp_header': "🔍 Transparency Analysis (Feature Importance)",
        's6_transp_desc': "Factors that most influence the model's decisions:",
        's6_bias_desc': "Analyzing {} similar pairs across all selected sensitive features.",
        's6_expand_title': "Analysis: {}",
        's6_same_group': "Same Group Inconsistency",
        's6_diff_group': "Diff Group Inconsistency",
        's6_total_pairs': "Total Pairs: {}",
        's6_bias_sev': "Bias Severity",
        's6_high': "High",
        's6_low': "Low",
        's6_err_bias': "⚠️ **Bias Detected for {}!** People with different {} are significantly more likely to get different results.",
        's6_success_bias': "✅ No significant discrimination found based on {}.",
        's6_warn_nopairs': "Not enough pairs or sensitive features for Pairwise Analysis.",
        's6_show_details': "🔍 Show Discordant Pairs (Diff Group)",
        's6_calculated': "Scores Calculated.",
        's6_next': "Generate Final Report ➡️",

        # Step 7
        's7_title': "Step 7: Final Ethics Report",
        's7_final_score': "Final Score",
        's7_breakdown': "### Breakdown",
        's7_summary_roi': "### Executive Summary",
        's7_summary_text': "The model achieved a score of **{:.2f}**. Similar individuals encountered inconsistent outcomes in **{:.1f}%** of cases. Standard fairness metrics indicate a statistical parity difference of **{:.3f}**.",
        's7_restart': "🔄 Start New Analysis",
        
        # Report Table
        's7_col_metric': "Metric",
        's7_col_score': "Score (0-100)",
        's7_col_weight': "Weight",
        's7_m_fair': "Fairness",
        's7_m_transp': "Transparency",
        's7_m_sim': "Similarity"
    },
    'tr': {
        # Sidebar
        'app_title': "🕵️ Yapay Zeka Etik Denetçisi",
        'step_progress': "Adım {} / 7",
        'start_over': "🔄 Baştan Başla",
        'language': "Dil / Language",
        
        # Step 1
        's1_title': "Adım 1: Veri Setini Yükle",
        's1_desc': "Yapay Zeka Etik Denetçisine hoş geldiniz. Başlamak için lütfen veri setini yükleyin.",
        's1_btn': "📂 Alman Kredi Veri Setini Yükle",
        's1_loading': "Veriler Çekiliyor...",
        's1_success': "Veri Başarıyla Yüklendi!",
        's1_success': "Veri Başarıyla Yüklendi!",
        's1_err': "Veri yükleme hatası: {}",
        's1_datasource': "Veri Kaynağı",
        's1_ds_default': "Örnek Veri (Alman Kredi)",
        's1_ds_upload': "Kendi Verini Yükle (CSV)",
        's1_upload_label': "CSV Dosyası Yükle",
        's1_target_col': "Hedef Sütun (Tahmin Edilecek Değer)",
        's1_btn_upload': "Veriyi İşle ve Eğit",
        
        # Step 2
        's2_title': "Adım 2: Veri İnceleme",
        's2_desc': "Veri seti insanlar tarafından okunabilir formata dönüştürüldü.",
        's2_total': "Toplam Satır: {}",
        's2_next': "Onayla ve İlerle ➡️",
        
        # Step 3
        's3_title': "Adım 3: Etik Özellik Seçimi",
        's3_desc': "Korumak / izlemek istediğiniz özellikleri (Hassas Nitelikler) seçin.",
        's3_label': "Hassas Nitelikler:",
        's3_err': "Lütfen en az bir özellik seçin.",
        's3_next': "İleri: Benzerlik Aramasını Başlat ➡️",
        
        # Step 4
        's4_title': "Adım 4: Küresel Benzerlik Listeleme",
        's4_desc': "Hassas olmayan özelliklerde **istatistiksel olarak tıpatıp aynı** olan birey çiftlerinin listelenmesi.",
        's4_thresh_label': "Benzerlik Eşiği (Mesafe)",
        's4_thresh_help': "Düşük değer (0.1-1.0) = Katı 'Tam İkizler'. Yüksek değer (2.0+) = 'Benzer Arkadaşlar'.",
        's4_guide_strict': "🟢 Çok Katı (Sadece İkizler)",
        's4_guide_bal': "🟡 Dengeli (Benzer Profiller)",
        's4_guide_loose': "🔴 Gevşek (Geniş Komşuluk)",
        's4_recalc': "🔄 Eşiği Uygula ve Tekrar Ara",
        's4_spinner': "İkizler bulunuyor...",
        's4_found': "Bulunan Benzer Çiftler",
        's4_inspect': "### Bir Çifti İncele",
        's4_select_pair': "İncelenecek Çifti Seçin:",
        's4_discordant': "🚨 Uyumsuz Sonuç! Bu ikizler farklı sonuçlar aldı.",
        's4_consistent': "✅ Tutarlı Sonuç.",
        's4_next': "İleri: Skorlamayı Yapılandır ➡️",

        # Step 5
        's5_title': "Adım 5: Skorlama Yapılandırması",
        's5_desc': "Nihai AHP Etik Skoru için ağırlıkları yapılandırın.",
        's5_w_fair': "Adillik Ağırlığı",
        's5_w_transp': "Şeffaflık Ağırlığı",
        's5_w_sim': "Benzerlik Tutarlılığı Ağırlığı",
        's5_next': "Skorları Hesapla ➡️",

        # Step 6
        's6_title': "Adım 6: Analiz Sonuçları",
        's6_bias_header': "👥 Komşuluk Yanlılık Analizi",
        's6_transp_header': "🔍 Şeffaflık Analizi (Özellik Önemi)",
        's6_transp_desc': "Modelin kararlarını en çok etkileyen faktörler aşağıdadır:",
        's6_bias_desc': "Seçilen tüm hassas özellikler üzerinden {} benzer çift analiz ediliyor.",
        's6_expand_title': "Analiz: {}",
        's6_same_group': "Aynı Grup Tutarsızlığı",
        's6_diff_group': "Farklı Grup Tutarsızlığı",
        's6_total_pairs': "Toplam Çift: {}",
        's6_bias_sev': "Yanlılık Şiddeti",
        's6_high': "Yüksek",
        's6_low': "Düşük",
        's6_err_bias': "⚠️ **{} için Yanlılık Tespit Edildi!** Farklı {} değerine sahip kişilerin farklı sonuç alma olasılığı önemli ölçüde daha yüksek.",
        's6_success_bias': "✅ {} bazında önemli bir ayrımcılık bulunamadı.",
        's6_warn_nopairs': "İkili Analiz için yeterli çift veya hassas özellik yok.",
        's6_show_details': "🔍 Uyumsuz Çiftleri Göster (Farklı Grup)",
        's6_calculated': "Skorlar Hesaplandı.",
        's6_next': "Nihai Raporu Oluştur ➡️",

        # Step 7
        's7_title': "Adım 7: Nihai Etik Raporu",
        's7_final_score': "Nihai Skor",
        's7_breakdown': "### Detaylar",
        's7_summary_roi': "### Yönetici Özeti",
        's7_summary_text': "Model **{:.2f}** skoruna ulaştı. Benzer bireyler, vakaların **%{:.1f}**'sinde tutarsız sonuçlarla karşılaştı. Standart adillik metrikleri, **{:.3f}** düzeyinde istatistiksel parite farkı gösteriyor.",
        's7_restart': "🔄 Yeni Analiz Başlat",
        
        # Report Table
        's7_col_metric': "Metrik",
        's7_col_score': "Skor (0-100)",
        's7_col_weight': "Ağırlık",
        's7_m_fair': "Adillik",
        's7_m_transp': "Şeffaflık",
        's7_m_sim': "Benzerlik"
    }
}

def get_text(lang, key):
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)
