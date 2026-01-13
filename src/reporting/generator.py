from fpdf import FPDF
import datetime

class EthicsReportPDF(FPDF):
    def __init__(self, lang='en'):
        super().__init__()
        self.lang = lang
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, "AI Ethics Inspector (AEI) - Audit Report", 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.cell(0, 6, f"Generated: {date_str}", 0, 1, 'C')
        self.line(10, 25, 200, 25)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        page_str = f'Page {self.page_no()}' if self.lang == 'en' else f'Sayfa {self.page_no()}'
        self.cell(0, 10, f"{page_str} - AEI Ethical Compliance System", 0, 0, 'C')

    def sanitize(self, text):
        """
        Replace Turkish characters for standard Arial font compatibility.
        """
        replacements = {
            'ğ': 'g', 'Ğ': 'G', 'ü': 'u', 'Ü': 'U', 'ş': 's', 'Ş': 'S',
            'ı': 'i', 'İ': 'I', 'ö': 'o', 'Ö': 'O', 'ç': 'c', 'Ç': 'C'
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text.encode('latin-1', 'replace').decode('latin-1')

    def section(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(30, 58, 95) # Dark Navy
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, f"  {self.sanitize(title)}", 0, 1, 'L', 1)
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def kv(self, key, value):
        self.set_font('Arial', 'B', 10)
        self.cell(60, 6, self.sanitize(key) + ":", 0)
        self.set_font('Arial', '', 10)
        self.cell(0, 6, self.sanitize(str(value)), 0, 1)

    def generate(self, metrics, weights, final_score, config):
        self.add_page()
        
        # --- 1. EXECUTIVE SUMMARY ---
        title = "Executive Summary" if self.lang == 'en' else "Yonetici Ozeti"
        self.section(title)
        
        # Just show the score without judgement
        self.set_font('Arial', 'B', 14)
        self.cell(100, 10, f"Final Score: {final_score:.2f} / 5.0", 0, 1)
        
        desc = (f"The AI model audit is complete. Quantitative results for Fairness, Transparency, and Similarity are detailed below."
                if self.lang == 'en' else 
                f"Yapay zeka model denetimi tamamlanmistir. Adillik, Seffaflik ve Benzerlik icin nicel sonuclar asagida detaylandirilmistir.")
        
        self.set_font('Arial', '', 10)
        effective_width = self.w - 2 * self.l_margin
        self.multi_cell(effective_width, 6, self.sanitize(desc))
        self.ln(6)

        # --- 2. AUDIT CONFIG ---
        self.section("Audit Configuration" if self.lang == 'en' else "Denetim Konfigurasyonu")
        
        sens_str = ", ".join(config.get('sensitive_features', []))
        self.kv("Sensitive Attributes", sens_str)
        self.kv("Transparency Weight", weights['Transparency'])
        self.kv("Fairness Weight", weights['Fairness'])
        self.kv("Similarity Weight", weights['Similarity'])
        self.ln(6)

        # --- 3. DETAILED ANALYSIS ---
        self.section("Detailed Findings" if self.lang == 'en' else "Detayli Bulgular")
        
        # Fairness
        f_score = 100 - abs(metrics['fairness'].get('statistical_parity_difference', 0) * 500)
        self.kv("Fairness Raw Score", f"{f_score:.1f} / 100")
        for k, v in metrics['fairness'].items():
            self.kv(f" - {k}", f"{v:.4f}")
            
        self.ln(2)
        
        # Similarity
        sim = metrics['similarity_score']
        bias = metrics['sim_bias_detected']
        self.kv("Similarity Consistency", f"{sim:.1f} / 100")
        
        # Just state if bias was detected or not, no color coding or judgment
        bias_str = "DETECTED / TESPIT EDILDI" if bias else "None / Yok"
        self.kv("Neighborhood Bias Status", bias_str)
        
        # If specific bias info exists, could list it here factually, but user said "results from each step".
        # Bias status is the key result from Step 6.
        
        self.ln(6)

        # Recommendations section REMOVED as per user request.

        # Output Logic
        out = self.output(dest='S')
        if isinstance(out, (bytes, bytearray)):
            return bytes(out)
        return out.encode('latin-1')
