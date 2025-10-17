# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from scipy.optimize import minimize
import json
import time
import requests  # Added for fetching data from URL
import io  # Added for in-memory file handling for downloads

from model_logic import IntegratedReactor

# ==============================================================================
# 1. Page Config, Language, and Theming
# ==============================================================================
st.set_page_config(page_title="KINEMOD-KIT", page_icon="🧪",
                   layout="wide", initial_sidebar_state="expanded")

# --- Translations (no changes)
translations = {
    "en": {
        "title": "KINEMOD-KIT", "subtitle": "UASB-FILTRATION-RBC REACTOR KINETIC MODELING KIT",
        "author_line": "By Rizky Mursyidan Baldan | Last Updated: {}",
        "tabs": ["📊 Dashboard", "🔬 Model Details", "🍃 Methane & Energy", "🔬 Sensitivity", "⚙️ Optimizer", "❓ Help"],
        "welcome_message": "👋 **Welcome!** Please upload your data, or use the default dataset from GitHub, then select an action in the sidebar.",
        "sidebar_controls": "⚙️ Controls", "sidebar_upload": "1. Upload Your Data (CSV)", "sidebar_upload_help": "Upload `Model_Data.csv`.",
        "sidebar_github": "Or Use Default Data", "sidebar_github_help": "Load Model_Data.csv directly from the GitHub repository.",
        "sidebar_actions": "2. Choose Action",
        "sidebar_advanced_config": "🔬 Advanced Config", "sidebar_rbc_vol": "RBC Volume (V_RBC)", "sidebar_rbc_area": "RBC Surface Area (A_RBC)",
        "sidebar_calibrate": "🚀 Calibrate", "sidebar_calibrate_help": "Run parameter estimation.",
        "sidebar_load": "📂 Load", "sidebar_load_help": "Load `calibrated_params.json`.",
        "kpi_header": "Key Performance Indicators", "kpi_compliance_header": "Effluent Compliance",
        "timeseries_header": "Time-Series Analysis", "log_axis_toggle": "Use logarithmic Y-axis",
        "compliance_input": "Set COD Discharge Limit (mg/L)", "date_slider": "Select Date Range to Analyze",
        "parity_header": "Parity & Error Analysis",
        "param_header": "🔬 Calibrated Model Details", "param_subheader": "📋 Kinetic Parameters", "eq_subheader": "🧪 Calibrated Kinetic Equations",
        "editable_param_info": "You can edit parameter values below. Click the button to see how your changes affect the predictions.",
        "rerun_button": "Re-run Predictions with Edited Values",
        "methane_header": "🍃 Methane & Energy Potential (UASB Stage)", "cod_removed_label": "Avg. COD Removed",
        "methane_yield_label": "Methane Yield (m³/kg COD)", "methane_prod_label": "Daily Methane Production",
        "energy_potential_label": "Daily Energy Potential", "uasb_volume_label": "UASB Reactor Volume (m³)",
        "sa_header": "🔬 Sensitivity Analysis", "sa_info": "Evaluate how uncertainty in model parameters affects the output. Results are cached after the first run.",
        "sa_type_select": "Select Analysis Type", "sa_model_select": "Select RBC Model", "run_sa_button": "🚀 Run Analysis",
        "what_if_header": "⚡ \"What-If\" Scenarios", "what_if_help": "Adjust inputs to see a quick prediction of the final effluent quality based on the pH model.",
        "what_if_cod": "Influent COD (mg/L)", "what_if_hrt_uasb": "UASB HRT (days)", "what_if_hrt_rbc": "RBC HRT (hours)", "what_if_result": "Predicted Final Effluent (pH Model)",
        "optimizer_header": "⚙️ Process Optimizer", "optimizer_intro": "This tool uses an optimization algorithm to find the best operational inputs to achieve a selected goal within your defined constraints.",
        "optimizer_goal": "Optimization Goal", "optimizer_goal_option": "Minimize Final Effluent COD",
        "optimizer_constraints": "Set Operational Constraints (Search Space)",
        "optimizer_button": "🚀 Find Optimal Settings", "optimizer_results_header": "🏆 Optimization Results", "optimizer_success": "Optimization Successful!", "optimizer_fail": "Optimization failed to converge.",
        "help_title": "❓ Help & Documentation",
        "help_intro": "This is a comprehensive tool for analyzing and simulating the performance of a three-stage POME treatment system. This guide will help you understand its features.",
        "help_usage_title": "Workflow Guide",
        "help_usage_step1_title": "1. Provide Data", "help_usage_step1_text": "Start by uploading your `Model_Data.csv` file, or check the box to use the default dataset from the project's GitHub repository.",
        "help_usage_step2_title": "2. Calibrate or Load", "help_usage_step2_text": "Click **Calibrate** to estimate model parameters from your data, or **Load** if you have a pre-calibrated `calibrated_params.json` file.",
        "help_usage_step3_title": "3. Analyze Dashboard", "help_usage_step3_text": "The Dashboard shows model performance via KPIs, time-series charts, and parity plots. Use the date slider to focus on specific periods. You can also download data and charts.",
        "help_usage_step4_title": "4. Explore Tabs", "help_usage_step4_text": "Dive deeper into other tabs to view model equations, run sensitivity analysis (results are cached), calculate energy potential, or optimize process parameters.",
        "help_model_docs_title": "Model Documentation",
        "help_uasb_title": "UASB Model", "help_uasb_obj": "**Objective**: Predict the Substrate Removal Rate (SRR) based on Organic Loading Rate (OLR) and VFA/Alkalinity inhibition.",
        "help_filter_title": "Adsorption & Filtration Model", "help_filter_obj": "**Objective**: Quantify COD removal through two mechanisms: physical filtration of suspended solids and adsorption of soluble COD.",
        "help_rbc_title": "RBC Biofilm Model", "help_rbc_obj": "**Objective**: Simulate substrate removal in the biofilm, considering factors like biomass concentration, oxygen levels, and an optional pH inhibition factor.",
        "help_params_title": "Key Parameters", "help_inputs_title": "Inputs", "help_outputs_title": "Outputs",
        "help_tab_guide_title": "Guide to App Features",
        "help_tab_dashboard_desc": "Your main analysis hub. It shows KPIs (R², RMSE), time-series predictions vs. actual data, and parity plots to evaluate model accuracy.",
        "help_tab_model_details_desc": "View the calibrated kinetic parameters in an editable table. You can tweak values and re-run predictions. Also displays the final governing equations with the calibrated values.",
        "help_tab_methane_desc": "Calculates the potential for methane production and energy generation from the COD removed in the UASB reactor. Adjust reactor volume and methane yield for scenario analysis.",
        "help_tab_sensitivity_desc": "Perform Global Sensitivity Analysis (GSA) or Monte Carlo simulations to identify which model parameters have the most significant impact on the final effluent COD.",
        "help_tab_optimizer_desc": "Find the optimal operational conditions (Influent COD, UASB HRT, RBC HRT) that minimize the final effluent COD, given a set of constraints you define.",
        "help_faq_title": "FAQ & Troubleshooting",
        "help_faq1_q": "Why does calibration fail or take a long time?", "help_faq1_a": "Calibration is a complex optimization process. It can fail if the data is very noisy, contains many outliers, or if the initial parameter guesses are far from the optimal values. Ensure your data is clean and covers a wide range of operational conditions.",
        "help_faq2_q": "What's the difference between RBC 'Original' and 'pH Mod' models?", "help_faq2_a": "The 'Original' model is a standard biofilm model. The 'pH Mod' version adds a pH inhibition factor (τ_pH), making the model's predictions sensitive to pH fluctuations, which is common in biological treatment systems."
    },
    "id": {
        "title": "KINEMOD-KIT", "subtitle": "KIT PEMODELAN KINETIK REAKTOR UASB-FILTRASI-RBC",
        "author_line": "Oleh Rizky Mursyidan Baldan | Terakhir Diperbarui: {}",
        "tabs": ["📊 Dasbor", "🔬 Detail Model", "🍃 Metana & Energi", "🔬 Sensitivitas", "⚙️ Pengoptimal", "❓ Bantuan"],
        "welcome_message": "👋 **Selamat datang!** Silakan unggah data Anda, atau gunakan dataset default dari GitHub, lalu pilih tindakan di bilah sisi.",
        "sidebar_controls": "⚙️ Kontrol", "sidebar_upload": "1. Unggah Data Anda (CSV)", "sidebar_upload_help": "Unggah file `Model_Data.csv`.",
        "sidebar_github": "Atau Gunakan Data Default", "sidebar_github_help": "Muat Model_Data.csv langsung dari repositori GitHub.",
        "sidebar_actions": "2. Pilih Tindakan",
        "sidebar_advanced_config": "🔬 Konfigurasi Lanjutan", "sidebar_rbc_vol": "Volume RBC (V_RBC)", "sidebar_rbc_area": "Luas Permukaan RBC (A_RBC)",
        "sidebar_calibrate": "🚀 Kalibrasi", "sidebar_calibrate_help": "Jalankan estimasi parameter.",
        "sidebar_load": "📂 Muat", "sidebar_load_help": "Muat `calibrated_params.json`.",
        "kpi_header": "Indikator Kinerja Utama", "kpi_compliance_header": "Kepatuhan Efluen",
        "timeseries_header": "Analisis Runtun Waktu", "log_axis_toggle": "Gunakan sumbu Y logaritmik",
        "compliance_input": "Atur Batas Buangan COD (mg/L)", "date_slider": "Pilih Rentang Tanggal untuk Analisis",
        "parity_header": "Analisis Paritas & Kesalahan",
        "param_header": "🔬 Detail Model Terkalibrasi", "param_subheader": "📋 Parameter Kinetik", "eq_subheader": "🧪 Persamaan Kinetik Terkalibrasi",
        "editable_param_info": "Anda dapat mengedit nilai parameter di bawah. Klik tombol untuk melihat pengaruhnya pada prediksi.",
        "rerun_button": "Jalankan Ulang Prediksi dengan Nilai Baru",
        "methane_header": "🍃 Potensi Metana & Energi (Tahap UASB)", "cod_removed_label": "Rata-rata COD Dihilangkan",
        "methane_yield_label": "Hasil Metana (m³/kg COD)", "methane_prod_label": "Produksi Metana Harian",
        "energy_potential_label": "Potensi Energi Harian", "uasb_volume_label": "Volume Reaktor UASB (m³)",
        "sa_header": "🔬 Analisis Sensitivitas", "sa_info": "Evaluasi bagaimana ketidakpastian parameter mempengaruhi output. Hasil di-cache setelah proses pertama.",
        "sa_type_select": "Pilih Jenis Analisis", "sa_model_select": "Pilih Model RBC", "run_sa_button": "🚀 Jalankan Analisis",
        "what_if_header": "⚡ Skenario \"What-If\"", "what_if_help": "Sesuaikan input ini untuk melihat prediksi cepat kualitas efluen akhir berdasarkan model pH.",
        "what_if_cod": "COD Influen (mg/L)", "what_if_hrt_uasb": "HRT UASB (hari)", "what_if_hrt_rbc": "HRT RBC (jam)", "what_if_result": "Prediksi Efluen Akhir (Model pH)",
        "optimizer_header": "⚙️ Pengoptimal Proses", "optimizer_intro": "Alat ini menggunakan algoritma optimisasi untuk menemukan input operasional terbaik untuk mencapai tujuan yang dipilih dalam batasan yang Anda tentukan.",
        "optimizer_goal": "Tujuan Optimisasi", "optimizer_goal_option": "Minimalkan Efluen Akhir COD",
        "optimizer_constraints": "Atur Batasan Operasional (Ruang Pencarian)",
        "optimizer_button": "🚀 Temukan Pengaturan Optimal", "optimizer_results_header": "🏆 Hasil Optimisasi", "optimizer_success": "Optimisasi Berhasil!", "optimizer_fail": "Optimisasi gagal mencapai konvergensi.",
        "help_title": "❓ Bantuan & Dokumentasi",
        "help_intro": "Ini adalah alat komprehensif untuk menganalisis dan menyimulasikan kinerja sistem pengolahan POME tiga tahap. Panduan ini akan membantu Anda memahami fitur-fiturnya.",
        "help_usage_title": "Panduan Alur Kerja",
        "help_usage_step1_title": "1. Sediakan Data", "help_usage_step1_text": "Mulailah dengan mengunggah file `Model_Data.csv` Anda, atau centang kotak untuk menggunakan dataset default dari repositori GitHub proyek.",
        "help_usage_step2_title": "2. Kalibrasi atau Muat", "help_usage_step2_text": "Klik **Kalibrasi** untuk mengestimasi parameter model dari data Anda, atau **Muat** jika Anda memiliki file `calibrated_params.json` yang sudah ada.",
        "help_usage_step3_title": "3. Analisis Dasbor", "help_usage_step3_text": "Dasbor menampilkan kinerja model melalui KPI, grafik runtun waktu, dan plot paritas. Gunakan slider tanggal untuk fokus pada periode tertentu. Anda juga dapat mengunduh data dan grafik.",
        "help_usage_step4_title": "4. Jelajahi Tab Lain", "help_usage_step4_text": "Pelajari lebih dalam di tab lain untuk melihat persamaan model, menjalankan analisis sensitivitas (hasilnya di-cache), menghitung potensi energi, atau mengoptimalkan parameter proses.",
        "help_model_docs_title": "Dokumentasi Model",
        "help_uasb_title": "Model UASB", "help_uasb_obj": "**Tujuan**: Memprediksi Laju Penghilangan Substrat (SRR) berdasarkan Beban Pemuatan Organik (OLR) dan inhibisi rasio VFA/Alkalinitas.",
        "help_filter_title": "Model Adsorpsi & Filtrasi", "help_filter_obj": "**Tujuan**: Mengukur penghilangan COD melalui dua mekanisme: filtrasi fisik padatan tersuspensi dan adsorpsi COD terlarut.",
        "help_rbc_title": "Model Biofilm RBC", "help_rbc_obj": "**Tujuan**: Menyimulasikan penghilangan substrat dalam biofilm, dengan mempertimbangkan faktor-faktor seperti konsentrasi biomassa, kadar oksigen, dan faktor inhibisi pH opsional.",
        "help_params_title": "Parameter Kunci", "help_inputs_title": "Input", "help_outputs_title": "Output",
        "help_tab_guide_title": "Panduan Fitur Aplikasi",
        "help_tab_dashboard_desc": "Pusat analisis utama Anda. Menampilkan KPI (R², RMSE), perbandingan prediksi vs data aktual dalam runtun waktu, dan plot paritas untuk mengevaluasi akurasi model.",
        "help_tab_model_details_desc": "Lihat parameter kinetik hasil kalibrasi dalam tabel yang dapat diedit. Anda dapat mengubah nilai dan menjalankan ulang prediksi. Juga menampilkan persamaan utama dengan nilai yang telah dikalibrasi.",
        "help_tab_methane_desc": "Menghitung potensi produksi metana dan pembangkitan energi dari COD yang dihilangkan di reaktor UASB. Sesuaikan volume reaktor dan hasil metana untuk analisis skenario.",
        "help_tab_sensitivity_desc": "Lakukan Analisis Sensitivitas Global (GSA) atau simulasi Monte Carlo untuk mengidentifikasi parameter model mana yang memiliki dampak paling signifikan terhadap COD efluen akhir.",
        "help_tab_optimizer_desc": "Temukan kondisi operasional optimal (COD Influen, HRT UASB, HRT RBC) yang meminimalkan COD efluen akhir, berdasarkan batasan yang Anda tentukan.",
        "help_faq_title": "FAQ & Penyelesaian Masalah",
        "help_faq1_q": "Mengapa kalibrasi gagal atau memakan waktu lama?", "help_faq1_a": "Kalibrasi adalah proses optimisasi yang kompleks. Bisa gagal jika data sangat bising, mengandung banyak pencilan, atau jika tebakan parameter awal jauh dari nilai optimal. Pastikan data Anda bersih dan mencakup berbagai kondisi operasional.",
        "help_faq2_q": "Apa perbedaan antara model RBC 'Original' dan 'pH Mod'?", "help_faq2_a": "Model 'Original' adalah model biofilm standar. Versi 'pH Mod' menambahkan faktor inhibisi pH (τ_pH), membuat prediksi model sensitif terhadap fluktuasi pH, yang umum terjadi pada sistem pengolahan biologis."
    }
}


if 'lang' not in st.session_state:
    st.session_state.lang = 'en'
lang_options = {"English": "en", "Bahasa Indonesia": "id"}
lang_selection = st.sidebar.selectbox(
    "Language / Bahasa", options=list(lang_options.keys()))
st.session_state.lang = lang_options[lang_selection]
def t(key): return translations[st.session_state.lang].get(key, key)


macos_colors = ['#0A84FF', '#30D158', '#FF9F0A',
                '#FF453A', '#BF5AF2', '#FFD60A', '#64D2FF']
pio.templates["macos"] = go.layout.Template(layout=go.Layout(
    colorway=macos_colors, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'))
pio.templates.default = "macos"

st.title(f"💻 {t('title')} 📊")
st.subheader(t('subtitle'))
st.caption(t('author_line').format(
    pd.to_datetime('today').strftime('%Y-%m-%d')))
st.markdown("---")

# ==============================================================================
# 2. Global CSS Styles & Animation Functions (no changes)
# ==============================================================================


def get_animated_class():
    animation_key = str(int(time.time() * 1000))
    st.markdown(f"""
        <style>
            @keyframes softFadeInUp_{animation_key} {{
                from {{ opacity: 0; transform: translateY(10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            .fade-in-up-{animation_key} {{
                animation: softFadeInUp_{animation_key} 0.6s ease-out forwards;
            }}
        </style>
    """, unsafe_allow_html=True)
    return f"fade-in-up-{animation_key}"


GLOBAL_STYLES = """
<style>
    /* --- Base & Font --- */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", sans-serif;
    }
    /* --- KPI Card (Dashboard Tab) --- */
    .kpi-card {
        display: flex; flex-direction: column; justify-content: space-between;
        background: rgba(255, 255, 255, 0.72); backdrop-filter: blur(10px);
        border-radius: 18px; padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid rgba(230,230,230,0.5);
        transition: transform 0.35s ease, box-shadow 0.35s ease;
        min-height: 220px;
    }
    .kpi-card:hover { transform: scale(1.03) translateY(-5px); box-shadow: 0 10px 30px rgba(0,0,0,0.12); }
    .card-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; }
    .kpi-icon {
        display: flex; align-items: center; justify-content: center;
        width: 42px; height: 42px; font-size: 1.5rem;
        background-color: #F0F0F5; color: #333; border-radius: 50%;
    }
    .kpi-title { font-weight: 600; font-size: 1rem; color: #333; }
    .metric-value { font-size: 2.0rem; font-weight: 700; color: #000; line-height: 1.2; margin-bottom: 0.1rem; }
    .metric-label { font-size: 0.9rem; color: #666; }
    .anova { margin-top: 1rem; }
    .anova-label { font-size: 0.8rem; color: #777; }
    .anova-value { font-size: 1rem; font-weight: 600; }
    .badge {
        display: inline-block; padding: 0.25rem 0.6rem;
        border-radius: 10px; font-size: 0.7rem; font-weight: 600; margin-top: 0.25rem;
    }
    .badge.green { background-color: #E2F6E9; color: #155724; }
    .badge.red { background-color: #FCE8E6; color: #7F1D1D; }
    /* --- Help Tab Cards --- */
    .help-card {
        background: rgba(255,255,255,0.7); backdrop-filter: blur(12px);
        border-radius: 16px; padding: 1.3rem 1.5rem; margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05); border: 1px solid rgba(255,255,255,0.4);
        transition: all 0.25s ease;
    }
    .help-card:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(0,0,0,0.08); }
    .mini-card {
        background: rgba(245,245,247,0.9); padding: 1rem; border-radius: 12px;
        border: 1px solid rgba(200,200,200,0.4); transition: all 0.25s ease;
        height: 100%;
    }
    .mini-card:hover { background: rgba(255,255,255,0.9); transform: translateY(-2px); }
    .styled-table { width:100%; border-collapse:collapse; margin-top: 0.5rem; }
    .styled-table th, .styled-table td { padding: 8px 12px; border: 1px solid rgba(200,200,200,0.4); text-align: left; }
    .styled-table th { font-weight: 600; background-color: rgba(240,240,245,0.8); }
    .styled-table tr:nth-child(even) { background-color: rgba(248,248,250,0.7); }
    .divider { margin: 2rem 0; border-top: 1px solid rgba(200,200,200,0.4); }
    /* macOS-style slider and selectbox tweaks */
    [data-testid="stSlider"] label, [data-testid="stNumberInput"] label { font-weight: 600; font-size: 0.9rem; }
    [data-testid="stSelectbox"] {
        background: rgba(245,245,247,0.85); border-radius: 8px !important;
        border: 1px solid rgba(220,220,220,0.6); box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        padding: 3px 8px !important; transition: all 0.25s ease;
    }
    [data-testid="stSelectbox"]:hover { transform: scale(1.01); box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
    [data-baseweb="select"] div { background: transparent !important; }
    /* Dark Mode Adjustments */
    @media (prefers-color-scheme: dark) {
        .help-card { background: rgba(40,40,45,0.6); color: #eee; }
        .mini-card { background: rgba(60,60,65,0.6); color: #ddd; }
        .styled-table th { background-color: rgba(70,70,75,0.8); }
        .styled-table tr:nth-child(even) { background-color: rgba(55,55,60,0.7); }
    }
</style>
"""
st.markdown(GLOBAL_STYLES, unsafe_allow_html=True)

# ==============================================================================
# 3. Caching & Backend Logic
# ==============================================================================

# FIX: Added function to get data from either upload or URL, and cached the result


@st.cache_data
def get_data_content(uploaded_file, use_github, github_url):
    """Loads data content from either a user upload or a GitHub URL."""
    if uploaded_file is not None:
        return uploaded_file.getvalue()
    if use_github:
        try:
            response = requests.get(github_url)
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.content
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data from GitHub: {e}")
            return None
    return None

# FIX: Caching is now based on the file content bytes (_uploaded_file_content) which can come from upload or URL


@st.cache_data
def run_full_analysis(_uploaded_file_content, config, action):
    """Performs the main model calibration and prediction, writing content to a temporary file."""
    if _uploaded_file_content is None:
        return None, "No data content provided."
    # Use an in-memory buffer instead of writing to disk for better compatibility
    data_buffer = io.BytesIO(_uploaded_file_content)
    reactor = IntegratedReactor(data_buffer, config["reactor_constants"])
    reactor.load_and_prepare_data()
    if action == "estimate":
        reactor.run_parameter_estimation(config["optimization_bounds"])
    elif action == "load":
        # Note: loading parameters still depends on a local file `calibrated_params.json`
        if not reactor.load_parameters():
            return None, "Could not find `calibrated_params.json`."
    reactor.run_full_predictions()
    return reactor, None

# FIX: The sensitivity analysis function is already cached. This prevents re-computation if the inputs don't change.
# The key is to store the result in session_state so it persists across re-runs.


@st.cache_data
def run_sensitivity_analysis(analysis_type, model_key, params_uasb, params_filter, params_rbc_orig, params_rbc_ph, _uploaded_file_content, config):
    """Runs computationally intensive sensitivity analysis. Cached to avoid re-runs."""
    if _uploaded_file_content is None:
        return "No data content available for sensitivity analysis.", None
    data_buffer = io.BytesIO(_uploaded_file_content)
    reactor = IntegratedReactor(data_buffer, config["reactor_constants"])
    reactor.load_and_prepare_data()
    reactor.params_uasb = params_uasb
    reactor.params_filter = params_filter
    reactor.params_rbc_orig = params_rbc_orig
    reactor.params_rbc_ph = params_rbc_ph
    reactor.params_estimated = True
    params = {**params_uasb, **params_filter, **
              (params_rbc_ph if model_key == 'ph' else params_rbc_orig)}
    if not params:
        return "No parameters available for the selected model.", None
    param_keys = sorted(params.keys())
    failure_message = "Analysis failed: The model solver could not converge for most parameter variations."
    if analysis_type == 'GSA':
        Mi, Si, problem = reactor._execute_gsa(params, param_keys, model_key)
        if Mi is None:
            return failure_message, None
        return None, {'Mi': Mi.to_df(), 'Si': Si, 'problem': problem}
    elif analysis_type == 'Monte Carlo':
        results_df, samples_df = reactor._execute_mc(
            params, param_keys, model_key)
        if results_df is None:
            return failure_message, None
        return None, {'mc_results': results_df, 'mc_samples': samples_df}
    return "Invalid analysis type", None


def optimization_objective_function(inputs, reactor, fixed_params):
    cod_in, hrt_uasb_days, hrt_rbc_hours = inputs
    srr_pred = reactor.uasb_model.predict(
        (cod_in / hrt_uasb_days, fixed_params['vfa_alk']), reactor.params_uasb)
    cod_uasb_pred = cod_in - srr_pred * hrt_uasb_days
    cod_filt_removed = reactor.filter_model.predict(
        (fixed_params['tss_in'], fixed_params['tss_out'], cod_uasb_pred), reactor.params_filter)
    cod_filt_pred = cod_uasb_pred - cod_filt_removed
    final_pred = reactor.rbc_model_ph.predict(
        {'So': cod_filt_pred, 'HRT_days': hrt_rbc_hours / 24.0,
            'Xa': fixed_params['xa'], 'Xs': fixed_params['xs'], 'pH': fixed_params['ph']},
        reactor.params_rbc_ph
    )
    return final_pred if not np.isnan(final_pred) else 1e9

# ==============================================================================
# 4. Helper & Plotting Functions
# ==============================================================================

# FIX: Added caching to plotting functions. The underscore in _df tells Streamlit to hash the dataframe's content.


@st.cache_data
def plot_interactive_timeseries(_df, compliance_limit=None, log_y=False):
    """Generates the main time-series plot. Cached for performance."""
    df = _df.copy()  # Work on a copy to avoid mutation issues with caching
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1, subplot_titles=("UASB", "Filter", "RBC"))
    marker_style = dict(symbol='x', size=8)
    fig.add_trace(go.Scatter(x=df['Day'], y=df['COD_UASB_Eff'], mode='markers',
                  name='UASB Measured', marker=marker_style), row=1, col=1)
    if 'COD_UASB_Pred' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Day'], y=df['COD_UASB_Pred'], mode='lines', name='UASB Predicted'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Day'], y=df['COD_Filt_Eff'], mode='markers',
                  name='Filter Measured', marker=marker_style), row=2, col=1)
    if 'COD_Filt_Pred' in df.columns:
        fig.add_trace(go.Scatter(x=df['Day'], y=df['COD_Filt_Pred'],
                      mode='lines', name='Filter Predicted'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Day'], y=df['COD_Final'], mode='markers',
                  name='RBC Measured', marker=marker_style), row=3, col=1)
    if 'COD_Final_Pred_Orig' in df.columns:
        fig.add_trace(go.Scatter(x=df['Day'], y=df['COD_Final_Pred_Orig'],
                      mode='lines', name='RBC Original Pred.'), row=3, col=1)
    if 'COD_Final_Pred_pH' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Day'], y=df['COD_Final_Pred_pH'], mode='lines', name='RBC pH Pred.'), row=3, col=1)
    if compliance_limit:
        fig.add_hline(y=compliance_limit, line_dash="dot", line_color="#FF453A",
                      annotation_text="Compliance Limit", annotation_position="bottom right", row=3, col=1)
    fig.update_layout(height=600, title_text=f"<b>{t('timeseries_header')}</b>", hovermode="x unified", legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    if log_y:
        fig.update_yaxes(type="log")
    fig.update_yaxes(title_text="COD (mg/L)", row=1, col=1)
    fig.update_yaxes(title_text="COD (mg/L)", row=2, col=1)
    fig.update_yaxes(title_text="COD (mg/L)", row=3, col=1)
    fig.update_xaxes(title_text="Time (Days)", row=3, col=1)
    return fig


@st.cache_data
def plot_interactive_parity(_df, stage_name, actual_col, pred_cols_dict):
    """Generates parity plots. Cached for performance."""
    df = _df.copy()
    required_cols = [actual_col, 'Day'] + list(pred_cols_dict.values())
    if not all(col in df.columns for col in required_cols):
        return go.Figure().update_layout(title=f'Parity Plot: {stage_name} (Missing Data)')
    data = df.dropna(subset=required_cols).copy()
    if data.empty:
        return go.Figure().update_layout(title=f'Parity Plot: {stage_name} (No valid data)')
    fig = go.Figure()
    all_values = pd.concat([data[actual_col]] + [data[col]
                           for col in pred_cols_dict.values()])
    min_val, max_val = all_values.min(), all_values.max()
    padding = (max_val - min_val) * 0.05
    fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val,
                  y1=max_val, line=dict(color="#FF453A", dash="dash"), name="1:1 Line")
    for i, (model_name, pred_col) in enumerate(pred_cols_dict.items()):
        fig.add_trace(go.Scatter(x=data[actual_col], y=data[pred_col], mode='markers', name=model_name, marker=dict(
            color=macos_colors[i]), customdata=data['Day'], hovertemplate='<b>Day %{customdata}</b><br>Measured: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'))
    fig.update_layout(title=f'<b>Parity Plot: {stage_name}</b>', xaxis_title='Measured COD (mg/L)', yaxis_title='Predicted COD (mg/L)',
                      legend_title='Model', xaxis=dict(constrain='domain'), yaxis=dict(scaleanchor="x", scaleratio=1))
    fig.update_xaxes(range=[min_val - padding, max_val + padding])
    fig.update_yaxes(range=[min_val - padding, max_val + padding])
    return fig


@st.cache_data
def plot_gsa_results(_mi_df, _si_dict, _problem_dict):
    """Generates GSA results plot using Plotly. Cached."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Morris Elementary Effects", "Sobol Sensitivity Indices"))
    # Morris Plot
    fig.add_trace(go.Scatter(
        x=_mi_df['mu_star'], y=_mi_df['sigma'], mode='markers+text',
        text=_mi_df.index, textposition='top right',
        marker=dict(size=10), hovertext=_mi_df.index, name='Morris'
    ), row=1, col=1)
    # Sobol Plot
    sobol_df = pd.DataFrame({'S1': _si_dict['S1'], 'ST': _si_dict['ST']},
                            index=_problem_dict['names']).sort_values('ST', ascending=True)
    fig.add_trace(go.Bar(y=sobol_df.index,
                  x=sobol_df['S1'], name='S1 (First-order)', orientation='h'), row=1, col=2)
    fig.add_trace(go.Bar(y=sobol_df.index,
                  x=sobol_df['ST'], name='ST (Total-order)', orientation='h'), row=1, col=2)
    fig.update_layout(barmode='group', height=500)
    fig.update_xaxes(title_text='μ* (Overall Influence)', row=1, col=1)
    fig.update_yaxes(title_text='σ (Interaction Effects)', row=1, col=1)
    fig.update_xaxes(title_text='Sobol Index Value', row=1, col=2)
    return fig


@st.cache_data
def plot_mc_results(_mc_df):
    """Generates Monte Carlo results plot using Plotly. Cached."""
    df_sorted = _mc_df.copy().reindex(
        _mc_df['Spearman_Correlation'].abs().sort_values(ascending=True).index)
    fig = px.bar(df_sorted, y='Parameter', x='Spearman_Correlation', color='Spearman_Correlation',
                 orientation='h', title='Monte Carlo: Spearman Rank Correlation',
                 color_continuous_scale=px.colors.diverging.Picnic)
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.update_layout(height=500, coloraxis_showscale=False)
    return fig


def create_kpi_card(icon, title, r2, rmse, p_value):
    sig_class = "red" if p_value < 0.05 else "green"
    sig_label = "Significant" if p_value < 0.05 else "Not Sig."
    card_html = f"""<div class="kpi-card"> <div class="card-header"> <div class="kpi-icon">{icon}</div> <div class="kpi-title">{title}</div> </div> <div class="card-body"> <div style="display:flex; justify-content:space-between;"> <div> <div class="metric-label">R²</div> <div class="metric-value">{r2:.3f}</div> </div> <div style="text-align:right;"> <div class="metric-label">RMSE</div> <div class="metric-value">{rmse:.1f}</div> </div> </div> </div> <div class="anova"> <span class="anova-label">ANOVA p-value:</span> <span class="anova-value">{p_value:.3f}</span> <span class="badge {sig_class}">{sig_label}</span> </div> </div>"""
    st.markdown(card_html, unsafe_allow_html=True)


def display_final_equations(params_uasb, params_filter, params_rbc_orig, params_rbc_ph):
    st.markdown("""<style> .equation-card { background-color: #F0F2F6; border-radius: 10px; padding: 10px; margin-bottom: 0px; border: 1px solid #E0E0E0; } .equation-card .title { font-weight: bold; font-size: 1.1em; margin-bottom: 15px; } .equation-card .variables { font-size: 0.9em; margin-top: 15px; columns: 2; -webkit-columns: 2; -moz-columns: 2; } </style> """, unsafe_allow_html=True)
    st.subheader(t('eq_subheader'))
    if params_uasb:
        p = params_uasb
        st.markdown(
            f"""<div class="equation-card"><div class="title">{t('help_uasb_title')}: Substrate Removal Rate</div></div>""", unsafe_allow_html=True)
        st.latex(
            fr''' SRR = \left( \dfrac{{ \textcolor{{#0A84FF}}{{{p['U_max']:.3f}}} \cdot OLR }}{{ \textcolor{{#0A84FF}}{{{p['K_B']:.3f}}} + OLR }} \right) \cdot \left( \dfrac{1}{{ 1 + \dfrac{{VFA/ALK}}{{ \textcolor{{#0A84FF}}{{{p['K_I']:.3f}}} }} }} \right) ''')
        st.markdown("""<div class="variables"><b>Where:</b><ul><li><em>SRR</em>: Substrate Removal Rate (g/L·day)</li><li><em>OLR</em>: Organic Loading Rate (g/L·day)</li><li><em>VFA/ALK</em>: VFA/Alkalinity Ratio (dimensionless)</li></ul></div>""", unsafe_allow_html=True)
    if params_filter:
        p = params_filter
        st.markdown(
            f"""<div class="equation-card"><div class="title">{t('help_filter_title')}: COD Removal</div></div>""", unsafe_allow_html=True)
        st.latex(
            fr'COD_{{Removed}} = \textcolor{{#0A84FF}}{{{p["R_cod_tss"]:.3f}}} \cdot (TSS_{{in}} - TSS_{{out}}) + \textcolor{{#0A84FF}}{{{p["k_ads"]:.3f}}} \cdot sCOD_{{in}}')
        st.markdown("""<div class="variables"><b>Where:</b><ul><li><em>COD<sub>Removed</sub></em>: COD Removed (mg/L)</li><li><em>TSS</em>: Total Suspended Solids (mg/L)</li><li><em>sCOD</em>: Soluble COD (mg/L)</li></ul></div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### RBC Models")
    if params_rbc_orig:
        p = params_rbc_orig
        st.markdown(
            f"""<div class="equation-card"><div class="title">RBC v1.0 (Original)</div></div>""", unsafe_allow_html=True)
        st.latex(fr'''\mu_a = \dfrac{{\textcolor{{#0A84FF}}{{{p['umxa']:.3f}}} \cdot S_e}}{{\textcolor{{#0A84FF}}{{{p['Ku']:.3f}}} + S_e}} \cdot \dfrac{{0.5 \cdot \textcolor{{#0A84FF}}{{{p['Ko']:.3f}}} + \textcolor{{#0A84FF}}{{{p['O']:.3f}}}}}{{ \textcolor{{#0A84FF}}{{{p['Ko']:.3f}}} + \textcolor{{#0A84FF}}{{{p['O']:.3f}}}}}''')
        st.latex(
            fr'''\dfrac{{S_o - S_e}}{{HRT}} = \dfrac{{\mu_a \cdot X_a \cdot A_b}}{{\textcolor{{#0A84FF}}{{{p['Ya']:.3f}}} \cdot V}} + \dfrac{{\mu_s \cdot X_s}}{{\textcolor{{#0A84FF}}{{{p['Ys']:.3f}}}}}''')
        st.markdown("""<div class="variables"><b>Where:</b><ul><li><em>&mu;<sub>a</sub></em>: Growth Rate (day<sup>-1</sup>)</li><li><em>S<sub>o</sub>, S<sub>e</sub></em>: Substrate Conc. (mg/L)</li><li><em>X<sub>a</sub></em>: Attached Biomass (g/m²)</li><li><em>X<sub>s</sub></em>: Suspended Biomass (mg/L)</li><li><em>HRT</em>: Retention Time (days)</li><li><em>V, A<sub>b</sub></em>: Volume (m³), Area (m²)</li></ul></div>""", unsafe_allow_html=True)
    if params_rbc_ph:
        p = params_rbc_ph
        exponent_val = 0.5 * (p['pH_min'] - p['pH_max'])
        st.markdown(
            f"""<div class="equation-card"><div class="title">RBC v1.1 (pH-Inhibited)</div></div>""", unsafe_allow_html=True)
        st.latex(
            fr'''\tau_{{pH}} = \dfrac{{ 1 + 2 \cdot 10^{{{exponent_val:.2f}}} }}{{ 1 + 10^{{pH - \textcolor{{#0A84FF}}{{{p['pH_max']:.2f}}}}} + 10^{{\textcolor{{#0A84FF}}{{{p['pH_min']:.2f}}} - pH}} }}''')
        st.latex(
            fr'''\mu_a = \tau_{{pH}} \cdot \dfrac{{\textcolor{{#0A84FF}}{{{p['umxa']:.3f}}} \cdot S_e}}{{\textcolor{{#0A84FF}}{{{p['Ku']:.3f}}} + S_e}} \cdot \dfrac{{0.5 \cdot \textcolor{{#0A84FF}}{{{p['Ko']:.3f}}} + \textcolor{{#0A84FF}}{{{p['O']:.3f}}}}}{{ \textcolor{{#0A84FF}}{{{p['Ko']:.3f}}} + \textcolor{{#0A84FF}}{{{p['O']:.3f}}}}}''')
        st.latex(
            fr'''\dfrac{{S_o - S_e}}{{HRT}} = \dfrac{{\mu_a \cdot X_a \cdot A_b}}{{\textcolor{{#0A84FF}}{{{p['Ya']:.3f}}} \cdot V}} + \dfrac{{\mu_s \cdot X_s}}{{\textcolor{{#0A84FF}}{{{p['Ys']:.3f}}}}}''')
        st.markdown("""<div class="variables"><b>Where:</b><ul><li><em>&tau;<sub>pH</sub></em>: pH Inhibition Factor (dimensionless)</li><li><em>&mu;<sub>a</sub></em>: Growth Rate (day<sup>-1</sup>)</li><li><em>S<sub>e</sub></em>: Effluent Substrate Conc. (mg/L)</li></ul></div>""", unsafe_allow_html=True)

# FIX: Added helper to convert dataframe to CSV for downloading


@st.cache_data
def to_csv(_df):
    return _df.to_csv(index=False).encode('utf-8')


# ==============================================================================
# 5. Sidebar and State Management
# ==============================================================================
if 'reactor' not in st.session_state:
    st.session_state.reactor = None
if 'sa_results' not in st.session_state:
    st.session_state.sa_results = None
if 'edited_params' not in st.session_state:
    st.session_state.edited_params = None
if 'optimizer_results' not in st.session_state:
    st.session_state.optimizer_results = None
if 'data_content' not in st.session_state:
    st.session_state.data_content = None


with st.sidebar:
    st.header(t('sidebar_controls'))
    # FIX: Added option to use default data from GitHub
    st.subheader(t('sidebar_upload'))
    uploaded_file = st.file_uploader(
        "", type=["csv"], help=t('sidebar_upload_help'))
    use_github_data = st.checkbox(
        t('sidebar_github'), value=True, help=t('sidebar_github_help'))
    github_url = "https://raw.githubusercontent.com/RizkyBaldan/KINEMOD-KIT/main/Model_Data.csv"

    # Store the latest data content in session state
    st.session_state.data_content = get_data_content(
        uploaded_file, use_github_data, github_url)

    if use_github_data and not uploaded_file:
        st.info(f"Using default data from [GitHub]({github_url})")

    st.subheader(t('sidebar_actions'))
    CONFIG = {"reactor_constants": {"V_RBC": 18, "A_RBC": 240}, "optimization_bounds": {"UASB": [(1.4, 75.9), (4.9, 25.0), (1.9, 8.0)], "Filter": [(0.01, 4.0), (0.01, 2.5)], "RBC": [(
        5, 30), (100, 500), (20, 60), (1, 15), (300, 450), (1, 10), (0.01, 1), (0.01, 2.5)], "RBC_pH": [(5, 50), (50, 100), (20, 30), (1, 5), (1, 5), (1, 10), (0.01, 2.5), (0.01, 2.5), (3.0, 5), (9, 11.5)]}}
    with st.expander(t('sidebar_advanced_config')):
        CONFIG["reactor_constants"]["V_RBC"] = st.number_input(
            t('sidebar_rbc_vol'), value=18.0)
        CONFIG["reactor_constants"]["A_RBC"] = st.number_input(
            t('sidebar_rbc_area'), value=240.0)
    col1, col2 = st.columns(2)
    with col1:
        run_estimation_button = st.button(
            t('sidebar_calibrate'), type="primary", use_container_width=True, help=t('sidebar_calibrate_help'))
    with col2:
        load_params_button = st.button(
            t('sidebar_load'), use_container_width=True, help=t('sidebar_load_help'))

    if st.session_state.reactor:
        st.markdown("---")
        st.header(t('what_if_header'))
        st.info(t('what_if_help'))
        what_if_cod = st.slider(t('what_if_cod'), 5000, 40000, int(
            st.session_state.reactor.df['COD_in'].mean()), 500)
        what_if_hrt_uasb = st.slider(
            t('what_if_hrt_uasb'), 1.0, 20.0, st.session_state.reactor.df['HRT_UASB'].mean(), 0.1)
        what_if_hrt_rbc = st.slider(
            t('what_if_hrt_rbc'), 1.0, 120.0, st.session_state.reactor.df['HRT_hours'].mean(), 0.5)
        srr_pred = st.session_state.reactor.uasb_model.predict(
            (what_if_cod / what_if_hrt_uasb, 0.3), st.session_state.reactor.params_uasb)
        cod_uasb_pred = what_if_cod - srr_pred * what_if_hrt_uasb
        final_pred = st.session_state.reactor.rbc_model_ph.predict(
            {'So': cod_uasb_pred * 0.8, 'HRT_days': what_if_hrt_rbc / 24.0, 'Xa': 400, 'Xs': 200, 'pH': 7.5}, st.session_state.reactor.params_rbc_ph)
        st.metric(label=t('what_if_result'),
                  value=f"{final_pred:.0f} mg/L" if not np.isnan(final_pred) else "N/A")

action_to_perform = None
if run_estimation_button:
    action_to_perform = "estimate"
if load_params_button:
    action_to_perform = "load"
if action_to_perform and st.session_state.data_content is not None:
    with st.spinner('🔬 Running analysis... This may take a moment.'):
        reactor_obj, error = run_full_analysis(
            st.session_state.data_content, CONFIG, action_to_perform)
        if error:
            st.error(error)
            st.session_state.reactor = None
        else:
            st.success("Analysis complete!")
            st.session_state.reactor = reactor_obj
            st.session_state.sa_results = None
            st.session_state.edited_params = None
            st.session_state.optimizer_results = None
elif action_to_perform and st.session_state.data_content is None:
    st.warning(
        "Please upload a data file or select the GitHub option first.", icon="⚠️")

# ==============================================================================
# 6. Main App Display
# ==============================================================================
if st.session_state.reactor:
    reactor = st.session_state.reactor
    df_results = reactor.df
    tabs = st.tabs(t('tabs'))

    with tabs[0]:  # Dashboard
        anim_class = get_animated_class()
        st.markdown(f"<div class='{anim_class}'>", unsafe_allow_html=True)

        st.header(f"🎯 {t('kpi_header')}")
        min_day, max_day = int(df_results['Day'].min()), int(
            df_results['Day'].max())
        selected_days = st.slider(
            t('date_slider'), min_day, max_day, (min_day, max_day), key='date_slider_dashboard')
        filtered_df = df_results[(df_results['Day'] >= selected_days[0]) & (
            df_results['Day'] <= selected_days[1])]
        stages = {"UASB": ("COD_UASB_Eff", "COD_UASB_Pred"), "Filter": ("COD_Filt_Eff", "COD_Filt_Pred"), "RBC (Orig)": (
            "COD_Final", "COD_Final_Pred_Orig"), "RBC (pH Mod)": ("COD_Final", "COD_Final_Pred_pH")}
        stage_icons = {"UASB": "🦠", "Filter": "✨",
                       "RBC (Orig)": "🔄", "RBC (pH Mod)": "🌿"}
        kpi_cols = st.columns(4)
        for i, (name, (actual_col, pred_col)) in enumerate(stages.items()):
            if pred_col in filtered_df.columns:
                valid_df = filtered_df.dropna(subset=[actual_col, pred_col])
                if not valid_df.empty:
                    r2, rmse, p_value = r2_score(valid_df[actual_col], valid_df[pred_col]), np.sqrt(mean_squared_error(
                        valid_df[actual_col], valid_df[pred_col])), stats.f_oneway(valid_df[actual_col], valid_df[pred_col])[1]
                    with kpi_cols[i]:
                        create_kpi_card(icon=stage_icons.get(
                            name, "📊"), title=name, r2=r2, rmse=rmse, p_value=p_value)
        st.markdown("---")
        st.subheader(t('timeseries_header'))
        col_left, col_right = st.columns([1, 2.5])
        with col_left:
            compliance_limit = st.number_input(
                t('compliance_input'), value=350, min_value=0, step=25)
            compliance_pct = ((filtered_df['COD_Final_Pred_pH'] <= compliance_limit).mean(
            ) * 100 if 'COD_Final_Pred_pH' in filtered_df.columns and not filtered_df['COD_Final_Pred_pH'].isna().all() else 0)
            st.metric(t('kpi_compliance_header'), f"{compliance_pct:.1f}%")
        with col_right:
            log_y = st.checkbox(t('log_axis_toggle'))
            fig_timeseries = plot_interactive_timeseries(
                filtered_df, compliance_limit, log_y)
            st.plotly_chart(fig_timeseries, use_container_width=True)

        st.markdown("---")
        st.subheader(t('parity_header'))
        stage_choice = st.selectbox(
            "Select Stage", ["UASB", "Filter", "RBC (Orig)", "RBC (pH Mod)"])
        pred_dict, actual = ({"UASB": "COD_UASB_Pred"}, "COD_UASB_Eff") if stage_choice == "UASB" else ({"Filter": "COD_Filt_Pred"}, "COD_Filt_Eff") if stage_choice == "Filter" else (
            {"Original": "COD_Final_Pred_Orig"}, "COD_Final") if stage_choice == "RBC (Orig)" else ({"pH Model": "COD_Final_Pred_pH"}, "COD_Final")
        par_col, err_col = st.columns([2, 1])
        with par_col:
            fig_parity = plot_interactive_parity(
                filtered_df, stage_choice, actual, pred_dict)
            st.plotly_chart(fig_parity, use_container_width=True)
        with err_col:
            error_dfs = [pd.DataFrame({"Error": filtered_df[pred_col] - filtered_df[actual], "Model": model_name}).dropna()
                         for model_name, pred_col in pred_dict.items() if pred_col in filtered_df.columns]
            if error_dfs:
                full_error_df = pd.concat(error_dfs)
                fig_hist = px.histogram(full_error_df, x="Error", color="Model",
                                        marginal="box", barmode="overlay", title="Error Distribution")
                fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_hist, use_container_width=True)

        # FIX: Added download section
        with st.expander("📥 Download Data & Charts"):
            st.download_button(
                label="Download Results Data as CSV",
                data=to_csv(filtered_df),
                file_name=f"filtered_results_{selected_days[0]}-{selected_days[1]}.csv",
                mime="text/csv",
            )
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.download_button(
                    label="Download Time-Series Chart (PNG)",
                    data=pio.to_image(fig_timeseries, format='png',
                                      width=1200, height=700, scale=2),
                    file_name="timeseries_plot.png",
                    mime="image/png"
                )
            with d_col2:
                st.download_button(
                    label="Download Parity Chart (PNG)",
                    data=pio.to_image(fig_parity, format='png',
                                      width=800, height=600, scale=2),
                    file_name="parity_plot.png",
                    mime="image/png"
                )

        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:  # Model Details (no changes)
        anim_class = get_animated_class()
        st.markdown(f"<div class='{anim_class}'>", unsafe_allow_html=True)
        st.header(t('param_header'))
        param_tab, eq_tab = st.tabs(
            ["Calibrated Parameters", "Governing Equations"])
        with param_tab:
            st.info(t('editable_param_info'))
            param_sets = [("UASB", reactor.params_uasb), ("Filter", reactor.params_filter), (
                "RBC (Orig)", reactor.params_rbc_orig), ("RBC (pH)", reactor.params_rbc_ph)]
            if 'edited_params' not in st.session_state or st.session_state.edited_params is None:
                all_params = [{"Model": model, "Parameter": key, "Value": value}
                              for model, params in param_sets if params for key, value in params.items()]
                if all_params:
                    st.session_state.edited_params = pd.DataFrame(all_params)
            if st.session_state.edited_params is not None:
                with st.expander("UASB Parameters", expanded=True):
                    uasb_params = st.data_editor(
                        st.session_state.edited_params[st.session_state.edited_params['Model'] == 'UASB'], hide_index=True)
                with st.expander("Filter Parameters"):
                    filter_params = st.data_editor(
                        st.session_state.edited_params[st.session_state.edited_params['Model'] == 'Filter'], hide_index=True)
                with st.expander("RBC Parameters"):
                    rbc_orig_params = st.data_editor(
                        st.session_state.edited_params[st.session_state.edited_params['Model'] == 'RBC (Orig)'], hide_index=True)
                    rbc_ph_params = st.data_editor(
                        st.session_state.edited_params[st.session_state.edited_params['Model'] == 'RBC (pH)'], hide_index=True)
                if st.button(t('rerun_button'), type="primary"):
                    edited_params_df = pd.concat(
                        [uasb_params, filter_params, rbc_orig_params, rbc_ph_params])
                    st.session_state.edited_params = edited_params_df
                    with st.spinner("Re-running predictions..."):
                        p_uasb, p_filter, p_rbc_o, p_rbc_ph = [dict(edited_params_df[edited_params_df['Model'] == m].set_index(
                            'Parameter')['Value']) for m in ["UASB", "Filter", "RBC (Orig)", "RBC (pH)"]]
                        # The original df is part of the reactor object, no need to rerun analysis
                        new_df = reactor.run_predictions_with_new_params(
                            reactor.df, p_uasb, p_filter, p_rbc_o, p_rbc_ph)
                        st.session_state.reactor.df = new_df
                        st.success("Predictions updated!")
        with eq_tab:
            display_final_equations(
                reactor.params_uasb, reactor.params_filter, reactor.params_rbc_orig, reactor.params_rbc_ph)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:  # Methane & Energy (no changes)
        anim_class = get_animated_class()
        st.markdown(f"<div class='{anim_class}'>", unsafe_allow_html=True)
        st.header(t('methane_header'))
        filtered_df = df_results[(df_results['Day'] >= st.session_state.get('date_slider_dashboard', (df_results['Day'].min(), df_results['Day'].max()))[
                                  0]) & (df_results['Day'] <= st.session_state.get('date_slider_dashboard', (df_results['Day'].min(), df_results['Day'].max()))[1])]
        uasb_cod_removed = (filtered_df['COD_in'] - filtered_df['COD_UASB_Pred']
                            ).mean() if 'COD_UASB_Pred' in filtered_df else 0
        uasb_volume = st.number_input(
            t('uasb_volume_label'), value=25.0, min_value=0.1)
        avg_flow_rate = uasb_volume / (filtered_df['HRT_UASB'].mean(
        ) if not filtered_df.empty and 'HRT_UASB' in filtered_df and filtered_df['HRT_UASB'].mean() > 0 else 1)
        cod_mass_removed_kg_day = (uasb_cod_removed * avg_flow_rate) / 1000
        methane_yield = st.slider(
            t('methane_yield_label'), 0.20, 0.40, 0.35, 0.01)
        methane_prod = cod_mass_removed_kg_day * methane_yield
        energy_potential = methane_prod * 9.7  # kWh/m3 CH4
        m_cols = st.columns(3)
        m_cols[0].metric(f"{t('cod_removed_label')} (kg/day)",
                         f"{cod_mass_removed_kg_day:.1f}")
        m_cols[1].metric(
            f"{t('methane_prod_label')} (m³/day)", f"{methane_prod:.1f}")
        m_cols[2].metric(
            f"{t('energy_potential_label')} (kWh/day)", f"{energy_potential:.1f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:  # Sensitivity
        anim_class = get_animated_class()
        st.markdown(f"<div class='{anim_class}'>", unsafe_allow_html=True)
        st.header(t('sa_header'))
        st.info(t('sa_info'))
        sa_col1, sa_col2 = st.columns([1, 2])
        with sa_col1:
            sa_type = st.selectbox(t('sa_type_select'), ["GSA", "Monte Carlo"])
            sa_model = st.selectbox(t('sa_model_select'), [
                                    "RBC Original", "RBC pH-Modified"])
            if st.button(t('run_sa_button'), type="primary", use_container_width=True):
                if st.session_state.data_content:
                    model_key = 'ph' if 'pH' in sa_model else 'orig'
                    with st.spinner(f"Running {sa_type}... this is computationally intensive and may take several minutes."):
                        sa_error, sa_data = run_sensitivity_analysis(
                            sa_type, model_key, reactor.params_uasb, reactor.params_filter, reactor.params_rbc_orig, reactor.params_rbc_ph, st.session_state.data_content, CONFIG)
                        if sa_error:
                            st.error(sa_error)
                            st.session_state.sa_results = None
                        else:
                            st.success("Analysis Complete!")
                            st.session_state.sa_results = sa_data
                else:
                    st.warning(
                        "Please provide data before running analysis.", icon="⚠️")

        if st.session_state.sa_results:
            sa_data = st.session_state.sa_results
            fig_sa = None
            with sa_col2:
                if 'Mi' in sa_data:  # GSA Results
                    st.subheader("GSA Results")
                    sobol_df = pd.DataFrame({k: sa_data['Si'][k] for k in [
                                            'S1', 'ST']}, index=sa_data['problem']['names'])
                    top_param = sobol_df.sort_values(
                        'ST', ascending=False).index[0]
                    st.metric(label="Most Influential Parameter (Sobol ST)",
                              value=top_param, delta=f"{sobol_df.loc[top_param, 'ST']:.3f}")
                    fig_sa = plot_gsa_results(
                        sa_data['Mi'], sa_data['Si'], sa_data['problem'])
                    st.plotly_chart(fig_sa, use_container_width=True)

                elif 'mc_results' in sa_data:  # Monte Carlo Results
                    st.subheader("Monte Carlo Results")
                    mc_df = sa_data['mc_results']
                    top_param = mc_df.reindex(mc_df['Spearman_Correlation'].abs(
                    ).sort_values(ascending=False).index).iloc[0]
                    st.metric(label="Most Correlated Parameter",
                              value=top_param['Parameter'], delta=f"{top_param['Spearman_Correlation']:.3f}")
                    fig_sa = plot_mc_results(mc_df)
                    st.plotly_chart(fig_sa, use_container_width=True)

            # FIX: Added download for sensitivity plot
            if fig_sa:
                st.download_button(
                    label=f"Download {sa_type} Chart (PNG)",
                    data=pio.to_image(fig_sa, format='png',
                                      width=1000, height=500, scale=2),
                    file_name=f"sensitivity_plot_{sa_type}.png",
                    mime="image/png"
                )
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[4]:  # Optimizer (no changes)
        anim_class = get_animated_class()
        st.markdown(f"<div class='{anim_class}'>", unsafe_allow_html=True)
        st.header(t('optimizer_header'))
        st.info(t('optimizer_intro'))
        opt_col1, opt_col2 = st.columns([1, 2])
        with opt_col1:
            st.selectbox(t('optimizer_goal'), [t('optimizer_goal_option')])
            st.subheader(t('optimizer_constraints'))
            cod_bounds = st.slider(
                t('what_if_cod'), 5000, 40000, (15000, 25000))
            hrt_uasb_bounds = st.slider(
                t('what_if_hrt_uasb'), 1.0, 20.0, (1.5, 5.0))
            hrt_rbc_bounds = st.slider(
                t('what_if_hrt_rbc'), 1.0, 120.0, (6.0, 24.0))
            if st.button(t('optimizer_button'), type='primary', use_container_width=True):
                with st.spinner("Finding optimal settings... this may take some time."):
                    bounds = [cod_bounds, hrt_uasb_bounds, hrt_rbc_bounds]
                    initial_guess, fixed_params = [np.mean(b) for b in bounds], {'vfa_alk': df_results['VFA_ALK_Ratio'].mean(), 'tss_in': df_results['TSS_UASB_Eff'].mean(
                    ), 'tss_out': df_results['TSS_Filt_Eff'].mean(), 'xa': df_results['Xa'].mean(), 'xs': df_results['Xs'].mean(), 'ph': df_results['pH_Eff_RBC'].mean()}
                    result = minimize(optimization_objective_function, initial_guess, args=(
                        reactor, fixed_params), bounds=bounds, method='L-BFGS-B')
                    st.session_state.optimizer_results = result
        with opt_col2:
            if st.session_state.optimizer_results:
                res = st.session_state.optimizer_results
                st.subheader(t('optimizer_results_header'))
                if res.success:
                    st.success(t('optimizer_success'))
                    r_cols = st.columns(3)
                    r_cols[0].metric("Optimal Influent COD",
                                     f"{res.x[0]:.0f} mg/L")
                    r_cols[1].metric("Optimal UASB HRT",
                                     f"{res.x[1]:.2f} days")
                    r_cols[2].metric("Optimal RBC HRT",
                                     f"{res.x[2]:.2f} hours")
                    st.metric("Predicted Minimum Final Effluent",
                              f"{res.fun:.1f} mg/L", help="This is the lowest possible effluent COD the optimizer could find within your constraints.")
                else:
                    st.error(t('optimizer_fail'))
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[5]:  # Help (updated text slightly)
        anim_class = get_animated_class()
        st.markdown(f"<div class='{anim_class}'>", unsafe_allow_html=True)
        st.header("❓ Help & Technical Reference")
        query = st.text_input("🔍 Search Technical Topics",
                              placeholder="e.g., 'UASB model', 'kinetic parameters', 'compliance'")

        def match(q, *texts): return (not q) or any(q.lower()
                                                    in (t or "").lower() for t in texts)

        search_anim_class = get_animated_class()
        st.markdown(
            f"<div class='{search_anim_class}'>", unsafe_allow_html=True)
        if match(query, "overview", "pome", "system", "workflow", "remediation"):
            st.subheader("🧭 System Overview & Remediation Strategy")
            st.markdown("""<div class="help-card">The <b>KINEMOD-KIT platform</b> provides a kinetic modeling environment for a synergistic, three-stage bioremediation system designed for high-strength Palm Oil Mill Effluent (POME). The treatment philosophy is based on sequential anaerobic, physical, and aerobic processes to achieve comprehensive pollutant removal.<br><br>🟤 <b>UASB Reactor (Anaerobic Digestion)</b> — This primary stage leverages methanogenic archaea under anaerobic conditions to achieve a substantial reduction in organic load, targeting approximately <b>70% COD removal</b>.<br>🪶 <b>EFB Filtration (Adsorptive Filtration)</b> — Effluent from the UASB is passed through a packed bed of Empty Fruit Bunch (EFB) fibers. This stage serves a dual purpose: physical interception of suspended solids and adsorption of residual oleaginous compounds, achieving over <b>97% TSS removal</b>.<br>🧬 <b>RBC Unit (Aerobic Polishing)</b> — The final stage employs a Rotating Biological Contactor, a fixed-film aerobic reactor, to polish the effluent. This process oxidizes remaining soluble COD to ensure the final discharge quality conforms to regulatory standards, specifically a COD concentration <b>at or below 350 mg/L</b>.</div>""", unsafe_allow_html=True)

        if match(query, "uasb", "filter", "rbc", "reactor", "stage", "model"):
            st.subheader("🔬 Reactor Stages & Kinetic Models")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("""<div class="mini-card"><b>🟤 UASB Reactor</b><br>• <b>Process</b>: Mesophilic (35–38°C) anaerobic digestion for biogas production.<br>• <b>Function</b>: Primary organic load reduction via methanogenesis.<br>• <b>Kinetic Model</b>: A modified Stover–Kincannon substrate utilization model, incorporating a non-competitive inhibition term for the VFA/Alkalinity ratio.</div>""", unsafe_allow_html=True)
            with c2:
                st.markdown("""<div class="mini-card"><b>🪶 EFB Adsorption Filter</b><br>• <b>Process</b>: Physico-chemical adsorption and filtration.<br>• <b>Function</b>: Removes recalcitrant organics, lipids, and total suspended solids (TSS).<br>• <b>Mechanism</b>: Utilizes lignocellulosic EFB fibers for physical interception and surface chemistry-driven adsorption of pollutants.</div>""", unsafe_allow_html=True)
            with c3:
                st.markdown("""<div class="mini-card"><b>🧬 RBC Unit</b><br>• <b>Process</b>: Aerobic fixed-film biological oxidation.<br>• <b>Function</b>: Final polishing of effluent to reduce soluble COD to regulatory levels.<br>• <b>Mechanism</b>: Rotating discs support a high-density, attached-growth biofilm, enhancing oxygen transfer and treatment efficiency.</div>""", unsafe_allow_html=True)

        if match(query, "operation", "performance", "guidelines", "troubleshooting"):
            st.subheader("⚙️ Operational & Performance Guidelines")
            st.markdown("""<div class="help-card"><b>Recommended Hydraulic Retention Time (HRT):</b> For optimal performance, the UASB reactor requires <b>2 to 6 days</b>, while the RBC unit requires <b>10 to 60 hours</b>.<br><b>Expected Biogas Yield:</b> The system is designed to produce approximately <b>0.31 cubic meters of methane</b> per kilogram of COD removed in the UASB stage.<br>• <b>Biofilm Sloughing:</b> Periodic detachment of excess biofilm is normal. If excessive, this may indicate hydraulic shock. Consider **increasing the backwash frequency**.<br>• <b>COD Rebound:</b> A sudden increase in effluent COD may indicate a nutrient imbalance. Verify the C:N:P ratio is maintained near the optimal <b>100:5:1</b>.<br>• <b>Low Methane Production:</b> Insufficient buffering capacity can inhibit methanogenesis. Ensure the process maintains an alkalinity that <b>exceeds 1500 mg/L as CaCO₃</b> to neutralize volatile fatty acids.</div>""", unsafe_allow_html=True)

        if match(query, "interpretation", "analysis", "metric", "dashboard", "statistics"):
            st.subheader("📊 Guide to Data Interpretation")
            st.markdown("""<div class="help-card">• <b>R² (Coefficient of Determination)</b> → This metric indicates the proportion of variance in the measured data that is predictable from the model. A value <b>approaching 1.0</b> signifies a strong model fit.<br>• <b>RMSE (Root Mean Square Error)</b> → Represents the standard deviation of the prediction errors (residuals). A <b>lower RMSE value</b> indicates higher prediction accuracy, in the same units as the response variable (mg/L COD).<br>• <b>ANOVA p-value</b> → Used to test the statistical significance between datasets. A p-value <b>less than 0.05</b> typically indicates that the observed differences between the model's predictions and the measured data are statistically significant.<br><br><b>Parity Plot Analysis:</b> This visualization plots predicted values against measured values. For a perfect model, all data points would lie on the <b>diagonal y=x line</b>. Deviations from this line indicate systemic under- or over-prediction.<br><b>Time-Series Analysis:</b> This chart tracks model performance over time, which is critical for identifying trends, cyclical patterns, or responses to operational changes.</div>""", unsafe_allow_html=True)

        if match(query, "standard", "kepmen", "limit", "compliance", "regulatory"):
            st.subheader("🌿 Regulatory Compliance Standards (Indonesia)")
            st.markdown("""<div class="help-card">The effluent quality targets are based on the Indonesian ministerial decree: <b>Kepmen LH No. 5 Tahun 2014</b>, regarding the Quality Standards for Liquid Waste from the Palm Oil Industry.<br><table class="styled-table"><tr><th>Parameter</th><th>Maximum Permissible Limit</th></tr><tr><td>COD (Chemical Oxygen Demand)</td><td>Must be <b>at or below 350 mg/L</b></td></tr><tr><td>BOD (Biochemical Oxygen Demand)</td><td>Must be <b>at or below 100 mg/L</b></td></tr><tr><td>TSS (Total Suspended Solids)</td><td>Must be <b>at or below 150 mg/L</b></td></tr><tr><td>Oil & Grease</td><td>Must be <b>at or below 25 mg/L</b></td></tr><tr><td>pH</td><td>Must be within the range of <b>6.0 to 9.0</b></td></tr></table></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.subheader("📤 Feedback & Support")
        with st.form("feedback_form"):
            email = st.text_input("Email Address (Optional)")
            msg = st.text_area("Provide feedback or report an issue")
            sent = st.form_submit_button("Submit Feedback 🍏")
            if sent:
                if not msg.strip():
                    st.warning("⚠️ Please enter a message before submitting.")
                else:
                    st.success(
                        "✅ Feedback submitted successfully. Thank you for helping to improve KINEMOD-KIT!")
        st.markdown("<center style='opacity:0.7; font-size:0.85rem;'>Built with ❤️ by Rizky Mursyidan Baldan — KINEMOD-KIT Thesis Project</center>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info(t('welcome_message'))
    st.image("https://raw.githubusercontent.com/RizkyBaldan/KINEMOD-KIT/main/Diagram.png",
             caption="Diagram of the multi-stage treatment process.", use_column_width=True)
