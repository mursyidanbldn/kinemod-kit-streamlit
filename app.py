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
import io
import time

from model_logic import IntegratedReactor

# ==============================================================================
# 1. Page Config, Language, and Theming
# ==============================================================================
st.set_page_config(page_title="KINEMOD-KIT", page_icon="🧪",
                   layout="wide", initial_sidebar_state="expanded")

# --- Translations for multi-language support ---
translations = {
    "en": {
        "title": "KINEMOD-KIT", "subtitle": "UASB-FILTRATION-RBC REACTOR KINETIC MODELING KIT",
        "author_line": "By Rizky Mursyidan Baldan | Last Updated: {}",
        "tabs": ["🏠 Home", "📊 Dashboard", "🔬 Model Details", "🍃 Methane & Energy", "🔬 Sensitivity", "⚙️ Optimizer", "❓ Help"],
        "dashboard_header": "Performance Dashboard",
        "welcome_message": "👋 **Welcome!** Please upload your data or select the option to use the default dataset in the sidebar to begin.",
        "sidebar_controls": "⚙️ Controls", "sidebar_upload": "Upload data (CSV)", "sidebar_upload_help": "Upload a custom `Model_Data.csv` file.",
        "sidebar_use_default": "Use default `Model_Data.csv` from GitHub",
        "sidebar_advanced_config": "🔬 Advanced Config", "sidebar_rbc_vol": "RBC Volume (V_RBC)", "sidebar_rbc_area": "RBC Surface Area (A_RBC)",
        "sidebar_calibrate": "🚀 Calibrate", "sidebar_calibrate_help": "Run parameter estimation.",
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
        "sa_header": "🔬 Sensitivity Analysis", "sa_info": "Evaluate how uncertainty in model parameters affects the output. This is computationally intensive.",
        "sa_type_select": "Select Analysis Type", "sa_model_select": "Select RBC Model", "run_sa_button": "🚀 Run Analysis",
        "sa_mode_select": "Select Analysis Mode", "sa_mode_specific": "Run a Specific Analysis", "sa_mode_all": "Run All Methods and Models",
        "what_if_header": "⚡ \"What-If\" Scenarios", "what_if_help": "Adjust inputs to see a quick prediction of the final effluent quality based on the pH model.",
        "what_if_cod": "Influent COD (mg/L)", "what_if_hrt_uasb": "UASB HRT (days)", "what_if_hrt_rbc": "RBC HRT (hours)", "what_if_result": "Predicted Final Effluent (pH Model)",
        "optimizer_header": "⚙️ Process Optimizer", "optimizer_intro": "This tool uses an optimization algorithm to find the best operational inputs to achieve a selected goal within your defined constraints.",
        "optimizer_goal": "Optimization Goal", "optimizer_goal_option": "Minimize Final Effluent COD",
        "optimizer_constraints": "Set Operational Constraints (Search Space)",
        "optimizer_button": "🚀 Find Optimal Settings", "optimizer_results_header": "🏆 Optimization Results", "optimizer_success": "Optimization Successful!", "optimizer_fail": "Optimization failed to converge.",
        "help_title": "❓ Help & Documentation",
        "help_intro": "This is a comprehensive tool for analyzing and simulating the performance of a three-stage POME treatment system. This guide will help you understand its features.",
        "help_usage_title": "Workflow Guide", "help_usage_step1_title": "1. Provide Data", "help_usage_step1_text": "Start by uploading your `Model_Data.csv` file, or check the box to use the default dataset from GitHub.",
        "help_usage_step2_title": "2. Calibrate Model", "help_usage_step2_text": "Click **Calibrate** to estimate model parameters from your data.",
        "help_usage_step3_title": "3. Analyze Dashboard", "help_usage_step3_text": "The Dashboard shows model performance via KPIs, time-series charts, and parity plots. Use the date slider to focus on specific periods.",
        "help_usage_step4_title": "4. Explore Tabs", "help_usage_step4_text": "Dive deeper into other tabs to view model equations, run sensitivity analysis, calculate energy potential, or optimize process parameters.",
        "help_model_docs_title": "Model Documentation", "help_uasb_title": "UASB Model", "help_uasb_obj": "**Objective**: Predict the Substrate Removal Rate (SRR) based on Organic Loading Rate (OLR) and VFA/Alkalinity inhibition.",
        "help_filter_title": "Adsorption & Filtration Model", "help_filter_obj": "**Objective**: Quantify COD removal through two mechanisms: physical filtration of suspended solids and adsorption of soluble COD.",
        "help_rbc_title": "RBC Biofilm Model", "help_rbc_obj": "**Objective**: Simulate substrate removal in the biofilm, considering factors like biomass concentration, oxygen levels, and an optional pH inhibition factor.",
        "help_params_title": "Key Parameters", "help_inputs_title": "Inputs", "help_outputs_title": "Outputs",
        "help_tab_guide_title": "Guide to App Features", "help_tab_dashboard_desc": "Your main analysis hub. It shows KPIs (R², RMSE), time-series predictions vs. actual data, and parity plots to evaluate model accuracy.",
        "help_tab_model_details_desc": "View the calibrated kinetic parameters in an editable table. You can tweak values and re-run predictions. Also displays the final governing equations with the calibrated values.",
        "help_tab_methane_desc": "Calculates the potential for methane production and energy generation from the COD removed in the UASB reactor. Adjust reactor volume and methane yield for scenario analysis.",
        "help_tab_sensitivity_desc": "Perform Global Sensitivity Analysis (GSA) or Monte Carlo simulations to identify which model parameters have the most significant impact on the final effluent COD.",
        "help_tab_optimizer_desc": "Find the optimal operational conditions (Influent COD, UASB HRT, RBC HRT) that minimize the final effluent COD, given a set of constraints you define.",
        "help_faq_title": "FAQ & Troubleshooting", "help_faq1_q": "Why does calibration fail or take a long time?", "help_faq1_a": "Calibration is a complex optimization process. It can fail if the data is very noisy, contains many outliers, or if the initial parameter guesses are far from the optimal values. Ensure your data is clean and covers a wide range of operational conditions.",
        "help_faq2_q": "What's the difference between RBC 'Original' and 'pH Mod' models?", "help_faq2_a": "The 'Original' model is a standard biofilm model. The 'pH Mod' version adds a pH inhibition factor (τ_pH), making the model's predictions sensitive to pH fluctuations, which is common in biological treatment systems.",
        "filtered_data_header": "📊 Filtered Data",
    },
    "id": {
        "title": "KINEMOD-KIT", "subtitle": "KIT PEMODELAN KINETIK REAKTOR UASB-FILTRASI-RBC",
        "author_line": "Oleh Rizky Mursyidan Baldan | Terakhir Diperbarui: {}",
        "tabs": ["🏠 Beranda", "📊 Dasbor", "🔬 Detail Model", "🍃 Metana & Energi", "🔬 Sensitivitas", "⚙️ Pengoptimal", "❓ Bantuan"],
        "dashboard_header": "Dasbor Kinerja",
        "welcome_message": "👋 **Selamat datang!** Silakan unggah data Anda atau pilih opsi untuk menggunakan dataset default di bilah sisi untuk memulai.",
        "sidebar_controls": "⚙️ Kontrol", "sidebar_upload": "Unggah data (CSV)", "sidebar_upload_help": "Unggah file `Model_Data.csv` kustom.",
        "sidebar_use_default": "Gunakan `Model_Data.csv` default dari GitHub",
        "sidebar_advanced_config": "🔬 Konfigurasi Lanjutan", "sidebar_rbc_vol": "Volume RBC (V_RBC)", "sidebar_rbc_area": "Luas Permukaan RBC (A_RBC)",
        "sidebar_calibrate": "🚀 Kalibrasi", "sidebar_calibrate_help": "Jalankan estimasi parameter.",
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
        "sa_header": "🔬 Analisis Sensitivitas", "sa_info": "Evaluasi bagaimana ketidakpastian parameter mempengaruhi output. Proses ini intensif secara komputasi.",
        "sa_type_select": "Pilih Jenis Analisis", "sa_model_select": "Pilih Model RBC", "run_sa_button": "🚀 Jalankan Analisis",
        "sa_mode_select": "Pilih Mode Analisis", "sa_mode_specific": "Jalankan Analisis Spesifik", "sa_mode_all": "Jalankan Semua Metode dan Model",
        "what_if_header": "⚡ Skenario \"What-If\"", "what_if_help": "Sesuaikan input ini untuk melihat prediksi cepat kualitas efluen akhir berdasarkan model pH.",
        "what_if_cod": "COD Influen (mg/L)", "what_if_hrt_uasb": "HRT UASB (hari)", "what_if_hrt_rbc": "HRT RBC (jam)", "what_if_result": "Prediksi Efluen Akhir (Model pH)",
        "optimizer_header": "⚙️ Pengoptimal Proses", "optimizer_intro": "Alat ini menggunakan algoritma optimisasi untuk menemukan input operasional terbaik untuk mencapai tujuan yang dipilih dalam batasan yang Anda tentukan.",
        "optimizer_goal": "Tujuan Optimisasi", "optimizer_goal_option": "Minimalkan Efluen Akhir COD",
        "optimizer_constraints": "Atur Batasan Operasional (Ruang Pencarian)",
        "optimizer_button": "🚀 Temukan Pengaturan Optimal", "optimizer_results_header": "🏆 Hasil Optimisasi", "optimizer_success": "Optimisasi Berhasil!", "optimizer_fail": "Optimisasi gagal mencapai konvergensi.",
        "help_title": "❓ Bantuan & Dokumentasi",
        "help_intro": "Ini adalah alat komprehensif untuk menganalisis dan menyimulasikan kinerja sistem pengolahan POME tiga tahap. Panduan ini akan membantu Anda memahami fitur-fiturnya.",
        "help_usage_title": "Panduan Alur Kerja", "help_usage_step1_title": "1. Sediakan Data", "help_usage_step1_text": "Mulailah dengan mengunggah file `Model_Data.csv` Anda, atau centang kotak untuk menggunakan dataset default dari GitHub.",
        "help_usage_step2_title": "2. Kalibrasi Model", "help_usage_step2_text": "Klik **Kalibrasi** untuk mengestimasi parameter model dari data Anda.",
        "help_usage_step3_title": "3. Analisis Dasbor", "help_usage_step3_text": "Dasbor menampilkan kinerja model melalui KPI, grafik runtun waktu, dan plot paritas. Gunakan slider tanggal untuk fokus pada periode tertentu.",
        "help_usage_step4_title": "4. Jelajahi Tab Lain", "help_usage_step4_text": "Pelajari lebih dalam di tab lain untuk melihat persamaan model, menjalankan analisis sensitivitas, menghitung potensi energi, atau mengoptimalkan parameter proses.",
        "help_model_docs_title": "Dokumentasi Model", "help_uasb_title": "Model UASB", "help_uasb_obj": "**Tujuan**: Memprediksi Laju Penghilangan Substrat (SRR) berdasarkan Beban Pemuatan Organik (OLR) dan inhibisi rasio VFA/Alkalinitas.",
        "help_filter_title": "Model Adsorpsi & Filtrasi", "help_filter_obj": "**Tujuan**: Mengukur penghilangan COD melalui dua mekanisme: filtrasi fisik padatan tersuspensi dan adsorpsi COD terlarut.",
        "help_rbc_title": "Model Biofilm RBC", "help_rbc_obj": "**Tujuan**: Menyimulasikan penghilangan substrat dalam biofilm, dengan mempertimbangkan faktor-faktor seperti konsentrasi biomassa, kadar oksigen, dan faktor inhibisi pH opsional.",
        "help_params_title": "Parameter Kunci", "help_inputs_title": "Input", "help_outputs_title": "Output",
        "help_tab_guide_title": "Panduan Fitur Aplikasi", "help_tab_dashboard_desc": "Pusat analisis utama Anda. Menampilkan KPI (R², RMSE), perbandingan prediksi vs data aktual dalam runtun waktu, dan plot paritas untuk mengevaluasi akurasi model.",
        "help_tab_model_details_desc": "Lihat parameter kinetik hasil kalibrasi dalam tabel yang dapat diedit. Anda dapat mengubah nilai dan menjalankan ulang prediksi. Juga menampilkan persamaan utama dengan nilai yang telah dikalibrasi.",
        "help_tab_methane_desc": "Menghitung potensi produksi metana dan pembangkitan energi dari COD yang dihilangkan di reaktor UASB. Sesuaikan volume reaktor dan hasil metana untuk analisis skenario.",
        "help_tab_sensitivity_desc": "Lakukan Analisis Sensitivitas Global (GSA) atau simulasi Monte Carlo untuk mengidentifikasi parameter model mana yang memiliki dampak paling signifikan terhadap COD efluen akhir.",
        "help_tab_optimizer_desc": "Temukan kondisi operasional optimal (COD Influen, HRT UASB, HRT RBC) yang meminimalkan COD efluen akhir, berdasarkan batasan yang Anda tentukan.",
        "help_faq_title": "FAQ & Penyelesaian Masalah", "help_faq1_q": "Mengapa kalibrasi gagal atau memakan waktu lama?", "help_faq1_a": "Kalibrasi adalah proses optimisasi yang kompleks. Bisa gagal jika data sangat bising, mengandung banyak pencilan, atau jika tebakan parameter awal jauh dari nilai optimal. Pastikan data Anda bersih dan mencakup berbagai kondisi operasional.",
        "help_faq2_q": "Apa perbedaan antara model RBC 'Original' dan 'pH Mod'?", "help_faq2_a": "Model 'Original' adalah model biofilm standar. Versi 'pH Mod' menambahkan faktor inhibisi pH (τ_pH), membuat prediksi model sensitif terhadap fluktuasi pH, yang umum terjadi pada sistem pengolahan biologis.",
        "filtered_data_header": "📊 Data Terfilter",
    }
}

# --- Initialize Session State ---
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'
if 'reactor' not in st.session_state:
    st.session_state.reactor = None
if 'sa_results' not in st.session_state:
    st.session_state.sa_results = None
if 'edited_params' not in st.session_state:
    st.session_state.edited_params = None
if 'optimizer_results' not in st.session_state:
    st.session_state.optimizer_results = None

# --- Language Selection ---
lang_options = {"English": "en", "Bahasa Indonesia": "id"}
lang_selection = st.sidebar.selectbox(
    "Language / Bahasa", options=list(lang_options.keys()))
st.session_state.lang = lang_options[lang_selection]
def t(key): return translations[st.session_state.lang].get(key, key)


# --- Plotly and Color Theming ---
pio.templates["macos"] = go.layout.Template(layout=go.Layout(
    colorway=['#0A84FF', '#30D158', '#FF9F0A',
              '#FF453A', '#BF5AF2', '#FFD60A', '#64D2FF'],
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'))
pio.templates.default = "macos"

model_colors = {
    'UASB': '#0A84FF', 'Filter': '#30D158', 'RBC_Orig': '#FF9F0A',
    'RBC_pH': '#FF453A', 'Original': '#FF9F0A', 'pH Model': '#FF453A',
    'Measured': '#555555'
}

# ==============================================================================
# 2. Global CSS Styles
# ==============================================================================
# REFACTOR: Replaced dynamic animations with a single, static CSS class for performance.
GLOBAL_STYLES = """
<style>
    /* Base & Font */
    html, body, [class*="css"] { font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", sans-serif; }
    /* Static Animation */
    @keyframes softFadeInUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .fade-in-up { animation: softFadeInUp 0.6s ease-out forwards; }
    /* KPI Card */
    .kpi-card { display: flex; flex-direction: column; justify-content: space-between; background: rgba(255, 255, 255, 0.72); backdrop-filter: blur(10px); border-radius: 18px; padding: 1.5rem; box-shadow: 0 4px 15px rgba(0,0,0,0.08); border: 1px solid rgba(230,230,230,0.5); transition: transform 0.35s ease, box-shadow 0.35s ease; height: 100%; }
    .kpi-card:hover { transform: translateY(-5px); box-shadow: 0 10px 30px rgba(0,0,0,0.12); }
    .card-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; }
    .kpi-icon { display: flex; align-items: center; justify-content: center; width: 42px; height: 42px; font-size: 1.5rem; background-color: #F0F0F5; color: #333; border-radius: 50%; }
    .kpi-title { font-weight: 600; font-size: 1rem; color: #333; }
    .metric-value { font-size: 2.0rem; font-weight: 700; color: #000; line-height: 1.2; margin-bottom: 0.1rem; }
    .metric-label { font-size: 0.9rem; color: #666; }
    .anova { margin-top: 1rem; }
    .anova-label { font-size: 0.8rem; color: #777; }
    .anova-value { font-size: 1rem; font-weight: 600; }
    .badge { display: inline-block; padding: 0.25rem 0.6rem; border-radius: 10px; font-size: 0.7rem; font-weight: 600; margin-top: 0.25rem; }
    .badge.green { background-color: #E2F6E9; color: #155724; }
    .badge.red { background-color: #FCE8E6; color: #7F1D1D; }
    /* Section Container */
    .section-container { background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.04); }
    /* Help Tab */
    .help-card { background: rgba(255,255,255,0.7); backdrop-filter: blur(12px); border-radius: 16px; padding: 1.3rem 1.5rem; margin-bottom: 1rem; box-shadow: 0 4px 16px rgba(0,0,0,0.05); border: 1px solid rgba(255,255,255,0.4); }
    .divider { margin: 2rem 0; border-top: 1px solid rgba(200,200,200,0.4); }
</style>
"""
st.markdown(GLOBAL_STYLES, unsafe_allow_html=True)

# ==============================================================================
# 3. Caching & Backend Logic
# ==============================================================================


@st.cache_data
def load_default_data():
    """OPTIMIZATION: Caches the default data file to prevent reading from disk on every rerun."""
    try:
        with open('Model_Data.csv', 'rb') as f:
            return f.read()
    except FileNotFoundError:
        return None


@st.cache_data
def run_full_analysis(_uploaded_file_content, config):
    data_buffer = io.BytesIO(_uploaded_file_content)
    reactor = IntegratedReactor(data_buffer, config["reactor_constants"])
    if not reactor.load_and_prepare_data():
        return None, "Failed to load or process data."
    reactor.run_parameter_estimation(config["optimization_bounds"])
    reactor.run_full_predictions()
    return reactor, None


@st.cache_data
def run_sensitivity_analysis(analysis_type, model_key, params_uasb, params_filter, rbc_params, _uploaded_file_content, config):
    data_buffer = io.BytesIO(_uploaded_file_content)
    reactor = IntegratedReactor(data_buffer, config["reactor_constants"])
    reactor.load_and_prepare_data()

    # We only need to set the parameters that are actually used in the analysis
    reactor.params_uasb = params_uasb
    reactor.params_filter = params_filter
    if model_key == 'ph':
        reactor.params_rbc_ph = rbc_params
    else:
        reactor.params_rbc_orig = rbc_params
    reactor.params_estimated = True

    # The logic here becomes much simpler and more direct
    params = {**params_uasb, **params_filter, **rbc_params}

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

# ==============================================================================
# 4. UI Helper & Plotting Functions
# ==============================================================================


@st.cache_data
def plot_interactive_timeseries(df, compliance_limit=None, log_y=False):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1, subplot_titles=("UASB", "Filter", "RBC"))

    marker_style = dict(symbol='x', size=8, color=model_colors['Measured'])

    # UASB Plot
    fig.add_trace(go.Scatter(x=df['Day'], y=df['COD_UASB_Eff'], mode='markers',
                  name='UASB Measured', marker=marker_style), row=1, col=1)
    if 'COD_UASB_Pred' in df.columns:
        fig.add_trace(go.Scatter(x=df['Day'], y=df['COD_UASB_Pred'], mode='lines',
                      name='UASB Predicted', line=dict(color=model_colors['UASB'])), row=1, col=1)
    # Filter Plot
    fig.add_trace(go.Scatter(x=df['Day'], y=df['COD_Filt_Eff'], mode='markers',
                  name='Filter Measured', marker=marker_style), row=2, col=1)
    if 'COD_Filt_Pred' in df.columns:
        fig.add_trace(go.Scatter(x=df['Day'], y=df['COD_Filt_Pred'], mode='lines',
                      name='Filter Predicted', line=dict(color=model_colors['Filter'])), row=2, col=1)
    # RBC Plot
    fig.add_trace(go.Scatter(x=df['Day'], y=df['COD_Final'], mode='markers',
                  name='RBC Measured', marker=marker_style), row=3, col=1)
    if 'COD_Final_Pred_Orig' in df.columns:
        fig.add_trace(go.Scatter(x=df['Day'], y=df['COD_Final_Pred_Orig'], mode='lines',
                      name='RBC Original Pred.', line=dict(color=model_colors['RBC_Orig'])), row=3, col=1)
    if 'COD_Final_Pred_pH' in df.columns:
        fig.add_trace(go.Scatter(x=df['Day'], y=df['COD_Final_Pred_pH'], mode='lines',
                      name='RBC pH Pred.', line=dict(color=model_colors['RBC_pH'])), row=3, col=1)

    if compliance_limit:
        fig.add_hline(y=compliance_limit, line_dash="dot", line_color="#FF453A",
                      annotation_text="Compliance Limit", annotation_position="bottom right", row=3, col=1)
    # UI/UX FIX: Reduced height and moved legend for better scannability.
    fig.update_layout(height=600, title_text=f"<b>{t('timeseries_header')}</b>",
                      hovermode="x unified", legend=dict(orientation="v", x=1.05, xanchor="left", y=1))
    if log_y:
        fig.update_yaxes(type="log")
    for i in range(1, 4):
        fig.update_yaxes(title_text="COD (mg/L)", row=i, col=1)
    fig.update_xaxes(title_text="Time (Days)", row=3, col=1)
    return fig


@st.cache_data
def plot_interactive_parity(df, stage_name, actual_col, pred_cols_dict, rmse):
    data = df.dropna(subset=[actual_col] +
                     list(pred_cols_dict.values())).copy()
    if data.empty:
        return go.Figure().update_layout(title=f'Parity Plot: {stage_name} (No valid data)')
    fig = go.Figure()
    y_true = data.sort_values(by=actual_col)[actual_col]
    # UI/UX FIX: Changed error band to a light grey.
    fig.add_trace(go.Scatter(x=np.concatenate([y_true, y_true[::-1]]), y=np.concatenate([y_true + rmse, (y_true - rmse)[::-1]]), fill='toself',
                  fillcolor='rgba(220, 220, 220, 0.3)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name=f'±{rmse:.1f} RMSE Band'))
    min_val, max_val = min(data[actual_col].min(), data[list(pred_cols_dict.values())[
                           0]].min()), max(data[actual_col].max(), data[list(pred_cols_dict.values())[0]].max())
    fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val,
                  y1=max_val, line=dict(color="#FF453A", dash="dash"))
    for model_name, pred_col in pred_cols_dict.items():
        fig.add_trace(go.Scatter(x=data[actual_col], y=data[pred_col], mode='markers', name=model_name, marker=dict(color=model_colors.get(
            model_name, '#0A84FF')), customdata=data['Day'], hovertemplate='<b>Day %{customdata}</b><br>Measured: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'))
    fig.update_layout(height=450, title=f'<b>Parity Plot: {stage_name}</b>', xaxis_title='Measured COD (mg/L)',
                      yaxis_title='Predicted COD (mg/L)', legend_title='Model', yaxis=dict(scaleanchor="x", scaleratio=1))
    return fig


def create_kpi_card(icon, title, r2, rmse, p_value):
    sig_class, sig_label = ("green", "Not Sig.") if p_value >= 0.05 else (
        "red", "Significant")
    st.markdown(f"""<div class="kpi-card"><div class="card-header"><div class="kpi-icon">{icon}</div><div class="kpi-title">{title}</div></div><div><div style="display:flex; justify-content:space-between;"><div><div class="metric-label">R²</div><div class="metric-value">{r2:.3f}</div></div><div style="text-align:right;"><div class="metric-label">RMSE</div><div class="metric-value">{rmse:.1f}</div></div></div></div><div class="anova"><span class="anova-label">ANOVA p-value:</span><span class="anova-value">{p_value:.3f}</span><span class="badge {sig_class}">{sig_label}</span></div></div>""", unsafe_allow_html=True)


def display_final_equations(params_uasb, params_filter, params_rbc_orig, params_rbc_ph):
    st.markdown("""
    <style>
        .equation-container {
            background-color: #FFFFFF;
            border: 1px solid #E0E0E0;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            transition: all 0.3s ease-in-out;
        }
        .equation-container:hover {
            box-shadow: 0 6px 16px rgba(0,0,0,0.08);
            transform: translateY(-3px);
        }
        .equation-title {
            font-weight: 600;
            font-size: 1.2rem;
            margin-bottom: 1rem;
            border-bottom: 2px solid #F0F2F6;
            padding-bottom: 0.5rem;
        }
        .legend-title {
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        }
        .legend-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.5rem 1.5rem;
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)

    st.info("These are the governing equations for each model stage, populated with the kinetic parameters derived from your data.", icon="🧪")
    st.markdown("---")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        # --- UASB Card ---
        if params_uasb:
            st.markdown('<div class="equation-container">',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="equation-title">🦠 {t("help_uasb_title")}</div>', unsafe_allow_html=True)
            # FIX: Escaped the literal {1} to {{1}} to avoid format string IndexError
            st.latex(r''' SRR = \left( \dfrac{{ {U_max:.3f} \cdot OLR }}{{ {K_B:.3f} + OLR }} \right) \cdot \left( \dfrac{{{1}}}{{ 1 + \dfrac{{VFA/ALK}}{{ {K_I:.3f} }} }} \right) '''.format(
                U_max=params_uasb.get('U_max', 0),
                K_B=params_uasb.get('K_B', 0),
                K_I=params_uasb.get('K_I', 0)
            ))
            st.markdown('<div class="legend-title">Variables</div>',
                        unsafe_allow_html=True)
            st.markdown("""
            <div class="legend-grid">
                <div><b>SRR</b>: Substrate Removal Rate</div>
                <div><b>OLR</b>: Organic Loading Rate</div>
                <div><b>VFA/ALK</b>: VFA to Alkalinity Ratio</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- Filter Card ---
        if params_filter:
            st.markdown('<div class="equation-container">',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="equation-title">✨ {t("help_filter_title")}</div>', unsafe_allow_html=True)
            st.latex(r'COD_{{Removed}} = {R_cod_tss:.3f} \cdot (TSS_{{in}} - TSS_{{out}}) + {k_ads:.3f} \cdot sCOD_{{in}}'.format(
                R_cod_tss=params_filter.get('R_cod_tss', 0),
                k_ads=params_filter.get('k_ads', 0)
            ))
            st.markdown('<div class="legend-title">Variables</div>',
                        unsafe_allow_html=True)
            st.markdown("""
            <div class="legend-grid">
                <div><b>TSS</b>: Total Suspended Solids</div>
                <div><b>sCOD</b>: Soluble COD</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # --- RBC Card ---
        if params_rbc_orig or params_rbc_ph:
            st.markdown('<div class="equation-container">',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="equation-title">🔄 {t("help_rbc_title")}</div>', unsafe_allow_html=True)

            if params_rbc_orig:
                p_orig = params_rbc_orig
                st.markdown("<h6>RBC v1.0 (Original)</h6>",
                            unsafe_allow_html=True)
                st.latex(r'''\mu_a = \left( \dfrac{{ {umxa:.3f} \cdot S_e }}{{ {Ku:.3f} + S_e }} \right) \cdot \left( \dfrac{{ 0.5 \cdot {Ko:.3f} + {O:.3f} }}{{ {Ko:.3f} + {O:.3f} }} \right)'''.format(
                    umxa=p_orig.get('umxa', 0), Ku=p_orig.get('Ku', 0),
                    Ko=p_orig.get('Ko', 0), O=p_orig.get('O', 0)
                ))

            if params_rbc_ph:
                p_ph = params_rbc_ph
                exponent_val = 0.5 * \
                    (p_ph.get('pH_min', 0) - p_ph.get('pH_max', 0))
                st.markdown("<h6>RBC v1.1 (pH-Inhibited)</h6>",
                            unsafe_allow_html=True)
                st.info(
                    "This model enhances the original by adding a pH inhibition factor `τ_pH`.", icon="💡")
                st.latex(r'''\tau_{{pH}} = \dfrac{{ 1 + 2 \cdot 10^{{{exp_val:.2f}}} }}{{ 1 + 10^{{pH - {pH_max:.2f}}} + 10^{{{pH_min:.2f} - pH}} }}'''.format(
                    exp_val=exponent_val,
                    pH_max=p_ph.get('pH_max', 0),
                    pH_min=p_ph.get('pH_min', 0)
                ))
                st.latex(r'''\mu_a = \tau_{{pH}} \cdot \left( \dfrac{{ {umxa:.3f} \cdot S_e }}{{ {Ku:.3f} + S_e }} \right) \cdot \left( \dfrac{{ 0.5 \cdot {Ko:.3f} + {O:.3f} }}{{ {Ko:.3f} + {O:.3f} }} \right)'''.format(
                    umxa=p_ph.get('umxa', 0), Ku=p_ph.get('Ku', 0),
                    Ko=p_ph.get('Ko', 0), O=p_ph.get('O', 0)
                ))

            st.markdown('<div class="legend-title">Variables</div>',
                        unsafe_allow_html=True)
            st.markdown("""
            <div class="legend-grid">
                <div><b>μa</b>: Autotrophic Growth Rate</div>
                <div><b>Se</b>: Effluent Substrate Conc.</div>
                <div><b>τ_pH</b>: pH Inhibition Factor</div>
                <div><b>Ko</b>: Oxygen half-saturation const.</div>
                <div><b>O</b>: Dissolved Oxygen conc.</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


def plot_gsa_results(_mi_df, _si, _problem):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        'Morris Elementary Effects', 'Sobol Sensitivity Indices'))
    _mi_df.sort_values('mu_star', inplace=True)
    fig.add_trace(go.Scatter(x=_mi_df['mu_star'], y=_mi_df['sigma'],
                  mode='markers+text', text=_mi_df.index, textposition="top right"), row=1, col=1)
    sobol_df = pd.DataFrame(
        {'S1': _si['S1'], 'ST': _si['ST']}, index=_problem['names']).sort_values('ST')
    fig.add_trace(go.Bar(y=sobol_df.index,
                  x=sobol_df['S1'], name='S1 (First-order)', orientation='h'), row=1, col=2)
    fig.add_trace(go.Bar(y=sobol_df.index,
                  x=sobol_df['ST'], name='ST (Total-order)', orientation='h'), row=1, col=2)
    fig.update_layout(barmode='group', height=500,
                      title_text="<b>Global Sensitivity Analysis (GSA) Results</b>")
    fig.update_xaxes(title_text='μ* (Overall Influence)', row=1, col=1)
    fig.update_yaxes(
        title_text='σ (Interaction/Non-linear Effects)', row=1, col=1)
    fig.update_xaxes(title_text='Sobol Index Value', row=1, col=2)
    return fig


def plot_mc_results(_results_df):
    df_sorted = _results_df.sort_values(
        by="Spearman_Correlation", key=abs, ascending=True)
    fig = px.bar(df_sorted, y='Parameter', x='Spearman_Correlation', color=np.where(df_sorted['Spearman_Correlation'] > 0, 'Positive', 'Negative'),
                 color_discrete_map={
                     'Positive': '#30D158', 'Negative': '#FF453A'},
                 title='<b>Monte Carlo Sensitivity Analysis</b>', orientation='h',
                 labels={'Spearman_Correlation': 'Spearman Rank Correlation'})
    fig.add_vline(x=0, line_dash="dash", line_color="grey")
    return fig

# ==============================================================================
# 5. TAB-SPECIFIC DISPLAY FUNCTIONS (MODULAR UI)
# ==============================================================================


def show_welcome_page():
    """
    Displays the Welcome Page (Home tab) of the application.
    This page serves as a guide for first-time users, explaining the app's purpose,
    data requirements, and features.
    """

    # --- Page Title and Introduction ---
    st.title("🧪 Welcome to KINEMOD-KIT")
    st.subheader("An Integrated Tool for Wastewater Treatment Modeling")

    st.markdown("""
        This application provides an end-to-end platform for the kinetic modeling and performance analysis of a three-stage POME (Palm Oil Mill Effluent) treatment system, consisting of **UASB**, **Filtration**, and **RBC** reactors.
    """)
    st.divider()

    # --- Use columns for a centered, focused layout ---
    _, mid_col, _ = st.columns([1, 3, 1])
    with mid_col:

        # --- Purpose Section ---
        st.header("🧭 Purpose")
        st.markdown("""
            The primary goal of KINEMOD-KIT is to empower engineers and researchers by:
            - **Calibrating** kinetic models against experimental data.
            - **Visualizing** key performance indicators (KPIs), time-series trends, and parity plots.
            - **Analyzing** model sensitivity to parameter uncertainty through GSA and Monte Carlo simulations.
            - **Optimizing** operational parameters to achieve specific treatment goals.
        """)

        # --- Diagram Section ---
        st.header("🧩 System Diagram")
        # Ensure the 'assets' folder is in the same directory as app.py or adjust path
        try:
            # Assuming diagram is in the root for simplicity based on original code.
            st.image("Diagram.png", use_column_width=True,
                     caption="System Workflow and Model Structure")
        except FileNotFoundError:
            st.warning(
                "`Diagram.png` not found. Please ensure the image is in the root directory.")

    st.divider()

    # --- How to Get Started Section ---
    st.header("🚀 How to Get Started")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        # --- Data Input Guide ---
        with st.container(border=True):
            st.subheader("📥 1. Provide Your Data")
            st.markdown("""
                The first step is to load your operational data. You have two options in the sidebar:
                1.  **Upload a CSV file:** Your data must contain the required columns.
                2.  **Use the default dataset:** A pre-loaded example file will be used.

                **File Requirements:**
                - **Format:** CSV (`.csv`) with semicolon (`;`) separators.
                - **Required Columns:** `Day`, `COD_in`, `HRT_UASB`, `VFA_Eff`, `ALK_Eff`, `TSS_UASB_Eff`, `TSS_Filt_Eff`, `COD_UASB_Eff`, `COD_Filt_Eff`, `COD_Final`, `Xa`, `Xs`, `pH_Eff_RBC`, `Steady_State`.
            """)
            st.info(
                "You can upload a new file at any time to restart the analysis with different data.", icon="ℹ️")

    with col2:
        # --- Running the Analysis Guide ---
        with st.container(border=True):
            st.subheader("⚙️ 2. Calibrate & Analyze")
            st.markdown("""
                Once your data is loaded, use the sidebar to **Calibrate** the models. After calibration, explore the tabs:

                - **Dashboard:** Your main hub for KPIs (`R²`, `RMSE`), time-series graphs, and model performance summaries.
                - **Model Details:** View and edit calibrated parameters and see the final kinetic equations.
                - **Sensitivity:** Perform GSA and Monte Carlo simulations to see which parameters impact the results most.
                - **Optimizer:** Find the optimal operational settings to minimize the final effluent COD.
            """)
            st.warning(
                "Calibration is a required step before you can view results in the other tabs.", icon="⚠️")

    st.divider()

    # --- Tips for Smooth Use ---
    st.header("💡 Tips for Smooth Use")
    tips_cols = st.columns(3)
    tips_cols[0].success(
        "**Correct Column Names:** Ensure your uploaded CSV file's column headers exactly match the required names.", icon="✔️")
    tips_cols[1].success(
        "**Persistent Data:** Switch between tabs anytime. Your uploaded data and calibration results will be kept in memory.", icon="🧠")
    tips_cols[2].success(
        "**Recalibrate as Needed:** If you adjust advanced parameters in the sidebar, remember to click **Calibrate** again.", icon="🚀")


def display_dashboard_tab(reactor):
    st.header(f"📊 {t('dashboard_header')}")
    st.markdown("This dashboard provides a comprehensive overview of the reactor's performance. Use the filters below to analyze specific time ranges.")
    min_day, max_day = int(reactor.df['Day'].min()), int(
        reactor.df['Day'].max())
    selected_days = st.slider(t('date_slider'), min_day, max_day,
                              (min_day, max_day), key='date_slider_dashboard')
    filtered_df = reactor.df[(reactor.df['Day'] >= selected_days[0]) & (
        reactor.df['Day'] <= selected_days[1])].copy()
    st.markdown("---")
    st.subheader(f"🎯 {t('kpi_header')}")
    kpi_cols = st.columns(4)
    stages = {"UASB": ("COD_UASB_Eff", "COD_UASB_Pred"), "Filter": ("COD_Filt_Eff", "COD_Filt_Pred"), "RBC (Orig)": (
        "COD_Final", "COD_Final_Pred_Orig"), "RBC (pH Mod)": ("COD_Final", "COD_Final_Pred_pH")}
    stage_icons = {"UASB": "🦠", "Filter": "✨",
                   "RBC (Orig)": "🔄", "RBC (pH Mod)": "🌿"}
    stage_rmses = {}
    for i, (name, (actual_col, pred_col)) in enumerate(stages.items()):
        with kpi_cols[i]:
            if pred_col in filtered_df.columns:
                valid_df = filtered_df.dropna(subset=[actual_col, pred_col])
                if not valid_df.empty:
                    r2, rmse, p_value = r2_score(valid_df[actual_col], valid_df[pred_col]), np.sqrt(mean_squared_error(
                        valid_df[actual_col], valid_df[pred_col])), stats.f_oneway(valid_df[actual_col], valid_df[pred_col])[1]
                    stage_rmses[name] = rmse
                    create_kpi_card(icon=stage_icons.get(
                        name, "📊"), title=name, r2=r2, rmse=rmse, p_value=p_value)

    with st.container(border=True):
        st.subheader(f"📈 {t('timeseries_header')}")
        ts_control_cols = st.columns([2, 1, 1])
        compliance_limit = ts_control_cols[0].number_input(
            t('compliance_input'), value=350, min_value=0, step=25)
        if 'COD_Final_Pred_pH' in filtered_df.columns and not filtered_df['COD_Final_Pred_pH'].isna().all():
            compliance_pct = (
                filtered_df['COD_Final_Pred_pH'] <= compliance_limit).mean() * 100
            ts_control_cols[1].metric(
                t('kpi_compliance_header'), f"{compliance_pct:.1f}%")
        log_y = ts_control_cols[2].checkbox(
            t('log_axis_toggle'), key='log_y_ts')
        st.plotly_chart(plot_interactive_timeseries(
            filtered_df, compliance_limit, log_y), use_container_width=True)

    with st.container(border=True):
        st.subheader(f"📉 {t('parity_header')}")
        stage_choice = st.selectbox("Select Stage to Analyze", stages.keys())
        selected_rmse = stage_rmses.get(stage_choice, 0)
        pred_dict, actual = ({stage_choice.split(' ')[0] if ' ' not in stage_choice else (
            'Original' if 'Orig' in stage_choice else 'pH Model'): stages[stage_choice][1]}, stages[stage_choice][0])
        par_col, err_col = st.columns(2)
        par_col.plotly_chart(plot_interactive_parity(
            filtered_df, stage_choice, actual, pred_dict, selected_rmse), use_container_width=True)
        error_dfs = [pd.DataFrame({"Error": filtered_df[pred_col] - filtered_df[actual], "Model": model_name}).dropna()
                     for model_name, pred_col in pred_dict.items() if pred_col in filtered_df.columns]
        if error_dfs:
            err_col.plotly_chart(px.histogram(pd.concat(error_dfs), x="Error", color="Model", marginal="box", barmode="overlay", title="<b>Error Distribution</b>",
                                 height=450, template="plotly_white", color_discrete_map=model_colors).add_vline(x=0, line_dash="dash", line_color="red"), use_container_width=True)

    # UI/UX FIX: Moved "What-If" scenarios to the dashboard for better visibility and workflow.
    with st.container(border=True):
        st.header(f"⚡ {t('what_if_header')}")
        st.info(t('what_if_help'))
        what_if_cols = st.columns(3)
        what_if_cod = what_if_cols[0].slider(
            t('what_if_cod'), 5000, 40000, int(reactor.df['COD_in'].mean()), 500)
        what_if_hrt_uasb = what_if_cols[1].slider(
            t('what_if_hrt_uasb'), 1.0, 20.0, reactor.df['HRT_UASB'].mean(), 0.1)
        what_if_hrt_rbc = what_if_cols[2].slider(
            t('what_if_hrt_rbc'), 1.0, 120.0, reactor.df['HRT_hours'].mean(), 0.5)
        if reactor.params_uasb and reactor.params_rbc_ph:
            srr_pred = reactor.uasb_model.predict(
                (what_if_cod / what_if_hrt_uasb, 0.3), reactor.params_uasb)
            cod_uasb_pred = what_if_cod - srr_pred * what_if_hrt_uasb
            final_pred = reactor.rbc_model_ph.predict(
                {'So': cod_uasb_pred * 0.8, 'HRT_days': what_if_hrt_rbc / 24.0, 'Xa': 400, 'Xs': 200, 'pH': 7.5}, reactor.params_rbc_ph)
            st.metric(label=t('what_if_result'),
                      value=f"{final_pred:.0f} mg/L" if not np.isnan(final_pred) else "N/A")

    st.markdown("---")
    st.header(t('filtered_data_header'))
    st.dataframe(filtered_df)


def display_model_details_tab(reactor, data_content):
    st.header(t('param_header'))
    param_tab, eq_tab = st.tabs([t('param_subheader'), t('eq_subheader')])
    with param_tab:
        st.info(t('editable_param_info'))
        param_sets = [("UASB", reactor.params_uasb), ("Filter", reactor.params_filter),
                      ("RBC (Orig)", reactor.params_rbc_orig), ("RBC (pH)", reactor.params_rbc_ph)]
        if 'edited_params' not in st.session_state or st.session_state.edited_params is None:
            st.session_state.edited_params = pd.DataFrame(
                [{"Model": model, "Parameter": key, "Value": value} for model, params in param_sets if params for key, value in params.items()])
        if 'edited_params' in st.session_state and st.session_state.edited_params is not None:
            with st.form("edit_params_form"):
                editors = {}
                with st.expander("UASB Parameters", expanded=True):
                    editors['UASB'] = st.data_editor(
                        st.session_state.edited_params[st.session_state.edited_params['Model'] == 'UASB'], hide_index=True, key='uasb_editor')
                with st.expander("Filter Parameters"):
                    editors['Filter'] = st.data_editor(
                        st.session_state.edited_params[st.session_state.edited_params['Model'] == 'Filter'], hide_index=True, key='filter_editor')
                with st.expander("RBC Parameters"):
                    editors['RBC (Orig)'] = st.data_editor(
                        st.session_state.edited_params[st.session_state.edited_params['Model'] == 'RBC (Orig)'], hide_index=True, key='rbc_o_editor')
                    editors['RBC (pH)'] = st.data_editor(
                        st.session_state.edited_params[st.session_state.edited_params['Model'] == 'RBC (pH)'], hide_index=True, key='rbc_ph_editor')
                if st.form_submit_button(t('rerun_button'), type="primary"):
                    edited_params_df = pd.concat(editors.values())
                    st.session_state.edited_params = edited_params_df
                    with st.spinner("Re-running predictions..."):
                        p_uasb = dict(edited_params_df[edited_params_df['Model'] == 'UASB'].set_index(
                            'Parameter')['Value'])
                        p_filter = dict(edited_params_df[edited_params_df['Model'] == 'Filter'].set_index(
                            'Parameter')['Value'])
                        p_rbc_o = dict(edited_params_df[edited_params_df['Model'] == 'RBC (Orig)'].set_index(
                            'Parameter')['Value'])
                        p_rbc_ph = dict(edited_params_df[edited_params_df['Model'] == 'RBC (pH)'].set_index(
                            'Parameter')['Value'])
                        new_df = reactor.run_predictions_with_new_params(pd.read_csv(
                            io.BytesIO(data_content), sep=';'), p_uasb, p_filter, p_rbc_o, p_rbc_ph)
                        st.session_state.reactor.df = new_df
                        st.success("Predictions updated!")
    with eq_tab:
        display_final_equations(reactor.params_uasb, reactor.params_filter,
                                reactor.params_rbc_orig, reactor.params_rbc_ph)


def display_methane_tab(reactor):
    st.header(t('methane_header'))
    with st.container(border=True):
        min_day, max_day = int(reactor.df['Day'].min()), int(
            reactor.df['Day'].max())
        selected_days = st.slider(
            t('date_slider'), min_day, max_day, (min_day, max_day), key='date_slider_methane')
        filtered_df = reactor.df[(reactor.df['Day'] >= selected_days[0]) & (
            reactor.df['Day'] <= selected_days[1])]
        uasb_cod_removed = (filtered_df['COD_in'] - filtered_df['COD_UASB_Pred']).mean(
        ) if 'COD_UASB_Pred' in filtered_df and not filtered_df.empty else 0
        uasb_volume = st.number_input(
            t('uasb_volume_label'), value=25.0, min_value=0.1)
        mean_hrt = filtered_df['HRT_UASB'].mean()
        # BUG FIX: Prevent ZeroDivisionError if mean_hrt is 0 or NaN
        avg_flow_rate = uasb_volume / mean_hrt if mean_hrt > 0 else 0
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


def display_sensitivity_tab(reactor, data_content, config):
    st.header(t('sa_header'))
    st.info(t('sa_info'))
    control_col, result_col = st.columns([1, 2.5])
    with control_col:
        analysis_mode = st.radio(t('sa_mode_select'), (t(
            'sa_mode_specific'), t('sa_mode_all')), key='sa_mode')
        sa_type, sa_model = (st.selectbox(t('sa_type_select'), ["GSA", "Monte Carlo"]), st.selectbox(t('sa_model_select'), [
                             "RBC Original", "RBC pH-Modified"])) if analysis_mode == t('sa_mode_specific') else (None, None)

        if st.button(t('run_sa_button'), type="primary", use_container_width=True):
            if data_content:
                st.session_state.sa_results = {}
                if analysis_mode == t('sa_mode_all'):
                    all_types, all_models = ["GSA", "Monte Carlo"], {
                        "RBC Original": "orig", "RBC pH-Modified": "ph"}
                    progress_bar, total_runs = st.progress(
                        0, text="Starting all analyses..."), len(all_types) * len(all_models)
                    for i, (model_name, model_key) in enumerate(all_models.items()):
                        if model_key == 'ph':
                            rbc_params_to_use = reactor.params_rbc_ph
                        else:
                            rbc_params_to_use = reactor.params_rbc_orig
                        for j, type_name in enumerate(all_types):
                            run_count = i * len(all_types) + j + 1
                            progress_bar.progress(
                                run_count / total_runs, text=f"Running {type_name} for {model_name}... ({run_count}/{total_runs})")
                            error, data = run_sensitivity_analysis(
                                analysis_type=type_name, model_key=model_key,
                                params_uasb=reactor.params_uasb, params_filter=reactor.params_filter,
                                rbc_params=rbc_params_to_use, _uploaded_file_content=data_content,
                                config=config)
                            st.session_state.sa_results[f"{type_name}_{model_key}"] = {
                                'error': error, 'data': data, 'type': type_name, 'model_name': model_name}
                    progress_bar.progress(1.0, text="All analyses complete!")
                else:
                    model_key = 'ph' if 'pH' in sa_model else 'orig'
                    if model_key == 'ph':
                        rbc_params_to_use = reactor.params_rbc_ph
                    else:
                        rbc_params_to_use = reactor.params_rbc_orig
                    with st.spinner(f"Running {sa_type} for {sa_model}..."):
                        error, data = run_sensitivity_analysis(
                            analysis_type=sa_type, model_key=model_key,
                            params_uasb=reactor.params_uasb, params_filter=reactor.params_filter,
                            rbc_params=rbc_params_to_use, _uploaded_file_content=data_content,
                            config=config)
                        st.session_state.sa_results[f"{type_name}_{model_key}"] = {
                            'error': error, 'data': data, 'type': sa_type, 'model_name': sa_model}
                st.success("Analysis complete!")
    with result_col:
        if st.session_state.get('sa_results'):
            for key, result in st.session_state.sa_results.items():
                if result['error']:
                    st.error(
                        f"Error in {result['type']} for {result['model_name']}: {result['error']}")
                    continue
                with st.expander(f"Results for {result['type']} - {result['model_name']}", expanded=True):
                    if result['type'] == 'GSA' and result['data']:
                        st.plotly_chart(plot_gsa_results(
                            result['data']['Mi'], result['data']['Si'], result['data']['problem']), use_container_width=True, key=f"gsa_chart_{key}")
                        st.caption(
                            "Table 1. Global Sensitivity Analysis Results")
                        sobol_df = pd.DataFrame({'S1': result['data']['Si']['S1'], 'ST': result['data']['Si']['ST'], 'S1_conf': result['data']
                                                ['Si']['S1_conf'], 'ST_conf': result['data']['Si']['ST_conf']}, index=result['data']['problem']['names'])
                        st.dataframe(sobol_df)
                    elif result['type'] == 'Monte Carlo' and result['data']:
                        st.plotly_chart(plot_mc_results(
                            result['data']['mc_results']), use_container_width=True, key=f"mc_chart_{key}")
                        st.caption("Table 2. Monte Carlo Output Summary")
                        summary_df = result['data']['mc_samples'].describe(
                        ).transpose()
                        summary_df['Spearman_Correlation'] = result['data']['mc_results'].set_index(
                            'Parameter')['Spearman_Correlation']
                        st.dataframe(
                            summary_df[['mean', 'std', 'Spearman_Correlation']])


def display_optimizer_tab(reactor):
    st.header(t('optimizer_header'))
    with st.container(border=True):
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
                with st.spinner("Finding optimal settings..."):
                    bounds = [cod_bounds, hrt_uasb_bounds, hrt_rbc_bounds]
                    fixed_params = {k: reactor.df[v].mean() for k, v in {
                        'vfa_alk': 'VFA_ALK_Ratio', 'tss_in': 'TSS_UASB_Eff', 'tss_out': 'TSS_Filt_Eff', 'xa': 'Xa', 'xs': 'Xs', 'ph': 'pH_Eff_RBC'}.items()}
                    result = minimize(optimization_objective_function, [np.mean(b) for b in bounds], args=(
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
                              f"{res.fun:.1f} mg/L")
                else:
                    st.error(t('optimizer_fail'))


def display_help_tab():
    st.header(t('help_title'))
    st.info(t('help_intro'))

    # UI/UX FIX: Replaced expanders with tabs for easier navigation.
    h_tab1, h_tab2, h_tab3 = st.tabs(
        [t('help_usage_title'), t('help_tab_guide_title'), t('help_faq_title')])
    with h_tab1:
        st.markdown(f"**{t('help_usage_step1_title')}**: {t('help_usage_step1_text')}<br>**{t('help_usage_step2_title')}**: {t('help_usage_step2_text')}<br>**{t('help_usage_step3_title')}**: {t('help_usage_step3_text')}<br>**{t('help_usage_step4_title')}**: {t('help_usage_step4_text')}", unsafe_allow_html=True)
    with h_tab2:
        st.markdown(f"**{t('tabs')[0]}**: {t('help_tab_dashboard_desc')}<br>**{t('tabs')[1]}**: {t('help_tab_model_details_desc')}<br>**{t('tabs')[2]}**: {t('help_tab_methane_desc')}<br>**{t('tabs')[3]}**: {t('help_tab_sensitivity_desc')}<br>**{t('tabs')[4]}**: {t('help_tab_optimizer_desc')}", unsafe_allow_html=True)
    with h_tab3:
        st.markdown(f"**{t('help_faq1_q')}**\n{t('help_faq1_a')}")
        st.markdown(f"**{t('help_faq2_q')}**\n{t('help_faq2_a')}")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.subheader("📤 Feedback & Support")
    with st.form("feedback_form"):
        st.text_input("Email Address (Optional)")
        st.text_area("Provide feedback or report an issue")
        if st.form_submit_button("Submit Feedback 🍏"):
            st.success("✅ Feedback submitted successfully. Thank you!")

# ==============================================================================
# 6. Main App Execution
# ==============================================================================


def main():
    st.title(f"💻 {t('title')} 📊")
    st.subheader(t('subtitle'))
    st.caption(t('author_line').format(
        pd.to_datetime('today').strftime('%Y-%m-%d')))
    st.markdown("---")

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header(t('sidebar_controls'))
        uploaded_file = st.file_uploader(t('sidebar_upload'), type=[
                                         "csv"], help=t('sidebar_upload_help'))
        use_default_data = st.checkbox(
            t('sidebar_use_default'), value=not uploaded_file)
        CONFIG = {"reactor_constants": {"V_RBC": 18, "A_RBC": 240}, "optimization_bounds": {"UASB": [(1.4, 75.9), (4.9, 25.0), (1.9, 8.0)], "Filter": [(0.01, 4.0), (0.01, 2.5)], "RBC": [(
            5, 30), (100, 500), (20, 60), (1, 15), (300, 450), (1, 10), (0.01, 1), (0.01, 2.5)], "RBC_pH": [(5, 50), (50, 100), (20, 30), (1, 5), (1, 5), (1, 10), (0.01, 2.5), (0.01, 2.5), (3.0, 5), (9, 11.5)]}}
        with st.expander(t('sidebar_advanced_config')):
            CONFIG["reactor_constants"]["V_RBC"] = st.number_input(
                t('sidebar_rbc_vol'), value=18.0)
            CONFIG["reactor_constants"]["A_RBC"] = st.number_input(
                t('sidebar_rbc_area'), value=240.0)
        run_estimation_button = st.button(
            t('sidebar_calibrate'), type="primary", use_container_width=True, help=t('sidebar_calibrate_help'))

    # --- Data Loading and Calibration ---
    data_content = uploaded_file.getvalue() if uploaded_file else load_default_data()
    if data_content is None and use_default_data:
        st.error(
            "Error: Default `Model_Data.csv` not found. Please ensure it is in the same directory as `app.py`.")
    if run_estimation_button and data_content:
        with st.spinner('🔬 Calibrating model... This may take a moment.'):
            reactor_obj, error = run_full_analysis(data_content, CONFIG)
            if error:
                st.error(error)
            else:
                st.session_state.reactor, st.session_state.sa_results, st.session_state.edited_params, st.session_state.optimizer_results = reactor_obj, None, None, None
    elif run_estimation_button:
        st.warning(
            "Please upload a data file or select the default data option.", icon="⚠️")

    # --- Main App Display ---
    if st.session_state.reactor:
        tab_titles = t('tabs')
        tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_titles)
        with tab0:
            show_welcome_page()
        with tab1:
            display_dashboard_tab(st.session_state.reactor)
        with tab2:
            display_model_details_tab(st.session_state.reactor, data_content)
        with tab3:
            display_methane_tab(st.session_state.reactor)
        with tab4:
            display_sensitivity_tab(
                st.session_state.reactor, data_content, CONFIG)
        with tab5:
            display_optimizer_tab(st.session_state.reactor)
        with tab6:
            display_help_tab()
    else:
        # Show only the welcome page before calibration
        show_welcome_page()


if __name__ == "__main__":
    main()
