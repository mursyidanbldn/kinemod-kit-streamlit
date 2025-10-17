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

translations = {
    "en": {
        "title": "KINEMOD-KIT", "subtitle": "UASB-FILTRATION-RBC REACTOR KINETIC MODELING KIT",
        "author_line": "By Rizky Mursyidan Baldan | Last Updated: {}",
        "tabs": ["📊 Dashboard", "🔬 Model Details", "🍃 Methane & Energy", "🔬 Sensitivity", "⚙️ Optimizer", "❓ Help"],
        "welcome_message": "👋 **Welcome!** Please upload your data or select the option to use the default dataset in the sidebar to begin.",
        "sidebar_controls": "⚙️ Controls", "sidebar_upload": "Upload data (CSV)", "sidebar_upload_help": "Upload a custom `Model_Data.csv` file.",
        "sidebar_use_default": "Use default `Model_Data.csv` from GitHub",
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
        "sa_header": "🔬 Sensitivity Analysis", "sa_info": "Evaluate how uncertainty in model parameters affects the output. This is computationally intensive.",
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
        "help_usage_step1_title": "1. Provide Data", "help_usage_step1_text": "Start by uploading your `Model_Data.csv` file, or check the box to use the default dataset from GitHub.",
        "help_usage_step2_title": "2. Calibrate or Load", "help_usage_step2_text": "Click **Calibrate** to estimate model parameters from your data, or **Load** if you have a pre-calibrated `calibrated_params.json` file.",
        "help_usage_step3_title": "3. Analyze Dashboard", "help_usage_step3_text": "The Dashboard shows model performance via KPIs, time-series charts, and parity plots. Use the date slider to focus on specific periods.",
        "help_usage_step4_title": "4. Explore Tabs", "help_usage_step4_text": "Dive deeper into other tabs to view model equations, run sensitivity analysis, calculate energy potential, or optimize process parameters.",
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
        "help_faq2_q": "What's the difference between RBC 'Original' and 'pH Mod' models?", "help_faq2_a": "The 'Original' model is a standard biofilm model. The 'pH Mod' version adds a pH inhibition factor (τ_pH), making the model's predictions sensitive to pH fluctuations, which is common in biological treatment systems.",
        "download_section_header": "📥 Download Data & Charts",
        "download_data_button": "Download Filtered Data (CSV)",
        "download_ts_chart_button": "Download Time-Series Chart (PNG)",
        "download_parity_chart_button": "Download Parity Chart (PNG)",
        "download_sa_chart_button": "Download Sensitivity Chart (PNG)"
    },
    "id": {
        "title": "KINEMOD-KIT", "subtitle": "KIT PEMODELAN KINETIK REAKTOR UASB-FILTRASI-RBC",
        "author_line": "Oleh Rizky Mursyidan Baldan | Terakhir Diperbarui: {}",
        "tabs": ["📊 Dasbor", "🔬 Detail Model", "🍃 Metana & Energi", "🔬 Sensitivitas", "⚙️ Pengoptimal", "❓ Bantuan"],
        "welcome_message": "👋 **Selamat datang!** Silakan unggah data Anda atau pilih opsi untuk menggunakan dataset default di bilah sisi untuk memulai.",
        "sidebar_controls": "⚙️ Kontrol", "sidebar_upload": "Unggah data (CSV)", "sidebar_upload_help": "Unggah file `Model_Data.csv` kustom.",
        "sidebar_use_default": "Gunakan `Model_Data.csv` default dari GitHub",
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
        "sa_header": "🔬 Analisis Sensitivitas", "sa_info": "Evaluasi bagaimana ketidakpastian parameter mempengaruhi output. Proses ini intensif secara komputasi.",
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
        "help_usage_step1_title": "1. Sediakan Data", "help_usage_step1_text": "Mulailah dengan mengunggah file `Model_Data.csv` Anda, atau centang kotak untuk menggunakan dataset default dari GitHub.",
        "help_usage_step2_title": "2. Kalibrasi atau Muat", "help_usage_step2_text": "Klik **Kalibrasi** untuk mengestimasi parameter model dari data Anda, atau **Muat** jika Anda memiliki file `calibrated_params.json` yang sudah ada.",
        "help_usage_step3_title": "3. Analisis Dasbor", "help_usage_step3_text": "Dasbor menampilkan kinerja model melalui KPI, grafik runtun waktu, dan plot paritas. Gunakan slider tanggal untuk fokus pada periode tertentu.",
        "help_usage_step4_title": "4. Jelajahi Tab Lain", "help_usage_step4_text": "Pelajari lebih dalam di tab lain untuk melihat persamaan model, menjalankan analisis sensitivitas, menghitung potensi energi, atau mengoptimalkan parameter proses.",
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
        "help_faq2_q": "Apa perbedaan antara model RBC 'Original' dan 'pH Mod'?", "help_faq2_a": "Model 'Original' adalah model biofilm standar. Versi 'pH Mod' menambahkan faktor inhibisi pH (τ_pH), membuat prediksi model sensitif terhadap fluktuasi pH, yang umum terjadi pada sistem pengolahan biologis.",
        "download_section_header": "📥 Unduh Data & Grafik",
        "download_data_button": "Unduh Data Terfilter (CSV)",
        "download_ts_chart_button": "Unduh Grafik Runtun Waktu (PNG)",
        "download_parity_chart_button": "Unduh Grafik Paritas (PNG)",
        "download_sa_chart_button": "Unduh Grafik Sensitivitas (PNG)"
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
# 2. Global CSS Styles & Animation Functions
# ==============================================================================
st.markdown("""
<style>
    @keyframes softFadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in-up {
        animation: softFadeInUp 0.6s ease-out forwards;
    }
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", sans-serif;
    }
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
    .help-card {
        background: rgba(255,255,255,0.7); backdrop-filter: blur(12px);
        border-radius: 16px; padding: 1.3rem 1.5rem; margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05); border: 1px solid rgba(255,255,255,0.4);
        transition: all 0.25s ease;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. Caching & Backend Logic
# ==============================================================================


@st.cache_data
def run_full_analysis(_uploaded_file_content, config, action):
    data_buffer = io.BytesIO(_uploaded_file_content)
    reactor = IntegratedReactor(data_buffer, config["reactor_constants"])
    if not reactor.load_and_prepare_data():
        return None, "Failed to load or process data."
    if action == "estimate":
        reactor.run_parameter_estimation(config["optimization_bounds"])
    elif action == "load":
        if not reactor.load_parameters():
            return None, "Could not find or load `calibrated_params.json`."
    reactor.run_full_predictions()
    return reactor, None


@st.cache_data
def run_sensitivity_analysis(analysis_type, model_key, params_uasb, params_filter, params_rbc_orig, params_rbc_ph, _uploaded_file_content, config):
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


@st.cache_data
def plot_interactive_timeseries(df, compliance_limit=None, log_y=False):
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
    fig.update_layout(
        height=600, title_text=f"<b>{t('timeseries_header')}</b>", hovermode="x unified")
    if log_y:
        fig.update_yaxes(type="log")
    fig.update_yaxes(title_text="COD (mg/L)", row=1, col=1)
    fig.update_yaxes(title_text="COD (mg/L)", row=2, col=1)
    fig.update_yaxes(title_text="COD (mg/L)", row=3, col=1)
    fig.update_xaxes(title_text="Time (Days)", row=3, col=1)
    return fig


@st.cache_data
def plot_interactive_parity(df, stage_name, actual_col, pred_cols_dict):
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


def create_kpi_card(icon, title, r2, rmse, p_value):
    sig_class = "red" if p_value < 0.05 else "green"
    sig_label = "Significant" if p_value < 0.05 else "Not Sig."
    card_html = f"""<div class="kpi-card"> <div class="card-header"> <div class="kpi-icon">{icon}</div> <div class="kpi-title">{title}</div> </div> <div class="card-body"> <div style="display:flex; justify-content:space-between;"> <div> <div class="metric-label">R²</div> <div class="metric-value">{r2:.3f}</div> </div> <div style="text-align:right;"> <div class="metric-label">RMSE</div> <div class="metric-value">{rmse:.1f}</div> </div> </div> </div> <div class="anova"> <span class="anova-label">ANOVA p-value:</span> <span class="anova-value">{p_value:.3f}</span> <span class="badge {sig_class}">{sig_label}</span> </div> </div>"""
    st.markdown(card_html, unsafe_allow_html=True)


def display_final_equations(params_uasb, params_filter, params_rbc_orig, params_rbc_ph):
    st.subheader(t('eq_subheader'))
    if params_uasb:
        st.markdown(f"**{t('help_uasb_title')}**")
        p = params_uasb
        st.latex(
            fr''' SRR = \left( \dfrac{{ \textcolor{{#0A84FF}}{{{p['U_max']:.3f}}} \cdot OLR }}{{ \textcolor{{#0A84FF}}{{{p['K_B']:.3f}}} + OLR }} \right) \cdot \left( \dfrac{1}{{ 1 + \dfrac{{VFA/ALK}}{{ \textcolor{{#0A84FF}}{{{p['K_I']:.3f}}} }} }} \right) ''')
    if params_filter:
        st.markdown(f"**{t('help_filter_title')}**")
        p = params_filter
        st.latex(
            fr'COD_{{Removed}} = \textcolor{{#0A84FF}}{{{p["R_cod_tss"]:.3f}}} \cdot (TSS_{{in}} - TSS_{{out}}) + \textcolor{{#0A84FF}}{{{p["k_ads"]:.3f}}} \cdot sCOD_{{in}}')
    if params_rbc_orig or params_rbc_ph:
        st.markdown("---")
    if params_rbc_orig:
        st.markdown(f"**RBC v1.0 (Original)**")
        p = params_rbc_orig
        st.latex(fr'''\mu_a = \dfrac{{\textcolor{{#0A84FF}}{{{p['umxa']:.3f}}} \cdot S_e}}{{\textcolor{{#0A84FF}}{{{p['Ku']:.3f}}} + S_e}} \cdot \dfrac{{0.5 \cdot \textcolor{{#0A84FF}}{{{p['Ko']:.3f}}} + \textcolor{{#0A84FF}}{{{p['O']:.3f}}}}}{{ \textcolor{{#0A84FF}}{{{p['Ko']:.3f}}} + \textcolor{{#0A84FF}}{{{p['O']:.3f}}}}}''')
    if params_rbc_ph:
        st.markdown(f"**RBC v1.1 (pH-Inhibited)**")
        p = params_rbc_ph
        exponent_val = 0.5 * (p['pH_min'] - p['pH_max'])
        st.latex(
            fr'''\tau_{{pH}} = \dfrac{{ 1 + 2 \cdot 10^{{{exponent_val:.2f}}} }}{{ 1 + 10^{{pH - \textcolor{{#0A84FF}}{{{p['pH_max']:.2f}}}}} + 10^{{\textcolor{{#0A84FF}}{{{p['pH_min']:.2f}}} - pH}} }}''')


@st.cache_data
def plot_gsa_results(mi_df, si, problem):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        'Morris Elementary Effects', 'Sobol Sensitivity Indices'))
    mi_df.sort_values('mu_star', inplace=True)
    fig.add_trace(go.Scatter(x=mi_df['mu_star'], y=mi_df['sigma'], mode='markers+text',
                  text=mi_df.index, textposition="top right"), row=1, col=1)
    sobol_df = pd.DataFrame(
        {'S1': si['S1'], 'ST': si['ST']}, index=problem['names']).sort_values('ST')
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


@st.cache_data
def plot_mc_results(results_df):
    df_sorted = results_df.sort_values(
        by="Spearman_Correlation", key=abs, ascending=True)
    fig = px.bar(df_sorted, y='Parameter', x='Spearman_Correlation', color=np.where(df_sorted['Spearman_Correlation'] > 0, 'Positive', 'Negative'),
                 color_discrete_map={
                     'Positive': '#30D158', 'Negative': '#FF453A'},
                 title='<b>Monte Carlo Sensitivity Analysis</b>', orientation='h',
                 labels={'Spearman_Correlation': 'Spearman Rank Correlation'})
    fig.add_vline(x=0, line_dash="dash", line_color="grey")
    return fig


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

with st.sidebar:
    st.header(t('sidebar_controls'))
    uploaded_file = st.file_uploader(t('sidebar_upload'), type=[
                                     "csv"], help=t('sidebar_upload_help'))
    use_default_data = st.checkbox(t('sidebar_use_default'), value=True)

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
        if st.session_state.reactor.params_uasb and st.session_state.reactor.params_rbc_ph:
            srr_pred = st.session_state.reactor.uasb_model.predict(
                (what_if_cod / what_if_hrt_uasb, 0.3), st.session_state.reactor.params_uasb)
            cod_uasb_pred = what_if_cod - srr_pred * what_if_hrt_uasb
            final_pred = st.session_state.reactor.rbc_model_ph.predict(
                {'So': cod_uasb_pred * 0.8, 'HRT_days': what_if_hrt_rbc / 24.0, 'Xa': 400, 'Xs': 200, 'pH': 7.5}, st.session_state.reactor.params_rbc_ph)
            st.metric(label=t('what_if_result'),
                      value=f"{final_pred:.0f} mg/L" if not np.isnan(final_pred) else "N/A")

data_content = None
if uploaded_file:
    data_content = uploaded_file.getvalue()
elif use_default_data:
    try:
        url = 'https://raw.githubusercontent.com/RizkyBaldan/KINEMOD-KIT/main/Model_Data.csv'
        response = requests.get(url)
        # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        response.raise_for_status()
        data_content = response.content
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from GitHub: {e}")
        data_content = None

action_to_perform = None
if run_estimation_button:
    action_to_perform = "estimate"
if load_params_button:
    action_to_perform = "load"
if action_to_perform and data_content is not None:
    with st.spinner('🔬 Running analysis...'):
        reactor_obj, error = run_full_analysis(
            data_content, CONFIG, action_to_perform)
        if error:
            st.error(error)
            st.session_state.reactor = None
        else:
            st.session_state.reactor = reactor_obj
            st.session_state.sa_results = None
            st.session_state.edited_params = None
            st.session_state.optimizer_results = None
elif action_to_perform and data_content is None:
    st.warning(
        "Please upload a data file or select the default data option.", icon="⚠️")

# ==============================================================================
# 6. Main App Display
# ==============================================================================
if st.session_state.reactor:
    reactor = st.session_state.reactor
    df_results = reactor.df
    tabs = st.tabs(t('tabs'))

    with tabs[0]:  # Dashboard
        st.markdown('<div class="fade-in-up">', unsafe_allow_html=True)
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
        ts_chart = plot_interactive_timeseries(
            filtered_df, log_y=st.checkbox(t('log_axis_toggle')))
        st.plotly_chart(ts_chart, use_container_width=True, key="timeseries")

        st.markdown("---")
        st.subheader(t('parity_header'))
        parity_chart = plot_interactive_parity(filtered_df, "All Stages", 'COD_Final', {
                                               "Original": "COD_Final_Pred_Orig", "pH Model": "COD_Final_Pred_pH"})
        st.plotly_chart(parity_chart, use_container_width=True, key="parity")

        st.markdown("---")
        st.subheader(t('download_section_header'))
        dl_cols = st.columns(3)
        with dl_cols[0]:
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(label=t('download_data_button'), data=csv,
                               file_name='filtered_data.csv', mime='text/csv', use_container_width=True)
        with dl_cols[1]:
            ts_img = pio.to_image(ts_chart, format='png',
                                  width=1200, height=700, scale=2)
            st.download_button(label=t('download_ts_chart_button'), data=ts_img,
                               file_name='timeseries_plot.png', mime='image/png', use_container_width=True)
        with dl_cols[2]:
            parity_img = pio.to_image(
                parity_chart, format='png', width=800, height=600, scale=2)
            st.download_button(label=t('download_parity_chart_button'), data=parity_img,
                               file_name='parity_plot.png', mime='image/png', use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]:  # Model Details
        st.markdown('<div class="fade-in-up">', unsafe_allow_html=True)
        st.header(t('param_header'))
        param_tab, eq_tab = st.tabs([t('param_subheader'), t('eq_subheader')])
        with param_tab:
            st.info(t('editable_param_info'))
            param_sets = [("UASB", reactor.params_uasb), ("Filter", reactor.params_filter), (
                "RBC (Orig)", reactor.params_rbc_orig), ("RBC (pH)", reactor.params_rbc_ph)]
            if 'edited_params' not in st.session_state or st.session_state.edited_params is None:
                all_params = [{"Model": model, "Parameter": key, "Value": value}
                              for model, params in param_sets if params for key, value in params.items()]
                if all_params:
                    st.session_state.edited_params = pd.DataFrame(all_params)

            if 'edited_params' in st.session_state and st.session_state.edited_params is not None:
                edited_df = st.data_editor(
                    st.session_state.edited_params, key="params_editor", num_rows="dynamic")
                if st.button(t('rerun_button'), type="primary"):
                    st.session_state.edited_params = edited_df
                    with st.spinner("Re-running predictions..."):
                        p_uasb = dict(edited_df[edited_df['Model'] == 'UASB'].set_index(
                            'Parameter')['Value'])
                        p_filter = dict(
                            edited_df[edited_df['Model'] == 'Filter'].set_index('Parameter')['Value'])
                        p_rbc_o = dict(edited_df[edited_df['Model'] == 'RBC (Orig)'].set_index(
                            'Parameter')['Value'])
                        p_rbc_ph = dict(
                            edited_df[edited_df['Model'] == 'RBC (pH)'].set_index('Parameter')['Value'])

                        # We need the original, unprocessed dataframe for rerunning predictions
                        original_unprocessed_df = pd.read_csv(
                            io.BytesIO(data_content), sep=';')
                        new_df = reactor.run_predictions_with_new_params(
                            original_unprocessed_df, p_uasb, p_filter, p_rbc_o, p_rbc_ph)
                        st.session_state.reactor.df = new_df
                        st.success("Predictions updated!")
                        st.experimental_rerun()

        with eq_tab:
            display_final_equations(
                reactor.params_uasb, reactor.params_filter, reactor.params_rbc_orig, reactor.params_rbc_ph)
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[2]:  # Methane & Energy
        st.markdown('<div class="fade-in-up">', unsafe_allow_html=True)
        st.header(t('methane_header'))
        filtered_df = df_results[(df_results['Day'] >= st.session_state.get('date_slider_dashboard', (df_results['Day'].min(), df_results['Day'].max()))[
                                  0]) & (df_results['Day'] <= st.session_state.get('date_slider_dashboard', (df_results['Day'].min(), df_results['Day'].max()))[1])]
        uasb_cod_removed = (filtered_df['COD_in'] - filtered_df['COD_UASB_Pred']).mean(
        ) if 'COD_UASB_Pred' in filtered_df and not filtered_df.empty else 0
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
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[3]:  # Sensitivity
        st.markdown('<div class="fade-in-up">', unsafe_allow_html=True)
        st.header(t('sa_header'))
        st.info(t('sa_info'))
        sa_col1, sa_col2 = st.columns([1, 2])
        sa_chart = None
        with sa_col1:
            sa_type = st.selectbox(t('sa_type_select'), ["GSA", "Monte Carlo"])
            sa_model = st.selectbox(t('sa_model_select'), [
                                    "RBC Original", "RBC pH-Modified"])
            if st.button(t('run_sa_button'), type="primary", use_container_width=True):
                if data_content:
                    model_key = 'ph' if 'pH' in sa_model else 'orig'
                    with st.spinner(f"Running {sa_type}..."):
                        sa_error, sa_data = run_sensitivity_analysis(
                            sa_type, model_key, reactor.params_uasb, reactor.params_filter, reactor.params_rbc_orig, reactor.params_rbc_ph, data_content, CONFIG)
                        if sa_error:
                            st.error(sa_error)
                            st.session_state.sa_results = None
                        else:
                            st.success("Complete!")
                            st.session_state.sa_results = sa_data
                else:
                    st.warning("No data available to run analysis.", icon="⚠️")

        if st.session_state.sa_results:
            with sa_col2:
                sa_data = st.session_state.sa_results
                if 'Mi' in sa_data:
                    sa_chart = plot_gsa_results(
                        sa_data['Mi'], sa_data['Si'], sa_data['problem'])
                    st.plotly_chart(sa_chart, use_container_width=True)
                elif 'mc_results' in sa_data:
                    sa_chart = plot_mc_results(sa_data['mc_results'])
                    st.plotly_chart(sa_chart, use_container_width=True)

            if sa_chart:
                sa_img = pio.to_image(
                    sa_chart, format='png', width=1000, height=600, scale=2)
                st.download_button(label=t('download_sa_chart_button'), data=sa_img,
                                   file_name='sensitivity_analysis.png', mime='image/png', use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[4]:  # Optimizer
        st.markdown('<div class="fade-in-up">', unsafe_allow_html=True)
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
                    initial_guess = [np.mean(b) for b in bounds]
                    fixed_params = {
                        'vfa_alk': df_results['VFA_ALK_Ratio'].mean(),
                        'tss_in': df_results['TSS_UASB_Eff'].mean(),
                        'tss_out': df_results['TSS_Filt_Eff'].mean(),
                        'xa': df_results['Xa'].mean(),
                        'xs': df_results['Xs'].mean(),
                        'ph': df_results['pH_Eff_RBC'].mean()
                    }
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
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[5]:  # Help
        st.markdown('<div class="fade-in-up">', unsafe_allow_html=True)
        st.header(t('help_title'))
        st.markdown(t('help_intro'))

        with st.expander(t('help_usage_title')):
            st.markdown(
                f"**{t('help_usage_step1_title')}**: {t('help_usage_step1_text')}")
            st.markdown(
                f"**{t('help_usage_step2_title')}**: {t('help_usage_step2_text')}")
            st.markdown(
                f"**{t('help_usage_step3_title')}**: {t('help_usage_step3_text')}")
            st.markdown(
                f"**{t('help_usage_step4_title')}**: {t('help_usage_step4_text')}")

        with st.expander(t('help_model_docs_title')):
            st.markdown(f"#### {t('help_uasb_title')}\n{t('help_uasb_obj')}")
            st.markdown(
                f"#### {t('help_filter_title')}\n{t('help_filter_obj')}")
            st.markdown(f"#### {t('help_rbc_title')}\n{t('help_rbc_obj')}")

        with st.expander(t('help_faq_title')):
            st.markdown(f"**{t('help_faq1_q')}**\n{t('help_faq1_a')}")
            st.markdown(f"**{t('help_faq2_q')}**\n{t('help_faq2_a')}")

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info(t('welcome_message'))
    # You might want to have this image locally in your repo
    st.image("Diagram.png",
             caption="Diagram of the multi-stage treatment process.", width=500)
