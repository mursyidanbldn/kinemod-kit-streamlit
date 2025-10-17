# 🧪 KINEMOD-KIT STREAMLIT DASHBOARD

**Integrated Kinetic Modeling and Sensitivity Analysis of a Multi-Stage POME Treatment System**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR-APP-URL-HERE)

---

## 📋 Overview

KINEMOD-KIT is a comprehensive web-based platform for analyzing and simulating the performance of a three-stage Palm Oil Mill Effluent (POME) treatment system:

- 🟤 **UASB Reactor** - Anaerobic digestion for primary organic load reduction
- 🪶 **EFB Filtration** - Adsorptive filtration for suspended solids and residual organics
- 🧬 **RBC Unit** - Aerobic polishing for final effluent quality compliance

---

## 🚀 Live Demo

**👉 [Access the Application Here](YOUR-APP-URL-HERE)**

No installation required - just click and use!

---

## 🎯 Features

- ✅ **Real-time Kinetic Modeling** - Calibrate model parameters from your data
- 📊 **Interactive Dashboard** - Visualize performance with dynamic charts
- 🔬 **Sensitivity Analysis** - Global Sensitivity Analysis (GSA) and Monte Carlo simulations
- ⚙️ **Process Optimizer** - Find optimal operational conditions
- 🌿 **Methane & Energy Potential** - Calculate biogas production and energy generation
- 🌐 **Bilingual Support** - English and Bahasa Indonesia

---

## 📖 How to Use

1. **Upload your data** - CSV file with reactor performance data
2. **Calibrate models** - Click "Calibrate" to estimate kinetic parameters
3. **Analyze results** - Explore the dashboard, view equations, and run analyses
4. **Optimize** - Find the best operational settings for your system

---

## 📊 Sample Data Format

Your CSV should include columns like:
- `Day`, `COD_in`, `COD_UASB_Eff`, `COD_Filt_Eff`, `COD_Final`
- `HRT_UASB`, `HRT_hours`, `TSS_UASB_Eff`, `TSS_Filt_Eff`
- `VFA_Eff`, `ALK_Eff`, `pH_Eff_RBC`, `Xa`, `Xs`

See `Model_Data.csv` in this repository for an example.

---

## 🛠️ Technology Stack

- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Modeling:** SciPy, Scikit-learn
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Sensitivity Analysis:** SALib

---

## 👨‍🔬 Author

**Rizky Mursyidan Baldan**

*Thesis Project - [Institut Pertanian Bogor]*

📧 Contact: [rizkymursyidan@gmail.com]

---

## 📄 License

This project is part of academic research. For academic use and collaboration inquiries, please contact the author.
