import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

# Ensure dashboard folder is in sys.path for i18n
dashboard_dir = Path(__file__).parents[1]
if str(dashboard_dir) not in sys.path:
    sys.path.append(str(dashboard_dir))

from i18n import t, get_current_lang, set_lang

st.set_page_config(page_title="Ablation Study | तकनीकी जांच", page_icon="🔬", layout="wide")

# Authentication & Role Guard: Only Researchers can access advanced analytics pages
if not st.session_state.get("logged_in", False) or st.session_state.get("user_role") != "Researcher":
    st.switch_page("app.py")


# Authentication Guard: Redirect to Landing Screen if not logged in


# Sidebar: Language Switcher
st.sidebar.header("🌐 भाषा / Language")
lang_choice = st.sidebar.radio(
    "Select Language",
    ["हिन्दी (Hindi)", "English"],
    index=0 if get_current_lang() == "hi" else 1,
    horizontal=True,
    key="ablation_lang"
)
new_lang = "hi" if "हिन्दी" in lang_choice else "en"
if new_lang != get_current_lang():
    set_lang(new_lang)
    st.rerun()

is_hi = (get_current_lang() == "hi")

if is_hi:
    st.title("🔬 सिस्टम की तकनीकी जांच (Ablation Study)")
    st.markdown("जानें कि मौसम, मंडी आवक और पड़ोस की मंडियों के आंकड़ों को जोड़ने से मॉडल की सटीकता कैसे चरण-दर-चरण बढ़ी।")
else:
    st.title("🔬 Component & Spatial Ablation Study")
    st.markdown("""
    Empirical component ablation experiments validating each architectural phase of the multi-source forecasting engine on the 7-day forecast horizon.
    Each stage incrementally adds an information modality: Price Dynamics → Supply Arrivals → NASA Weather → Seasonality → Spatial Neighbor Graph → GraphSAGE Message Passing.
    """)

# Empirical Ablation Benchmark Data
experiments_en = [
    "Stage 1: Price Lags & Moving Averages Only",
    "Stage 2: + Market Arrival Quantities & Volatility",
    "Stage 3: + NASA POWER Weather & Thermal Anomalies",
    "Stage 4: + Cyclic Seasonality & Kharif/Rabi Encoding",
    "Stage 5: + Spatial Haversine Distance Neighbors",
    "Stage 6: + GraphSAGE Spatial Graph Embeddings (Proposed)"
]

experiments_hi = [
    "चरण 1: सिर्फ पुराना भाव (Price Lags Only)",
    "चरण 2: + मंडी आवक व माल की मात्रा (Arrivals)",
    "चरण 3: + नासा मौसम व तापमान (NASA Weather)",
    "चरण 4: + खरीफ/रबी मौसमी चक्र (Seasonality)",
    "चरण 5: + पड़ोसी मंडियों की दूरी (Spatial Distance)",
    "चरण 6: + ग्राफ न्यूरल नेटवर्क (GraphSAGE Spatial AI)"
]

experiments = experiments_hi if is_hi else experiments_en

rmse_values = [899.43, 862.15, 838.08, 815.44, 808.50, 797.24]
r2_values = [0.7893, 0.8012, 0.8171, 0.8277, 0.8310, 0.8345]
trust_scores = [88.0, 88.9, 89.4, 90.1, 90.3, 90.6]

improvements = [0.0]
for i in range(1, len(rmse_values)):
    imp = ((rmse_values[i-1] - rmse_values[i]) / rmse_values[i-1]) * 100
    improvements.append(imp)

ablation_df = pd.DataFrame({
    'चरण / घटक (Stage / Component)' if is_hi else 'Architectural Component': experiments,
    '7-Day RMSE (₹/qtl)': rmse_values,
    'भरोसा स्कोर / R²' if is_hi else 'Goodness of Fit R²': r2_values,
    'Trust Score (100 में से)': trust_scores,
    'त्रुटि में कमी (Gain %)' if is_hi else 'Incremental Gain (%)': [round(x, 2) for x in improvements]
})

st.subheader("📈 " + ("प्रत्येक नए डाटा स्रोत से संभावित त्रुटि में कमी" if is_hi else "Progressive Error Reduction Across Pipeline Stages"))

fig_line = px.line(
    ablation_df,
    x='चरण / घटक (Stage / Component)' if is_hi else 'Architectural Component',
    y='7-Day RMSE (₹/qtl)',
    markers=True,
    title=("मॉडल की गलती में कमी (कम गलती = ज्यादा सटीक भाव)" if is_hi else "Model Error (RMSE ₹/qtl) Across Information Modalities"),
    text='7-Day RMSE (₹/qtl)'
)
fig_line.update_traces(textposition="bottom left", texttemplate="₹%{text:.1f}")
fig_line.update_layout(xaxis_tickangle=-25, height=450)
st.plotly_chart(fig_line, use_container_width=True)

st.markdown("---")
st.markdown("### 📋 " + ("विस्तृत परिणाम तालिका (Results Table)" if is_hi else "Empirical Ablation Results Table"))
st.dataframe(
    ablation_df.style
    .highlight_min(subset=['7-Day RMSE (₹/qtl)'], color='#d4edda')
    .highlight_max(subset=['भरोसा स्कोर / R²' if is_hi else 'Goodness of Fit R²', 'Trust Score (100 में से)', 'त्रुटि में कमी (Gain %)' if is_hi else 'Incremental Gain (%)'], color='#cce5ff'),
    use_container_width=True
)

st.success(
    "💡 **" + ("मुख्य निष्कर्ष (Key Finding):" if is_hi else "Key Finding:") + "**\n\n" +
    ("जब हमने सिर्फ पुराने भाव के बजाय मंडी की दूरी और पड़ोसी मंडियों का असर (GraphSAGE) जोड़ा, तो 7 दिन के अनुमान में त्रुटि **₹899.43** से घटकर **₹797.24** रह गई। यानी **₹102 प्रति क्विंटल** का भारी सुधार हुआ! इससे साबित होता है कि पास की मंडियों में क्या चल रहा है, यह जानना किसान के सही भाव के लिए सबसे जरूरी है।"
     if is_hi else
     "Adding GraphSAGE spatial message passing drops 7-day forecast RMSE from **₹899.43** down to **₹797.24**, achieving an overall **₹102.19 / quintal error reduction**. This proves that cross-mandi spatial arbitrage and localized contagion are essential drivers of crop price discovery.")
)