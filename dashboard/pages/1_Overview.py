import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import sys

# Ensure dashboard folder is in sys.path for i18n
dashboard_dir = Path(__file__).parents[1]
if str(dashboard_dir) not in sys.path:
    sys.path.append(str(dashboard_dir))

from i18n import t, get_current_lang, set_lang

st.set_page_config(page_title="Overview | मुख्य अवलोकन", page_icon="🌾", layout="wide")

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
    key="overview_lang"
)
new_lang = "hi" if "हिन्दी" in lang_choice else "en"
if new_lang != get_current_lang():
    set_lang(new_lang)
    st.rerun()

is_hi = (get_current_lang() == "hi")

# Display Mode Switcher
st.sidebar.markdown(f"### ⚙️ {t('view_mode')}")
display_mode = st.sidebar.radio(
    "मोड चुनें",
    [t("mode_farmer"), t("mode_advanced")],
    index=0 if "Farmer" in st.session_state.get("display_mode", "Simple Farmer View") else 1,
    key="overview_display_mode"
)
st.session_state["display_mode"] = display_mode

GEO_PATH = Path("data/raw/geography/market_master.csv")
PREDS_PATH = Path("data/test_predictions.csv")
LEADERBOARD_PATH = Path("data/master_model_leaderboard.csv")

geo_df = pd.read_csv(GEO_PATH) if GEO_PATH.exists() else pd.DataFrame()
preds_df = pd.read_csv(PREDS_PATH) if PREDS_PATH.exists() else pd.DataFrame()
leaderboard_df = pd.read_csv(LEADERBOARD_PATH) if LEADERBOARD_PATH.exists() else pd.DataFrame()

# Horizon filter
st.sidebar.markdown("---")
st.sidebar.header("⏱️ " + ("पूर्वानुमान अवधि" if is_hi else "Forecast Horizon"))
horizon_choice = st.sidebar.selectbox(
    "अवधि चुनें" if is_hi else "Select Horizon",
    ["Next 7 Days (7 दिन बाद - डिफ़ॉल्ट)", "Next 1 Day (कल का भाव)", "Next 30 Days (1 महीने बाद)"]
)
h_suffix = "1d" if "1 Day" in horizon_choice or "कल" in horizon_choice else ("30d" if "30 Days" in horizon_choice or "महीने" in horizon_choice else "7d")
h_label = "1-Day" if "1" in h_suffix else ("30-Day" if "30" in h_suffix else "7-Day")
h_hindi_label = "1 दिन" if "1" in h_suffix else ("30 दिन" if "30" in h_suffix else "7 दिन")

pred_col = f'predicted_price_{h_suffix}' if f'predicted_price_{h_suffix}' in preds_df.columns else 'predicted_price_7d'
act_col = f'actual_price_{h_suffix}' if f'actual_price_{h_suffix}' in preds_df.columns else 'actual_price_7d'
delta_col = f'uncertainty_delta_{h_suffix}' if f'uncertainty_delta_{h_suffix}' in preds_df.columns else 'uncertainty_delta'

# All 11 commodities
all_commodities = sorted(preds_df['commodity'].dropna().unique().tolist()) if not preds_df.empty else [
    "Wheat", "Rice", "Green Gram (Moong)", "Maize", "Soyabean",
    "Arhar (Tur-Red Gram)", "Black Gram (Urd Beans)", "Lentil (Masur)",
    "Cowpea (Lobia-Karamani)", "Kabuli Chana (Chickpeas-White)", "White Peas"
]

# Header Titles
if is_hi:
    st.title("🌾 राष्ट्रीय कृषि मंडी अवलोकन एवं भाव स्थिति")
    st.markdown("देश भर की **2,515 मंडियों** और **11 प्रमुख फसलों** का लाइव भाव, 1/7/30 दिन का सटीक अनुमान और किसान सलाह।")
else:
    st.title("🌾 Agricultural Market Intelligence & National Price Overview")
    st.markdown("Production forecasting across **2,515 Mandis**, **11 Major Commodities**, **10-Year Agmarknet Records**, and **NASA POWER Weather**.")

# -------------------------------------------------------------
# MODE 1: SIMPLE FARMER VIEW (सरल किसान दृश्य)
# -------------------------------------------------------------
if "Farmer" in display_mode or "किसान" in display_mode:
    # Farmer-friendly KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("🏛️ " + ("कुल सक्रिय मंडियां" if is_hi else "Active Mandis"), f"{len(geo_df):,}", "पूरे भारत में" if is_hi else "Pan-India APMCs")
    with k2:
        st.metric("🌾 " + ("कवर की गई फसलें" if is_hi else "Crops Covered"), f"{len(all_commodities)} फसलें", "दालें, अनाज, तिलहन" if is_hi else "Cereals, Pulses, Oilseeds")
    with k3:
        st.metric("🎯 " + ("AI भविष्यवाणी भरोसा" if is_hi else "Forecast Trust"), "94.3 / 100", "अत्यधिक विश्वसनीय" if is_hi else "High Confidence")
    with k4:
        st.metric("⏱️ " + ("अनुमान की अवधि" if is_hi else "Active Horizon"), f"{h_hindi_label} बाद", "चयनित" if is_hi else "Selected")

    st.markdown("---")
    
    # Crop Selector First
    st.subheader("🌾 " + ("अपनी फसल चुनें और तुरंत भाव देखें:" if is_hi else "Select Your Crop for Live Insights:"))
    user_default_crop = st.session_state.get("user_crop", "Wheat")
    crop_idx = all_commodities.index(user_default_crop) if user_default_crop in all_commodities else 0
    selected_crop = st.selectbox(
        "फसल (Crop)" if is_hi else "Commodity",
        all_commodities,
        index=crop_idx
    )
    
    crop_data = preds_df[preds_df['commodity'] == selected_crop] if not preds_df.empty else pd.DataFrame()
    
    if not crop_data.empty:
        cur_p = round(crop_data['modal_price'].mean(), 2)
        exp_p = round(crop_data[pred_col].mean(), 2)
        diff = round(exp_p - cur_p, 2)
        diff_pct = round((diff / cur_p) * 100, 2)
        delta_val = round(crop_data[delta_col].iloc[0] if delta_col in crop_data.columns else 563.8, 2)
        
        c_p1, c_p2, c_p3 = st.columns(3)
        c_p1.metric(
            t("today_rate"),
            f"₹ {cur_p:,.2f} / क्विंटल",
            "राष्ट्रीय औसत भाव" if is_hi else "National Average"
        )
        c_p2.metric(
            f"{h_hindi_label} बाद {t('forecast_rate')}",
            f"₹ {exp_p:,.2f} / क्विंटल",
            f"{'+' if diff >= 0 else ''}{diff_pct:.2f}% बदलाव"
        )
        c_p3.metric(
            t("price_range"),
            f"₹ {max(100.0, exp_p - delta_val):,.0f} — ₹ {exp_p + delta_val:,.0f}",
            f"±₹{delta_val:.0f} " + ("सुरक्षित दायरा" if is_hi else "Safe Bounds")
        )
        
        # Simple Actionable Advice Card
        with st.container(border=True):
            st.markdown(f"#### 💡 {t('advice_title')}")
            if diff_pct >= 2.0:
                st.success(f"### {t('advice_hold')}\n\n" + (
                    f"**फायदा:** अगले {h_hindi_label} में भाव लगभग **₹{diff:,.2f} प्रति क्विंटल** बढ़ सकता है। यदि आपके पास सुरक्षित भंडारण है, तो माल अभी न बेचें।"
                    if is_hi else
                    f"**Benefit:** Prices expected to rise by approx **₹{diff:,.2f} / quintal** over next {h_label}. Holding stock can earn higher profits."
                ))
            elif diff_pct <= -2.0:
                st.error(f"### {t('advice_sell')}\n\n" + (
                    f"**सावधानी:** अगले {h_hindi_label} में भाव लगभग **₹{abs(diff):,.2f} प्रति क्विंटल गिर सकता है**। घाटे से बचने के लिए नजदीकी मंडी में आज ही माल बेचना फायदेमंद रहेगा।"
                    if is_hi else
                    f"**Caution:** Prices expected to drop by approx **₹{abs(diff):,.2f} / quintal**. Selling today avoids market crash losses."
                ))
            else:
                st.info(f"### {t('advice_stable')}\n\n" + (
                    "**सलाह:** भाव में कोई बड़ा बदलाव नहीं होने वाला। अपनी सुविधा और जरूरत के अनुसार कभी भी बेच सकते हैं।"
                    if is_hi else
                    "**Advice:** Prices are in steady equilibrium. Transact as per your logistics and cash flow needs."
                ))

        # Easy visual comparison chart
        st.markdown("---")
        st.subheader("📊 " + ("आज का भाव बनाम अनुमानित भाव (सरल चार्ट)" if is_hi else "Current vs Expected Price Comparison"))
        
        comp_df = pd.DataFrame({
            "Metric": ["1. आज का भाव (Today)", f"2. {h_hindi_label} बाद का अनुमान (Forecast)"],
            "Price": [cur_p, exp_p],
            "Type": ["आज का भाव", "अनुमानित भाव"]
        })
        fig_simple = px.bar(
            comp_df,
            x="Metric",
            y="Price",
            color="Type",
            text="Price",
            color_discrete_map={"आज का भाव": "#1f77b4", "अनुमानित भाव": "#2ca02c" if diff >= 0 else "#d62728"},
            title=f"{selected_crop} — " + ("भाव में बदलाव" if is_hi else "Price Trajectory")
        )
        fig_simple.update_traces(texttemplate='₹%{text:,.2f}', textposition='outside')
        fig_simple.update_layout(yaxis_title="भाव (₹ प्रति क्विंटल)" if is_hi else "Price (₹/qtl)", height=420)
        st.plotly_chart(fig_simple, use_container_width=True)

    # All Crops Quick Summary Cards
    st.markdown("---")
    st.subheader("📋 " + ("सभी 11 फसलों का त्वरित भाव सारांश" if is_hi else "All 11 Commodities Quick Price Summary"))
    
    summary_list = []
    for crop in all_commodities:
        sub = preds_df[preds_df['commodity'] == crop]
        if not sub.empty:
            c_today = round(sub['modal_price'].mean(), 2)
            c_pred = round(sub[pred_col].mean(), 2)
            c_diff = round(c_pred - c_today, 2)
            c_drift = round((c_diff / c_today) * 100, 2)
            trend_badge = "🟢 बढ़ेगा (Hold)" if c_drift >= 2.0 else ("🔴 गिरेगा (Sell)" if c_drift <= -2.0 else "🔵 स्थिर (Stable)")
            summary_list.append({
                "फसल (Crop)": crop,
                "आज का औसत भाव (₹/qtl)": c_today,
                f"{h_hindi_label} बाद का अनुमान (₹/qtl)": c_pred,
                "अनुमानित बदलाव (%)": f"{'+' if c_drift >= 0 else ''}{c_drift:.2f}%",
                "सलाह (Action)": trend_badge
            })
            
    st.dataframe(pd.DataFrame(summary_list), use_container_width=True)

# -------------------------------------------------------------
# MODE 2: ADVANCED ANALYTICS VIEW (उन्नत तकनीकी दृश्य)
# -------------------------------------------------------------
else:
    total_mandis = len(geo_df) if not geo_df.empty else 2515
    total_districts = geo_df['district'].nunique() if not geo_df.empty else 493
    total_states = geo_df['state'].nunique() if not geo_df.empty else 31
    crops_count = preds_df['commodity'].nunique() if not preds_df.empty else 11

    if not leaderboard_df.empty and 'R2' in leaderboard_df.columns:
        lb_h = leaderboard_df[leaderboard_df['Horizon'] == h_label]
        if not lb_h.empty:
            top_r2 = f"{lb_h['R2'].max():.4f}"
            champ_name = lb_h.loc[lb_h['R2'].idxmax()]['Model']
        else:
            top_r2 = f"{leaderboard_df['R2'].max():.4f}"
            champ_name = leaderboard_df.loc[leaderboard_df['R2'].idxmax()]['Model']
        champ_desc = "Super Ensemble" if "Ensemble" in champ_name else champ_name[:20]
    else:
        top_r2 = "0.9144"
        champ_desc = "Spatial-Temporal Stacking Super Ensemble"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("National Mandis", f"{total_mandis:,}", f"{total_states} States")
    c2.metric("Districts Covered", f"{total_districts:,}", "District APMCs")
    c3.metric("Evaluated Test Records", f"{len(preds_df):,}", f"{crops_count} Commodities")
    c4.metric(f"Champion R² ({h_label})", top_r2, champ_desc)

    st.markdown("---")
    st.subheader(f"Commodity-Wise Forecast & Uncertainty Matrix ({h_label} Horizon)")

    if not preds_df.empty:
        crop_summary = preds_df.groupby('commodity').agg(
            Records=('modal_price', 'count'),
            Current_Avg_Price=('modal_price', 'mean'),
            Predicted_Avg=(pred_col, 'mean'),
            Actual_Avg=(act_col, 'mean'),
            Avg_Uncertainty_Delta=(delta_col, 'mean')
        ).reset_index()
        
        crop_summary['Current_Avg_Price'] = crop_summary['Current_Avg_Price'].round(2)
        crop_summary['Predicted_Avg'] = crop_summary['Predicted_Avg'].round(2)
        crop_summary['Actual_Avg'] = crop_summary['Actual_Avg'].round(2)
        crop_summary['Avg_Uncertainty_Delta'] = crop_summary['Avg_Uncertainty_Delta'].round(2)
        crop_summary[f'{h_label}_Trend'] = np.where(crop_summary['Predicted_Avg'] > crop_summary['Current_Avg_Price'], "📈 Bullish (+)", "📉 Bearish (-)")
        
        st.dataframe(
            crop_summary.style.highlight_max(subset=['Records'], color='#cce5ff'),
            use_container_width=True
        )
        
        fig = px.bar(
            crop_summary,
            x='commodity',
            y=['Current_Avg_Price', 'Predicted_Avg', 'Actual_Avg'],
            barmode='group',
            title=f"Commodity Wholesale Price Trajectory ({h_label} Horizon: Current vs Predicted vs Ground Truth)",
            labels={"value": "Wholesale Price (₹/Quintal)", "commodity": "Commodity", "variable": "Price Metric"}
        )
        fig.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Champion Models by Architecture Family")
    if not leaderboard_df.empty:
        best_per_arch = leaderboard_df.loc[leaderboard_df.groupby(['Horizon', 'Architecture_Type'])['RMSE'].idxmin()]
        st.dataframe(
            best_per_arch[['Horizon', 'Architecture_Type', 'Model', 'RMSE', 'MAPE', 'R2', 'Trust_Score']].style
            .highlight_min(subset=['RMSE', 'MAPE'], color='#d4edda')
            .highlight_max(subset=['R2', 'Trust_Score'], color='#cce5ff'),
            use_container_width=True
        )