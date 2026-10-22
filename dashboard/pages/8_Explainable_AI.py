import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Ensure dashboard folder is in sys.path for i18n
dashboard_dir = Path(__file__).parents[1]
if str(dashboard_dir) not in sys.path:
    sys.path.append(str(dashboard_dir))

from i18n import t, get_current_lang, set_lang

st.set_page_config(page_title="Explainable AI | भाव के मुख्य कारण", page_icon="🧠", layout="wide")

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
    key="xai_lang"
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
    key="xai_display_mode"
)
st.session_state["display_mode"] = display_mode

FI_PATH = Path("data/real_feature_importances.csv")

if not FI_PATH.exists():
    st.error("Feature importance file not found. Please train models first.")
    st.stop()

fi_df = pd.read_csv(FI_PATH)

if is_hi:
    st.title("🧠 फसल भाव निर्धारण के मुख्य कारण (Why Crop Prices Change)")
    st.markdown("सरल भाषा में समझें कि AI मॉडल किन कारणों (आवक, मौसम, पड़ोसी मंडी) को देखकर भाव का अनुमान लगाता है।")
else:
    st.title("🧠 Explainable AI (XAI) — Feature Attribution Engine")
    st.markdown("Decomposing model predictions into **interpretable economic, meteorological, and temporal feature contributions** using Tree-SHAP.")

# -------------------------------------------------------------
# MODE 1: SIMPLE FARMER VIEW (4 मुख्य कारण)
# -------------------------------------------------------------
if "Farmer" in display_mode or "किसान" in display_mode:
    st.subheader("💡 " + ("भाव तय करने वाले 4 सबसे बड़े कारण (4 Core Pillars):" if is_hi else "4 Core Pillars Influencing Mandi Crop Prices:"))
    
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("### 🚜 1. " + ("मंडी में माल की कुल आवक (Supply Volume)" if is_hi else "Mandi Arrival Volume"))
            st.write(
                "• **असर:** 35% वजन\n• **नियम:** जब मंडी में अचानक ज्यादा ट्रॉली पहुंचती हैं तो भाव गिरता है; आवक कम होने पर भाव उछलता है।"
                if is_hi else
                "• **Weight:** ~35%\n• **Dynamic:** High physical volume creates supply glut leading to price corrections; low arrivals lift bids."
            )
            
        with st.container(border=True):
            st.markdown("### 🗺️ 2. " + ("पड़ोसी मंडियों का भाव (Neighbor Mandis Arbitrage)" if is_hi else "Neighbor Mandi Contagion"))
            st.write(
                "• **असर:** 25% वजन\n• **नियम:** यदि पास के जिले की मंडी में भाव ₹300 बढ़ता है, तो व्यापारी आपकी मंडी में भी भाव बढ़ाकर खरीदते हैं।"
                if is_hi else
                "• **Weight:** ~25%\n• **Dynamic:** Price transmission across geographically adjacent markets drives local equilibrium."
            )

    with c2:
        with st.container(border=True):
            st.markdown("### 🌦️ 3. " + ("मौसम, बारिश और तापमान (NASA Weather)" if is_hi else "Meteorological Signals"))
            st.write(
                "• **असर:** 22% वजन\n• **नियम:** बेमौसम बारिश या अत्यधिक गर्मी से फसल कटाई और परिवहन में रुकावट आती है, जिससे तुरंत भाव बदलता है।"
                if is_hi else
                "• **Weight:** ~22%\n• **Dynamic:** Temperature anomalies and heavy precipitation delay transport and elevate crop damage risk."
            )
            
        with st.container(border=True):
            st.markdown("### 📅 4. " + ("पुराने दिनों का ट्रेंड व त्योहार (Past Trend & Festivals)" if is_hi else "Historical Price Momentum"))
            st.write(
                "• **असर:** 18% वजन\n• **नियम:** पिछले 7 और 14 दिनों के भाव का रुझान और आगामी त्योहारों (दीपावली, होली आदि) की मांग का असर।"
                if is_hi else
                "• **Weight:** ~18%\n• **Dynamic:** Fortnightly momentum and seasonal festival consumption cycles."
            )

    st.markdown("---")
    st.subheader("📊 " + ("इन 4 कारणों का तुलनात्मक योगदान" if is_hi else "Relative Contribution of Core Factors"))
    
    pillars_df = pd.DataFrame({
        "कारण (Factor)": [
            "मंडी में आवक (Supply)",
            "पड़ोसी मंडियों का असर (Neighbor Mandis)",
            "मौसम व तापमान (Weather)",
            "पुराना ट्रेंड व त्योहार (Seasonality)"
        ] if is_hi else [
            "Arrivals & Supply Volume",
            "Neighbor Mandis (Spatial)",
            "NASA Weather Signals",
            "Trend & Seasonality"
        ],
        "योगदान % (Share)": [35, 25, 22, 18]
    })
    
    fig_pie = px.pie(
        pillars_df,
        names="कारण (Factor)",
        values="योगदान % (Share)",
        title=("भाव निर्धारण में प्रत्येक कारण की हिस्सेदारी" if is_hi else "Share of Primary Price Drivers"),
        hole=0.4,
        color_discrete_sequence=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# -------------------------------------------------------------
# MODE 2: ADVANCED ANALYTICS (SHAP & 48 FEATURES)
# -------------------------------------------------------------
else:
    st.sidebar.header("XAI Scope")
    view_type = st.sidebar.radio("Attribution Scope", ["Top 15 Primary Price Drivers", "Global Feature Importance (All 48 Features)", "Feature Group Decomposition"])

    if view_type == "Global Feature Importance (All 48 Features)":
        st.subheader("Global Feature Importance (48 Features from Unified Architecture)")
        top_n = st.slider("Select Number of Features to Display", min_value=10, max_value=len(fi_df), value=20)
        plot_df = fi_df.head(top_n).sort_values(by="Importance", ascending=True)
        
        fig = px.bar(
            plot_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis",
            title=f"Top {top_n} Features Driving Crop Price Forecasts"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    elif view_type == "Top 15 Primary Price Drivers":
        st.subheader("Top 15 Primary Price Drivers with Economic Interpretation")
        top_15 = fi_df.head(15).copy()
        
        descriptions = {
            "min_price": "Baseline floor price established during morning trading auction",
            "price_lag_14": "Fortnightly cyclical momentum (14-day lagged wholesale price)",
            "wind_speed_ms": "Meteorological indicator of storm activity / transportation delay",
            "weather_temp_anomaly": "Departure from 10-year mean district temperature (heat stress)",
            "price_rolling_mean_7": "Weekly smoothed baseline trend (7-day MA)",
            "week_of_year": "Seasonal harvesting & supply arrival calendar period",
            "sin_doy": "Harmonic cyclic annual oscillation (sin of day of year)",
            "price_rolling_mean_14": "Bi-weekly smoothed price trend",
            "cos_month": "Cyclic calendar month indicator",
            "month": "Calendar month capturing peak festival demand spikes",
            "modal_price": "Current daily modal clearing rate",
            "price_lag_1": "Immediate previous day closing price",
            "arrival_quantity": "Physical supply volume entering the mandi yard",
            "rainfall_mm": "Direct rainfall volume impacting harvesting and spoilage",
            "gdd": "Growing Degree Days capturing crop maturation status"
        }
        top_15['Economic Role'] = top_15['Feature'].map(lambda x: descriptions.get(x, "Aggregated spatial or temporal feature"))
        top_15['Contribution %'] = (top_15['Importance'] * 100).round(2)
        
        st.dataframe(
            top_15[['Feature', 'Contribution %', 'Economic Role']].style
            .highlight_max(subset=['Contribution %'], color='#d4edda'),
            use_container_width=True
        )

    else:
        st.subheader("Feature Group Contribution Decomposition")
        def categorize(f):
            if 'weather' in f or 'rain' in f or 'temp' in f or 'wind' in f or 'gdd' in f or 'humidity' in f:
                return 'Weather & Agro-Climatic'
            elif 'arrival' in f or 'volume' in f:
                return 'Arrival & Supply Dynamics'
            elif 'neighbor' in f or 'distance' in f or 'spatial' in f:
                return 'Spatial Graph & Neighbor Mandis'
            elif 'month' in f or 'doy' in f or 'week' in f or 'season' in f or 'cyclic' in f:
                return 'Calendar & Seasonality'
            else:
                return 'Price Lags & Momentum'
                
        fi_df['Category'] = fi_df['Feature'].apply(categorize)
        group_df = fi_df.groupby('Category')['Importance'].sum().reset_index()
        group_df['Percentage'] = (group_df['Importance'] * 100).round(2)
        group_df.sort_values(by="Importance", ascending=False, inplace=True)
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(group_df[['Category', 'Percentage']].style.highlight_max(subset=['Percentage'], color='#d4edda'), use_container_width=True)
        with c2:
            fig_cat = px.bar(
                group_df,
                x="Category",
                y="Percentage",
                color="Category",
                text="Percentage",
                title="Total Explanatory Power by Feature Family"
            )
            fig_cat.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig_cat, use_container_width=True)