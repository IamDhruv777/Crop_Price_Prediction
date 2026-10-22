import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Ensure dashboard folder is in sys.path for i18n import
dashboard_dir = Path(__file__).parents[1]
if str(dashboard_dir) not in sys.path:
    sys.path.append(str(dashboard_dir))

from i18n import t, get_current_lang, set_lang

st.set_page_config(page_title="Price Forecast | फसल भाव अनुमान", layout="wide", page_icon="🌾")

# Authentication & Role Guard: Only Researchers can access advanced analytics pages
if not st.session_state.get("logged_in", False) or st.session_state.get("user_role") != "Researcher":
    st.switch_page("app.py")


# Authentication Guard: Redirect to Landing Screen if not logged in


# Language Toggle in Sidebar
st.sidebar.header(t("sidebar_lang_title"))
lang_choice = st.sidebar.radio(
    "Language",
    ["English", "हिन्दी (Hindi)"],
    index=0 if get_current_lang() == "en" else 1,
    horizontal=True,
    key="forecast_lang"
)
new_lang = "hi" if "हिन्दी" in lang_choice else "en"
if new_lang != get_current_lang():
    set_lang(new_lang)
    st.rerun()

is_hi = (get_current_lang() == "hi")

# Display Mode Toggle
st.sidebar.markdown(f"### ⚙️ {t('view_mode')}")
display_mode = st.sidebar.radio(
    t("view_mode"),
    [t("mode_farmer"), t("mode_advanced")],
    index=0 if "Farmer" in st.session_state.get("display_mode", "Simple Farmer View") or "किसान" in st.session_state.get("display_mode", "") else 1,
    key="forecast_display_mode"
)
st.session_state["display_mode"] = display_mode
is_farmer_view = ("Farmer" in display_mode or "किसान" in display_mode)

if is_hi:
    st.title("🌾 फसल भाव का सटीक अनुमान एवं किसान सलाह")
    st.markdown("आने वाले 1 दिन, 7 दिन और 30 दिन का सटीक भाव जानें, ताकि आप सही समय पर फसल बेचकर ज्यादा मुनाफा कमा सकें।")
else:
    st.title("🌾 Multi-Horizon Crop Price Forecasting & Farmer Decision Engine")
    st.markdown("Accurate 1-Day, 7-Day, and 30-Day price forecasts with actionable holding/selling guidance across Indian mandis.")

# Load real pipeline outputs
DATA_DIR = Path("data")
PREDS_FILE = DATA_DIR / "test_predictions.csv"
METRICS_FILE = DATA_DIR / "master_model_leaderboard.csv"

try:
    preds_df = pd.read_csv(PREDS_FILE)
    
    st.sidebar.header("⚙️ " + ("फसल व मंडी चुनाव" if is_hi else "Forecast Filters"))
    
    # 1. SELECT CROP FIRST (Reveals all 11 crops)
    all_commodities = sorted(preds_df['commodity'].dropna().unique().tolist())
    user_def_crop = st.session_state.get("user_crop", "Wheat")
    default_crop_idx = all_commodities.index(user_def_crop) if user_def_crop in all_commodities else (all_commodities.index("Wheat") if "Wheat" in all_commodities else 0)
    selected_crop = st.sidebar.selectbox(
        "1. " + ("फसल चुनें" if is_hi else "Select Crop"),
        all_commodities,
        index=default_crop_idx
    )
    
    # 2. SELECT MANDI SECOND (Filtered for that crop)
    crop_sub = preds_df[preds_df['commodity'] == selected_crop]
    available_markets = sorted(crop_sub['market'].dropna().unique().tolist())
    market = st.sidebar.selectbox(
        "2. " + ("मंडी चुनें" if is_hi else "Select Mandi"),
        available_markets
    )
    
    # 3. SELECT HORIZON
    horizon_options = ["7 दिन बाद (Next 7 Days)", "1 दिन बाद (Next 1 Day)", "30 दिन बाद (Next 30 Days)"] if is_hi else ["Next 7 Days (Default)", "Next 1 Day", "Next 30 Days"]
    horizon_choice = st.sidebar.selectbox(
        "3. " + ("समय सीमा" if is_hi else "Forecast Horizon"),
        horizon_options
    )
    h_suffix = "1d" if "1 Day" in horizon_choice or "1 दिन" in horizon_choice else ("30d" if "30 Days" in horizon_choice or "30 दिन" in horizon_choice else "7d")
    h_label = ("1 दिन" if "1" in h_suffix else ("30 दिन" if "30" in h_suffix else "7 दिन")) if is_hi else ("1-Day" if "1" in h_suffix else ("30-Day" if "30" in h_suffix else "7-Day"))

    # Filter for selected market and crop
    market_df = crop_sub[crop_sub['market'] == market].sort_values('date')
    if market_df.empty:
        market_df = crop_sub.sort_values('date')
        
    latest_row = market_df.iloc[-1]
    cur_p = float(latest_row['modal_price'])
    pred_col = f'predicted_price_{h_suffix}' if f'predicted_price_{h_suffix}' in market_df.columns else 'predicted_price_7d'
    actual_col = f'actual_price_{h_suffix}' if f'actual_price_{h_suffix}' in market_df.columns else 'actual_price_7d'
    delta_col = f'uncertainty_delta_{h_suffix}' if f'uncertainty_delta_{h_suffix}' in market_df.columns else 'uncertainty_delta_7d'
    
    pred_p = float(latest_row[pred_col]) if pred_col in latest_row else cur_p
    delta_val = float(latest_row[delta_col]) if delta_col in latest_row else 563.81
    pct_drift = ((pred_p - cur_p) / cur_p) * 100 if cur_p > 0 else 0.0
    diff_rupees = round(pred_p - cur_p, 2)
    trolley_rupees = round(abs(diff_rupees) * 50, 0)
    
    st.subheader(f"🌾 **{selected_crop}** @ {market} — " + (f"{h_label} का भाव पूर्वानुमान" if is_hi else f"{h_label} Price Forecast"))
    
    # -------------------------------------------------------------
    # 1. Actionable Advice Card (Hold vs Sell)
    # -------------------------------------------------------------
    with st.container(border=True):
        st.markdown("### 💡 " + ("किसान के लिए सीधी सलाह (Actionable Advice):" if is_hi else "Actionable Market Advice:"))
        if pct_drift >= 2.0:
            st.success(
                f"### 🟢 **" + ("फसल अभी न बेचें! माल रोकें (HOLD PRODUCE)" if is_hi else "HOLD PRODUCE! Price Expected to Rise") + f"**\n\n" +
                (f"आने वाले **{h_label}** में भाव लगभग **₹{diff_rupees:+,} प्रति क्विंटल (+{pct_drift:.1f}%) बढ़ने का अनुमान है**।\n"
                 f"• **50 क्विंटल ट्रॉली पर अतिरिक्त मुनाफा:** लगभग **₹{trolley_rupees:,.0f}** अधिक मिलेंगे।\n"
                 f"• यदि आपके पास सुरक्षित गोदाम या भंडारण है, तो माल रोकना बहुत फायदेमंद रहेगा।"
                 if is_hi else
                 f"Prices projected to rise by **₹{diff_rupees:+,} / quintal (+{pct_drift:.1f}%)** over the next **{h_label}**.\n"
                 f"• **Net Gain per 50-qtl Trolley:** Approx **+₹{trolley_rupees:,.0f}** extra revenue.\n"
                 f"• If you have covered on-farm storage, holding stock will maximize your returns.")
            )
        elif pct_drift <= -2.0:
            st.error(
                f"### 🔴 **" + ("आज ही बेचें! भाव में गिरावट का खतरा (SELL TODAY)" if is_hi else "SELL TODAY! Price Expected to Drop") + f"**\n\n" +
                (f"आने वाले **{h_label}** में भाव लगभग **₹{abs(diff_rupees):,} प्रति क्विंटल (-{abs(pct_drift):.1f}%) गिरने का खतरा है**।\n"
                 f"• **50 क्विंटल ट्रॉली पर संभावित नुकसान:** लगभग **₹{trolley_rupees:,.0f}** का घाटा हो सकता है।\n"
                 f"• नुकसान से बचने के लिए आज या कल में नजदीकी मंडी में माल निकाल देना ही समझदारी है।"
                 if is_hi else
                 f"Prices projected to drop by **₹{abs(diff_rupees):,} / quintal (-{abs(pct_drift):.1f}%)** over the next **{h_label}**.\n"
                 f"• **Potential Loss per 50-qtl Trolley:** Approx **-₹{trolley_rupees:,.0f}** in lost value.\n"
                 f"• Selling immediately in your local mandi protects your operational margins.")
            )
        else:
            st.info(
                f"### 🔵 **" + ("भाव स्थिर रहेगा (PRICE STABLE)" if is_hi else "PRICE STABLE — Normal Market Equilibrium") + f"**\n\n" +
                (f"आने वाले **{h_label}** में भाव लगभग स्थिर (₹{pred_p:,.0f} / क्विंटल) रहने की संभावना है। बड़ा फेरबदल नहीं होगा।\n"
                 f"• आप अपनी नकदी जरूरत और सुविधा अनुसार कभी भी माल बेच सकते हैं।"
                 if is_hi else
                 f"Prices projected to remain steady around ₹{pred_p:,.0f} / quintal over the next {h_label}.\n"
                 f"• Transact comfortably according to your storage space and immediate cash requirements.")
            )

    # -------------------------------------------------------------
    # 2. Key Metrics
    # -------------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        ("आज का मंडी भाव" if is_hi else "Current Modal Price"),
        f"₹ {cur_p:,.2f}",
        ("प्रति क्विंटल" if is_hi else "Per Quintal")
    )
    k2.metric(
        (f"{h_label} बाद अनुमान" if is_hi else f"{h_label} Forecast"),
        f"₹ {pred_p:,.2f}",
        f"{diff_rupees:+,} ({pct_drift:+.1f}%)"
    )
    k3.metric(
        ("सुरक्षित भाव दायरा (95%)" if is_hi else "95% Safe Bounds"),
        f"₹ {max(100.0, pred_p - delta_val):,.0f} - ₹ {pred_p + delta_val:,.0f}",
        f"±₹{delta_val:.0f} " + ("दायरा" if is_hi else "Margin")
    )
    k4.metric(
        ("ट्रॉली का मुनाफा/नुकसान" if is_hi else "Trolley Gain (50 qtl)"),
        f"{'+' if diff_rupees >= 0 else ''}₹ {trolley_rupees:,.0f}",
        ("50 क्विंटल पर" if is_hi else "Per 50 Quintals")
    )

    st.markdown("---")
    
    # -------------------------------------------------------------
    # 3. Simple Visual Chart
    # -------------------------------------------------------------
    st.subheader("📊 " + ("भाव का रुझान एवं 95% सुरक्षित दायरा" if is_hi else "Price Trajectory & 95% Confidence Bounds"))
    
    recent_history = market_df.tail(30).copy()
    fig = go.Figure()
    
    # Historical line
    fig.add_trace(go.Scatter(
        x=recent_history['date'],
        y=recent_history['modal_price'],
        mode='lines+markers',
        name=("वास्तविक मंडी भाव" if is_hi else "Historical Price"),
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Forecast point
    last_date = pd.to_datetime(recent_history['date'].iloc[-1])
    days_fwd = 1 if "1" in h_suffix else (30 if "30" in h_suffix else 7)
    future_date = (last_date + pd.Timedelta(days=days_fwd)).strftime('%Y-%m-%d')
    
    fig.add_trace(go.Scatter(
        x=[recent_history['date'].iloc[-1], future_date],
        y=[cur_p, pred_p],
        mode='lines+markers',
        name=(f"{h_label} का अनुमानित भाव" if is_hi else f"{h_label} Predicted Price"),
        line=dict(color='#10b981' if diff_rupees >= 0 else '#ef4444', width=3, dash='dash'),
        marker=dict(size=10)
    ))
    
    # 95% Confidence interval
    fig.add_trace(go.Scatter(
        x=[future_date, future_date],
        y=[max(100.0, pred_p - delta_val), pred_p + delta_val],
        mode='lines+markers',
        name=("95% सुरक्षित दायरा" if is_hi else "95% Safe Bounds"),
        line=dict(color='#f59e0b', width=4),
        marker=dict(symbol='line-ew', size=14)
    ))
    
    fig.update_layout(
        title=f"{selected_crop} @ {market} — " + ("दैनिक भाव एवं आगामी अनुमान" if is_hi else "Daily Price & Forecast"),
        xaxis_title=("तारीख" if is_hi else "Date"),
        yaxis_title=("भाव (₹ प्रति क्विंटल)" if is_hi else "Price (₹/Quintal)"),
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------
    # 4. Interactive What-If Scenario Simulation
    # -------------------------------------------------------------
    st.markdown("---")
    st.subheader("🧪 " + ("'व्हाट-इफ' परिस्थिति अनुमान (What-If Market Simulator)" if is_hi else "Interactive What-If Scenario Simulator"))
    st.markdown(
        ("यदि मंडी में आवक अचानक बढ़ जाए या बारिश आ जाए, तो भाव पर क्या असर पड़ेगा? स्लाइडर खिसकाकर देखें:" if is_hi else
         "Simulate how sudden arrival surges or rainfall anomalies impact future prices in real-time:")
    )
    
    sim_c1, sim_c2 = st.columns(2)
    with sim_c1:
        arrival_change = st.slider(
            ("मंडी में आवक में बदलाव (%)" if is_hi else "Arrival Volume Shock (%)"),
            -50, 100, 0, step=10,
            help=("+50% का मतलब मंडी में अचानक बहुत ज्यादा माल आ गया।" if is_hi else "Simulate supply shocks.")
        )
    with sim_c2:
        rain_change = st.slider(
            ("अतिरिक्त वर्षा (मिमी)" if is_hi else "Rainfall Shock (mm)"),
            0, 100, 0, step=10,
            help=("अचानक बेमौसम बारिश।" if is_hi else "Simulate unseasonal rain.")
        )
        
    # Elasticity coefficients
    sim_elasticity = -0.12 * (arrival_change / 100.0) - 0.08 * (rain_change / 100.0)
    simulated_price = round(pred_p * (1.0 + sim_elasticity), 2)
    sim_diff = round(simulated_price - cur_p, 2)
    
    s1, s2 = st.columns(2)
    s1.metric(
        ("नई परिस्थिति में अनुमानित भाव" if is_hi else "Simulated Forecast Price"),
        f"₹ {simulated_price:,.2f} / qtl",
        f"{sim_diff:+,} " + ("आज से बदलाव" if is_hi else "vs Current")
    )
    s2.metric(
        ("परिस्थिति का शुद्ध प्रभाव" if is_hi else "Net Shock Impact"),
        f"{sim_elasticity * 100:+.2f}%",
        ("आवक व बारिश का असर" if is_hi else "Elasticity Effect")
    )

except Exception as e:
    st.error(f"Error loading forecasting module: {e}")