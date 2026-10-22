import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import sys

# Ensure dashboard folder is in sys.path for i18n import
dashboard_dir = Path(__file__).parents[1]
if str(dashboard_dir) not in sys.path:
    sys.path.append(str(dashboard_dir))

from i18n import t, get_current_lang, set_lang

st.set_page_config(page_title="Mandi Comparison | मंडी रेट मुकाबला", page_icon="⚖️", layout="wide")

# Authentication & Role Guard: Only Researchers can access advanced analytics pages
if not st.session_state.get("logged_in", False) or st.session_state.get("user_role") != "Researcher":
    st.switch_page("app.py")


# Authentication Guard: Redirect to Landing Screen if not logged in


PREDS_PATH = Path("data/test_predictions.csv")

if not PREDS_PATH.exists():
    st.error("Predictions file not found at `data/test_predictions.csv`.")
    st.stop()

preds_df = pd.read_csv(PREDS_PATH)

# Language Toggle in Sidebar
st.sidebar.header(t("sidebar_lang_title"))
lang_choice = st.sidebar.radio(
    "Language",
    ["English", "हिन्दी (Hindi)"],
    index=0 if get_current_lang() == "en" else 1,
    horizontal=True,
    key="mandi_comp_lang"
)
new_lang = "hi" if "हिन्दी" in lang_choice else "en"
if new_lang != get_current_lang():
    set_lang(new_lang)
    st.rerun()

is_hi = (get_current_lang() == "hi")

# Display Mode: Simple Farmer View vs Advanced Analytics
st.sidebar.markdown(f"### ⚙️ {t('view_mode')}")
display_mode = st.sidebar.radio(
    t("view_mode"),
    [t("mode_farmer"), t("mode_advanced")],
    index=0 if "Farmer" in st.session_state.get("display_mode", "Simple Farmer View") or "किसान" in st.session_state.get("display_mode", "") else 1,
    key="mandi_display_mode"
)
st.session_state["display_mode"] = display_mode
is_farmer_view = ("Farmer" in display_mode or "किसान" in display_mode)

# Title
if is_hi:
    st.title("⚖️ आस-पास की सभी मंडियों का लाइव रेट मुकाबला")
    st.markdown("अपनी फसल बेचने से पहले जानें कि किस मंडी में सबसे ज्यादा भाव मिल रहा है, ताकि आपको अपनी मेहनत का पूरा दाम मिले।")
else:
    st.title("⚖️ Cross-Mandi Price & Arbitrage Comparison")
    st.markdown("Compare price disparities, spatial spreads, and model forecasts across real nationwide mandis.")

st.sidebar.header("⚙️ " + ("फसल व मंडी चुनाव" if is_hi else "Filters"))
commodities = sorted(preds_df['commodity'].dropna().unique().tolist())
user_def_crop = st.session_state.get("user_crop", "Wheat")
default_crop_idx = commodities.index(user_def_crop) if user_def_crop in commodities else (commodities.index("Wheat") if "Wheat" in commodities else 0)
selected_crop = st.sidebar.selectbox(
    "1. " + ("फसल चुनें" if is_hi else "Select Crop"),
    commodities,
    index=default_crop_idx
)

horizon_options = ["7 दिन बाद (Next 7 Days)", "1 दिन बाद (Next 1 Day)", "30 दिन बाद (Next 30 Days)"] if is_hi else ["Next 7 Days (Default)", "Next 1 Day", "Next 30 Days"]
comparison_horizon = st.sidebar.selectbox(
    "2. " + ("समय सीमा" if is_hi else "Forecast Horizon"),
    horizon_options
)
h_suffix = "1d" if "1 Day" in comparison_horizon or "1 दिन" in comparison_horizon else ("30d" if "30 Days" in comparison_horizon or "30 दिन" in comparison_horizon else "7d")
h_label = ("1 दिन" if "1" in h_suffix else ("30 दिन" if "30" in h_suffix else "7 दिन")) if is_hi else ("1-Day" if "1" in h_suffix else ("30-Day" if "30" in h_suffix else "7-Day"))

pred_col = f'predicted_price_{h_suffix}' if f'predicted_price_{h_suffix}' in preds_df.columns else 'predicted_price_7d'
act_col = f'actual_price_{h_suffix}' if f'actual_price_{h_suffix}' in preds_df.columns else 'actual_price_7d'

crop_df = preds_df[preds_df['commodity'] == selected_crop]
available_markets = sorted(crop_df['market'].dropna().unique().tolist())

default_selection = available_markets[:min(5, len(available_markets))]
selected_markets = st.sidebar.multiselect(
    "3. " + ("मंडियां चुनें" if is_hi else "Select Mandis to Compare"),
    available_markets,
    default=default_selection
)

if not selected_markets:
    st.warning("कृपया कम से कम एक मंडी चुनें।" if is_hi else "Please select at least one mandi from the sidebar.")
else:
    st.subheader(f"🌾 **{selected_crop}** — " + (f"{h_label} का भाव मुकाबला" if is_hi else f"{h_label} Price Comparison"))
    
    # Aggregate real metrics for selected markets dynamically
    comp_list = []
    for m in selected_markets:
        sub = crop_df[crop_df['market'] == m]
        if not sub.empty:
            cur_price = sub['modal_price'].iloc[-1]
            pred_h = sub[pred_col].iloc[-1] if pred_col in sub.columns else cur_price
            spread = pred_h - cur_price
            drift_pct = (spread / cur_price) * 100 if cur_price > 0 else 0.0
            
            if is_hi:
                comp_list.append({
                    "मंडी का नाम": m,
                    "आज का भाव (₹/क्विंटल)": int(round(cur_price)),
                    f"{h_label} बाद अनुमानित भाव": int(round(pred_h)),
                    "भाव में बदलाव (₹)": int(round(spread)),
                    "अनुमानित बदलाव (%)": f"{drift_pct:+.1f}%",
                    "सलाह": "🟢 माल रोकें" if drift_pct >= 2 else ("🔴 आज ही बेचें" if drift_pct <= -2 else "🔵 स्थिर")
                })
            else:
                comp_list.append({
                    "Mandi / Market": m,
                    "Current Price (₹/qtl)": int(round(cur_price)),
                    f"{h_label} Forecast (₹/qtl)": int(round(pred_h)),
                    "Price Drift (₹)": int(round(spread)),
                    "Drift (%)": f"{drift_pct:+.1f}%",
                    "Recommendation": "🟢 HOLD" if drift_pct >= 2 else ("🔴 SELL" if drift_pct <= -2 else "🔵 STABLE")
                })
                
    comp_df = pd.DataFrame(comp_list)
    
    # Identify highest paying market for actionable farmer advice
    price_key = "आज का भाव (₹/क्विंटल)" if is_hi else "Current Price (₹/qtl)"
    mandi_key = "मंडी का नाम" if is_hi else "Mandi / Market"
    
    best_mandi = comp_df.loc[comp_df[price_key].idxmax()]
    worst_mandi = comp_df.loc[comp_df[price_key].idxmin()]
    price_diff_per_qtl = best_mandi[price_key] - worst_mandi[price_key]
    extra_profit_per_trolley = price_diff_per_qtl * 50  # 50 quintal tractor trolley
    
    # Actionable Trolley Profit Card
    with st.container(border=True):
        st.markdown("### 🚜 " + ("ट्रॉली मुनाफा गाइड (Trolley Profit Opportunity):" if is_hi else "Trolley Profit Arbitrage Guide:"))
        if price_diff_per_qtl > 0:
            st.success(
                f"💰 **" + (f"सबसे ज्यादा भाव देने वाली मंडी: {best_mandi[mandi_key]} (₹{best_mandi[price_key]:,}/क्विंटल)" if is_hi else f"Highest Paying Mandi: {best_mandi[mandi_key]} (₹{best_mandi[price_key]:,}/qtl)") + f"**\n\n" +
                (f"• **भाव का सीधा अंतर:** **{worst_mandi[mandi_key]}** के मुकाबले **₹{price_diff_per_qtl:,} प्रति क्विंटल** ज्यादा भाव मिल रहा है।\n"
                 f"• **1 ट्रैक्टर ट्रॉली (50 क्विंटल) पर अतिरिक्त बचत:** लगभग **+₹{extra_profit_per_trolley:,} का शुद्ध मुनाफा**!\n"
                 f"• यदि दोनों मंडियों के बीच डीजल का खर्च ₹2,000–₹3,000 भी हो, तो भी आपको **₹{max(0, extra_profit_per_trolley - 2500):,} की सीधी बचत** होगी।"
                 if is_hi else
                 f"• **Direct Price Spread:** Mandi **{best_mandi[mandi_key]}** offers **₹{price_diff_per_qtl:,} / quintal** more than **{worst_mandi[mandi_key]}**.\n"
                 f"• **Extra Net Margin per 50-qtl Trolley:** Approx **+₹{extra_profit_per_trolley:,}** additional revenue!\n"
                 f"• Even after deducting ₹2,500 for diesel transportation, you still pocket **₹{max(0, extra_profit_per_trolley - 2500):,} in net savings**.")
            )
        else:
            st.info("सभी चुनी हुई मंडियों में भाव लगभग समान है।" if is_hi else "All selected mandis currently share equal prices.")

    st.markdown("---")
    
    # Comparison Table
    st.subheader("📋 " + ("चुनी हुई मंडियों की तुलना तालिका" if is_hi else "Mandi Comparison Table"))
    st.dataframe(comp_df, use_container_width=True)
    
    # Bar Chart: Today vs Forecast Price
    st.markdown("---")
    st.subheader("📊 " + ("मंडी अनुसार भाव का तुलनात्मक चार्ट" if is_hi else "Cross-Mandi Comparative Bar Chart"))
    
    chart_df = []
    for _, row in comp_df.iterrows():
        chart_df.append({
            "Mandi": row[mandi_key],
            "Type": ("आज का भाव" if is_hi else "Current Price"),
            "Price": row[price_key]
        })
        forecast_key = f"{h_label} बाद अनुमानित भाव" if is_hi else f"{h_label} Forecast (₹/qtl)"
        chart_df.append({
            "Mandi": row[mandi_key],
            "Type": (f"{h_label} का अनुमान" if is_hi else f"{h_label} Forecast"),
            "Price": row[forecast_key]
        })
    plot_df = pd.DataFrame(chart_df)
    
    fig = px.bar(
        plot_df,
        x="Mandi",
        y="Price",
        color="Type",
        barmode="group",
        text="Price",
        title=(f"{selected_crop} — विभिन्न मंडियों में आज का भाव बनाम आगामी अनुमान" if is_hi else f"{selected_crop} — Current vs Forecast Price Across Selected Mandis"),
        labels={"Price": "₹ / Quintal", "Mandi": ("मंडी" if is_hi else "Mandi / APMC"), "Type": ("भाव का प्रकार" if is_hi else "Price Type")}
    )
    fig.update_traces(texttemplate='₹%{text:,}', textposition='outside')
    fig.update_layout(xaxis_tickangle=-25, height=450)
    st.plotly_chart(fig, use_container_width=True)