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

st.set_page_config(page_title="Historical Trends | ऐतिहासिक भाव", page_icon="📅", layout="wide")

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
    key="hist_lang"
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
    key="hist_display_mode"
)
st.session_state["display_mode"] = display_mode

HIST_SAMPLE_PATH = Path("data/processed/historical_multi_crop_sample.csv")
FALLBACK_PATH = Path("data/processed/unified_splits/train_unified.csv")

if is_hi:
    st.title("📅 पिछले 10 वर्षों के मंडी भाव का इतिहास एवं मौसमी रुझान")
    st.markdown("देखें कि पिछले 10 सालों में आपकी फसल का भाव किस महीने में सबसे ज्यादा रहा और साल-दर-साल भाव कैसे बढ़े हैं।")
else:
    st.title("📅 Multi-Year Historical Wholesale Trends & Seasonality")
    st.markdown("Exploring real historical time-series from the **10-Year AGMARKNET Master Database** with rolling baselines, volatility bands, and seasonality.")

@st.cache_data
def load_historical_data():
    if HIST_SAMPLE_PATH.exists():
        df = pd.read_csv(HIST_SAMPLE_PATH)
    elif FALLBACK_PATH.exists():
        df = pd.read_csv(FALLBACK_PATH, nrows=50000)
    else:
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'])
    return df

df_hist = load_historical_data()

if df_hist.empty:
    st.error("Historical dataset not found. Please run the data preparation pipeline.")
    st.stop()

# 1. Commodity Selection (First!)
st.sidebar.header("🔍 " + ("फसल और मंडी चुनें" if is_hi else "Filters"))
commodities = sorted(df_hist['commodity'].dropna().unique().tolist())
user_def_crop = st.session_state.get("user_crop", "Wheat")
def_idx = commodities.index(user_def_crop) if user_def_crop in commodities else 0

selected_crop = st.sidebar.selectbox(
    "1. फसल चुनें (Select Crop):" if is_hi else "1. Select Commodity:",
    commodities,
    index=def_idx
)

crop_df = df_hist[df_hist['commodity'] == selected_crop]
markets = sorted(crop_df['market'].dropna().unique().tolist())
selected_market = st.sidebar.selectbox(
    "2. मंडी चुनें (Select Mandi):" if is_hi else "2. Select Mandi:",
    markets
)

mandi_df = crop_df[crop_df['market'] == selected_market].sort_values(by='date')

if mandi_df.empty:
    st.warning(f"No records found for {selected_crop} at {selected_market}.")
else:
    st.subheader(f"🌾 {selected_crop} @ {selected_market} — " + ("भाव का ऐतिहासिक रिकॉर्ड" if is_hi else "Historical Price Track Record"))
    
    # 4 Simple Key Metric Cards
    k1, k2, k3, k4 = st.columns(4)
    avg_price = mandi_df['modal_price'].mean()
    max_price = mandi_df['modal_price'].max()
    min_price = mandi_df['modal_price'].min()
    total_days = len(mandi_df)
    
    k1.metric("💰 " + ("औसत ऐतिहासिक भाव" if is_hi else "Historical Avg Price"), f"₹ {avg_price:,.2f} / qtl")
    k2.metric("📈 " + ("सर्वोच्च भाव (All-Time High)" if is_hi else "All-Time High"), f"₹ {max_price:,.2f} / qtl", "रिकॉर्ड दाम" if is_hi else "Peak")
    k3.metric("📉 " + ("न्यूनतम भाव (All-Time Low)" if is_hi else "All-Time Low"), f"₹ {min_price:,.2f} / qtl", "न्यूनतम" if is_hi else "Trough")
    k4.metric("📅 " + ("कुल दर्ज कारोबारी दिन" if is_hi else "Total Trading Days"), f"{total_days:,} दिन" if is_hi else f"{total_days:,} Days")
    
    st.markdown("---")
    
    # Farmer Insight Card: Which month pays best?
    if 'month' in mandi_df.columns:
        month_avg = mandi_df.groupby('month')['modal_price'].mean()
        best_month_num = int(month_avg.idxmax())
        best_month_val = round(month_avg.max(), 2)
        worst_month_num = int(month_avg.idxmin())
        worst_month_val = round(month_avg.min(), 2)
        
        month_names_hi = {1: "जनवरी", 2: "फरवरी", 3: "मार्च", 4: "अप्रैल", 5: "मई", 6: "जून", 7: "जुलाई", 8: "अगस्त", 9: "सितंबर", 10: "अक्टूबर", 11: "नवंबर", 12: "दिसंबर"}
        month_names_en = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
        
        best_m_str = month_names_hi[best_month_num] if is_hi else month_names_en[best_month_num]
        worst_m_str = month_names_hi[worst_month_num] if is_hi else month_names_en[worst_month_num]
        
        with st.container(border=True):
            st.markdown(f"#### 💡 " + ("किसान के लिए सबसे महत्वपूर्ण मौसमी सलाह (Seasonality Insight)" if is_hi else "Key Seasonal Selling Advice"))
            st.success(
                f"🌟 **" + (f"सबसे ज्यादा भाव का महीना: {best_m_str}" if is_hi else f"Best Selling Month: {best_m_str}") + f"**\n\n" +
                (f"पिछले 10 वर्षों के आंकड़ों के अनुसार, इस मंडी में **{selected_crop}** का सबसे ऊंचा औसत भाव **{best_m_str}** में (लगभग **₹{best_month_val:,.2f}/क्विंटल**) मिलता है।\n\n"
                 f"जबकि **{worst_m_str}** में नई फसल की भारी आवक से भाव घटकर **₹{worst_month_val:,.2f}/क्विंटल** पर आ जाते हैं।"
                 if is_hi else
                 f"Historical data reveals that **{selected_crop}** at this market historically peaks in **{best_m_str}** at **₹{best_month_val:,.2f}/qtl**, while dropping to **₹{worst_month_val:,.2f}/qtl** in **{worst_m_str}** due to harvest glut.")
            )

    # Price Trajectory Plot
    st.markdown("---")
    st.subheader("📈 " + ("दैनिक भाव एवं मूविंग एवरेज चार्ट" if is_hi else "Daily Price Trajectory & Moving Averages"))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mandi_df['date'],
        y=mandi_df['modal_price'],
        mode='lines',
        name='दैनिक भाव (Daily Price)' if is_hi else 'Daily Modal Price',
        line=dict(color='#1f77b4', width=1.5)
    ))
    
    if 'price_rolling_mean_7' in mandi_df.columns:
        fig.add_trace(go.Scatter(
            x=mandi_df['date'],
            y=mandi_df['price_rolling_mean_7'],
            mode='lines',
            name='7 दिन का औसत (7-Day MA)' if is_hi else '7-Day Moving Avg',
            line=dict(color='#ff7f0e', width=2)
        ))
        
    if 'price_rolling_mean_14' in mandi_df.columns:
        fig.add_trace(go.Scatter(
            x=mandi_df['date'],
            y=mandi_df['price_rolling_mean_14'],
            mode='lines',
            name='14 दिन का औसत (14-Day MA)' if is_hi else '14-Day Moving Avg',
            line=dict(color='#2ca02c', width=2, dash='dot')
        ))
        
    fig.update_layout(
        title=f"{selected_crop} @ {selected_market} — " + ("भाव में बदलाव" if is_hi else "Price Trajectory"),
        xaxis_title="तारीख (Date)" if is_hi else "Date",
        yaxis_title="मंडी भाव (₹/क्विंटल)" if is_hi else "Wholesale Price (₹/Quintal)",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonality Boxplot (Available in both, but simplified in farmer view)
    if 'month' in mandi_df.columns:
        st.markdown("---")
        st.subheader("📊 " + ("महीने अनुसार भाव वितरण (किस महीने क्या भाव मिलता है?)" if is_hi else "Monthly Seasonality Distribution"))
        
        mandi_df['Month_Name'] = mandi_df['month'].map(month_names_hi if is_hi else month_names_en)
        fig_month = px.box(
            mandi_df,
            x='Month_Name',
            y='modal_price',
            title=f"{selected_crop} — " + ("माहवार भाव दायरा" if is_hi else "Monthly Price Distribution"),
            labels={"Month_Name": "महीना" if is_hi else "Month", "modal_price": "भाव (₹/qtl)" if is_hi else "Modal Price (₹/qtl)"},
            color="Month_Name"
        )
        fig_month.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_month, use_container_width=True)