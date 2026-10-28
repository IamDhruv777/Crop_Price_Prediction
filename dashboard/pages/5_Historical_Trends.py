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
    index=0 if "Farmer" in st.session_state.get("display_mode", "Simple Farmer View") or "किसान" in st.session_state.get("display_mode", "") else 1,
    key="hist_display_mode"
)
st.session_state["display_mode"] = display_mode
is_farmer_view = ("Farmer" in display_mode or "किसान" in display_mode)

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
    if 'year' not in df.columns:
        df['year'] = df['date'].dt.year
    if 'month' not in df.columns:
        df['month'] = df['date'].dt.month
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
    month_names_hi = {1: "जनवरी", 2: "फरवरी", 3: "मार्च", 4: "अप्रैल", 5: "मई", 6: "जून", 7: "जुलाई", 8: "अगस्त", 9: "सितंबर", 10: "अक्टूबर", 11: "नवंबर", 12: "दिसंबर"}
    month_names_en = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}

    avg_price = mandi_df['modal_price'].mean()
    max_price = mandi_df['modal_price'].max()
    min_price = mandi_df['modal_price'].min()
    total_days = len(mandi_df)

    # Calculate seasonality stats
    if 'month' in mandi_df.columns:
        month_avg = mandi_df.groupby('month')['modal_price'].mean().reset_index()
        month_avg['Month_Name'] = month_avg['month'].map(month_names_hi if is_hi else month_names_en)
        best_month_row = month_avg.loc[month_avg['modal_price'].idxmax()]
        worst_month_row = month_avg.loc[month_avg['modal_price'].idxmin()]
        best_m_str = best_month_row['Month_Name']
        best_month_val = round(best_month_row['modal_price'], 2)
        worst_m_str = worst_month_row['Month_Name']
        worst_month_val = round(worst_month_row['modal_price'], 2)
        month_spread = round(best_month_val - worst_month_val, 2)
    else:
        best_m_str, worst_m_str = "N/A", "N/A"
        best_month_val, worst_month_val, month_spread = 0, 0, 0

    # ---------------------------------------------------------
    # MODE 1: SIMPLE FARMER VIEW (सरल किसान दृश्य)
    # ---------------------------------------------------------
    if is_farmer_view:
        st.subheader(f"🌾 {selected_crop} @ {selected_market} — " + ("किसान ऐतिहासिक सारांश" if is_hi else "Farmer Historical Overview"))

        # 4 High-level, crystal-clear metric cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("💰 " + ("औसत ऐतिहासिक भाव" if is_hi else "Average Historical Price"), f"₹ {avg_price:,.0f} / qtl", "10 वर्षों का औसत" if is_hi else "10-Year Average")
        k2.metric("🏆 " + ("रिकॉर्ड सबसे ऊंचा भाव" if is_hi else "Record Peak Price"), f"₹ {max_price:,.0f} / qtl", "सर्वोच्च दाम" if is_hi else "All-Time High")
        k3.metric("🔻 " + ("सबसे कम भाव" if is_hi else "Lowest Price"), f"₹ {min_price:,.0f} / qtl", "मंदी का स्तर" if is_hi else "All-Time Low")
        k4.metric("📅 " + ("कुल दर्ज कारोबारी दिन" if is_hi else "Trading History"), f"{total_days:,} दिन" if is_hi else f"{total_days:,} Days", "आंकड़े उपलब्ध" if is_hi else "Records Analyzed")

        st.markdown("---")

        # Farmer Actionable Card: Best Month to Sell
        with st.container(border=True):
            st.markdown("### 💡 " + ("फसल कब बेचें? — 10 वर्षों के आधार पर सबसे फायदेमंद महीना" if is_hi else "When to Sell? — 10-Year Optimal Selling Month"))
            st.success(
                f"🌟 **" + (f"फसल बेचने का सबसे फायदेमंद महीना: {best_m_str}" if is_hi else f"Best Selling Month: {best_m_str}") + f"**\n\n" +
                (f"• **अधिकतम भाव:** पिछले 10 वर्षों के आंकड़ों के अनुसार, इस मंडी में **{selected_crop}** का भाव **{best_m_str}** में औसतन **₹{best_month_val:,.0f} प्रति क्विंटल** रहता है।\n"
                 f"• **मंदी का महीना:** नई फसल की आवक के कारण **{worst_m_str}** में भाव घटकर **₹{worst_month_val:,.0f} प्रति क्विंटल** रह जाता है।\n"
                 f"• **सीधा मुनाफा:** यदि आप फसल को {worst_m_str} में बेचने के बजाय सुरक्षित रखकर **{best_m_str}** में बेचते हैं, तो आपको **प्रति क्विंटल लगभग ₹{month_spread:,.0f} (1 ट्रॉली पर ₹{month_spread * 50:,.0f}) का अतिरिक्त लाभ** मिल सकता है!"
                 if is_hi else
                 f"• **Peak Window:** Historically, **{selected_crop}** at this market reaches its highest average rate in **{best_m_str}** at **₹{best_month_val:,.0f} / quintal**.\n"
                 f"• **Harvest Glut:** In **{worst_m_str}**, prices dip to **₹{worst_month_val:,.0f} / quintal** due to heavy post-harvest arrivals.\n"
                 f"• **Net Gain:** Holding grain until **{best_m_str}** can yield an extra **₹{month_spread:,.0f} / qtl (₹{month_spread * 50:,.0f} per 50-qtl trolley)**!")
            )

        st.markdown("---")
        # Visual Intuitive Monthly Bar Chart (Green for peak, Red for lowest, Blue for others)
        st.subheader("📊 " + ("महीने अनुसार औसत भाव (बेचने का सबसे सही समय)" if is_hi else "Average Price by Month (Optimal Selling Calendar)"))
        
        if 'month' in mandi_df.columns:
            colors = []
            for m in month_avg['month']:
                if m == best_month_row['month']:
                    colors.append("#2ca02c")
                elif m == worst_month_row['month']:
                    colors.append("#d62728")
                else:
                    colors.append("#1f77b4")
            month_avg['Color'] = colors

            fig_farmer_month = px.bar(
                month_avg,
                x='Month_Name',
                y='modal_price',
                color='Month_Name',
                color_discrete_sequence=colors,
                text='modal_price',
                title=(f"{selected_crop} — 12 महीनों का औसत भाव (हरा = सबसे ज्यादा, लाल = सबसे कम)" if is_hi else f"{selected_crop} — 12-Month Historical Average Rate (Green = Peak, Red = Trough)"),
                labels={"Month_Name": "महीना" if is_hi else "Month", "modal_price": "औसत भाव (₹/क्विंटल)" if is_hi else "Average Price (₹/qtl)"}
            )
            fig_farmer_month.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside', showlegend=False)
            fig_farmer_month.update_layout(xaxis_tickangle=-30, height=450)
            st.plotly_chart(fig_farmer_month, use_container_width=True)

        # Smooth Multi-Year Annual Growth Trend for farmers
        st.markdown("---")
        st.subheader("📈 " + ("10 वर्षों में भाव की सालाना बढ़त" if is_hi else "Yearly Average Price Trend (10-Year Historical Growth)"))
        yearly_avg = mandi_df.groupby('year')['modal_price'].mean().reset_index()
        fig_yearly = px.line(
            yearly_avg,
            x='year',
            y='modal_price',
            markers=True,
            text='modal_price',
            title=(f"{selected_crop} @ {selected_market} — साल-दर-साल भाव में वृद्धि" if is_hi else f"{selected_crop} @ {selected_market} — Year-on-Year Growth"),
            labels={"year": "वर्ष" if is_hi else "Year", "modal_price": "सालाना औसत भाव (₹/क्विंटल)" if is_hi else "Annual Avg Price (₹/qtl)"}
        )
        fig_yearly.update_traces(texttemplate='₹%{text:,.0f}', textposition='top center', line=dict(color="#2ca02c", width=3))
        fig_yearly.update_layout(height=420)
        st.plotly_chart(fig_yearly, use_container_width=True)

    # ---------------------------------------------------------
    # MODE 2: ADVANCED ANALYTICS / RESEARCHER VIEW (उन्नत तकनीकी दृश्य)
    # ---------------------------------------------------------
    else:
        st.subheader(f"🔬 Advanced Econometric Time-Series Analysis — {selected_crop} @ {selected_market}")

        # Quantitative Summary Metrics
        std_price = mandi_df['modal_price'].std()
        cv_price = (std_price / avg_price) * 100 if avg_price > 0 else 0
        q25 = mandi_df['modal_price'].quantile(0.25)
        median_p = mandi_df['modal_price'].median()
        q75 = mandi_df['modal_price'].quantile(0.75)
        iqr = q75 - q25

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mean Price (μ)", f"₹ {avg_price:,.2f}", f"Std (σ): ±₹{std_price:,.1f}")
        c2.metric("Coefficient of Variation", f"{cv_price:.1f}%", "Annual Volatility")
        c3.metric("Median (50th %ile)", f"₹ {median_p:,.2f}", f"IQR: ₹{iqr:,.0f}")
        c4.metric("25th – 75th Spread", f"₹ {q25:,.0f} – ₹ {q75:,.0f}", "Interquartile Band")
        c5.metric("Data Sample Points", f"{total_days:,}", "Agmarknet Days")

        st.markdown("---")
        st.subheader("📈 Multi-Horizon Rolling Baselines & Volatility Envelopes")

        # Compute rolling stats if missing
        m_df_sorted = mandi_df.copy().sort_values('date')
        if 'price_rolling_mean_7' not in m_df_sorted.columns:
            m_df_sorted['price_rolling_mean_7'] = m_df_sorted['modal_price'].rolling(7, min_periods=1).mean()
        if 'price_rolling_mean_14' not in m_df_sorted.columns:
            m_df_sorted['price_rolling_mean_14'] = m_df_sorted['modal_price'].rolling(14, min_periods=1).mean()
        m_df_sorted['price_rolling_mean_30'] = m_df_sorted['modal_price'].rolling(30, min_periods=1).mean()
        m_df_sorted['rolling_std_30'] = m_df_sorted['modal_price'].rolling(30, min_periods=1).std().fillna(0)
        m_df_sorted['upper_band'] = m_df_sorted['price_rolling_mean_30'] + (1.96 * m_df_sorted['rolling_std_30'])
        m_df_sorted['lower_band'] = (m_df_sorted['price_rolling_mean_30'] - (1.96 * m_df_sorted['rolling_std_30'])).clip(lower=0)

        fig_adv = go.Figure()
        fig_adv.add_trace(go.Scatter(
            x=m_df_sorted['date'], y=m_df_sorted['modal_price'],
            mode='lines', name='Daily Modal Price',
            line=dict(color='rgba(31, 119, 180, 0.45)', width=1)
        ))
        fig_adv.add_trace(go.Scatter(
            x=m_df_sorted['date'], y=m_df_sorted['price_rolling_mean_7'],
            mode='lines', name='7-Day Rolling Mean',
            line=dict(color='#ff7f0e', width=1.5)
        ))
        fig_adv.add_trace(go.Scatter(
            x=m_df_sorted['date'], y=m_df_sorted['price_rolling_mean_14'],
            mode='lines', name='14-Day Rolling Mean',
            line=dict(color='#2ca02c', width=1.5, dash='dash')
        ))
        fig_adv.add_trace(go.Scatter(
            x=m_df_sorted['date'], y=m_df_sorted['price_rolling_mean_30'],
            mode='lines', name='30-Day Trend Baseline',
            line=dict(color='#9467bd', width=2)
        ))
        fig_adv.add_trace(go.Scatter(
            x=m_df_sorted['date'], y=m_df_sorted['upper_band'],
            mode='lines', name='Upper 95% Envelope (+1.96σ)',
            line=dict(color='rgba(148, 103, 189, 0.2)', width=0),
            showlegend=False
        ))
        fig_adv.add_trace(go.Scatter(
            x=m_df_sorted['date'], y=m_df_sorted['lower_band'],
            mode='lines', name='95% Volatility Envelope',
            line=dict(color='rgba(148, 103, 189, 0.2)', width=0),
            fill='tonexty', fillcolor='rgba(148, 103, 189, 0.12)'
        ))
        fig_adv.update_layout(
            title=f"{selected_crop} @ {selected_market} — Econometric Moving Averages & Volatility Envelope",
            xaxis_title="Transaction Date",
            yaxis_title="Wholesale Price (₹/Quintal)",
            height=520,
            hovermode="x unified"
        )
        st.plotly_chart(fig_adv, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Empirical Seasonality Distribution (Monthly Boxplot & Interquartile Spread)")
        m_df_sorted['Month_Name'] = m_df_sorted['month'].map(month_names_en)
        fig_box = px.box(
            m_df_sorted,
            x='Month_Name',
            y='modal_price',
            color='Month_Name',
            title=f"{selected_crop} — Monthly Price Distribution, Dispersion & Outliers",
            labels={"Month_Name": "Month", "modal_price": "Modal Price (₹/Quintal)"}
        )
        fig_box.update_layout(xaxis_tickangle=-30, height=480, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

        with st.expander("🔍 View Monthly Statistical Breakdown Table", expanded=False):
            month_stats = m_df_sorted.groupby(['month', 'Month_Name'])['modal_price'].agg(
                Count='count',
                Mean='mean',
                Std='std',
                Min='min',
                Median='median',
                Max='max'
            ).reset_index().drop(columns=['month'])
            for num_c in ['Mean', 'Std', 'Min', 'Median', 'Max']:
                month_stats[num_c] = month_stats[num_c].round(2)
            st.dataframe(month_stats, use_container_width=True)

        st.markdown("---")
        st.subheader("📅 Longitudinal Annual Trend & Summary Table")
        year_stats = m_df_sorted.groupby('year').agg(
            Trading_Days=('modal_price', 'count'),
            Annual_Mean_Price=('modal_price', 'mean'),
            Annual_Min_Price=('modal_price', 'min'),
            Annual_Max_Price=('modal_price', 'max'),
            Annual_Std_Dev=('modal_price', 'std')
        ).reset_index()
        for num_c in ['Annual_Mean_Price', 'Annual_Min_Price', 'Annual_Max_Price', 'Annual_Std_Dev']:
            year_stats[num_c] = year_stats[num_c].round(2)

        st.dataframe(year_stats, use_container_width=True)