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

st.set_page_config(page_title="Weather Analysis | मौसम विश्लेषण", page_icon="🌦️", layout="wide")

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
    key="weather_lang"
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
    key="weather_display_mode"
)
st.session_state["display_mode"] = display_mode

WEATHER_PATH = Path("data/raw/weather/mandi_daily_weather_aligned.csv")

if not WEATHER_PATH.exists():
    st.error("Weather dataset not found at `data/raw/weather/mandi_daily_weather_aligned.csv`.")
    st.stop()

if is_hi:
    st.title("🌦️ मौसम विश्लेषण एवं कृषि-जलवायु प्रभाव")
    st.markdown("नासा उपग्रह (NASA POWER) के 10 वर्षों के मौसम आंकड़े: तापमान, बारिश, नमी और हवा की गति का आपकी मंडी पर असर।")
else:
    st.title("🌦️ Meteorological Impact & Climate Dynamics")
    st.markdown("Analyzing **10-Year Daily Meteorological Data (NASA POWER)**: Temperature Range, Cumulative Rainfall, Relative Humidity, and Wind Velocity.")

st.sidebar.header("🔍 " + ("जिला व राज्य चुनें" if is_hi else "Select Location"))

@st.cache_data
def load_weather_districts():
    df_sample = pd.read_csv(WEATHER_PATH, nrows=50000)
    return df_sample

weather_df = load_weather_districts()

states = sorted(weather_df['state'].dropna().unique().tolist())
selected_state = st.sidebar.selectbox("राज्य चुनें (State)" if is_hi else "Select State", states)

state_districts = sorted(weather_df[weather_df['state'] == selected_state]['district'].dropna().unique().tolist())
selected_district = st.sidebar.selectbox("जिला चुनें (District)" if is_hi else "Select District", state_districts)

district_df = weather_df[(weather_df['state'] == selected_state) & (weather_df['district'] == selected_district)].copy()
district_df['date'] = pd.to_datetime(district_df['date'])
district_df.sort_values(by='date', inplace=True)

st.subheader(f"📍 {selected_district}, {selected_state} — " + ("मौसम का हाल व रिकॉर्ड" if is_hi else "10-Year Meteorological Profile"))

# 4 Intuitive Metric Cards
m1, m2, m3, m4 = st.columns(4)
avg_temp = district_df['temperature_mean'].mean()
max_temp = district_df['temperature_max'].max()
annual_rain = district_df['rainfall_mm'].mean() * 365
avg_humidity = district_df['relative_humidity'].mean()

m1.metric("🌡️ " + ("औसत तापमान" if is_hi else "Mean Temp"), f"{avg_temp:.1f} °C", "सामान्य" if is_hi else "Historical Norm")
m2.metric("☀️ " + ("अधिकतम तापमान" if is_hi else "Peak Temp"), f"{max_temp:.1f} °C", "गर्मियों में" if is_hi else "Summer Peak")
m3.metric("🌧️ " + ("वार्षिक औसत बारिश" if is_hi else "Annual Rainfall"), f"{annual_rain:.0f} mm", "प्रति वर्ष" if is_hi else "Per Year")
m4.metric("💧 " + ("हवा में नमी (आर्द्रता)" if is_hi else "Avg Humidity"), f"{avg_humidity:.1f}%", "वातावरण में" if is_hi else "Moisture")

# Farmer Agro-Weather Advisory Card
st.markdown("---")
with st.container(border=True):
    st.markdown("#### 🌾 " + ("किसान मौसम सलाह (Weather Impact on Crop Price)" if is_hi else "Agro-Weather Market Advisory"))
    recent_rain = district_df['rainfall_mm'].tail(30).sum()
    if recent_rain > 100:
        st.warning(
            "🌧️ **" + ("भारी बारिश की चेतावनी — फसल सुरक्षा सलाह:" if is_hi else "High Rainfall Warning — Market Advisory:") + "**\n\n" +
            ("पिछले 30 दिनों में इस क्षेत्र में पर्याप्त वर्षा दर्ज हुई है। मंडी में माल लाते समय तिरपाल से ढककर लाएं ताकि गीला होने पर आढ़ती भाव कम न लगाएं।"
             if is_hi else
             "Significant precipitation observed recently. Ensure produce is well-covered during mandi transport to prevent moisture deductions.")
        )
    else:
        st.success(
            "☀️ **" + ("मौसम अनुकूल — सामान्य व्यापार की स्थिति:" if is_hi else "Favorable Weather Conditions:") + "**\n\n" +
            ("तापमान और मौसम सामान्य दायरे में है। आवक सामान्य रूप से चलने की संभावना है।"
             if is_hi else
             "Atmospheric temperatures and moisture are within typical historical bounds. Supply chain arrivals are proceeding smoothly.")
        )

# Visualizations
st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    st.markdown("### 🌡️ " + ("दैनिक तापमान उतार-चढ़ाव (पिछले 365 दिन)" if is_hi else "Daily Temperature Dynamics"))
    fig_temp = px.line(
        district_df.tail(365),
        x="date",
        y=["temperature_mean", "temperature_max", "temperature_min"],
        labels={"value": "तापमान (°C)" if is_hi else "Temperature (°C)", "date": "तारीख" if is_hi else "Date", "variable": "पैमाना" if is_hi else "Metric"},
        title="तापमान का वार्षिक बदलाव" if is_hi else "Annual Temperature Fluctuation"
    )
    st.plotly_chart(fig_temp, use_container_width=True)

with c2:
    st.markdown("### 🌧️ " + ("दैनिक वर्षा (पिछले 365 दिन)" if is_hi else "Daily Precipitation Events"))
    fig_rain = px.bar(
        district_df.tail(365),
        x="date",
        y="rainfall_mm",
        labels={"rainfall_mm": "बारिश (mm)" if is_hi else "Rainfall (mm)", "date": "तारीख" if is_hi else "Date"},
        title="बारिश के दिन" if is_hi else "Daily Precipitation Events",
        color="rainfall_mm",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_rain, use_container_width=True)

# Correlation Matrix (Visible only in Advanced View)
if "Advanced" in display_mode or "उन्नत" in display_mode:
    st.markdown("---")
    st.subheader("📊 Meteorological Correlation Matrix (उन्नत मौसम संबंध)")
    weather_num = district_df[['temperature_mean', 'temperature_max', 'temperature_min', 'rainfall_mm', 'relative_humidity', 'wind_speed_ms']].corr().round(2)
    fig_corr = px.imshow(
        weather_num,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=f"Climate Parameter Correlations ({selected_district})"
    )
    st.plotly_chart(fig_corr, use_container_width=True)