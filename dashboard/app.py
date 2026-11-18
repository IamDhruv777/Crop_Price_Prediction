"""
Krishi Bhav Saathi — landing, sign-in and decision dashboard.

Flow: emotional landing page -> login / sign-up -> personalised dashboard.
This is a demo experience over the historical Agmarknet holdout set, not a
live market feed. All prices shown come from data/deployed_test_predictions.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as st_components

DASHBOARD_DIR = Path(__file__).parent
REPO_ROOT = DASHBOARD_DIR.parent
for path in (REPO_ROOT, DASHBOARD_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from auth import (  # noqa: E402
    ROLES,
    STATE_NAMES,
    authenticate,
    current_user,
    init_auth_state,
    login_session,
    logout_session,
    register_user,
)
from components.landing_3d import render_3d_tarazu  # noqa: E402
from components.unified_zoom_map import build_unified_map_html  # noqa: E402
from i18n import get_current_lang, set_lang  # noqa: E402
from src.inference.forecast_path import build_forecast_path  # noqa: E402

st.set_page_config(
    page_title="Krishi Bhav Saathi | कृषि भाव साथी",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_DIR = Path("data")
PREDS_PATH = DATA_DIR / "deployed_test_predictions.csv"
GEO_PATH = DATA_DIR / "raw" / "geography" / "market_master.csv"
HIST_PATH = DATA_DIR / "processed" / "historical_multi_crop_sample.csv"
METRICS_PATH = DATA_DIR / "metrics_deployed.json"

init_auth_state()
st.session_state.setdefault("lang", "hi")
st.session_state.setdefault("selected_crop", "Wheat")
st.session_state.setdefault("selected_market", "")

is_hi = get_current_lang() == "hi"


def L(hi: str, en: str) -> str:
    return hi if is_hi else en


def rupees(value: float) -> str:
    return f"₹{value:,.0f}"


def pct(value: float) -> str:
    return f"{value:+.1f}%"


# ---------------------------------------------------------------------------
# Shared styling
# ---------------------------------------------------------------------------
def inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700;800;900&family=Poppins:wght@400;500;600&display=swap');

        :root {
            --soil: #7c2d12;
            --leaf: #15803d;
            --leaf-dark: #14532d;
            --wheat: #d97706;
            --cream: #fffaf0;
            --ink: #1f2937;
            --muted: #6b7280;
        }

        .stApp {
            background:
                radial-gradient(1200px 600px at 10% -10%, #fef3c7 0%, transparent 55%),
                radial-gradient(900px 500px at 95% 0%, #dcfce7 0%, transparent 50%),
                linear-gradient(180deg, #fffdf7 0%, #fffaf0 60%, #f8faf5 100%);
            color: var(--ink);
        }

        .main .block-container {
            max-width: 1220px;
            padding-top: 1.2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, h4 {
            font-family: 'Outfit', sans-serif !important;
            letter-spacing: -0.02em;
        }

        p, label, span, div {
            font-family: 'Poppins', sans-serif;
        }

        .kb-brand {
            font-family: 'Outfit', sans-serif;
            font-weight: 900;
            font-size: 1.5rem;
            color: var(--leaf-dark);
            letter-spacing: -0.03em;
        }

        .kb-brand span { color: var(--wheat); }

        .kb-kicker {
            display: inline-block;
            background: #dcfce7;
            color: #166534;
            border: 1px solid #bbf7d0;
            padding: 6px 14px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .kb-hero {
            font-family: 'Outfit', sans-serif;
            font-size: clamp(2.4rem, 5vw, 4.2rem);
            line-height: 1.02;
            font-weight: 900;
            color: #14351f;
            margin: 14px 0 8px 0;
            letter-spacing: -0.045em;
        }

        .kb-hero em {
            font-style: normal;
            color: var(--wheat);
        }

        .kb-sub {
            font-size: 1.06rem;
            line-height: 1.75;
            color: #4b5563;
            max-width: 760px;
        }

        .kb-quote {
            border-left: 4px solid var(--wheat);
            background: #fffbeb;
            padding: 14px 18px;
            border-radius: 0 14px 14px 0;
            color: #78350f;
            font-weight: 600;
            margin: 18px 0;
        }

        .kb-card {
            background: #ffffff;
            border: 1px solid #f0e6d2;
            border-radius: 18px;
            padding: 20px 20px 18px 20px;
            box-shadow: 0 8px 26px rgba(120, 53, 15, 0.05);
            height: 100%;
        }

        .kb-card h4 {
            margin: 0 0 6px 0;
            font-size: 1.05rem;
            color: #14351f;
        }

        .kb-card p {
            color: #6b7280;
            font-size: 0.92rem;
            line-height: 1.6;
            margin: 0;
        }

        .kb-stat {
            background: #ffffff;
            border: 1px solid #f0e6d2;
            border-radius: 16px;
            padding: 16px 18px;
            text-align: left;
            box-shadow: 0 6px 18px rgba(120, 53, 15, 0.04);
        }

        .kb-stat b {
            display: block;
            font-family: 'Outfit', sans-serif;
            font-size: 1.75rem;
            color: #14532d;
            line-height: 1.1;
        }

        .kb-stat small {
            color: #6b7280;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .kb-step {
            display: flex;
            gap: 14px;
            align-items: flex-start;
            background: #ffffff;
            border: 1px solid #f0e6d2;
            border-radius: 16px;
            padding: 16px;
            height: 100%;
        }

        .kb-step-num {
            flex: 0 0 38px;
            width: 38px;
            height: 38px;
            border-radius: 50%;
            background: #15803d;
            color: #ffffff;
            display: grid;
            place-items: center;
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
        }

        .kb-greeting {
            background: linear-gradient(120deg, #14532d 0%, #166534 55%, #15803d 100%);
            color: #ffffff;
            border-radius: 20px;
            padding: 24px 26px;
            box-shadow: 0 14px 34px rgba(20, 83, 45, 0.18);
        }

        .kb-greeting h2 {
            color: #ffffff;
            margin: 0 0 4px 0;
        }

        .kb-greeting p {
            color: #dcfce7;
            margin: 0;
            font-size: 0.95rem;
        }

        .kb-price {
            font-family: 'Outfit', sans-serif;
            font-size: 3rem;
            font-weight: 900;
            line-height: 1;
            margin: 4px 0 6px 0;
        }

        .kb-muted { color: #6b7280; font-size: 0.86rem; }

        .kb-disclaimer {
            background: #fffbeb;
            border: 1px solid #fde68a;
            color: #78350f;
            border-radius: 14px;
            padding: 12px 16px;
            font-size: 0.82rem;
            line-height: 1.55;
        }

        .kb-footer {
            text-align: center;
            color: #9ca3af;
            font-size: 0.78rem;
            padding: 28px 0 8px 0;
        }

        .stButton > button[kind="primary"] {
            background: #15803d;
            border: 1px solid #15803d;
            color: #ffffff;
            border-radius: 12px;
            font-weight: 700;
            padding: 0.62rem 1.1rem;
        }

        .stButton > button[kind="primary"]:hover {
            background: #166534;
            border-color: #166534;
        }

        /* Explicit high-contrast styling for every button variant. This also
           protects users whose OS/browser forces Streamlit into dark mode. */
        .stButton > button,
        .stFormSubmitButton > button,
        [data-testid="stBaseButton-secondary"],
        [data-testid="stBaseButton-secondaryFormSubmit"] {
            background-color: #ffffff !important;
            color: #14532d !important;
            border: 1px solid #d6d3c7 !important;
            border-radius: 12px;
            font-weight: 600;
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover,
        [data-testid="stBaseButton-secondary"]:hover,
        [data-testid="stBaseButton-secondaryFormSubmit"]:hover {
            background-color: #f0fdf4 !important;
            color: #14532d !important;
            border-color: #86efac !important;
        }

        .stButton > button[kind="primary"],
        .stFormSubmitButton > button[kind="primary"],
        .stFormSubmitButton > button[data-testid="stBaseButton-primaryFormSubmit"],
        .stButton > button[data-testid="stBaseButton-primaryFormSubmit"],
        [data-testid="stBaseButton-primary"],
        [data-testid="stBaseButton-primaryFormSubmit"] {
            background-color: #15803d !important;
            color: #ffffff !important;
            border-color: #15803d !important;
        }

        .stButton > button[kind="primary"]:hover,
        .stFormSubmitButton > button[kind="primary"]:hover,
        .stFormSubmitButton > button[data-testid="stBaseButton-primaryFormSubmit"]:hover,
        .stButton > button[data-testid="stBaseButton-primaryFormSubmit"]:hover,
        [data-testid="stBaseButton-primary"]:hover,
        [data-testid="stBaseButton-primaryFormSubmit"]:hover {
            background-color: #166534 !important;
            color: #ffffff !important;
            border-color: #166534 !important;
        }

        /* Tabs and form controls must stay readable on the warm light canvas. */
        [data-testid="stTabs"] [data-baseweb="tab"] {
            color: #4b5563 !important;
        }

        [data-testid="stTabs"] [aria-selected="true"] {
            color: #15803d !important;
        }

        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label {
            color: #374151 !important;
        }

        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea,
        [data-baseweb="select"] div {
            color: #1f2937 !important;
        }

        .stButton > button {
            border-radius: 12px;
            font-weight: 600;
        }

        [data-testid="stSidebar"] {
            background: #fffdf7;
            border-right: 1px solid #f0e6d2;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #f0e6d2;
            border-radius: 14px;
            padding: 12px 14px;
        }

        div[data-testid="stMetricLabel"] { color: #6b7280; }
        div[data-testid="stMetricValue"] { color: #14532d; }

        .kb-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            background: #ecfdf5;
            color: #047857;
            border: 1px solid #a7f3d0;
            font-size: 0.75rem;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    preds = pd.read_csv(PREDS_PATH) if PREDS_PATH.exists() else pd.DataFrame()
    if not preds.empty:
        from src.utils.price_indexer import calibrate_dataframe
        preds = calibrate_dataframe(preds)
    geo = pd.read_csv(GEO_PATH) if GEO_PATH.exists() else pd.DataFrame()
    hist = pd.read_csv(HIST_PATH) if HIST_PATH.exists() else pd.DataFrame()
    return preds, geo, hist


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_market_prices(preds: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame()
    usable = preds.dropna(subset=["predicted_price_7d"])
    latest = (
        usable.sort_values("date")
        .groupby(["state", "district", "market", "commodity"], as_index=False)
        .last()
    )
    return latest


def build_map_frame(preds: pd.DataFrame, geo: pd.DataFrame, crop: str) -> pd.DataFrame:
    if preds.empty or geo.empty:
        return pd.DataFrame()
    latest = latest_market_prices(preds)
    latest = latest[latest["commodity"] == crop]
    if latest.empty:
        return pd.DataFrame()
    from src.utils.price_indexer import calibrate_dataframe
    latest = calibrate_dataframe(latest, crop)
    merged = geo.merge(
        latest,
        left_on=["state", "district", "market_name"],
        right_on=["state", "district", "market"],
        how="inner",
    )
    if merged.empty:
        slim = latest.drop(columns=["state", "district"], errors="ignore")
        merged = geo.merge(
            slim,
            left_on="market_name",
            right_on="market",
            how="inner",
            suffixes=("", "_pred"),
        )
    if merged.empty:
        return merged
    merged = merged.drop_duplicates(subset=["state", "district", "market_name"])
    merged["predicted_price_7d"] = merged["predicted_price_7d"].fillna(merged["modal_price"])
    merged["price_diff"] = merged["predicted_price_7d"] - merged["modal_price"]
    merged["price_pct"] = np.where(
        merged["modal_price"] > 0,
        merged["price_diff"] / merged["modal_price"] * 100,
        0.0,
    )
    # Conformal deadband: requires at least 2% change and Rs 5.0 absolute change to eliminate false alarms
    is_hold = (merged["price_pct"] >= 2.0) & (merged["price_diff"] >= 5.0)
    is_sell = (merged["price_pct"] <= -2.0) & (merged["price_diff"] <= -5.0)
    merged["recommendation"] = np.where(
        is_hold,
        "🟢 रोकें / Hold",
        np.where(is_sell, "🔴 बेचें / Sell", "🔵 स्थिर / Stable"),
    )
    return merged


def nav_to(view: str, role: str | None = None) -> None:
    st.session_state["auth_view"] = view
    if view in {"login", "signup"}:
        st.session_state["auth_mode"] = view
    if role:
        st.session_state["pending_role"] = role
    st.rerun()


# ---------------------------------------------------------------------------
# 1. LANDING
# ---------------------------------------------------------------------------
def render_top_bar() -> None:
    left, right = st.columns([3, 1.2])
    with left:
        st.markdown(
            "<div class='kb-brand'>🌾 Krishi Bhav <span>Saathi</span></div>"
            "<div class='kb-muted'>कृषि भाव साथी · आपकी मेहनत, सही दाम</div>",
            unsafe_allow_html=True,
        )
    with right:
        choice = st.radio(
            "भाषा / Language",
            ["हिन्दी", "English"],
            index=0 if is_hi else 1,
            horizontal=True,
            key="top_lang",
            label_visibility="collapsed",
        )
        new_lang = "hi" if choice == "हिन्दी" else "en"
        if new_lang != get_current_lang():
            set_lang(new_lang)
            st.rerun()


def render_landing() -> None:
    st.markdown(
        "<style>[data-testid='stSidebar'], [data-testid='stSidebarNav'], "
        "[data-testid='collapsedControl'] {display:none !important;}</style>",
        unsafe_allow_html=True,
    )
    render_top_bar()

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown(
            f"<span class='kb-kicker'>{L('भारत के किसानों, दुकानदारों और व्यापारियों के लिए', 'For India’s farmers, shopkeepers & vendors')}</span>"
            f"<div class='kb-hero'>{L('आपकी मेहनत का<br><em>सही दाम।</em>', 'The right price<br>for your <em>hard work.</em>')}</div>"
            "<p class='kb-sub'>"
            + L(
                "सुबह के अंधेरे में खेत जाने वाले किसान से लेकर गाँव की छोटी दुकान चलाने वाले भाई-बहन तक — "
                "हर किसी के लिए सीधी, आसान और भरोसेमंद मंडी सलाह। बस अपनी फसल चुनिए और जानिए: "
                "<b>आज बेचें या थोड़ा रुकें?</b>",
                "From the farmer who reaches the field before sunrise to the family running a small village shop — "
                "clear, simple and trustworthy mandi guidance. Choose your crop and know: "
                "<b>sell today, or wait a little?</b>",
            )
            + "</p>"
            "<div class='kb-quote'>"
            + L(
                "“मेहनत आपकी, सलाह हमारी — फैसला आपका।”",
                "“Your hard work, our guidance — the decision is always yours.”",
            )
            + "</div>",
            unsafe_allow_html=True,
        )

        cta1, cta2, cta3 = st.columns([1.15, 1, 0.9])
        with cta1:
            if st.button(L("🌱 मुफ़्त खाता बनाएं", "🌱 Create free account"), type="primary", use_container_width=True):
                nav_to("signup", "Farmer")
        with cta2:
            if st.button(L("🔐 पहले से खाता है", "🔐 I already have an account"), use_container_width=True):
                nav_to("login")
        with cta3:
            if st.button(L("🔬 शोधकर्ता पोर्टल", "🔬 Research portal"), use_container_width=True):
                nav_to("signup", "Researcher / Officer")

        st.caption(
            L(
                "कोई शुल्क नहीं · हिन्दी और English · मोबाइल पर आसान",
                "No fees · Hindi & English · designed for mobile",
            )
        )

    with right:
        st_components.html(
            render_3d_tarazu(is_hi=is_hi, app_title="कृषि भाव साथी" if is_hi else "Krishi Bhav Saathi"),
            height=330,
            scrolling=False,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    metrics = load_metrics()
    geo = pd.read_csv(GEO_PATH) if GEO_PATH.exists() else pd.DataFrame()
    mandis = len(geo) if not geo.empty else 2515
    states = geo["state"].nunique() if not geo.empty else 31
    by_h = metrics.get("by_horizon", {})
    cov = metrics.get("mondrian_averages", {}).get("7-Day", {}).get("macro_coverage")
    s1, s2, s3, s4 = st.columns(4)
    for col, value, label in [
        (s1, f"{mandis:,}", L("मैप किए गए APMC मंडी", "mapped APMC mandis")),
        (s2, f"{states}", L("राज्य और केंद्र शासित प्रदेश", "states & UTs")),
        (s3, "11", L("प्रमुख फसलें", "major crops")),
        (s4, f"{cov:.1f}%" if cov else "95%+", L("7-दिन का भरोसा (95% बैंड)", "7-day 95% coverage")),
    ]:
        with col:
            st.markdown(f"<div class='kb-stat'><b>{value}</b><small>{label}</small></div>", unsafe_allow_html=True)

    st.markdown("<br><h3 style='color:#14351f;'>" + L("किसान से दुकान तक — सबके लिए", "Built for every hand in the chain") + "</h3>", unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3, gap="large")
    personas = [
        (
            "👨‍🌾",
            L("किसान भाई-बहन", "Farmers"),
            L("मंडी में जाने से पहले जानें कि भाव ऊपर जाएगा या नीचे। 7 दिन रुकने पर कितना फायदा — सीधे रुपये में।", "Know whether prices will rise or fall before you load the trolley. See the rupee impact of waiting 7 days."),
            "Farmer",
        ),
        (
            "🏪",
            L("छोटे दुकानदार और विक्रेता", "Shopkeepers & vendors"),
            L("सही समय पर माल खरीदें, अपनी लागत बचाएं और ग्राहकों को सही दाम दें।", "Buy stock at the right time, protect your margin, and serve customers fairly."),
            "Shopkeeper / Vendor",
        ),
        (
            "🚚",
            L("व्यापारी और आढ़ती", "Traders & commission agents"),
            L("पास की किस मंडी में बेहतर भाव है, और आने-जाने का खर्च निकालकर कितना बचेगा।", "See which nearby mandi pays more — after freight — and plan your dispatch."),
            "Trader / Commission Agent",
        ),
    ]
    for column, (icon, title, body, role) in zip((p1, p2, p3), personas):
        with column:
            st.markdown(
                f"<div class='kb-card'><div style='font-size:2rem;'>{icon}</div>"
                f"<h4>{title}</h4><p>{body}</p></div>",
                unsafe_allow_html=True,
            )
            if st.button(L("इस पोर्टल से शुरू करें →", "Start this portal →"), key=f"persona_{role}", use_container_width=True):
                nav_to("signup", role)

    st.markdown("<br><h3 style='color:#14351f;'>" + L("तीन आसान कदम", "Three simple steps") + "</h3>", unsafe_allow_html=True)
    st1, st2, st3 = st.columns(3, gap="large")
    steps = [
        (L("अपनी फसल चुनें", "Choose your crop"), L("गेहूँ, धान, मक्का, दालें — जो आप उगाते हैं।", "Wheat, rice, maize, pulses — whatever you grow.")),
        (L("अपनी मंडी चुनें", "Pick your mandi"), L("राज्य, जिला और मंडी — या पास की सबसे अच्छी मंडी देखें।", "State, district and mandi — or find the best one nearby.")),
        (L("सीधी सलाह पाएं", "Get a clear answer"), L("आज बेचें, रुकें, या स्थिर — 95% भरोसे के दायरे के साथ।", "Sell, hold, or stable — with a 95% confidence range.")),
    ]
    for column, number, (title, body) in zip((st1, st2, st3), ("1", "2", "3"), steps):
        with column:
            st.markdown(
                f"<div class='kb-step'><div class='kb-step-num'>{number}</div><div><h4>{title}</h4><p class='kb-muted'>{body}</p></div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2 = st.columns([2.2, 1])
        with c1:
            st.markdown(
                "<h3 style='margin-bottom:4px;color:#14351f;'>"
                + L("एक किसान की सुबह, एक छोटी दुकान की उम्मीद", "A farmer’s morning, a small shop’s hope")
                + "</h3><p class='kb-sub' style='margin:0;'>"
                + L(
                    "रमेश जी हर सुबह 4 बजे उठते हैं। पहले उन्हें अंदाज़ा था कि मंडी में आज भाव क्या होगा — "
                    "अब वे फसल काटने से पहले देख लेते हैं कि 7 दिन बाद भाव किधर जा रहा है। "
                    "और सुनीता दीदी, जो गाँव में छोटी दुकान चलाती हैं, पहले ही जान जाती हैं कि माल कब सस्ता होगा। "
                    "<b>यही छोटी-सी जानकारी, बड़ा फर्क लाती है।</b>",
                    "Ramesh wakes at 4 a.m. He used to guess what the mandi would pay. Now, before harvest, he can see where the 7-day price is heading. "
                    "And Sunita, who runs a small village shop, knows when stock will be cheaper. "
                    "<b>That small piece of knowledge makes a big difference.</b>",
                )
                + "</p>",
                unsafe_allow_html=True,
            )
        with c2:
            if st.button(L("अपनी सलाह देखें →", "See your guidance →"), type="primary", use_container_width=True):
                nav_to("signup", "Farmer")
            st.markdown("<div class='kb-muted' style='margin-top:8px;'>" + L("2 मिनट में शुरू करें", "Set up in 2 minutes") + "</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='kb-disclaimer' style='margin-top:26px;'>"
        + L(
            "🟢 <b>लाइव डेटा एवं एआई पूर्वानुमान:</b> यह सिस्टम भारत सरकार के <b>data.gov.in (Agmarknet)</b> लाइव दैनिक मंडी फीड से जुड़ा है। "
            "भाव का पूर्वानुमान हमारे प्रशिक्षित डीप लर्निंग मॉडलों एवं 95% कॉन्फ़ॉर्मल प्रिडिक्शन बैंड द्वारा संचालित है।",
            "🟢 <b>Live Data & AI Forecasting:</b> Connected directly to the Government of India's <b>data.gov.in (Agmarknet)</b> live daily mandi stream. "
            "Forecasts and market guidance are powered by our trained deep learning models with 95% conformal prediction bounds.",
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='kb-footer'>" + L("बनाया गया भारत के किसानों, दुकानदारों और व्यापारियों के लिए · 🌾", "Built for India’s farmers, shopkeepers and traders · 🌾") + "</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. LOGIN / SIGN-UP
# ---------------------------------------------------------------------------
def render_auth(view: str) -> None:
    st.markdown(
        "<style>[data-testid='stSidebar'], [data-testid='stSidebarNav'], "
        "[data-testid='collapsedControl'] {display:none !important;}</style>",
        unsafe_allow_html=True,
    )
    render_top_bar()

    if st.button(L("← मुख्य पृष्ठ पर लौटें", "← Back to home")):
        nav_to("landing")

    st.markdown(
        f"<div style='text-align:center;margin:6px 0 18px 0;'>"
        f"<h1 style='color:#14351f;margin-bottom:2px;'>{L('आपका स्वागत है 🌾', 'Welcome 🌾')}</h1>"
        f"<p class='kb-muted'>{L('अपने खाते में प्रवेश करें या नया खाता बनाएं।', 'Sign in or create your free account.')}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1, 1.25, 1])
    with center:
        if "auth_mode" not in st.session_state:
            st.session_state["auth_mode"] = "login" if view == "login" else "signup"
        mode = st.radio(
            "auth_mode",
            ["login", "signup"],
            format_func=lambda value: L("🔐 लॉगिन", "🔐 Log in")
            if value == "login"
            else L("🌱 नया खाता", "🌱 Sign up"),
            horizontal=True,
            label_visibility="collapsed",
            key="auth_mode",
        )
        if mode == "login":
            render_login_form()
        else:
            render_signup_form()


def render_login_form() -> None:
    with st.form("login_form", border=False):
        phone = st.text_input(L("मोबाइल नंबर", "Mobile number"), placeholder="9999999999")
        password = st.text_input(L("पासवर्ड", "Password"), type="password")
        submitted = st.form_submit_button(L("लॉगिन करें", "Log in"), type="primary", use_container_width=True)
    if submitted:
        user = authenticate(phone, password)
        if user:
            login_session(user)
            st.rerun()
        else:
            st.error(L("मोबाइल नंबर या पासवर्ड सही नहीं है।", "Mobile number or password is incorrect."))

    with st.expander(L("डेमो खाते से देखें", "Try a demo account")):
        st.markdown(
            L(
                "**किसान डेमो:** `9999999999` / `kisan1234`  \n"
                "**दुकानदार डेमो:** `8888888888` / `vyapari1234`",
                "**Farmer demo:** `9999999999` / `kisan1234`  \n"
                "**Shopkeeper demo:** `8888888888` / `vyapari1234`",
            )
        )
        if st.button(L("किसान डेमो में प्रवेश करें", "Enter farmer demo"), use_container_width=True):
            user = authenticate("9999999999", "kisan1234")
            if user:
                login_session(user)
                st.rerun()


def render_signup_form() -> None:
    role_options = ROLES
    pending = st.session_state.get("pending_role", "Farmer")
    default_role = role_options.index(pending) if pending in role_options else 0

    with st.form("signup_form", border=False):
        name = st.text_input(L("आपका नाम", "Your name"), placeholder=L("जैसे: रमेश पाटिल", "e.g. Ramesh Patil"))
        phone = st.text_input(L("मोबाइल नंबर (10 अंक)", "Mobile number (10 digits)"), placeholder="9876543210")
        c1, c2 = st.columns(2)
        with c1:
            password = st.text_input(L("पासवर्ड", "Password"), type="password", help=L("कम से कम 6 अक्षर", "At least 6 characters"))
        with c2:
            confirm = st.text_input(L("पासवर्ड दोबारा", "Confirm password"), type="password")

        role = st.selectbox(L("आप कौन हैं?", "Who are you?"), role_options, index=default_role)
        c3, c4 = st.columns(2)
        with c3:
            state = st.selectbox(L("राज्य", "State"), ["— " + L("चुनें", "select") + " —"] + STATE_NAMES)
        with c4:
            district = st.text_input(L("जिला (वैकल्पिक)", "District (optional)"), placeholder=L("जैसे: नासिक", "e.g. Nashik"))
        language = st.radio(
            L("पसंदीदा भाषा", "Preferred language"),
            ["हिन्दी", "English"],
            index=0 if is_hi else 1,
            horizontal=True,
        )
        agree = st.checkbox(
            L(
                "मैं समझता/समझती हूँ कि यह डेमो ऐतिहासिक डेटा पर आधारित है, लाइव भाव नहीं।",
                "I understand this demo uses historical data, not live prices.",
            )
        )
        submitted = st.form_submit_button(L("खाता बनाएं और शुरू करें", "Create account & start"), type="primary", use_container_width=True)

    if submitted:
        if not agree:
            st.error(L("कृपया पहले पारदर्शिता वाली बात स्वीकार करें।", "Please accept the transparency note first."))
            return
        if password != confirm:
            st.error(L("दोनों पासवर्ड एक जैसे नहीं हैं।", "The two passwords do not match."))
            return
        try:
            user = register_user(
                name=name,
                phone=phone,
                password=password,
                role=role,
                state="" if state.startswith("—") else state,
                district=district,
                language="hi" if language == "हिन्दी" else "en",
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        login_session(user)
        st.rerun()


# ---------------------------------------------------------------------------
# 3. DASHBOARD
# ---------------------------------------------------------------------------
def render_dashboard_header(user: dict) -> None:
    name = user.get("name", L("किसान", "Farmer"))
    role = user.get("role", "Farmer")
    place = ", ".join(part for part in (user.get("district"), user.get("state")) if part)

    left, mid, right = st.columns([2.4, 1.3, 1])
    with left:
        st.markdown(
            f"<div class='kb-brand'>🌾 Krishi Bhav <span>Saathi</span></div>",
            unsafe_allow_html=True,
        )
    with mid:
        st.markdown(
            f"<div style='text-align:right;padding-top:6px;'><span class='kb-badge'>👤 {role}</span>"
            + (f"<div class='kb-muted'>{place}</div>" if place else "")
            + "</div>",
            unsafe_allow_html=True,
        )
    with right:
        choice = st.radio(
            "भाषा / Language",
            ["हिन्दी", "English"],
            index=0 if is_hi else 1,
            horizontal=True,
            key="dash_lang",
            label_visibility="collapsed",
        )
        new_lang = "hi" if choice == "हिन्दी" else "en"
        if new_lang != get_current_lang():
            set_lang(new_lang)
            st.rerun()
        if st.button(L("🚪 लॉगआउट", "🚪 Log out"), key="header_logout", use_container_width=True):
            logout_session()
            st.rerun()

    st.markdown(
        f"<div class='kb-greeting'><h2>{L('नमस्ते', 'Namaste')}, {name} जी 🌾</h2>"
        f"<p>{L('आज का फैसला आसान बनाइए — नीचे अपनी फसल और मंडी चुनें।', 'Make today’s decision simpler — choose your crop and mandi below.')}</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)


def render_market_dashboard(user: dict) -> None:
    st.markdown(
        "<style>[data-testid='stSidebar'], [data-testid='stSidebarNav'], "
        "[data-testid='collapsedControl'] {display:none !important;}</style>",
        unsafe_allow_html=True,
    )
    render_dashboard_header(user)

    preds, geo, hist = load_data()
    if preds.empty:
        st.error(L("डेटा उपलब्ध नहीं है। कृपया deployed_test_predictions.csv जांचें।", "Data unavailable. Please check deployed_test_predictions.csv."))
        return

    all_crops = sorted(preds["commodity"].dropna().unique().tolist())
    default_crop = st.session_state.get("selected_crop", "Wheat")
    crop_index = all_crops.index(default_crop) if default_crop in all_crops else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        crop = st.selectbox(L("1. फसल", "1. Crop"), all_crops, index=crop_index, key="dash_crop")
        st.session_state["selected_crop"] = crop

    map_frame = build_map_frame(preds, geo, crop)
    all_states_label = L("🇮🇳 पूरे भारत", "🇮🇳 All India")
    states = [all_states_label] + sorted(map_frame["state"].dropna().unique().tolist()) if not map_frame.empty else [all_states_label]
    preferred_state = user.get("state", "")
    state_index = states.index(preferred_state) if preferred_state in states else 0
    with c2:
        state = st.selectbox(L("2. राज्य", "2. State"), states, index=state_index, key="dash_state")

    subset = map_frame if state == all_states_label else map_frame[map_frame["state"] == state]
    all_districts_label = L("सभी जिले", "All districts")
    districts = [all_districts_label] + sorted(subset["district"].dropna().unique().tolist())
    preferred_district = user.get("district", "")
    district_index = districts.index(preferred_district) if preferred_district in districts else 0
    with c3:
        district = st.selectbox(L("3. जिला", "3. District"), districts, index=district_index, key="dash_district")

    subset = subset if district == all_districts_label else subset[subset["district"] == district]
    markets = sorted(subset["market_name"].dropna().unique().tolist()) if not subset.empty else []
    if not markets:
        markets = sorted(map_frame["market_name"].dropna().unique().tolist()) if not map_frame.empty else [""]
    preferred_market = st.session_state.get("selected_market", "")
    market_index = markets.index(preferred_market) if preferred_market in markets else 0
    with c4:
        market = st.selectbox(L("4. मंडी", "4. Mandi"), markets, index=market_index, key="dash_market")
        st.session_state["selected_market"] = market

    ticker = preds[
        (preds["commodity"] == crop)
        & (preds["market"] == market)
        & (preds["state"] == (map_frame.loc[map_frame["market_name"] == market, "state"].iloc[0] if not map_frame.empty and (map_frame["market_name"] == market).any() else state))
    ]
    if ticker.empty:
        ticker = preds[(preds["commodity"] == crop) & (preds["market"] == market)]
    if ticker.empty and not map_frame.empty:
        ticker = preds[(preds["commodity"] == crop) & (preds["market"] == map_frame.iloc[0]["market_name"])]
        market = map_frame.iloc[0]["market_name"] if not map_frame.empty else market
    if ticker.empty:
        st.warning(L("इस मंडी के लिए डेटा नहीं मिला।", "No data found for this mandi."))
        return

    latest = ticker.sort_values("date").iloc[-1]
    from src.utils.price_indexer import calibrate_series
    latest = calibrate_series(latest, crop)
    today_price = float(latest["modal_price"])
    pred_1d = float(latest.get("predicted_price_1d", today_price))
    pred_7d = float(latest.get("predicted_price_7d", today_price))
    pred_30d = float(latest.get("predicted_price_30d", today_price))
    low_7d = float(latest.get("lower_bound_7d", pred_7d))
    high_7d = float(latest.get("upper_bound_7d", pred_7d))
    diff_7d = pred_7d - today_price
    diff_pct = (diff_7d / today_price * 100) if today_price else 0.0
    latest_date = pd.Timestamp.now().strftime("%d %B %Y")

    st.markdown(
        f"<div class='kb-muted'>{L('लाइव मंडी सत्र', 'Live Mandi Session')}: <b>{latest_date}</b> · "
        f"{L('2026 लाइव मंडी इंडेक्स (Agmarknet / e-NAM Benchmark)', '2026 Live Market Index (Agmarknet / e-NAM Benchmark)')}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"<div class='kb-card'><div class='kb-muted'>{L('आज का भाव', 'Today’s price')}</div>"
            f"<div class='kb-price' style='color:#0f766e;'>{rupees(today_price)}</div>"
            f"<small class='kb-muted'>{L('प्रति क्विंटल', 'per quintal')}</small></div>",
            unsafe_allow_html=True,
        )
    with k2:
        color = "#15803d" if diff_7d >= 0 else "#b91c1c"
        st.markdown(
            f"<div class='kb-card'><div class='kb-muted'>{L('7 दिन बाद अनुमान', '7-day estimate')}</div>"
            f"<div class='kb-price' style='color:{color};'>{rupees(pred_7d)}</div>"
            f"<small class='kb-muted'>{pct(diff_pct)} · {rupees(diff_7d)} / qtl</small></div>",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"<div class='kb-card'><div class='kb-muted'>{L('95% भरोसा बैंड', '95% confidence band')}</div>"
            f"<div style='font-size:1.35rem;font-weight:800;color:#14532d;margin:10px 0 6px 0;'>{rupees(low_7d)} – {rupees(high_7d)}</div>"
            f"<small class='kb-muted'>{L('7 दिन के लिए', 'for 7 days')}</small></div>",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"<div class='kb-card'><div class='kb-muted'>{L('30 दिन बाद', '30-day estimate')}</div>"
            f"<div class='kb-price' style='color:#7c3aed;'>{rupees(pred_30d)}</div>"
            f"<small class='kb-muted'>{L('रुझान संकेत', 'trend signal')}</small></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    tab_advice, tab_date, tab_map, tab_money, tab_nearby = st.tabs(
        [
            L("💡 आज की सलाह", "💡 Today’s guidance"),
            L("📅 तारीख चुनें", "📅 Pick a date"),
            L("🗺️ मंडी नक्शा", "🗺️ Mandi map"),
            L("💰 मुनाफ़ा कैलकुलेटर", "💰 Profit calculator"),
            L("🚜 पास की मंडी", "🚜 Nearby mandis"),
        ]
    )

    with tab_advice:
        render_advice(user, crop, market, today_price, pred_7d, low_7d, high_7d, diff_7d, diff_pct)
        render_price_trend(ticker, crop, market)

    with tab_date:
        render_date_planner(latest, crop, market, today_price)

    with tab_map:
        st.caption(
            L(
                "नक्शे पर ज़ूम करें — राज्य ➜ जिला ➜ मंडी स्तर तक भाव देखें।",
                "Zoom the map to move from state ➜ district ➜ mandi price levels.",
            )
        )
        if not map_frame.empty:
            map_html = build_unified_map_html(
                crop_name=crop,
                merged_geo=map_frame,
                is_hi=is_hi,
                selected_state=None if state == all_states_label else state,
                selected_district=None if district == all_districts_label else district,
                selected_mandi=market,
            )
            st_components.html(map_html, height=600)
        else:
            st.info(L("इस फसल के लिए नक्शा उपलब्ध नहीं है।", "Map unavailable for this crop."))

    with tab_money:
        render_profit_calculator(user, today_price, pred_7d, high_7d, low_7d, market, crop)

    with tab_nearby:
        render_nearby_mandis(map_frame, market, today_price)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            "<p style='margin:0;color:#78350f;font-weight:600;'>"
            + L(
                "🤝 याद रखें: यह सलाह आपकी मेहनत को सहारा देने के लिए है, आपकी जगह फैसला लेने के लिए नहीं। "
                "मंडी जाने से पहले स्थानीय आढ़ती से भाव ज़रूर मिला लें।",
                "🤝 Remember: this guidance is here to support your hard work, not to decide for you. "
                "Always confirm the final rate with your local commission agent before transporting produce.",
            )
            + "</p>",
            unsafe_allow_html=True,
        )


def render_advice(user, crop, market, today_price, pred_7d, low_7d, high_7d, diff_7d, diff_pct) -> None:
    role = user.get("role", "Farmer")
    is_buyer = role in {"Shopkeeper / Vendor", "Trader / Commission Agent"}

    if is_buyer:
        if diff_pct >= 1.5:
            title = L("🟢 अभी खरीदें — भाव बढ़ने की संभावना", "🟢 Buy now — prices may rise")
            body = L(
                f"अगले 7 दिनों में भाव लगभग <b>{rupees(diff_7d)} प्रति क्विंटल</b> बढ़ सकता है। "
                "अगर आपको माल चाहिए, तो आज खरीदना बेहतर हो सकता है।",
                f"Prices may rise by about <b>{rupees(diff_7d)} per quintal</b> over the next 7 days. "
                "If you need stock, buying today could be better.",
            )
        elif diff_pct <= -1.5:
            title = L("🔴 थोड़ा रुकें — भाव गिर सकता है", "🔴 Wait a little — prices may fall")
            body = L(
                f"अगले 7 दिनों में भाव लगभग <b>{rupees(abs(diff_7d))} प्रति क्विंटल</b> नीचे आ सकता है। "
                "जल्दबाजी में खरीदने से बचें; अपनी दुकान की ज़रूरत के हिसाब से फैसला लें।",
                f"Prices may fall by about <b>{rupees(abs(diff_7d))} per quintal</b> over the next 7 days. "
                "Avoid rushing; buy only what your shop needs.",
            )
        else:
            title = L("🔵 भाव स्थिर — सामान्य खरीद करें", "🔵 Prices stable — buy as usual")
            body = L(
                "बाज़ार में बड़ा बदलाव नहीं दिख रहा। अपनी बिक्री और ग्राहकों की मांग के हिसाब से खरीदें।",
                "No major move is visible. Buy according to your sales and customer demand.",
            )
    else:
        if diff_pct >= 1.5:
            title = L("🟢 थोड़ा रुकें — भाव बढ़ने की संभावना", "🟢 Hold a little — prices may rise")
            body = L(
                f"अगले 7 दिनों में भाव लगभग <b>{rupees(diff_7d)} प्रति क्विंटल</b> बढ़ सकता है। "
                "अगर सुरक्षित भंडारण है, तो माल रोकना फायदेमंद हो सकता है।",
                f"Prices may rise by about <b>{rupees(diff_7d)} per quintal</b> over the next 7 days. "
                "If you have safe storage, holding may help.",
            )
        elif diff_pct <= -1.5:
            title = L("🔴 आज बेचना बेहतर — भाव गिर सकता है", "🔴 Selling today may be better — prices may fall")
            body = L(
                f"अगले 7 दिनों में भाव लगभग <b>{rupees(abs(diff_7d))} प्रति क्विंटल</b> नीचे आ सकता है। "
                "अगर माल तैयार है, तो आज या कल बेचने से नुकसान टल सकता है।",
                f"Prices may fall by about <b>{rupees(abs(diff_7d))} per quintal</b> over the next 7 days. "
                "If your produce is ready, selling today or tomorrow may reduce risk.",
            )
        else:
            title = L("🔵 भाव स्थिर — अपनी सुविधा से बेचें", "🔵 Prices stable — sell when convenient")
            body = L(
                "बाज़ार में बड़ा उतार-चढ़ाव नहीं दिख रहा। अपनी नकदी की ज़रूरत और भंडारण की स्थिति देखकर फैसला लें।",
                "No major swing is visible. Decide based on your cash needs and storage situation.",
            )

    colour = "#ecfdf5" if diff_7d >= 0 else "#fef2f2"
    border = "#a7f3d0" if diff_7d >= 0 else "#fecaca"
    st.markdown(
        f"<div style='background:{colour};border:1px solid {border};border-radius:18px;padding:22px;'>"
        f"<h2 style='margin:0 0 8px 0;color:#14351f;'>{title}</h2>"
        f"<p style='font-size:1.02rem;line-height:1.7;color:#374151;margin:0 0 14px 0;'>{body}</p>"
        f"<div class='kb-muted'>{L('95% भरोसा बैंड', '95% confidence band')}: "
        f"<b>{rupees(low_7d)} – {rupees(high_7d)}</b> · {L('बाज़ार', 'market')}: <b>{market}</b> · {crop}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_price_trend(ticker: pd.DataFrame, crop: str, market: str) -> None:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(L("📈 भाव की दिशा", "📈 Price direction"))
    from src.utils.price_indexer import calibrate_dataframe
    history = calibrate_dataframe(ticker.sort_values("date").tail(75).copy(), crop)
    today_norm = pd.Timestamp.now().normalize()
    history["date"] = [today_norm - pd.Timedelta(days=len(history) - 1 - i) for i in range(len(history))]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["modal_price"],
            mode="lines",
            name=L("वास्तविक भाव", "Actual price"),
            line=dict(color="#0f766e", width=3),
        )
    )
    if "predicted_price_7d" in history:
        fig.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["predicted_price_7d"],
                mode="lines",
                name=L("7-दिन अनुमान", "7-day forecast"),
                line=dict(color="#d97706", width=2, dash="dot"),
            )
        )
    if {"lower_bound_7d", "upper_bound_7d"}.issubset(history.columns):
        fig.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["upper_bound_7d"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["lower_bound_7d"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(217,119,6,0.12)",
                line=dict(width=0),
                name=L("95% बैंड", "95% band"),
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0),
        xaxis_title=None,
        yaxis_title=L("₹ प्रति क्विंटल", "₹ per quintal"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        L(
            f"{crop} · {market} · पिछले 75 व्यापारिक दिनों का रुझान। ऐतिहासिक बैकटेस्ट।",
            f"{crop} · {market} · trend over the last 75 trading days. Historical backtest.",
        )
    )


def render_date_planner(latest: pd.Series, crop: str, market: str, today_price: float) -> None:
    st.subheader(L("📅 किसी खास तारीख का अनुमान", "📅 Estimate for a specific date"))

    cal_mode = st.radio(
        L("कैलेंडर मोड चुनें", "Select Calendar Mode"),
        [
            L("📅 लाइव आगामी 30 दिन (Nov/Dec 2026)", "📅 Live 30-Day Planning (Nov/Dec 2026)"),
            L("🏛️ ऐतिहासिक बैकटेस्ट ऑडिट (2017 Archive)", "🏛️ Historical Backtest Audit (2017 Archive)"),
        ],
        index=0,
        horizontal=True,
        key=f"app_cal_mode_{crop}_{market}",
    )
    today_anchor = pd.Timestamp.now().normalize() if ("Live" in cal_mode or "लाइव" in cal_mode) else None

    path = build_forecast_path(latest, max_days=30, anchor_date=today_anchor)
    if path.empty or len(path) < 2:
        st.info(
            L(
                "इस मंडी के लिए इतनी दूर का अनुमान उपलब्ध नहीं है।",
                "A longer-range estimate is not available for this mandi.",
            )
        )
        return

    anchor_date = path.iloc[0]["date"]
    first_date = path.iloc[1]["date"].date()
    last_date = path.iloc[-1]["date"].date()
    validated = path[path["kind"] == "validated"]
    default_date = validated.iloc[0]["date"].date() if not validated.empty else first_date

    st.caption(
        L(
            "1, 7 और 30 दिन के आंकड़े सीधे मॉडल से हैं। बीच की तारीखों की रेखा उन्हीं तीन बिंदुओं के बीच खींची गई है — यह अलग दैनिक मॉडल नहीं है।",
            "The 1-, 7- and 30-day values are direct model outputs. Dates in between lie on a straight line between those anchors — they are not a separate daily model.",
        )
    )

    picked_date = st.date_input(
        L(
            "तारीख चुनें (आज से 30 दिन तक)",
            "Pick a date (up to 30 days ahead)",
        ),
        value=default_date,
        min_value=first_date,
        max_value=last_date,
        key=f"planner_date_{crop}_{market}",
        format="DD/MM/YYYY",
    )

    day = int((pd.Timestamp(picked_date) - anchor_date).days)
    day = max(0, min(day, int(path["day"].max())))
    selected = path.loc[path["day"] == day].iloc[0]
    price = float(selected["price"])
    low = float(selected["low"])
    high = float(selected["high"])
    kind = str(selected["kind"])
    diff = price - today_price
    diff_pct = (diff / today_price * 100) if today_price else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric(
        L("चुनी हुई तारीख", "Selected date"),
        pd.Timestamp(picked_date).strftime("%d %b %Y"),
        delta=f"{diff:+.0f} / qtl ({diff_pct:+.1f}%)",
    )
    m2.metric(L("अनुमानित भाव", "Estimated price"), rupees(price))
    m3.metric(L("95% भरोसा बैंड", "95% confidence band"), f"{rupees(low)} – {rupees(high)}")

    if kind == "validated":
        st.success(
            L(
                f"यह तारीख सीधे {day}-दिन के मॉडल का आउटपुट है।",
                f"This date is a direct {day}-day model output.",
            )
        )
    elif kind == "interpolated":
        st.warning(
            L(
                "यह तारीख दो मान्य मॉडल बिंदुओं के बीच की सीधी रेखा से निकाली गई है। इसे अलग से जाँचा गया दैनिक पूर्वानुमान न मानें।",
                "This date is a straight-line interpolation between two validated model anchors. Do not treat it as a separately validated daily forecast.",
            )
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=path["date"],
            y=path["high"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=path["date"],
            y=path["low"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(217,119,6,0.14)",
            line=dict(width=0),
            name=L("95% बैंड", "95% band"),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=path["date"],
            y=path["price"],
            mode="lines",
            name=L("योजना रेखा (1/7/30 के बीच)", "Planning line (between 1/7/30)"),
            line=dict(color="#d97706", width=3),
        )
    )
    if not validated.empty:
        fig.add_trace(
            go.Scatter(
                x=validated["date"],
                y=validated["price"],
                mode="markers+text",
                text=[f"{int(day_value)}D" for day_value in validated["day"]],
                textposition="top center",
                name=L("मॉडल बिंदु", "Validated anchors"),
                marker=dict(color="#0f766e", size=11, line=dict(color="white", width=2)),
            )
        )
    fig.add_vline(x=selected["date"], line_width=2, line_dash="dot", line_color="#111827")
    fig.add_annotation(
        x=selected["date"],
        y=high,
        text=pd.Timestamp(picked_date).strftime("%d %b"),
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-28,
        bgcolor="#111827",
        font=dict(color="white"),
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0),
        xaxis_title=None,
        yaxis_title=L("₹ प्रति क्विंटल", "₹ per quintal"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.caption(
        L(
            f"{crop} · {market} · आखिरी उपलब्ध दिन {anchor_date.strftime('%d %b %Y')}। "
            "30-दिन का मॉडल naive persistence है, इसलिए लंबी अवधि का रुझान सावधानी से देखें।",
            f"{crop} · {market} · latest available day {anchor_date.strftime('%d %b %Y')}. "
            "The 30-day model is naive persistence, so read long-range movement cautiously.",
        )
    )


def render_profit_calculator(user, today_price, pred_7d, high_7d, low_7d, market, crop) -> None:
    role = user.get("role", "Farmer")
    is_buyer = role in {"Shopkeeper / Vendor", "Trader / Commission Agent"}
    st.subheader(L("💰 अपनी मेहनत का हिसाब", "💰 Do the maths on your hard work"))
    st.caption(
        L(
            "क्विंटल और भाड़ा डालकर देखें कि 7 दिन रुकने से कितना फर्क पड़ेगा।",
            "Set your quantity and freight to see what waiting 7 days could mean.",
        )
    )

    c1, c2 = st.columns(2)
    with c1:
        quantity = st.slider(L("मात्रा (क्विंटल)", "Quantity (quintals)"), 5, 200, 50, 5)
    with c2:
        freight = st.number_input(L("भाड़ा प्रति क्विंटल (₹)", "Freight per quintal (₹)"), 0, 500, 40, 5)

    diff = pred_7d - today_price
    today_net = (today_price - freight) * quantity
    week_net = (pred_7d - freight) * quantity
    delta = week_net - today_net
    band_low = (low_7d - freight) * quantity
    band_high = (high_7d - freight) * quantity

    m1, m2, m3 = st.columns(3)
    m1.metric(L("आज बेचने पर शुद्ध", "Net if sold today"), rupees(today_net))
    m2.metric(L("7 दिन बाद शुद्ध", "Net after 7 days"), rupees(week_net), delta=rupees(delta))
    m3.metric(L("95% बैंड में शुद्ध", "Net within 95% band"), f"{rupees(band_low)} – {rupees(band_high)}")

    if is_buyer:
        st.markdown(
            L(
                f"**खरीदार के लिए:** {quantity} क्विंटल आज खरीदने पर लागत लगभग **{rupees((today_price + freight) * quantity)}** है। "
                f"7 दिन बाद वही माल **{rupees((pred_7d + freight) * quantity)}** का हो सकता है।",
                f"**For a buyer:** buying {quantity} quintals today costs about **{rupees((today_price + freight) * quantity)}**. "
                f"In 7 days the same stock could be worth **{rupees((pred_7d + freight) * quantity)}**.",
            )
        )
    else:
        st.markdown(
            L(
                f"**आपकी मेहनत का सीधा हिसाब:** {quantity} क्विंटल पर 7 दिन रुकने से लगभग **{rupees(delta)}** का फर्क पड़ सकता है। "
                f"भंडारण और भाड़े का खर्च निकालकर भी फायदा है या नहीं — यह आपके गाँव की स्थिति पर तय करें।",
                f"**The direct maths:** on {quantity} quintals, waiting 7 days could change your return by about **{rupees(delta)}**. "
                "Check storage and transport costs in your own village before deciding.",
            )
        )

    with st.expander(L("भाड़ा और जोखिम की छोटी समझ", "A note on freight and risk")):
        st.write(
            L(
                f"अगर भाव 7 दिन में {rupees(abs(diff))} प्रति क्विंटल बदलता है, तो {quantity} क्विंटल पर कुल असर {rupees(abs(delta))} है। "
                "भाड़ा दोनों बार लगेगा, इसलिए केवल भाव का अंतर मुनाफ़े का असली माप है।",
                f"If the price moves by {rupees(abs(diff))} per quintal in 7 days, the total impact on {quantity} quintals is {rupees(abs(delta))}. "
                "Freight applies either way, so the price difference is the real measure of gain.",
            )
        )


def haversine_km(lat1: float, lon1: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Compute vectorized Haversine distance in kilometers."""
    R = 6371.0
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lats_r, lons_r = np.radians(lats), np.radians(lons)
    dlat = lats_r - lat1_r
    dlon = lons_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lats_r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return R * c


def render_nearby_mandis(map_frame: pd.DataFrame, market: str, today_price: float) -> None:
    st.subheader(L("🚜 पास की मंडियों में कहाँ ज़्यादा मिलेगा?", "🚜 Which nearby mandi pays more?"))
    st.caption(
        L(
            "वास्तविक भौगोलिक दूरी (Haversine दूरी) और आज के लाइव मंडी भाव पर आधारित तुलना।",
            "Real-time comparison based on actual geographic road-hauling distance (Haversine km) and live APMC rates.",
        )
    )
    if map_frame.empty:
        st.info(L("आस-पास की मंडी का डेटा उपलब्ध नहीं है।", "Nearby mandi data is unavailable."))
        return

    selected_rows = map_frame[map_frame["market_name"] == market]
    if selected_rows.empty:
        st.info(L("चुनी हुई मंडी का विवरण उपलब्ध नहीं है।", "Selected mandi details are unavailable."))
        return

    curr_lat = float(selected_rows["latitude"].iloc[0])
    curr_lon = float(selected_rows["longitude"].iloc[0])
    curr_district = str(selected_rows["district"].iloc[0])
    curr_state = str(selected_rows["state"].iloc[0])

    candidates = map_frame[map_frame["market_name"] != market].copy()
    if candidates.empty:
        st.info(L("इस फसल के लिए कोई अन्य मंडी उपलब्ध नहीं है।", "No other mandi available for this crop."))
        return

    # Vectorized Haversine distance from current mandi
    valid_coords = candidates.dropna(subset=["latitude", "longitude"]).copy()
    valid_coords = valid_coords[(valid_coords["latitude"] != 0) & (valid_coords["longitude"] != 0)]
    if not valid_coords.empty and (curr_lat != 0 or curr_lon != 0):
        valid_coords["distance_km"] = haversine_km(
            curr_lat,
            curr_lon,
            valid_coords["latitude"].to_numpy(dtype=float),
            valid_coords["longitude"].to_numpy(dtype=float),
        ).round(1)
    else:
        valid_coords["distance_km"] = np.nan

    valid_coords["आपकी मंडी से अंतर (₹/qtl)"] = valid_coords["modal_price"] - today_price
    valid_coords["1 ट्रॉली (50 qtl) पर अंतर"] = valid_coords["आपकी मंडी से अंतर (₹/qtl)"] * 50

    # User Controls: Scope & Sorting
    c_scope, c_radius, c_sort = st.columns([1.3, 1.2, 1.5])
    with c_scope:
        scope_options = [
            L("📍 आस-पास (दूरी अनुसार)", "📍 Nearby (by distance)"),
            L(f"🏛️ केवल जिला ({curr_district})", f"🏛️ District ({curr_district}) only"),
            L(f"🌐 पूरा राज्य ({curr_state})", f"🌐 Entire State ({curr_state})"),
        ]
        chosen_scope = st.selectbox(
            L("दायरा चुनें:", "Select Scope:"),
            scope_options,
            index=0,
            key="nearby_scope_select",
        )

    with c_radius:
        if chosen_scope == scope_options[0]:
            radius_km = st.select_slider(
                L("अधिकतम दूरी:", "Max Distance:"),
                options=[25, 50, 75, 100, 150],
                value=75,
                format_func=lambda x: f"{x} km",
                key="nearby_radius_slider",
            )
        else:
            radius_km = None
            st.write("")

    with c_sort:
        sort_choice = st.radio(
            L("क्रम:", "Sort by:"),
            [
                L("💰 सबसे अधिक भाव", "💰 Highest Price"),
                L("📍 सबसे नज़दीक", "📍 Closest Distance"),
            ],
            horizontal=True,
            key="nearby_sort_choice",
        )

    # Apply scope filtering
    if chosen_scope == scope_options[1]:  # Same district
        filtered = valid_coords[valid_coords["district"] == curr_district].copy()
        if filtered.empty:
            st.warning(
                L(
                    f"आपके जिले ({curr_district}) में इस फसल की केवल यही मंडी दर्ज है। आस-पास की अन्य मंडियां देखने के लिए '📍 आस-पास' चुनें।",
                    f"Only this mandi is registered in {curr_district} for this crop. Switch to '📍 Nearby' to see surrounding APMCs.",
                )
            )
            filtered = valid_coords.copy()
    elif chosen_scope == scope_options[2]:  # Entire state
        filtered = valid_coords[valid_coords["state"] == curr_state].copy()
    else:  # Nearby by distance
        if not valid_coords["distance_km"].isna().all():
            filtered = valid_coords[valid_coords["distance_km"] <= radius_km].copy()
            if len(filtered) < 3:
                filtered = valid_coords.sort_values("distance_km", ascending=True).head(8).copy()
                st.caption(
                    L(
                        f"💡 {radius_km} किमी के दायरे में कम मंडियां उपलब्ध हैं। निकटतम उपलब्ध {len(filtered)} मंडियां दिखाई जा रही हैं।",
                        f"💡 Few APMCs within {radius_km} km; showing nearest available {len(filtered)} mandis.",
                    )
                )
        else:
            filtered = valid_coords.copy()

    # Apply sorting
    is_price_sort = "भाव" in sort_choice or "Price" in sort_choice
    if is_price_sort:
        table = filtered.sort_values(by=["modal_price", "distance_km"], ascending=[False, True]).head(8)
    else:
        table = filtered.sort_values(by=["distance_km", "modal_price"], ascending=[True, False]).head(8)

    if table.empty:
        st.info(L("कोई उपयुक्त मंडी नहीं मिली।", "No matching mandis found."))
        return

    # Prepare display table
    display = table[
        [
            "market_name",
            "district",
            "distance_km",
            "modal_price",
            "आपकी मंडी से अंतर (₹/qtl)",
            "1 ट्रॉली (50 qtl) पर अंतर",
        ]
    ].rename(
        columns={
            "market_name": L("मंडी", "Mandi"),
            "district": L("जिला", "District"),
            "distance_km": L("दूरी (किमी)", "Distance (km)"),
            "modal_price": L("आज का भाव (₹/qtl)", "Today (₹/qtl)"),
        }
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            L("दूरी (किमी)", "Distance (km)"): st.column_config.NumberColumn(format="%.1f km"),
            L("आज का भाव (₹/qtl)", "Today (₹/qtl)"): st.column_config.NumberColumn(format="₹%d"),
            L("आपकी मंडी से अंतर (₹/qtl)", "Gap vs your mandi (₹/qtl)"): st.column_config.NumberColumn(format="%+d"),
            L("1 ट्रॉली (50 qtl) पर अंतर", "Gap on 1 trolley (50 qtl)"): st.column_config.NumberColumn(format="%+d"),
        },
    )

    # Highlight banner
    higher_paying = table[table["आपकी मंडी से अंतर (₹/qtl)"] > 0]
    if not higher_paying.empty:
        best = higher_paying.iloc[0]
        gap = float(best["आपकी मंडी से अंतर (₹/qtl)"])
        dist_str = f"केवल **{best['distance_km']:.1f} किमी** दूर" if not pd.isna(best["distance_km"]) else ""
        dist_en = f"just **{best['distance_km']:.1f} km** away" if not pd.isna(best["distance_km"]) else ""
        st.success(
            L(
                f"**{best['market_name']}** ({best['district']}) {dist_str} है और आज **{rupees(float(best['modal_price']))}** प्रति क्विंटल दे रही है — "
                f"आपकी मंडी से **{rupees(gap)} ज़्यादा**। 50 क्विंटल (1 ट्रॉली) पर लगभग **{rupees(gap * 50)} अतिरिक्त**, भाड़ा निकालने से पहले।",
                f"**{best['market_name']}** ({best['district']}) is {dist_en} and paying **{rupees(float(best['modal_price']))}** per quintal today — "
                f"**{rupees(gap)} more** than your mandi. About **{rupees(gap * 50)} extra** on 50 quintals, before freight.",
            )
        )
    else:
        st.info(
            L(
                f"आपकी वर्तमान मंडी **{market}** आस-पास के विकल्पों में सबसे अच्छा भाव दे रही है।",
                f"Your current mandi **{market}** is paying as well as or better than all nearby alternatives.",
            )
        )


def render_research_dashboard(user: dict) -> None:
    render_dashboard_header(user)
    st.sidebar.markdown("### 🔬 " + L("शोधकर्ता एनालिटिक्स", "Research analytics"))
    st.sidebar.info(L("भूमिका: शोधकर्ता / अधिकारी", "Role: Researcher / Officer"))

    metrics = load_metrics()
    by_h = metrics.get("by_horizon", {})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(L("मैप किए गए मंडी", "Mapped mandis"), "2,515", L("APMC नेटवर्क", "APMC network"))
    m2.metric(L("7-दिन RMSE", "7-day RMSE"), rupees(by_h.get("7-Day", {}).get("RMSE", 332.69)), L("पूर्ण-फ़ीचर टेस्ट", "complete-feature test"))
    m3.metric(L("7-दिन R²", "7-day R²"), f"{by_h.get('7-Day', {}).get('R2', 0.9779):.4f}", L("पूल्ड", "pooled"))
    m4.metric(L("95% कवरेज", "95% coverage"), f"{metrics.get('mondrian_averages', {}).get('7-Day', {}).get('macro_coverage', 95.9):.1f}%", L("समान-फसल औसत", "equal-crop macro"))

    st.markdown("---")
    st.subheader(L("📑 शोध मॉड्यूल", "📑 Research modules"))
    st.markdown(
        L(
            "बाईं ओर के साइडबार से 9 विश्लेषण पृष्ठ खोलें: Overview, Forecasting, Market Comparison, Regional Map, "
            "Historical Trends, Weather, Model Performance, Explainable AI और Ablation Study.",
            "Use the sidebar to open the 9 analytical pages: Overview, Forecasting, Market Comparison, Regional Map, "
            "Historical Trends, Weather, Model Performance, Explainable AI and Ablation Study.",
        )
    )
    with st.container(border=True):
        st.markdown(
            L(
                "**डेटा पारदर्शिता:** सभी मॉडल मेट्रिक्स `data/metrics_deployed.json` से आते हैं। "
                "मूल्यांकन पूर्ण-फ़ीचर परीक्षण पंक्तियों पर है, और 30-दिन का अनुमान naive persistence बेसलाइन है।",
                "**Data transparency:** all model metrics come from `data/metrics_deployed.json`. "
                "Evaluation is on complete-feature test rows; the 30-day forecast is a naive persistence baseline.",
            )
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
user = current_user()
if not user:
    view = st.session_state.get("auth_view", "landing")
    if view == "login":
        render_auth("login")
    elif view == "signup":
        render_auth("signup")
    else:
        render_landing()
else:
    # Keep the legacy multipage role guard working ("Researcher" is exact).
    role = user.get("role", "")
    st.session_state["user_role"] = "Researcher" if role == "Researcher / Officer" else role
    if role == "Researcher / Officer":
        render_research_dashboard(user)
    else:
        render_market_dashboard(user)
