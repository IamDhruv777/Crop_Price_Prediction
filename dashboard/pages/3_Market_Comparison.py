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


PREDS_PATH = Path("data/deployed_test_predictions.csv")

if not PREDS_PATH.exists():
    st.error("Predictions file not found at `data/deployed_test_predictions.csv`.")
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

pred_col = f'predicted_price_{h_suffix}' if f'predicted_price_{h_suffix}' in preds_df.columns else ('predicted_price_7d' if 'predicted_price_7d' in preds_df.columns else 'modal_price')
act_col = f'target_price_{h_suffix}' if f'target_price_{h_suffix}' in preds_df.columns else (f'actual_price_{h_suffix}' if f'actual_price_{h_suffix}' in preds_df.columns else ('modal_price' if 'modal_price' in preds_df.columns else None))

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
    # Aggregate real metrics for selected markets dynamically
    comp_list = []
    for m in selected_markets:
        sub = crop_df[crop_df['market'] == m]
        if not sub.empty:
            cur_price = float(sub['modal_price'].iloc[-1])
            pred_h = float(sub[pred_col].iloc[-1]) if pred_col in sub.columns else cur_price
            delta_val = float(sub[f'uncertainty_delta_{h_suffix}'].iloc[-1]) if f'uncertainty_delta_{h_suffix}' in sub.columns else (cur_price * 0.05)
            spread = pred_h - cur_price
            drift_pct = (spread / cur_price) * 100 if cur_price > 0 else 0.0
            state = sub['state'].iloc[-1] if 'state' in sub.columns else "N/A"
            district = sub['district'].iloc[-1] if 'district' in sub.columns else "N/A"

            comp_list.append({
                "market": m,
                "state": state,
                "district": district,
                "current_price": round(cur_price, 2),
                "forecast_price": round(pred_h, 2),
                "spread": round(spread, 2),
                "drift_pct": round(drift_pct, 2),
                "delta": round(delta_val, 2),
                "advice": "🟢 माल रोकें (Hold)" if drift_pct >= 2.0 else ("🔴 आज बेचें (Sell)" if drift_pct <= -2.0 else "🔵 स्थिर (Stable)")
            })
    comp_df = pd.DataFrame(comp_list)

    if comp_df.empty:
        st.warning("चुनी हुई मंडियों के लिए डेटा उपलब्ध नहीं है।" if is_hi else "No data available for selected mandis.")
        st.stop()

    # Sort descending by current price
    comp_df_sorted = comp_df.sort_values(by='current_price', ascending=False).reset_index(drop=True)
    best_mandi = comp_df_sorted.iloc[0]

    # -------------------------------------------------------------
    # MODE 1: SIMPLE FARMER VIEW (सरल किसान दृश्य)
    # -------------------------------------------------------------
    if is_farmer_view:
        st.subheader(f"🌾 **{selected_crop}** — " + ("किसान मंडी भाव मुकाबला एवं सर्वश्रेष्ठ मंडी" if is_hi else "Farmer Mandi Comparison & Top APMC"))

        # Top 3 Recommended Mandi Cards
        st.markdown("#### 🏆 " + ("शीर्ष अनुशंसित मंडियां (जहाँ सबसे ज्यादा भाव मिल रहा है):" if is_hi else "Top Recommended Mandis by Price:"))
        t_cols = st.columns(min(3, len(comp_df_sorted)))
        medals = ["🥇 1. " + ("सर्वश्रेष्ठ मंडी" if is_hi else "Top Mandi"), "🥈 2. " + ("दूसरी मंडी" if is_hi else "2nd Best"), "🥉 3. " + ("तीसरी मंडी" if is_hi else "3rd Best")]
        
        for i, col in enumerate(t_cols):
            m_row = comp_df_sorted.iloc[i]
            with col:
                st.metric(
                    f"{medals[i]}: {m_row['market']}",
                    f"₹ {m_row['current_price']:,.0f} / qtl",
                    f"{h_label} बाद: ₹{m_row['forecast_price']:,.0f}"
                )

        st.markdown("---")

        # Interactive Trolley Profit Arbitrage Calculator
        with st.container(border=True):
            st.markdown("### 🚜 " + ("ट्रॉली मुनाफा कैलकुलेटर (Trolley Profit Arbitrage):" if is_hi else "Trolley Profit Arbitrage Calculator:"))
            
            calc_c1, calc_c2, calc_c3 = st.columns(3)
            with calc_c1:
                local_mandi = st.selectbox(
                    "अपनी नजदीकी / वर्तमान मंडी चुनें:" if is_hi else "Your Local Reference Mandi:",
                    selected_markets,
                    index=len(selected_markets)-1 if len(selected_markets) > 1 else 0,
                    key="farmer_calc_local_mandi"
                )
            with calc_c2:
                trolley_qtl = st.slider(
                    "ट्रॉली का वजन (क्विंटल)" if is_hi else "Trolley Load (Quintals)",
                    min_value=20, max_value=100, value=50, step=5
                )
            with calc_c3:
                diesel_cost = st.number_input(
                    "अनुमानित डीजल / भाड़ा खर्च (₹)" if is_hi else "Round-Trip Diesel Expense (₹)",
                    min_value=0, max_value=15000, value=2500, step=500
                )

            local_price = comp_df[comp_df['market'] == local_mandi]['current_price'].iloc[0]
            best_price = best_mandi['current_price']
            price_diff_per_qtl = best_price - local_price
            gross_gain = price_diff_per_qtl * trolley_qtl
            net_gain = gross_gain - diesel_cost

            if local_mandi == best_mandi['market']:
                st.info(
                    f"🌟 **" + (f"बधाई! आपकी चुनी हुई मंडी **{local_mandi}** ही इस क्षेत्र में सबसे अधिक भाव दे रही है।" if is_hi else f"Excellent! Your local mandi **{local_mandi}** is already the highest-paying market.") + f"**\n\n" +
                    (f"यहाँ आज का भाव **₹{best_price:,.0f} प्रति क्विंटल** है। आपको माल किसी अन्य मंडी में ले जाने की आवश्यकता नहीं है।"
                     if is_hi else
                     f"Current price is **₹{best_price:,.0f} / quintal**. No transit required to other mandis.")
                )
            elif net_gain > 0:
                st.success(
                    f"💰 **" + (f"माल **{best_mandi['market']}** ले जाने पर शुद्ध लाभ: +₹{net_gain:,.0f}!" if is_hi else f"Net Arbitrage Profit at **{best_mandi['market']}**: +₹{net_gain:,.0f}!") + f"**\n\n" +
                    (f"• **भाव का अंतर:** **{best_mandi['market']}** में **{local_mandi}** की तुलना में **₹{price_diff_per_qtl:,.0f} प्रति क्विंटल** ज्यादा भाव मिल रहा है।\n"
                     f"• **कुल अतिरिक्त बिक्री ({trolley_qtl} क्विंटल):** ₹{gross_gain:,.0f}\n"
                     f"• **डीजल खर्च (₹{diesel_cost:,.0f}) काटने के बाद शुद्ध बचत:** **+₹{net_gain:,.0f}** आपके हाथ में!\n\n"
                     f"👉 **किसान सलाह:** अपनी फसल **{best_mandi['market']}** ले जाना बहुत फायदेमंद रहेगा।"
                     if is_hi else
                     f"• **Price Spread:** **{best_mandi['market']}** offers **₹{price_diff_per_qtl:,.0f} / qtl** higher than **{local_mandi}**.\n"
                     f"• **Gross Trolley Gain ({trolley_qtl} qtl):** ₹{gross_gain:,.0f}\n"
                     f"• **Net Profit after Diesel (₹{diesel_cost:,.0f}):** **+₹{net_gain:,.0f} net savings**.\n\n"
                     f"👉 **Recommendation:** Transporting to **{best_mandi['market']}** delivers verified profit.")
                )
            else:
                st.warning(
                    f"⚠️ **" + (f"स्थानीय मंडी में ही बेचना बेहतर:" if is_hi else f"Local Mandi Preferred:") + f"**\n\n" +
                    (f"• भाव का अंतर केवल **₹{price_diff_per_qtl:,.0f} प्रति क्विंटल** (कुल ₹{gross_gain:,.0f}) है, जबकि आने-जाने का डीजल खर्च **₹{diesel_cost:,.0f}** है।\n"
                     f"• दूर की मंडी ले जाने से **₹{abs(net_gain):,.0f} का नुकसान** हो सकता है। अतः माल स्थानीय मंडी **{local_mandi}** में ही बेचें।"
                     if is_hi else
                     f"• Gross gain (₹{gross_gain:,.0f}) is insufficient to cover transport cost (₹{diesel_cost:,.0f}), leading to a net deficit of ₹{abs(net_gain):,.0f}.\n"
                     f"• Selling locally at **{local_mandi}** is the economically optimal decision.")
                )

        st.markdown("---")

        # Visual Mandi Rate Bar Chart (Sorted, Green for #1)
        st.subheader("📊 " + ("मंडियों का सीधा भाव मुकाबला (हरा = नंबर 1 मंडी)" if is_hi else "Mandi Rate Leaderboard (Green = Best Rate)"))
        
        bar_colors = ["#2ca02c" if m == best_mandi['market'] else "#1f77b4" for m in comp_df_sorted['market']]
        fig_farmer_bar = px.bar(
            comp_df_sorted,
            x='market',
            y='current_price',
            color='market',
            color_discrete_sequence=bar_colors,
            text='current_price',
            title=(f"{selected_crop} — विभिन्न मंडियों में आज का भाव (रुपये प्रति क्विंटल)" if is_hi else f"{selected_crop} — Current Wholesale Price by Mandi (₹/Quintal)"),
            labels={"market": "मंडी" if is_hi else "Mandi", "current_price": "आज का भाव (₹/qtl)" if is_hi else "Current Price (₹/qtl)"}
        )
        fig_farmer_bar.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside', showlegend=False)
        fig_farmer_bar.update_layout(xaxis_tickangle=-25, height=430)
        st.plotly_chart(fig_farmer_bar, use_container_width=True)

        # Simplified Farmer Table
        st.markdown("---")
        st.subheader("📋 " + ("मंडी निर्णय तालिका (कहाँ बेचना सबसे सही?)" if is_hi else "Mandi Decision Table"))
        
        farmer_table = []
        for _, row in comp_df_sorted.iterrows():
            diff_from_local = row['current_price'] - local_price
            trolley_diff = diff_from_local * trolley_qtl
            decision_badge = "🥇 सर्वश्रेष्ठ विकल्प" if row['market'] == best_mandi['market'] else ("🟢 फायदेमंद" if trolley_diff > diesel_cost else ("🔴 घाटा (डीजल ज्यादा)" if trolley_diff < 0 else "🔵 सामान्य"))
            if not is_hi:
                decision_badge = "🥇 Best Option" if row['market'] == best_mandi['market'] else ("🟢 Profitable" if trolley_diff > diesel_cost else ("🔴 Deficit" if trolley_diff < 0 else "🔵 Neutral"))
            farmer_table.append({
                "मंडी का नाम" if is_hi else "Mandi": row['market'],
                "जिला / राज्य" if is_hi else "District / State": f"{row['district']}, {row['state']}",
                "आज का भाव (₹/qtl)" if is_hi else "Today's Rate (₹/qtl)": int(round(row['current_price'])),
                f"{h_label} बाद अनुमान" if is_hi else f"{h_label} Forecast": int(round(row['forecast_price'])),
                f"1 ट्रॉली ({trolley_qtl}q) पर अंतर (₹)" if is_hi else f"Spread vs Local ({trolley_qtl}q)": f"{'+' if trolley_diff >= 0 else ''}{int(round(trolley_diff)):,}",
                "सलाह" if is_hi else "Decision": decision_badge
            })
        st.dataframe(pd.DataFrame(farmer_table), use_container_width=True)

    # -------------------------------------------------------------
    # MODE 2: ADVANCED ANALYTICS / RESEARCHER VIEW (उन्नत तकनीकी दृश्य)
    # -------------------------------------------------------------
    else:
        st.subheader(f"⚖️ Cross-Mandi Spatial Arbitrage & Dispersion Analysis — {selected_crop} ({h_label})")

        # Spatial Dispersion KPIs
        mean_p = comp_df['current_price'].mean()
        max_p = comp_df['current_price'].max()
        min_p = comp_df['current_price'].min()
        spread_p = max_p - min_p
        spread_pct = (spread_p / min_p) * 100 if min_p > 0 else 0.0
        std_p = comp_df['current_price'].std() if len(comp_df) > 1 else 0.0
        spatial_cv = (std_p / mean_p) * 100 if mean_p > 0 else 0.0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Spatial Mean", f"₹ {mean_p:,.2f}", "Across Mandis")
        c2.metric("Max Price Spread", f"₹ {spread_p:,.2f}", f"{spread_pct:.1f}% Max Divergence")
        c3.metric("Standard Deviation", f"±₹ {std_p:,.2f}", "Spatial Dispersion")
        c4.metric("Spatial CV", f"{spatial_cv:.1f}%", "Market Heterogeneity")
        c5.metric("Mandis Analyzed", f"{len(comp_df)}", f"{selected_crop}")

        st.markdown("---")
        # Full Analytical Table
        st.subheader("📋 Comprehensive Cross-Mandi Forecast & Arbitrage Matrix")
        
        adv_display_df = comp_df[[
            "market", "district", "state", "current_price", "forecast_price", "spread", "drift_pct", "delta", "advice"
        ]].copy()
        
        adv_display_df.columns = [
            "Mandi", "District", "State", "Current Price (₹/qtl)", f"{h_label} Forecast (₹/qtl)",
            "Price Drift (₹)", "Drift (%)", "Conformal Delta (±₹)", "Directional Signal"
        ]

        st.dataframe(
            adv_display_df.style
            .highlight_max(subset=["Current Price (₹/qtl)", f"{h_label} Forecast (₹/qtl)"], color="#d4edda")
            .highlight_min(subset=["Current Price (₹/qtl)"], color="#f8d7da"),
            use_container_width=True
        )

        st.markdown("---")
        # Comparative Grouped Bar Chart
        st.subheader(f"📊 Multi-Horizon Price Trajectory: Current vs {h_label} Forecast")
        
        chart_rows = []
        for _, row in comp_df.iterrows():
            chart_rows.append({"Mandi": row['market'], "Type": "Current Price", "Price": row['current_price']})
            chart_rows.append({"Mandi": row['market'], "Type": f"{h_label} Forecast", "Price": row['forecast_price']})
        grouped_chart_df = pd.DataFrame(chart_rows)

        fig_adv_bar = px.bar(
            grouped_chart_df,
            x="Mandi",
            y="Price",
            color="Type",
            barmode="group",
            text="Price",
            title=f"{selected_crop} — Current Price vs {h_label} Horizon Forecast Across APMCs",
            labels={"Price": "Price (₹/Quintal)", "Mandi": "Mandi / APMC", "Type": "Horizon"}
        )
        fig_adv_bar.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
        fig_adv_bar.update_layout(xaxis_tickangle=-25, height=460)
        st.plotly_chart(fig_adv_bar, use_container_width=True)

        st.markdown("---")
        # Spatial Premia / Discount Chart (Deviation from regional mean)
        st.subheader("📍 Spatial Price Premia / Discount (Deviation from Regional Mean)")
        comp_df['Mean_Deviation'] = comp_df['current_price'] - mean_p
        comp_df['Deviation_Type'] = np.where(comp_df['Mean_Deviation'] >= 0, "Premium (+)", "Discount (-)")

        fig_dev = px.bar(
            comp_df,
            x="market",
            y="Mean_Deviation",
            color="Deviation_Type",
            color_discrete_map={"Premium (+)": "#2ca02c", "Discount (-)": "#d62728"},
            text="Mean_Deviation",
            title=f"{selected_crop} — APMC Basis Premia / Discount vs Regional Spatial Mean (₹{mean_p:,.2f})",
            labels={"market": "Mandi", "Mean_Deviation": "Basis Spread from Mean (₹/qtl)", "Deviation_Type": "Pricing Tier"}
        )
        fig_dev.update_traces(texttemplate='%{text:+,.1f} ₹/qtl', textposition='outside')
        fig_dev.update_layout(xaxis_tickangle=-25, height=430)
        st.plotly_chart(fig_dev, use_container_width=True)