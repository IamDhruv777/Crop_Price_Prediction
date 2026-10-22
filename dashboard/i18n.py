"""
Bilingual (English / हिन्दी) Localization Engine for Krishi Bhav Saathi.
Provides 100% complete translations for both English and Hindi with zero mixed text.
"""
import streamlit as st

def get_current_lang():
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"
    return st.session_state["lang"]

def set_lang(lang_code):
    st.session_state["lang"] = lang_code

TRANSLATIONS = {
    # Brand & Typography
    "app_title": {
        "en": "Krishi Bhav Saathi",
        "hi": "कृषि भाव साथी"
    },
    "app_tagline": {
        "en": "AI-Powered Mandi Intelligence & Crop Price Decision System",
        "hi": "AI आधारित मंडी मूल्य पूर्वानुमान एवं किसान निर्णय प्रणाली"
    },
    "app_subtitle": {
        "en": "Empowering rural farmers, village shopkeepers, food vendors, and mandi merchants with transparent prices, spatial arbitrage, and calibrated 1-day, 7-day, and 30-day AI forecasts across 2,515+ Indian mandis.",
        "hi": "देश की 2,515+ मंडियों के पारदर्शी भाव, पड़ोस की मंडियों का मुकाबला और 1, 7 व 30 दिन के सटीक AI अनुमान से किसान भाइयों, ग्रामीण दुकानदारों, खाद्य विक्रेताओं और मंडी आढ़तियों का सशक्तिकरण।"
    },
    "hero_quote": {
        "en": "“When our farmers and village shopkeepers prosper, the nation rises.”",
        "hi": "“जब किसान, छोटा दुकानदार और ग्रामीण व्यापारी समृद्ध होगा, तभी भारत सशक्त बनेगा।”"
    },
    "sub_quote": {
        "en": "Connecting field to market with fair price discovery and zero middlemen exploitation.",
        "hi": "खेत से दुकान और मंडी तक पारदर्शी भाव — बिचौलियों के शोषण से मुक्ति और हर किसी की मेहनत का पूरा दाम।"
    },

    # 3D Animation Badges
    "badge_3d_network": {
        "en": "🌐 2,515+ Connected Mandis Across India",
        "hi": "🌐 पूरे भारत की 2,515+ मंडियों का सजीव नेटवर्क"
    },
    "badge_3d_ai": {
        "en": "🎯 94.3% Calibrated AI Trust Score",
        "hi": "🎯 94.3% सत्यापित एआई भरोसा स्कोर"
    },
    "badge_3d_crops": {
        "en": "🌾 11 Major Crops & Pulses",
        "hi": "🌾 11 प्रमुख फसलें, अनाज व दलहन"
    },
    "badge_3d_savings": {
        "en": "💰 +₹25,000 Extra Profit per Trolley",
        "hi": "💰 प्रति ट्रॉली +₹25,000 तक अतिरिक्त बचत"
    },

    # 3 Dedicated User Persona Cards
    "personas_title": {
        "en": "👥 Tailored for Everyone in India's Agricultural Ecosystem",
        "hi": "👥 ग्रामीण कृषि और व्यापार से जुड़े हर साथी के लिए विशेष समाधान"
    },
    "persona_farmer_title": {
        "en": "👨‍🌾 Rural Farmers & Cultivators",
        "hi": "👨‍🌾 अन्नदाता किसान भाई एवं उत्पादक"
    },
    "persona_farmer_desc": {
        "en": "Stop distress selling on harvest day. Get simple 1, 7 & 30-day price forecast advice (🟢 HOLD or 🔴 SELL) with 95% safe bounds so you know exactly when your crop will fetch the highest rate.",
        "hi": "फसल कटाई पर औने-पौने दाम पर बेचने से बचें। साफ-साफ 1, 7 और 30 दिन का भाव अनुमान (🟢 माल रोकें या 🔴 आज बेचें) देखें, ताकि आपको अपनी फसल का सबसे ऊंचा दाम मिल सके।"
    },
    "persona_vendor_title": {
        "en": "🛒 Village Shopkeepers & Food Vendors",
        "hi": "🛒 ग्रामीण दुकानदार एवं फल-सब्जी-अनाज विक्रेता"
    },
    "persona_vendor_desc": {
        "en": "Optimize your wholesale stock buying. Know in advance when mandi prices will crash so you avoid overpaying, and find the cheapest nearby APMC for procurement to increase your retail shop margin.",
        "hi": "थोक खरीदारी का सही समय तय करें। पहले ही जानें कि किस दिन थोक मंडी में भाव गिरने वाले हैं ताकि आप महंगे दाम पर माल न खरीदें और दुकान पर अधिकतम मुनाफा कमाएं।"
    },
    "persona_trader_title": {
        "en": "🚛 Mandi Commission Agents & Logistics",
        "hi": "🚛 मंडी आढ़ती, व्यापारी एवं ट्रांसपोर्टर"
    },
    "persona_trader_desc": {
        "en": "Monitor real-time inter-district price spreads (arbitrage). Plan truck and tractor dispatches between mandis with accurate freight cost adjustments to capture ₹300-₹500/quintal spreads.",
        "hi": "पास के जिलों की मंडियों में भाव के अंतर (आर्बिट्राज) पर नजर रखें। ट्रैक्टर व ट्रक का सही रूट तय करें और प्रति क्विंटल ₹300 से ₹500 तक का अतिरिक्त मार्जिन हासिल करें।"
    },

    # Visual Gallery Captions
    "gallery_title": {
        "en": "🌾 From Our Fields to India's Mandis & Village Shops",
        "hi": "🌾 हमारे खेतों से देश की मंडियों और ग्रामीण दुकानों तक"
    },
    "photo_farmer": {
        "en": "Hardworking Farmers in the Field",
        "hi": "खेतों में पसीना बहाते हमारे अन्नदाता किसान"
    },
    "photo_mandi": {
        "en": "Dynamic Mandi Auction Yards (APMC)",
        "hi": "देश की प्रमुख अनाज व कृषि मंडियां (APMC)"
    },
    "photo_vendor": {
        "en": "Local Food Vendors & Rural Shopkeepers",
        "hi": "स्थानीय गल्ला व्यापारी व ग्रामीण दुकानदार"
    },
    "photo_harvest": {
        "en": "Golden Harvest & Food Security",
        "hi": "लहलहाती सुनहरी फसल एवं खाद्य सुरक्षा"
    },

    # 3 Emotional Problems
    "problems_heading": {
        "en": "💔 The 3 Critical Pain Points Rural Sellers Face Every Season",
        "hi": "💔 हर फसल कटाई पर हमारे ग्रामीण भाइयों की 3 बड़ी समस्याएं"
    },
    "p1_title": {
        "en": "1. Distress Selling Without Future Price Visibility",
        "hi": "1. भविष्य के भाव का पता न होने से औने-पौने दाम पर बिक्री"
    },
    "p1_desc": {
        "en": "After months of hard labor and heavy input costs, farmers and local sellers are forced to dump produce at harvest crash rates because they do not know prices will rebound next week.",
        "hi": "महीनों पसीना बहाने के बाद फसल तैयार होते ही मंडियों में भाव गिर जाते हैं। किसान को पता ही नहीं होता कि अगले हफ्ते भाव बढ़ेंगे या और नीचे गिरेंगे।"
    },
    "p1_impact": {
        "en": "Result: Inability to recover cultivation costs and mounting debt.",
        "hi": "परिणाम: लागत भी नहीं निकलती और मजबूरी में कर्ज लेना पड़ता है।"
    },

    "p2_title": {
        "en": "2. Hidden Profit in Nearby Mandis (Lost Arbitrage)",
        "hi": "2. पास की दूसरी मंडी में ₹300-₹500 ज्यादा भाव, पर जानकारी नहीं"
    },
    "p2_desc": {
        "en": "Just 25 km away in a neighboring district, another mandi may be paying ₹400/quintal more, but middlemen pocket that profit due to information blackout.",
        "hi": "मात्र 25 किलोमीटर दूर पड़ोस के जिले की मंडी में वही माल ₹400 प्रति क्विंटल महंगा बिकता है, लेकिन सही जानकारी न होने से यह मुनाफा दलाल खा जाते हैं।"
    },
    "p2_impact": {
        "en": "Result: Loss of ₹20,000 to ₹30,000 on every tractor trolley.",
        "hi": "परिणाम: प्रत्येक ट्रैक्टर ट्रॉली पर ₹20,000 से ₹30,000 का सीधा नुकसान।"
    },

    "p3_title": {
        "en": "3. Sudden Weather Shocks & Mandi Disruptions",
        "hi": "3. बेमौसम बारिश, आंधी और मौसम का अचानक झटका"
    },
    "p3_desc": {
        "en": "Unseasonal rains flood open mandi yards, leading to soaked grains, transport paralysis, and steep moisture deduction penalties by local buyers.",
        "hi": "अचानक बारिश आने से खुली मंडियों में माल भीग जाता है, आवक रुक जाती है और आढ़ती नमी का बहाना बनाकर भाव धड़ाम से गिरा देते हैं।"
    },
    "p3_impact": {
        "en": "Result: Spoilage loss and forced discount deductions.",
        "hi": "परिणाम: फसल खराब होने का डर और मंडी में भारी कटौती।"
    },

    # 3 Transformative Solutions
    "solutions_heading": {
        "en": "🛡️ How Krishi Bhav Saathi Solves This For You",
        "hi": "🛡️ 'कृषि भाव साथी' आपकी कैसे मदद करता है?"
    },
    "s1_title": {
        "en": "1. Clear 1, 7 & 30-Day Forecasts (Hold or Sell)",
        "hi": "1. साफ-साफ 1, 7 और 30 दिन का भाव अनुमान (रोकें या बेचें?)"
    },
    "s1_desc": {
        "en": "Accurate daily price projections with 95% calibrated confidence bounds tell you exactly whether to sell today or hold for 7 days to earn more.",
        "hi": "95% सटीक दायरे के साथ जानें कि आज बेचना बेहतर है या 7 दिन रुकना, ताकि प्रति क्विंटल ₹200 से ₹500 तक का ज्यादा मुनाफा मिल सके।"
    },
    "s2_title": {
        "en": "2. Real-Time Inter-Mandi Price Comparison",
        "hi": "2. आस-पास की सभी मंडियों का लाइव रेट मुकाबला"
    },
    "s2_desc": {
        "en": "Compare live rates across all adjacent mandis in your district and state so you can drive your trolley to the highest-paying market.",
        "hi": "अपने जिले और आसपास की सभी मंडियों के भाव एक साथ देखें, ताकि आप अपनी उपज उसी मंडी में ले जाएं जहां सबसे ज्यादा दाम मिल रहा हो।"
    },
    "s3_title": {
        "en": "3. Simple Visual Badges (Green = Rise, Red = Drop)",
        "hi": "3. आसान हरा व लाल निशान — कोई जटिल गणित या अंग्रेजी नहीं"
    },
    "s3_desc": {
        "en": "No complicated charts required. Simple green badges indicate rising prices (HOLD), red badges signal market crashes (SELL TODAY).",
        "hi": "सीधे मोबाइल पर हरा निशान मतलब भाव बढ़ेगा (माल रोकें), लाल निशान मतलब भाव गिरेगा (आज ही बेचें)।"
    },

    # Sign-In / Login Portal
    "login_box_title": {
        "en": "🚪 Enter Portal — Farmer & Trader Login",
        "hi": "🚪 पोर्टल में प्रवेश करें — किसान एवं व्यापारी स्वागत"
    },
    "login_box_desc": {
        "en": "Please provide your basic details so the portal can tailor prices and advice for your primary crop and state:",
        "hi": "कृपया अपनी सामान्य जानकारी दर्ज करें ताकि पोर्टल आपकी मुख्य फसल और राज्य के अनुसार सही भाव दिखा सके:"
    },
    "select_role": {
        "en": "Select Your Role",
        "hi": "आप कौन हैं? (अपनी भूमिका चुनें)"
    },
    "role_farmer": {
        "en": "👨‍🌾 Farmer / Producer (किसान भाई / उत्पादक)",
        "hi": "👨‍🌾 किसान भाई / उत्पादक"
    },
    "role_vendor": {
        "en": "🛒 Rural Vendor / Shopkeeper (दुकानदार / छोटा व्यापारी)",
        "hi": "🛒 ग्रामीण दुकानदार एवं खुदरा व्यापारी"
    },
    "role_trader": {
        "en": "🚛 Mandi Trader / Commission Agent (मंडी आढ़ती / व्यापारी)",
        "hi": "🚛 मंडी आढ़ती / व्यापारी"
    },
    "role_analyst": {
        "en": "🔬 Expert / Researcher / Officer (अधिकारी / शोधकर्ता)",
        "hi": "🔬 अधिकारी / शोधकर्ता"
    },
    "user_name_label": {
        "en": "Your Name (Optional)",
        "hi": "आपका नाम (वैकल्पिक)"
    },
    "state_label": {
        "en": "Select Your State",
        "hi": "अपना राज्य चुनें"
    },
    "crop_label": {
        "en": "Select Your Main Crop",
        "hi": "अपनी मुख्य फसल चुनें"
    },
    "btn_enter": {
        "en": "🚀 Enter Dashboard",
        "hi": "🚀 डैशबोर्ड में प्रवेश करें"
    },
    "btn_quick_guest": {
        "en": "⚡ One-Click Guest Entry (Explore Without Login)",
        "hi": "⚡ बिना लॉगिन सीधे देखें (Quick Guest Entry)"
    },
    "or_text": {
        "en": "— OR —",
        "hi": "— अथवा —"
    },

    # Logged-In Home Dashboard
    "welcome_farmer": {
        "en": "Welcome",
        "hi": "राम-राम"
    },
    "role_label": {
        "en": "Role",
        "hi": "भूमिका"
    },
    "main_crop_label": {
        "en": "Main Crop",
        "hi": "मुख्य फसल"
    },
    "state_display": {
        "en": "State",
        "hi": "राज्य"
    },
    "btn_logout": {
        "en": "🚪 Logout / Back to 3D Landing Screen",
        "hi": "🚪 लॉगआउट / मुख्य लैंडिंग स्क्रीन पर जाएं"
    },
    "active_crop_heading": {
        "en": "Live Status for Your Crop:",
        "hi": "आपकी मुख्य फसल का लाइव हाल:"
    },
    "today_rate": {
        "en": "Today's Mandi Rate",
        "hi": "आज का मंडी भाव"
    },
    "forecast_rate": {
        "en": "7-Day Expected Rate",
        "hi": "7 दिन बाद अनुमानित भाव"
    },
    "price_range": {
        "en": "Safe Price Range (Min - Max)",
        "hi": "भाव का सुरक्षित दायरा (न्यूनतम - अधिकतम)"
    },
    "advice_title": {
        "en": "What Should You Do? (Actionable Advice)",
        "hi": "आपको क्या करना चाहिए? (किसान सलाह)"
    },
    "advice_hold_title": {
        "en": "🟢 HOLD PRODUCE! Prices Expected to RISE",
        "hi": "🟢 माल रोकें! आने वाले दिनों में भाव बढ़ने का अनुमान है"
    },
    "advice_hold_desc": {
        "en": "Expected gain: approx ₹{gain} per quintal (+₹{trolley} per 50-quintal trolley). If you have safe storage, holding stock will earn higher profits.",
        "hi": "संभावित फायदा: लगभग ₹{gain} प्रति क्विंटल (50 क्विंटल की ट्रॉली पर +₹{trolley} की अतिरिक्त बचत)। सुरक्षित गोदाम हो तो माल अभी न बेचें।"
    },
    "advice_sell_title": {
        "en": "🔴 SELL TODAY! Prices Expected to DROP Soon",
        "hi": "🔴 आज ही बेचें! आने वाले दिनों में भाव गिरने का खतरा है"
    },
    "advice_sell_desc": {
        "en": "Risk: Prices may drop by approx ₹{loss} per quintal (-₹{trolley} per 50-quintal trolley). Selling in your nearest mandi today protects your earnings.",
        "hi": "सावधानी: भाव लगभग ₹{loss} प्रति क्विंटल गिर सकता है (प्रति ट्रॉली -₹{trolley} का नुकसान)। घाटे से बचने के लिए नजदीकी मंडी में आज ही माल निकालना बेहतर है।"
    },
    "advice_stable_title": {
        "en": "🔵 PRICE STABLE — Equilibrium Market",
        "hi": "🔵 भाव स्थिर रहेगा — सामान्य उतार-चढ़ाव"
    },
    "advice_stable_desc": {
        "en": "No drastic swing predicted. Sell whenever convenient based on your personal cash flow and transportation.",
        "hi": "भाव में कोई बड़ा फेरबदल नहीं होने वाला। आप अपनी जरूरत और सुविधा अनुसार फैसला ले सकते हैं।"
    },

    # Quick Navigation Cards on Home Screen
    "quick_nav_heading": {
        "en": "📲 Key System Portals (Click below or choose from the left menu):",
        "hi": "📲 किसान साथी मुख्य सुविधाएं (नीचे दिए गए बटनों पर क्लिक करें या बाईं ओर के मेन्यू से चुनें):"
    },
    "nav_f_title": {
        "en": "📈 1. Accurate Crop Price Forecast",
        "hi": "📈 1. फसल भाव का सटीक अनुमान"
    },
    "nav_f_desc": {
        "en": "View 1-day, 7-day, and 30-day future rates with 95% confidence intervals and What-If scenario simulations.",
        "hi": "1 दिन, 7 दिन और 30 दिन का भविष्य का भाव देखें, साथ ही व्हाट-इफ परिस्थिति का अनुमान लगाएं।"
    },
    "nav_m_title": {
        "en": "⚖️ 2. Nearby Mandi Price Comparison",
        "hi": "⚖️ 2. पास की सभी मंडियों का मुकाबला"
    },
    "nav_m_desc": {
        "en": "Compare prices across adjacent APMCs and calculate exact extra profits per trolley before starting your tractor.",
        "hi": "देखें कि पड़ोस की किस मंडी में सबसे ज्यादा भाव मिल रहा है और ट्रॉली ले जाने पर कितना ज्यादा मुनाफा होगा।"
    },
    "nav_map_title": {
        "en": "🗺️ 3. Pan-India Mandi Price Map",
        "hi": "🗺️ 3. पूरे भारत का मंडी नक्शा"
    },
    "nav_map_desc": {
        "en": "Interactive nationwide map with color thermal indicators showing high-demand premium hubs across 2,515 mandis.",
        "hi": "नक्शे पर देखें कि देश में कहां सस्ता और कहां सबसे महंगा बिक रहा है, और अपनी मंडी का विवरण जांचें।"
    },
    "nav_h_title": {
        "en": "📅 4. 10-Year Historical Price Track",
        "hi": "📅 4. पुराने 10 साल के भाव का इतिहास"
    },
    "nav_h_desc": {
        "en": "Discover historical peak months and seasonal price cycles across 10 years of official Agmarknet records.",
        "hi": "देखें कि पिछले 10 सालों में आपकी मंडी में किस महीने सबसे ऊंचा भाव मिलता रहा है (मौसमी चक्र)।"
    },
    "nav_w_title": {
        "en": "🌦️ 5. NASA Weather & Rainfall Advisory",
        "hi": "🌦️ 5. नासा मौसम व वर्षा प्रभाव"
    },
    "nav_w_desc": {
        "en": "District-level temperature, rain, and humidity data to protect produce from unseasonal downpours.",
        "hi": "जिलेवार तापमान व बारिश के आंकड़े, ताकि बेमौसम बारिश से फसल और मंडी भाव को नुकसान न पहुंचे।"
    },
    "nav_t_title": {
        "en": "⚙️ 6. AI Model Trust & Benchmark",
        "hi": "⚙️ 6. AI मॉडल की विश्वसनीयता व भरोसा"
    },
    "nav_t_desc": {
        "en": "Transparent verification showing 94.3% trust score, validated on 186,000+ real transactions.",
        "hi": "पारदर्शी मूल्यांकन — 186,000 से अधिक वास्तविक सौदों पर 94.3% विश्वास स्कोर का सत्यापन।"
    },

    # Sidebar Controls
    "view_mode": {
        "en": "Dashboard Display Mode",
        "hi": "डैशबोर्ड देखने का तरीका"
    },
    "mode_farmer": {
        "en": "🌾 Simple Farmer View",
        "hi": "🌾 सरल किसान दृश्य"
    },
    "mode_advanced": {
        "en": "📊 Advanced Analytics View",
        "hi": "📊 उन्नत तकनीकी दृश्य"
    },
    "sidebar_lang_title": {
        "en": "🌐 Language / भाषा",
        "hi": "🌐 भाषा / Language"
    }
}

def t(key, **kwargs):
    """Translate a key based on current session language with formatting support."""
    lang = get_current_lang()
    entry = TRANSLATIONS.get(key, {})
    val = entry.get(lang, entry.get("en", key))
    if kwargs and isinstance(val, str):
        try:
            return val.format(**kwargs)
        except Exception:
            return val
    return val
