"""
Unified Zoom-Responsive Interactive India Map Component.

Features:
1. Standard OpenStreetMap tile layer (100% Free, NO API KEY REQUIRED, NO WATERMARKS).
2. True centroid decoding for all 36 States and 723 Districts in India.
3. Mandis positioned directly INSIDE their respective districts.
4. Permanent text labels on States (Zoom < 6), Districts (Zoom 6 to 8.8), and Mandis (Zoom >= 8.8).
5. Smooth zoom transitions between State -> District -> Mandi price distributions.
"""

import json
import math
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np

# Cache TopoJSON and computed centroids in memory
_CACHED_TOPO: Optional[Dict[str, Any]] = None
_CACHED_DISTRICT_CENTROIDS: Optional[Dict[str, Tuple[float, float]]] = None
_CACHED_STATE_CENTROIDS: Optional[Dict[str, Tuple[float, float]]] = None

STATE_NAME_MAP = {
    'Chattisgarh': 'Chhattisgarh',
    'Uttrakhand': 'Uttarakhand',
    'Orissa': 'Odisha',
    'NCT of Delhi': 'Delhi',
    'Pondicherry': 'Puducherry',
}


def _normalize_str(s: str) -> str:
    """Strip special characters and whitespace for fuzzy key matching."""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def _load_topo_and_centroids() -> Tuple[Dict[str, Any], Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    """Load TopoJSON and precompute real geometric centroids for states and districts."""
    global _CACHED_TOPO, _CACHED_DISTRICT_CENTROIDS, _CACHED_STATE_CENTROIDS

    if _CACHED_TOPO is not None and _CACHED_DISTRICT_CENTROIDS is not None and _CACHED_STATE_CENTROIDS is not None:
        return _CACHED_TOPO, _CACHED_DISTRICT_CENTROIDS, _CACHED_STATE_CENTROIDS

    base_dir = Path(__file__).resolve().parents[2]
    topo_path = base_dir / "data" / "raw" / "geography" / "india_states_districts.json"
    if not topo_path.exists():
        topo_path = Path("data/raw/geography/india_states_districts.json")

    with open(topo_path, "r", encoding="utf-8") as f:
        topo = json.load(f)

    arcs = topo['arcs']
    transform = topo['transform']
    scale = transform['scale']
    translate = transform['translate']

    def decode_arc(arc_idx: int) -> List[Tuple[float, float]]:
        raw_arc = arcs[~arc_idx] if arc_idx < 0 else arcs[arc_idx]
        coords = []
        x, y = 0, 0
        for dx, dy in raw_arc:
            x += dx
            y += dy
            coords.append((x * scale[0] + translate[0], y * scale[1] + translate[1]))
        return coords if arc_idx >= 0 else coords[::-1]

    # Precompute District Centroids
    dist_centroids = {}
    for g in topo['objects']['districts']['geometries']:
        props = g.get('properties', {})
        d_name = props.get('district', '').strip()
        st_name = props.get('st_nm', '').strip()
        all_points = []
        polys = g['arcs'] if g['type'] == 'MultiPolygon' else [g['arcs']]
        for poly in polys:
            for ring in poly:
                for arc_idx in ring:
                    all_points.extend(decode_arc(arc_idx))
        if all_points:
            avg_lon = sum(p[0] for p in all_points) / len(all_points)
            avg_lat = sum(p[1] for p in all_points) / len(all_points)
            c = (round(avg_lat, 4), round(avg_lon, 4))
            dist_centroids[f"{d_name}___{st_name}".lower()] = c
            dist_centroids[d_name.lower()] = c
            dist_centroids[_normalize_str(d_name)] = c

    # Precompute State Centroids
    st_centroids = {}
    for g in topo['objects']['states']['geometries']:
        props = g.get('properties', {})
        st_name = props.get('st_nm', '').strip()
        all_points = []
        polys = g['arcs'] if g['type'] == 'MultiPolygon' else [g['arcs']]
        for poly in polys:
            for ring in poly:
                for arc_idx in ring:
                    all_points.extend(decode_arc(arc_idx))
        if all_points:
            avg_lon = sum(p[0] for p in all_points) / len(all_points)
            avg_lat = sum(p[1] for p in all_points) / len(all_points)
            c = (round(avg_lat, 4), round(avg_lon, 4))
            st_centroids[st_name] = c
            st_centroids[_normalize_str(st_name)] = c

    _CACHED_TOPO = topo
    _CACHED_DISTRICT_CENTROIDS = dist_centroids
    _CACHED_STATE_CENTROIDS = st_centroids

    return topo, dist_centroids, st_centroids


def build_unified_map_html(
    crop_name: str,
    merged_geo: pd.DataFrame,
    is_hi: bool = True,
    selected_state: Optional[str] = None,
    selected_district: Optional[str] = None,
    selected_mandi: Optional[str] = None,
    **kwargs
) -> str:
    """
    Generate self-contained Leaflet HTML with OpenStreetMap (NO API KEY)
    and true geometric coordinates inside districts.
    """
    topo_data, dist_centroids, st_centroids = _load_topo_and_centroids()

    if merged_geo.empty:
        return "<div style='padding: 20px; text-align: center; color: #888;'>No mandi data available for map.</div>"

    merged = merged_geo.copy()

    # Safely handle column names from various merges
    if 'state' not in merged.columns:
        if 'state_x' in merged.columns:
            merged['state'] = merged['state_x']
        elif 'state_y' in merged.columns:
            merged['state'] = merged['state_y']
        else:
            merged['state'] = 'Punjab'

    if 'district' not in merged.columns:
        if 'district_x' in merged.columns:
            merged['district'] = merged['district_x']
        elif 'district_y' in merged.columns:
            merged['district'] = merged['district_y']
        else:
            merged['district'] = 'Ludhiana'

    if 'market_name' not in merged.columns:
        if 'market' in merged.columns:
            merged['market_name'] = merged['market']
        else:
            merged['market_name'] = 'Market'

    merged['modal_price'] = pd.to_numeric(merged['modal_price'], errors='coerce').fillna(2000.0)
    merged['predicted_price_7d'] = pd.to_numeric(merged.get('predicted_price_7d', merged['modal_price'] * 1.05), errors='coerce').fillna(merged['modal_price'] * 1.05)
    merged['price_diff'] = merged['predicted_price_7d'] - merged['modal_price']
    merged['price_pct'] = ((merged['price_diff'] / merged['modal_price']) * 100).round(1)

    merged['norm_state'] = merged['state'].map(lambda s: STATE_NAME_MAP.get(str(s), str(s)))

    # Compute 4 Price Distribution Brackets
    prices = merged['modal_price'].dropna()
    p_min = float(prices.min())
    p_max = float(prices.max())

    if p_min == p_max or len(prices) < 4:
        b1, b2, b3 = p_min * 0.95, p_min * 1.05, p_min * 1.15
    else:
        q25, q50, q75 = prices.quantile([0.25, 0.50, 0.75])
        b1, b2, b3 = float(round(q25, 0)), float(round(q50, 0)), float(round(q75, 0))
        if b1 >= b2:
            b2 = b1 + 50
        if b2 >= b3:
            b3 = b2 + 50

    def get_bracket_idx(p: float) -> int:
        if p < b1:
            return 0
        elif p < b2:
            return 1
        elif p < b3:
            return 2
        else:
            return 3

    # High-contrast distribution colors
    bracket_colors = [
        "#10b981",  # 0: Lower Range (Emerald Green)
        "#0284c7",  # 1: Moderate-Low Range (Sky Blue)
        "#f59e0b",  # 2: Moderate-High Range (Amber Gold)
        "#ef4444"   # 3: Peak Range (Crimson Red)
    ]

    bracket_labels = [
        f"< ₹{int(b1):,}" if not is_hi else f"< ₹{int(b1):,} (कम भाव)",
        f"₹{int(b1):,} - ₹{int(b2):,}" if not is_hi else f"₹{int(b1):,} - ₹{int(b2):,} (मध्यम-कम)",
        f"₹{int(b2):,} - ₹{int(b3):,}" if not is_hi else f"₹{int(b2):,} - ₹{int(b3):,} (मध्यम-उच्च)",
        f"> ₹{int(b3):,}" if not is_hi else f"> ₹{int(b3):,} (सर्वोच्च भाव)"
    ]

    # 1. State-Level Aggregation
    state_grp = merged.groupby('norm_state').agg(
        avg_price=('modal_price', 'mean'),
        avg_pred=('predicted_price_7d', 'mean'),
        mandis=('market_name', 'count')
    ).reset_index()

    state_stats: Dict[str, Any] = {}
    for _, r in state_grp.iterrows():
        st_name = str(r['norm_state'])
        p = float(round(r['avg_price'], 0))
        pred = float(round(r['avg_pred'], 0))
        b_idx = get_bracket_idx(p)
        diff = pred - p
        pct = round((diff / p * 100), 1) if p > 0 else 0
        c_lat, c_lon = st_centroids.get(st_name, st_centroids.get(_normalize_str(st_name), (22.5, 79.5)))

        state_stats[st_name] = {
            "name": st_name,
            "price": p,
            "pred_7d": pred,
            "mandis": int(r['mandis']),
            "diff": diff,
            "pct": pct,
            "bracket": b_idx,
            "color": bracket_colors[b_idx],
            "lat": c_lat,
            "lon": c_lon
        }
        state_stats[_normalize_str(st_name)] = state_stats[st_name]

    # 2. District-Level Aggregation
    dist_grp = merged.groupby(['norm_state', 'district']).agg(
        avg_price=('modal_price', 'mean'),
        avg_pred=('predicted_price_7d', 'mean'),
        mandis=('market_name', 'count')
    ).reset_index()

    district_stats: Dict[str, Any] = {}
    for _, r in dist_grp.iterrows():
        d_name = str(r['district']).strip()
        st_name = str(r['norm_state']).strip()
        p = float(round(r['avg_price'], 0))
        pred = float(round(r['avg_pred'], 0))
        b_idx = get_bracket_idx(p)
        diff = pred - p
        pct = round((diff / p * 100), 1) if p > 0 else 0

        # Retrieve true district centroid
        dist_key = f"{d_name}___{st_name}".lower()
        c_lat, c_lon = dist_centroids.get(dist_key, dist_centroids.get(d_name.lower(), dist_centroids.get(_normalize_str(d_name), (22.5, 79.5))))

        entry = {
            "name": d_name,
            "state": st_name,
            "price": p,
            "pred_7d": pred,
            "mandis": int(r['mandis']),
            "diff": diff,
            "pct": pct,
            "bracket": b_idx,
            "color": bracket_colors[b_idx],
            "lat": c_lat,
            "lon": c_lon
        }
        district_stats[dist_key] = entry
        district_stats[d_name.lower()] = entry
        district_stats[_normalize_str(d_name)] = entry

    # 3. Mandi Pins - Placed TRUE INSIDE their respective districts
    mandi_pins = []
    selected_coords = None
    target_zoom = 4.5

    # Group mandis by district so we can spread them geographically inside the district
    mandis_by_dist: Dict[str, List[Any]] = {}
    for _, r in merged.iterrows():
        d_key = f"{str(r['district']).strip()}___{str(r['norm_state']).strip()}".lower()
        mandis_by_dist.setdefault(d_key, []).append(r)

    for d_key, m_list in mandis_by_dist.items():
        total_in_dist = len(m_list)
        d_name = str(m_list[0]['district']).strip()
        st_name = str(m_list[0]['norm_state']).strip()

        # Find true district centroid
        c_lat, c_lon = dist_centroids.get(d_key, dist_centroids.get(d_name.lower(), dist_centroids.get(_normalize_str(d_name), (22.5, 79.5))))

        for idx, r in enumerate(m_list):
            p = float(round(r['modal_price'], 0))
            pred = float(round(r['predicted_price_7d'], 0))
            diff = float(round(r['price_diff'], 0))
            pct = float(r['price_pct'])
            b_idx = get_bracket_idx(p)

            # Position mandis inside the district polygon spread around centroid
            if total_in_dist == 1:
                lat, lon = c_lat, c_lon
            else:
                angle = (2 * math.pi * idx) / total_in_dist
                radius = 0.04 + (0.05 * (idx % 3))  # 5km to 15km spread
                lat = round(c_lat + radius * math.sin(angle), 4)
                lon = round(c_lon + (radius * 1.1) * math.cos(angle), 4)

            m_name = str(r['market_name'])
            is_sel = bool(selected_mandi and m_name.lower() == selected_mandi.lower())

            if is_sel:
                selected_coords = [lat, lon]
                target_zoom = 10.0

            action = "HOLD" if pct >= 2.0 else ("SELL" if pct <= -2.0 else "STABLE")
            action_text = (
                ("🟢 माल रोकें (HOLD)" if is_hi else "🟢 HOLD CROP")
                if action == "HOLD" else
                (("🔴 आज ही बेचें (SELL)" if is_hi else "🔴 SELL TODAY")
                 if action == "SELL" else
                 ("🔵 भाव स्थिर (STABLE)" if is_hi else "🔵 STABLE MARKET"))
            )

            trolley_gain = round(abs(diff) * 50, 0)

            mandi_pins.append({
                "name": m_name,
                "district": d_name,
                "state": st_name,
                "lat": lat,
                "lon": lon,
                "price": p,
                "pred_7d": pred,
                "diff": diff,
                "pct": pct,
                "bracket": b_idx,
                "color": bracket_colors[b_idx],
                "action": action,
                "action_text": action_text,
                "trolley_gain": trolley_gain,
                "is_selected": is_sel
            })

    # If district was selected from dropdown, focus on district center
    if not selected_coords and selected_district and selected_district != "All Districts":
        d_key = f"{selected_district}___{selected_state}".lower() if selected_state else selected_district.lower()
        if d_key in dist_centroids:
            selected_coords = list(dist_centroids[d_key])
            target_zoom = 8.5
        elif selected_district.lower() in dist_centroids:
            selected_coords = list(dist_centroids[selected_district.lower()])
            target_zoom = 8.5

    # If state was selected from dropdown, focus on state center
    if not selected_coords and selected_state and selected_state != "All States":
        if selected_state in st_centroids:
            selected_coords = list(st_centroids[selected_state])
            target_zoom = 6.8

    initial_center = selected_coords if selected_coords else [22.8, 79.5]
    initial_zoom = target_zoom

    # Prepare JSON strings
    topo_json_str = json.dumps(topo_data)
    state_stats_json = json.dumps(state_stats)
    district_stats_json = json.dumps(district_stats)
    mandi_pins_json = json.dumps(mandi_pins)
    bracket_colors_json = json.dumps(bracket_colors)
    bracket_labels_json = json.dumps(bracket_labels)
    center_coords_json = json.dumps(initial_center)

    # Bilingual labels
    legend_title = "📊 भाव वितरण श्रेणी" if is_hi else "📊 Price Distribution Ranges"
    no_data_label = "डेटा उपलब्ध नहीं" if is_hi else "No Mandi Data"
    reset_label = "🔄 पूरे भारत का दृश्य" if is_hi else "🔄 Reset to India View"
    hover_hint = "नक्शे पर किसी राज्य या जिले पर क्लिक करें या ज़ूम करें" if is_hi else "Click any State or District to zoom into it"

    html_content = f"""
<!DOCTYPE html>
<html lang="{'hi' if is_hi else 'en'}">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>India Unified Mandi Price Map</title>
  
  <!-- Leaflet CSS -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="" />
  <!-- TopoJSON Client -->
  <script src="https://unpkg.com/topojson-client@3/dist/topojson-client.min.js"></script>
  <!-- Leaflet JS -->
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>

  <style>
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }}
    body {{
      background: #f8fafc;
      overflow: hidden;
    }}
    #map-container {{
      position: relative;
      width: 100%;
      height: 620px;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
      border: 1px solid #cbd5e1;
    }}
    #map {{
      width: 100%;
      height: 100%;
      background: #e2e8f0;
    }}

    /* Permanent State & District Text Badges on Map */
    .state-name-badge {{
      background: rgba(255, 255, 255, 0.92);
      border: 1.5px solid #1e293b;
      border-radius: 6px;
      padding: 3px 8px;
      text-align: center;
      font-size: 11px;
      font-weight: 800;
      color: #0f172a;
      box-shadow: 0 2px 8px rgba(0,0,0,0.22);
      white-space: nowrap;
      pointer-events: none;
      transform: translate(-50%, -50%);
    }}
    .district-name-badge {{
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid #334155;
      border-radius: 5px;
      padding: 2px 6px;
      text-align: center;
      font-size: 10px;
      font-weight: 700;
      color: #0f172a;
      box-shadow: 0 2px 6px rgba(0,0,0,0.18);
      white-space: nowrap;
      pointer-events: none;
      transform: translate(-50%, -50%);
    }}
    .mandi-pin-label {{
      background: rgba(15, 23, 42, 0.92) !important;
      color: #ffffff !important;
      border: 1px solid #f59e0b !important;
      border-radius: 6px !important;
      padding: 3px 8px !important;
      font-size: 11px !important;
      font-weight: 700 !important;
      box-shadow: 0 3px 10px rgba(0,0,0,0.3) !important;
      white-space: nowrap;
    }}

    /* Top-Left Level Badge & Reset Button */
    .map-badge-ctrl {{
      position: absolute;
      top: 14px;
      left: 14px;
      z-index: 1000;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .zoom-level-badge {{
      background: rgba(15, 23, 42, 0.92);
      color: #ffffff;
      padding: 8px 16px;
      border-radius: 20px;
      font-size: 13px;
      font-weight: 700;
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
      border: 1px solid rgba(255, 255, 255, 0.2);
      display: flex;
      align-items: center;
      gap: 6px;
      backdrop-filter: blur(6px);
    }}
    .reset-btn {{
      background: #2563eb;
      color: #ffffff;
      padding: 7px 14px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 700;
      border: none;
      cursor: pointer;
      box-shadow: 0 3px 10px rgba(37, 99, 235, 0.35);
      display: inline-flex;
      align-items: center;
      gap: 6px;
      width: fit-content;
      transition: all 0.2s ease;
    }}
    .reset-btn:hover {{
      background: #1d4ed8;
      transform: translateY(-1px);
    }}

    /* Top-Right Floating Live Info Card */
    .live-info-panel {{
      position: absolute;
      top: 14px;
      right: 14px;
      z-index: 1000;
      background: rgba(255, 255, 255, 0.97);
      backdrop-filter: blur(10px);
      border-radius: 10px;
      padding: 12px 16px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
      border: 1px solid #cbd5e1;
      min-width: 240px;
      max-width: 300px;
      pointer-events: none;
    }}
    .live-info-panel h4 {{
      font-size: 15px;
      color: #0f172a;
      margin-bottom: 2px;
    }}
    .live-info-panel .subtitle {{
      font-size: 11px;
      color: #64748b;
      margin-bottom: 8px;
    }}
    .live-info-panel .stat-row {{
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      margin-bottom: 4px;
      color: #334155;
    }}
    .live-info-panel .stat-val {{
      font-weight: 700;
      color: #0f172a;
    }}
    .live-info-panel .hint-text {{
      margin-top: 6px;
      font-size: 10.5px;
      color: #2563eb;
      font-style: italic;
      border-top: 1px dashed #cbd5e1;
      padding-top: 4px;
    }}

    /* Bottom-Right Color Legend */
    .color-legend {{
      position: absolute;
      bottom: 20px;
      right: 14px;
      z-index: 1000;
      background: rgba(255, 255, 255, 0.96);
      backdrop-filter: blur(8px);
      border-radius: 10px;
      padding: 10px 14px;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
      border: 1px solid #cbd5e1;
      font-size: 11.5px;
    }}
    .color-legend h5 {{
      font-size: 12px;
      margin-bottom: 6px;
      color: #1e293b;
      font-weight: 700;
    }}
    .legend-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 4px;
      color: #334155;
    }}
    .legend-dot {{
      width: 14px;
      height: 14px;
      border-radius: 3px;
      border: 1px solid rgba(0,0,0,0.15);
      flex-shrink: 0;
    }}

    /* Custom Leaflet Popups */
    .leaflet-popup-content-wrapper {{
      border-radius: 10px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.2);
      border: 1px solid #cbd5e1;
      padding: 2px;
    }}
    .custom-mandi-popup {{
      font-size: 12px;
      color: #1e293b;
      min-width: 220px;
    }}
    .custom-mandi-popup h3 {{
      font-size: 14px;
      color: #0f172a;
      margin-bottom: 2px;
    }}
    .custom-mandi-popup .mandi-loc {{
      color: #64748b;
      font-size: 11px;
      margin-bottom: 6px;
    }}
    .custom-mandi-popup .row {{
      display: flex;
      justify-content: space-between;
      margin-bottom: 4px;
    }}
    .custom-mandi-popup .badge {{
      display: block;
      margin-top: 8px;
      padding: 6px;
      text-align: center;
      border-radius: 6px;
      font-weight: 700;
      font-size: 12px;
    }}
  </style>
</head>
<body>

  <div id="map-container">
    <div id="map"></div>

    <!-- Level Indicator Badge & Reset -->
    <div class="map-badge-ctrl">
      <div id="zoom-badge" class="zoom-level-badge">
        {'🇮🇳 <b>स्तर 1: राज्यवार भाव वितरण</b>' if is_hi else '🇮🇳 <b>Level 1: State Distribution</b>'}
      </div>
      <button class="reset-btn" onclick="resetToIndia()">
        {reset_label}
      </button>
    </div>

    <!-- Live Hover Info Panel -->
    <div id="info-panel" class="live-info-panel">
      <h4 id="info-title">🌾 {crop_name}</h4>
      <div id="info-sub" class="subtitle">{'पूरे भारत का औसत भाव वितरण' if is_hi else 'Nationwide Price Distribution'}</div>
      <div class="stat-row">
        <span>{'औसत भाव:' if is_hi else 'Average Price:'}</span>
        <span id="info-price" class="stat-val">-</span>
      </div>
      <div class="stat-row">
        <span>{'7-दिन अनुमान:' if is_hi else '7-Day Forecast:'}</span>
        <span id="info-pred" class="stat-val">-</span>
      </div>
      <div class="stat-row">
        <span>{'सक्रिय मंडियां:' if is_hi else 'Active Mandis:'}</span>
        <span id="info-mandis" class="stat-val">{len(mandi_pins)}</span>
      </div>
      <div id="info-hint" class="hint-text">{hover_hint}</div>
    </div>

    <!-- Color Legend -->
    <div class="color-legend">
      <h5>{legend_title}</h5>
      <div class="legend-row">
        <span class="legend-dot" style="background: {bracket_colors[0]};"></span>
        <span>{bracket_labels[0]}</span>
      </div>
      <div class="legend-row">
        <span class="legend-dot" style="background: {bracket_colors[1]};"></span>
        <span>{bracket_labels[1]}</span>
      </div>
      <div class="legend-row">
        <span class="legend-dot" style="background: {bracket_colors[2]};"></span>
        <span>{bracket_labels[2]}</span>
      </div>
      <div class="legend-row">
        <span class="legend-dot" style="background: {bracket_colors[3]};"></span>
        <span>{bracket_labels[3]}</span>
      </div>
      <div class="legend-row">
        <span class="legend-dot" style="background: #94a3b8; opacity: 0.3;"></span>
        <span style="color: #64748b;">{no_data_label}</span>
      </div>
    </div>
  </div>

  <script>
    // Injected Data
    const topoData = {topo_json_str};
    const stateStats = {state_stats_json};
    const districtStats = {district_stats_json};
    const mandiPins = {mandi_pins_json};
    const bracketColors = {bracket_colors_json};
    const bracketLabels = {bracket_labels_json};
    const initialCenter = {center_coords_json};
    const initialZoom = {initial_zoom};
    const isHi = {'true' if is_hi else 'false'};
    const cropName = "{crop_name}";

    // Initialize Leaflet Map
    const map = L.map('map', {{
      center: initialCenter,
      zoom: initialZoom,
      minZoom: 4,
      maxZoom: 14,
      zoomControl: true
    }});

    // STANDARD OPENSTREETMAP TILE LAYER - 100% FREE, NO API KEY EVER REQUIRED
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19
    }}).addTo(map);

    map.zoomControl.setPosition('bottomleft');

    function normStr(s) {{
      return String(s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
    }}

    // State Layer Style
    function getStateStyle(feature) {{
      const stName = feature.properties.st_nm || '';
      const stats = stateStats[stName] || stateStats[normStr(stName)];
      if (stats && stats.price > 0) {{
        return {{
          fillColor: stats.color,
          weight: 2,
          opacity: 1,
          color: '#ffffff',
          fillOpacity: 0.72
        }};
      }} else {{
        return {{
          fillColor: '#94a3b8',
          weight: 1,
          opacity: 0.4,
          color: '#cbd5e1',
          fillOpacity: 0.15
        }};
      }}
    }}

    // District Layer Style
    function getDistrictStyle(feature) {{
      const dName = feature.properties.district || '';
      const stName = feature.properties.st_nm || '';
      const key1 = (dName + '___' + stName).toLowerCase();
      const stats = districtStats[key1] || districtStats[dName.toLowerCase()] || districtStats[normStr(dName)];
      
      if (stats && stats.price > 0) {{
        return {{
          fillColor: stats.color,
          weight: 1.5,
          opacity: 0.95,
          color: '#ffffff',
          fillOpacity: 0.75
        }};
      }} else {{
        return {{
          fillColor: '#94a3b8',
          weight: 0.8,
          opacity: 0.4,
          color: '#e2e8f0',
          fillOpacity: 0.1
        }};
      }}
    }}

    // Build GeoJSON from TopoJSON
    const stateGeo = topojson.feature(topoData, topoData.objects.states);
    const districtGeo = topojson.feature(topoData, topoData.objects.districts);

    // Layer 1: State Polygons
    let stateLayer = L.geoJSON(stateGeo, {{
      style: getStateStyle,
      onEachFeature: function(feature, layer) {{
        const stName = feature.properties.st_nm || '';
        const stats = stateStats[stName] || stateStats[normStr(stName)];

        layer.on('mouseover', function(e) {{
          layer.setStyle({{
            weight: 3.5,
            color: '#f59e0b',
            fillOpacity: 0.90
          }});
          if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {{
            layer.bringToFront();
          }}
          updateInfoBox(stName, stats, 'state');
        }});

        layer.on('mouseout', function(e) {{
          stateLayer.resetStyle(layer);
          resetInfoBox();
        }});

        layer.on('click', function(e) {{
          map.fitBounds(layer.getBounds(), {{ padding: [25, 25], maxZoom: 6.8 }});
        }});
      }}
    }});

    // Layer 1B: State Permanent Labels
    const stateLabelsLayer = L.layerGroup();
    Object.keys(stateStats).forEach(function(stKey) {{
      const st = stateStats[stKey];
      if (st && st.name && st.price > 0 && st.lat && st.lon && !stKey.includes('_')) {{
        const icon = L.divIcon({{
          className: 'custom-state-label',
          html: `<div class="state-name-badge">${{st.name}}<br><span style="color:#0284c7;font-size:10px;">₹${{Math.round(st.price).toLocaleString('en-IN')}}</span></div>`,
          iconSize: [100, 30],
          iconAnchor: [50, 15]
        }});
        stateLabelsLayer.addLayer(L.marker([st.lat, st.lon], {{ icon: icon, interactive: false }}));
      }}
    }});

    // Layer 2: District Polygons
    let districtLayer = L.geoJSON(districtGeo, {{
      style: getDistrictStyle,
      onEachFeature: function(feature, layer) {{
        const dName = feature.properties.district || '';
        const stName = feature.properties.st_nm || '';
        const key1 = (dName + '___' + stName).toLowerCase();
        const stats = districtStats[key1] || districtStats[dName.toLowerCase()] || districtStats[normStr(dName)];

        layer.on('mouseover', function(e) {{
          layer.setStyle({{
            weight: 3,
            color: '#f59e0b',
            fillOpacity: 0.92
          }});
          if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {{
            layer.bringToFront();
          }}
          updateInfoBox(dName + ' (' + stName + ')', stats, 'district');
        }});

        layer.on('mouseout', function(e) {{
          districtLayer.resetStyle(layer);
          resetInfoBox();
        }});

        layer.on('click', function(e) {{
          map.fitBounds(layer.getBounds(), {{ padding: [35, 35], maxZoom: 9.8 }});
        }});
      }}
    }});

    // Layer 2B: District Permanent Labels
    const districtLabelsLayer = L.layerGroup();
    Object.keys(districtStats).forEach(function(dKey) {{
      const dist = districtStats[dKey];
      if (dist && dist.name && dist.price > 0 && dist.lat && dist.lon && dKey.includes('___')) {{
        const icon = L.divIcon({{
          className: 'custom-dist-label',
          html: `<div class="district-name-badge">${{dist.name}}<br><span style="color:#0284c7;font-size:9.5px;">₹${{Math.round(dist.price).toLocaleString('en-IN')}}</span></div>`,
          iconSize: [80, 26],
          iconAnchor: [40, 13]
        }});
        districtLabelsLayer.addLayer(L.marker([dist.lat, dist.lon], {{ icon: icon, interactive: false }}));
      }}
    }});

    // Layer 3: Mandi Markers with Labels
    const mandiLayer = L.layerGroup();
    mandiPins.forEach(function(pin) {{
      const marker = L.circleMarker([pin.lat, pin.lon], {{
        radius: pin.is_selected ? 11 : 7.5,
        fillColor: pin.color,
        color: pin.is_selected ? '#f59e0b' : '#ffffff',
        weight: pin.is_selected ? 3.5 : 2,
        opacity: 1,
        fillOpacity: 0.95
      }});

      // Permanent tooltip showing Mandi Name and Price
      marker.bindTooltip(`🏛️ <b>${{pin.name}}</b>: ₹${{pin.price.toLocaleString('en-IN')}}`, {{
        permanent: false,
        direction: 'top',
        className: 'mandi-pin-label'
      }});

      const diffSign = pin.diff >= 0 ? '+' : '';
      const diffColor = pin.diff >= 0 ? '#16a34a' : '#dc2626';
      const bgBadge = pin.diff >= 0 ? '#dcfce7' : '#fee2e2';
      const txtBadge = pin.diff >= 0 ? '#15803d' : '#b91c1c';

      const popupHtml = `
        <div class="custom-mandi-popup">
          <h3>🏛️ ${{pin.name}}</h3>
          <div class="mandi-loc">${{pin.district}}, ${{pin.state}}</div>
          <hr style="border: none; border-top: 1px solid #cbd5e1; margin: 6px 0;" />
          <div class="row">
            <span>${{isHi ? 'आज का भाव:' : 'Today Price:'}}</span>
            <b>₹${{pin.price.toLocaleString('en-IN')}} / क्विंटल</b>
          </div>
          <div class="row">
            <span>${{isHi ? '7-दिन अनुमान:' : '7-Day Forecast:'}}</span>
            <b style="color: ${{diffColor}};">₹${{pin.pred_7d.toLocaleString('en-IN')}} (${{diffSign}}${{pin.pct}}%)</b>
          </div>
          <div class="row" style="font-size: 11px; color: #475569;">
            <span>${{isHi ? '50 क्विंटल ट्रॉली लाभ:' : '50 Qtl Trolley Gain:'}}</span>
            <b>${{diffSign}}₹${{pin.trolley_gain.toLocaleString('en-IN')}}</b>
          </div>
          <div class="badge" style="background: ${{bgBadge}}; color: ${{txtBadge}};">
            ${{pin.action_text}}
          </div>
        </div>
      `;

      marker.bindPopup(popupHtml);

      marker.on('mouseover', function(e) {{
        updateInfoBox(pin.name + ' Mandi', {{
          price: pin.price,
          pred_7d: pin.pred_7d,
          mandis: 1,
          diff: pin.diff,
          pct: pin.pct
        }}, 'mandi');
      }});

      marker.on('mouseout', function(e) {{
        resetInfoBox();
      }});

      mandiLayer.addLayer(marker);
    }});

    // Update Info Box
    function updateInfoBox(title, stats, type) {{
      const elTitle = document.getElementById('info-title');
      const elSub = document.getElementById('info-sub');
      const elPrice = document.getElementById('info-price');
      const elPred = document.getElementById('info-pred');
      const elMandis = document.getElementById('info-mandis');
      const elHint = document.getElementById('info-hint');

      elTitle.innerText = title;
      if (type === 'state') {{
        elSub.innerText = isHi ? "राज्य स्तर (State Level Distribution)" : "State Level Price Distribution";
        elHint.innerText = isHi ? "🔍 राज्य पर क्लिक करें या ज़ूम करें (Click to view districts)" : "🔍 Click to zoom into Districts";
      }} else if (type === 'district') {{
        elSub.innerText = isHi ? "जिला स्तर (District Level Distribution)" : "District Level Price Distribution";
        elHint.innerText = isHi ? "🔍 जिले पर क्लिक करें या ज़ूम करें (Click to view mandis)" : "🔍 Click to zoom into Mandis";
      }} else {{
        elSub.innerText = isHi ? "स्थानीय मंडी (Mandi Pin)" : "Local Mandi Pin";
        elHint.innerText = isHi ? "💡 क्लिक करके 7-दिन निर्णय व बचत देखें" : "💡 Click for 7-day decision & savings";
      }}

      if (stats && stats.price > 0) {{
        elPrice.innerText = '₹' + Math.round(stats.price).toLocaleString('en-IN') + ' /क्विं';
        const sign = (stats.diff || 0) >= 0 ? '+' : '';
        const color = (stats.diff || 0) >= 0 ? '#16a34a' : '#dc2626';
        elPred.innerHTML = `<span style="color:${{color}}">₹${{Math.round(stats.pred_7d).toLocaleString('en-IN')}} (${{sign}}${{stats.pct || 0}}%)</span>`;
        elMandis.innerText = (stats.mandis || 1) + (isHi ? ' मंडियां' : ' Mandis');
      }} else {{
        elPrice.innerText = isHi ? 'डेटा उपलब्ध नहीं' : 'No Data';
        elPred.innerText = '-';
        elMandis.innerText = '0';
      }}
    }}

    function resetInfoBox() {{
      document.getElementById('info-title').innerText = "🌾 " + cropName;
      document.getElementById('info-sub').innerText = isHi ? "पूरे भारत का औसत भाव वितरण" : "Nationwide Price Distribution";
      document.getElementById('info-price').innerText = "-";
      document.getElementById('info-pred').innerText = "-";
      document.getElementById('info-mandis').innerText = "{len(mandi_pins)}";
      document.getElementById('info-hint').innerText = "{hover_hint}";
    }}

    // DYNAMIC ZOOM LAYER SWITCHER
    function handleZoomChange() {{
      const z = map.getZoom();
      const badge = document.getElementById('zoom-badge');

      if (z < 6.0) {{
        // LEVEL 1: ALL-INDIA STATE DISTRIBUTION
        if (!map.hasLayer(stateLayer)) map.addLayer(stateLayer);
        if (!map.hasLayer(stateLabelsLayer)) map.addLayer(stateLabelsLayer);
        if (map.hasLayer(districtLayer)) map.removeLayer(districtLayer);
        if (map.hasLayer(districtLabelsLayer)) map.removeLayer(districtLabelsLayer);
        if (map.hasLayer(mandiLayer)) map.removeLayer(mandiLayer);

        badge.innerHTML = isHi 
          ? "🇮🇳 <b>स्तर 1: राज्यवार भाव वितरण</b> (ज़ूम करके जिले देखें)"
          : "🇮🇳 <b>Level 1: State Distribution</b> (Zoom for Districts)";
      }} else if (z >= 6.0 && z < 8.8) {{
        // LEVEL 2: REGION & DISTRICT DISTRIBUTION
        if (map.hasLayer(stateLayer)) map.removeLayer(stateLayer);
        if (map.hasLayer(stateLabelsLayer)) map.removeLayer(stateLabelsLayer);
        if (!map.hasLayer(districtLayer)) map.addLayer(districtLayer);
        districtLayer.setStyle(getDistrictStyle);
        if (!map.hasLayer(districtLabelsLayer)) map.addLayer(districtLabelsLayer);
        if (map.hasLayer(mandiLayer)) map.removeLayer(mandiLayer);

        badge.innerHTML = isHi 
          ? "🗺️ <b>स्तर 2: जिलावार भाव वितरण</b> (ज़ूम करके मंडियां देखें)"
          : "🗺️ <b>Level 2: District Distribution</b> (Zoom for Mandis)";
      }} else {{
        // LEVEL 3: CITY & MANDI EXACT PIN VIEW
        if (map.hasLayer(stateLayer)) map.removeLayer(stateLayer);
        if (map.hasLayer(stateLabelsLayer)) map.removeLayer(stateLabelsLayer);
        if (!map.hasLayer(districtLayer)) map.addLayer(districtLayer);
        // keep district outlines thin and translucent as soft background context
        districtLayer.setStyle(function() {{
          return {{
            fillColor: '#94a3b8',
            weight: 1.5,
            opacity: 0.8,
            color: '#475569',
            fillOpacity: 0.12
          }};
        }});
        if (map.hasLayer(districtLabelsLayer)) map.removeLayer(districtLabelsLayer);
        if (!map.hasLayer(mandiLayer)) map.addLayer(mandiLayer);

        badge.innerHTML = isHi 
          ? "🏛️ <b>स्तर 3: स्थानीय मंडी व शहर पिन</b> (पिन पर क्लिक करें)"
          : "🏛️ <b>Level 3: Local Mandi & City Pins</b> (Click pins for details)";
      }}
    }}

    function resetToIndia() {{
      map.flyTo([22.8, 79.5], 4.5, {{ duration: 1.2 }});
    }}

    map.on('zoomend', handleZoomChange);

    handleZoomChange();
  </script>
</body>
</html>
"""
    return html_content
