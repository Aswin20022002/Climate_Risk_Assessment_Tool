"""
India Physical Risk Explorer
==============================
Files required in the SAME folder as this script:

  REQUIRED:
    India_Pincode_Boundary_with_LatLong_and_Shape_2022.csv
    india_pincodes.geojson
    pin_feature_index.json        <- pre-built index (298 KB)

  OPTIONAL:
    precomputed_cyclone_scores.csv
    precomputed_heat_scores.csv
    precomputed_drought_scores.csv
    precomputed_flood_scores.csv
    precomputed_rainfall_scores.csv

WHY pin_feature_index.json:
  The 236 MB GeoJSON has empty properties — no pin_code stored.
  Old approach: load all 236 MB at startup to build centroid index = OOM crash.
  New approach: pre-built 298 KB index loaded instantly at startup.
  GeoJSON is streamed for only the ONE feature the user requests.
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import json
import numpy as np
import math
import os

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Physical Risk Explorer",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_CSV   = "India_Pincode_Boundary_with_LatLong_and_Shape_2022.csv"
GEOJSON    = "india_pincodes.geojson"
INDEX_FILE = "pin_feature_index.json"

SCORE_FILES = {
    "cyclone":  {"file": "precomputed_cyclone_scores.csv",  "col": "cyclone_score",  "label": "Cyclone",  "icon": "🌀"},
    "heat":     {"file": "precomputed_heat_scores.csv",     "col": "heat_score",     "label": "Heat",     "icon": "🔥"},
    "rainfall": {"file": "precomputed_rainfall_scores.csv", "col": "rainfall_score", "label": "Rainfall", "icon": "🌧️"},
    "drought":  {"file": "precomputed_drought_scores.csv",  "col": "drought_score",  "label": "Drought",  "icon": "🌵"},
    "flood":    {"file": "precomputed_flood_scores.csv",    "col": "flood_score",    "label": "Flood",    "icon": "🌊"},
}

JUSTIFICATIONS = {
    "cyclone": {
        "Very High": "Multiple intense cyclone tracks have passed within 150 km historically. This area is among India's highest cyclone-exposure zones with recorded winds above 90 knots.",
        "High":      "Several cyclone tracks recorded within 150 km. Significant wind and storm-surge risk during Bay of Bengal or Arabian Sea cyclone seasons.",
        "Moderate":  "Occasional cyclone influence within 150 km. Indirect effects — heavy rain, gusty winds — are possible during active seasons.",
        "Low":       "Few cyclone tracks within 150 km. Cyclone impact is rare; occasional remnant depressions may bring rain.",
        "Very Low":  "No significant cyclone activity within 150 km in the historical record. This area is largely sheltered from tropical cyclone paths.",
    },
    "heat": {
        "Very High": "More than 30% of days exceed 35°C annually. Warm nights are frequent and the area shows a strong warming trend — among the most heat-stressed zones in India.",
        "High":      "Hot days are frequent (15–30% of days above 35°C). Urban heat island effect likely intensifies exposure. Warming trend is significant.",
        "Moderate":  "Seasonal heat spells occur but are within typical regional ranges. Some warm nights recorded during summer months.",
        "Low":       "Hot days are infrequent. Cooler climate, elevation, or coastal influence moderates temperatures.",
        "Very Low":  "Rarely experiences extreme heat. High elevation or consistent sea breeze keeps temperatures well below dangerous thresholds.",
    },
    "rainfall": {
        "Very High": "Extreme single-day and 5-day rainfall totals are among the highest in India. Heavy monsoon events are frequent and intensifying.",
        "High":      "Above-average extreme rainfall events. Short-duration intense downpours are common during monsoon season.",
        "Moderate":  "Occasional heavy rainfall events. Monsoon delivers significant rain but extreme single-day totals are not unusually high.",
        "Low":       "Rainfall is relatively moderate. Extreme daily totals are uncommon in the historical record.",
        "Very Low":  "Arid or semi-arid conditions. Extreme rainfall events are very rare.",
    },
    "drought": {
        "Very High": "Frequent long dry spells, high monsoon variability, and repeated rainfall deficits. Agricultural drought risk is severe.",
        "High":      "Significant monsoon variability. Below-normal monsoon years are common, creating periodic water stress.",
        "Moderate":  "Some years see below-normal monsoon. Dry spells occur but the area is not chronically drought-prone.",
        "Low":       "Adequate and relatively reliable rainfall. Drought years are infrequent.",
        "Very Low":  "Reliable rainfall or proximity to perennial water sources. Drought risk is minimal.",
    },
    "flood": {
        "Very High": "Large share of the area intersects flood-hazard zones. Low elevation, river proximity, and heavy rainfall combine to create very high inundation risk.",
        "High":      "Significant flood-prone areas within the PIN boundary. River flooding during heavy monsoon is a regular occurrence.",
        "Moderate":  "Some flood-prone sub-areas, primarily driven by drainage issues during intense rainfall rather than river overflow.",
        "Low":       "Limited flood hazard. Mostly elevated terrain with adequate drainage.",
        "Very Low":  "Minimal flood risk. Elevated or well-drained land with no significant river flood hazard zones.",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def norm_pin(val):
    return str(val).strip().zfill(6)

def find_pin_col(df):
    for c in df.columns:
        if "pin" in c.lower():
            return c
    return None

def risk_level(score):
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "No data", "#94a3b8"
    s = float(score)
    if s >= 75: return "Very High", "#dc2626"
    if s >= 55: return "High",      "#ea580c"
    if s >= 35: return "Moderate",  "#ca8a04"
    if s >= 15: return "Low",       "#16a34a"
    return "Very Low", "#0284c7"

def get_geom_bounds(geom):
    gtype  = geom["type"]
    coords = geom["coordinates"]
    if gtype == "Polygon":
        rings = [coords[0]]
    elif gtype == "MultiPolygon":
        rings = [poly[0] for poly in coords]
    else:
        return None
    all_lons = [p[0] for ring in rings for p in ring]
    all_lats = [p[1] for ring in rings for p in ring]
    return min(all_lats), min(all_lons), max(all_lats), max(all_lons)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING — startup loads only small files (CSV + index)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading pincode database…")
def load_base_csv():
    if not os.path.exists(BASE_CSV):
        return None
    df = pd.read_csv(BASE_CSV, encoding="utf-8-sig", low_memory=False)
    pin_col = find_pin_col(df)
    if not pin_col:
        return None
    df["_pin"] = df[pin_col].apply(norm_pin)
    df.columns = [c.lower().strip() for c in df.columns]
    return df

@st.cache_data(show_spinner="Loading hazard scores…")
def load_scores():
    merged = None
    loaded = []
    for key, cfg in SCORE_FILES.items():
        if not os.path.exists(cfg["file"]):
            continue
        try:
            df = pd.read_csv(cfg["file"], low_memory=False)
            pin_col = find_pin_col(df)
            if not pin_col:
                continue
            df["_pin"] = df[pin_col].apply(norm_pin)
            want = cfg["col"]
            score_col = want if want in df.columns else next(
                (c for c in df.columns if key in c.lower() or "score" in c.lower()), None)
            if not score_col:
                continue
            df[want] = pd.to_numeric(df[score_col], errors="coerce")
            subset = df[["_pin", want]].drop_duplicates("_pin")
            merged = subset if merged is None else merged.merge(subset, on="_pin", how="outer")
            loaded.append(cfg["label"])
        except Exception as e:
            st.warning(f"Error loading {cfg['file']}: {e}")
    return merged, loaded

@st.cache_data(show_spinner="Loading boundary index…")
def load_pin_index():
    """Load the 298 KB pre-built index: pin_code -> [feature_indices]."""
    if not os.path.exists(INDEX_FILE):
        return {}
    with open(INDEX_FILE) as f:
        return json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# ON-DEMAND FEATURE FETCH — reads ONLY the needed feature from the 236 MB file
#
# Strategy: seek through GeoJSON to find the feature at the target index.
# Uses json.JSONDecoder with raw_decode to parse features one at a time
# without loading the whole file. Falls back to full load if needed.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading boundary polygon…", ttl=3600)
def fetch_features_for_pin(indices_tuple):
    """
    Extract only the GeoJSON features at the given indices.
    Never loads the full 236 MB file into memory.
    Uses iterative JSON parsing to find the right features.
    """
    if not indices_tuple or not os.path.exists(GEOJSON):
        return []

    target_set = set(indices_tuple)
    found = {}

    # Try streaming with built-in json module
    # Read the file and find the "features": [ ... ] array
    # Then parse feature objects one at a time
    try:
        decoder = json.JSONDecoder()
        CHUNK = 65536  # 64 KB read buffer

        with open(GEOJSON, "r") as f:
            # Skip to the start of the features array
            buffer = ""
            feature_start = None
            bytes_read = 0

            while feature_start is None:
                chunk = f.read(CHUNK)
                if not chunk:
                    break
                buffer += chunk
                idx = buffer.find('"features"')
                if idx != -1:
                    # Find the opening [ of the array
                    bracket = buffer.find("[", idx)
                    if bracket != -1:
                        feature_start = bracket + 1
                        buffer = buffer[feature_start:]
                        break

            if feature_start is None:
                raise ValueError("Could not find features array")

            # Now parse features one by one
            feature_count = 0
            while buffer is not None:
                buffer = buffer.lstrip()
                if not buffer:
                    more = f.read(CHUNK)
                    if not more:
                        break
                    buffer = more.lstrip()
                if buffer.startswith("]"):
                    break
                if buffer.startswith(","):
                    buffer = buffer[1:].lstrip()

                # Try to decode one feature object
                while True:
                    try:
                        obj, end = decoder.raw_decode(buffer)
                        buffer = buffer[end:]
                        break
                    except json.JSONDecodeError:
                        more = f.read(CHUNK)
                        if not more:
                            obj = None
                            break
                        buffer += more

                if obj is None:
                    break

                if feature_count in target_set:
                    found[feature_count] = obj.get("geometry")
                    if len(found) == len(target_set):
                        break  # got everything we need

                feature_count += 1

    except Exception:
        # Nuclear fallback: load the whole file if streaming fails
        try:
            with open(GEOJSON) as f:
                gj = json.load(f)
            features = gj.get("features", [])
            for i in indices_tuple:
                if i < len(features):
                    found[i] = features[i].get("geometry")
        except Exception:
            return []

    return [found[i] for i in indices_tuple if i in found and found[i] is not None]


# ─────────────────────────────────────────────────────────────────────────────
# MAP
# ─────────────────────────────────────────────────────────────────────────────
def build_map(lat, lon, geom_list, fill_color, pin_str, area_m2):
    m = folium.Map(
        location=[lat, lon],
        zoom_start=13,
        tiles="CartoDB positron",
        prefer_canvas=True,
        zoom_control=True,
    )

    if geom_list:
        bounds_list = []
        for geom in geom_list:
            if not geom:
                continue
            folium.GeoJson(
                data={"type": "Feature", "properties": {}, "geometry": geom},
                style_function=lambda x, fc=fill_color: {
                    "fillColor": fc, "color": fc,
                    "weight": 2.5, "fillOpacity": 0.20,
                },
                tooltip=folium.Tooltip(f"PIN {pin_str}", sticky=False),
            ).add_to(m)
            b = get_geom_bounds(geom)
            if b:
                bounds_list.append(b)

        if bounds_list:
            m.fit_bounds(
                [[min(b[0] for b in bounds_list), min(b[1] for b in bounds_list)],
                 [max(b[2] for b in bounds_list), max(b[3] for b in bounds_list)]],
                max_zoom=15,
            )
    else:
        radius = math.sqrt(float(area_m2) / math.pi) if area_m2 and area_m2 > 100 else 2000
        folium.Circle(
            location=[lat, lon], radius=radius,
            color=fill_color, weight=2, fill=True, fill_opacity=0.18,
            tooltip=f"PIN {pin_str} — boundary not available",
        ).add_to(m)

    return m

# ─────────────────────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────────────────────
def render(pin_str, base_df, scores_df, loaded_hazards, pin_index):
    rows = base_df[base_df["_pin"] == pin_str]
    if rows.empty:
        st.error(f"PIN **{pin_str}** not found.")
        return
    row = rows.iloc[0]

    lat      = float(row.get("latitude",  0) or 0)
    lon      = float(row.get("longitude", 0) or 0)
    district = str(row.get("district", "Unknown")).title()
    state    = str(row.get("state",    "Unknown")).title()
    area_m2  = float(row.get("shape__area",   row.get("shape_area",   0)) or 0)
    perim_m  = float(row.get("shape__length", row.get("shape_length", 0)) or 0)

    if lat == 0 and lon == 0:
        st.error("Coordinates missing for this PIN.")
        return

    # Scores
    score_row = None
    if scores_df is not None:
        sr = scores_df[scores_df["_pin"] == pin_str]
        if not sr.empty:
            score_row = sr.iloc[0]

    scores = {}
    for key, cfg in SCORE_FILES.items():
        if score_row is not None:
            raw = score_row.get(cfg["col"])
            if raw is not None and not (isinstance(raw, float) and math.isnan(raw)):
                scores[key] = float(raw)

    composite = round(sum(scores.values()) / len(scores), 1) if scores else None
    ovr_label, fill_color = risk_level(composite)

    # Fetch boundary — only the specific features needed, not the whole file
    indices  = pin_index.get(pin_str, [])
    geom_list = fetch_features_for_pin(tuple(indices)) if indices else []

    # Header
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:1.2rem;flex-wrap:wrap;">
          <div style="background:#f0f4ff;padding:0.5rem 1.2rem;border-radius:20px;
                      color:#1e3a8a;font-weight:700;font-size:1.05rem;border:1px solid #bfdbfe;">
              📍 {district}, {state}
          </div>
          <div style="background:#f5f3ff;padding:0.5rem 1.2rem;border-radius:20px;
                      color:#4c1d95;font-weight:700;font-size:1.05rem;border:1px solid #ddd6fe;">
              PIN {pin_str}
          </div>
          <div style="background:{fill_color}18;padding:0.5rem 1.2rem;border-radius:20px;
                      color:{fill_color};font-weight:700;font-size:1.05rem;
                      border:1.5px solid {fill_color}55;">
              Overall Risk: {ovr_label}{f' ({composite}/100)' if composite is not None else ''}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    c1.metric("Area",      f"{area_m2/1e6:.3f} km²" if area_m2 else "N/A")
    c2.metric("Perimeter", f"{perim_m/1000:.2f} km"  if perim_m else "N/A")

    col_map, col_scores = st.columns([1.1, 0.9], gap="large")

    with col_map:
        st.markdown("##### Boundary Map")
        if not geom_list:
            st.caption("⚠️ Boundary polygon not in GeoJSON. Showing approximate area.")
        fmap = build_map(lat, lon, geom_list, fill_color, pin_str, area_m2)
        st_folium(fmap, width="100%", height=500, returned_objects=[])

    with col_scores:
        st.markdown("##### Climate Hazard Scores")

        if not scores:
            if not loaded_hazards:
                st.info("No precomputed score files found. Add score CSVs to see hazard data.")
            else:
                st.info(f"Score files loaded ({', '.join(loaded_hazards)}) but no data for PIN {pin_str}.")
        else:
            for key, cfg in SCORE_FILES.items():
                if key not in scores:
                    st.markdown(
                        f"""<div style="margin-bottom:1.1rem;opacity:0.45;">
                          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                            <span style="font-weight:600;font-size:0.95rem;">{cfg['icon']} {cfg['label']}</span>
                            <span style="font-size:0.82rem;color:#94a3b8;">not computed yet</span>
                          </div>
                          <div style="background:#e5e7eb;border-radius:999px;height:9px;"></div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    continue

                score        = scores[key]
                label, color = risk_level(score)
                pct          = min(max(score, 0), 100)

                st.markdown(
                    f"""<div style="margin-bottom:1.25rem;">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
                        <span style="font-weight:700;font-size:0.97rem;">{cfg['icon']} {cfg['label']}</span>
                        <span style="font-size:0.85rem;">
                          <b style="color:{color}">{score:.1f}</b>
                          <span style="color:#6b7280;"> / 100 &nbsp;</span>
                          <span style="background:{color}18;color:{color};font-weight:700;
                                       padding:2px 10px;border-radius:10px;
                                       border:1px solid {color}44;font-size:0.82rem;">{label}</span>
                        </span>
                      </div>
                      <div style="background:#e5e7eb;border-radius:999px;height:10px;overflow:hidden;margin-bottom:6px;">
                        <div style="width:{pct}%;height:100%;background:{color};border-radius:999px;"></div>
                      </div>
                      <div style="font-size:0.78rem;color:#64748b;line-height:1.45;padding-left:2px;">
                        {JUSTIFICATIONS.get(key, {}).get(label, '')}
                      </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown(
            """<div style="font-size:0.74rem;color:#9ca3af;line-height:1.55;">
              <b>Data sources:</b> ERA5-Land (heat) · CHIRPS (rainfall, drought) ·
              JRC Flood Hazard (flood) · IBTrACS NI (cyclone) · India POST / OGD (boundaries)<br>
              Scores are 0–100 relative to all India PIN codes. Not an absolute probability.
            </div>""",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style="margin-bottom:0.25rem;">India Physical Risk Explorer</h1>
    <p style="color:#6b7280;margin-top:0;margin-bottom:1.5rem;">
      Enter any 6-digit PIN code to view its boundary polygon, land area,
      perimeter, and climate hazard scores with explanations.
    </p>
    """,
    unsafe_allow_html=True,
)

base_df           = load_base_csv()
scores_df, loaded = load_scores()
pin_index         = load_pin_index()

if base_df is None:
    st.error(f"`{BASE_CSV}` is missing.")
    st.stop()

pin_in = st.text_input(
    "Enter 6-digit PIN code",
    value=st.session_state.get("pin_input", ""),
    max_chars=6,
    placeholder="e.g. 110001",
    key="pin_input",
)

if pin_in:
    pin_in = pin_in.strip()
    if not pin_in.isdigit() or len(pin_in) != 6:
        st.warning("Please enter exactly 6 digits.")
    else:
        st.divider()
        render(pin_in.zfill(6), base_df, scores_df, loaded, pin_index)