"""GeoAI-Prospector | presentation-ready geospatial intelligence dashboard.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import xarray as xr
from folium.plugins import DualMap, Fullscreen, MousePosition
from pyproj import Transformer
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geoai_prospector.model.explain import explain_pixel  # noqa: E402
from geoai_prospector.model.fusion import FEATURES  # noqa: E402

OUT, PROC = ROOT / "outputs", ROOT / "data" / "processed"
st.set_page_config(page_title="GeoAI | Exploration Intelligence", page_icon="🧭", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""<style>
.block-container{max-width:1600px;padding-top:1.5rem;padding-bottom:3rem}
[data-testid="stSidebar"]{background:#f6f8fb;border-right:1px solid #e2e8f0}
[data-testid="stSidebar"] .block-container{padding-top:1.5rem}
h1,h2,h3{letter-spacing:-.025em}
h1{font-size:2rem!important}h2{font-size:1.35rem!important}
[data-testid="stMetric"]{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:14px 18px}
[data-testid="stMetricLabel"]{font-size:.82rem}
[data-testid="stMetricValue"]{font-size:1.55rem}
.stTabs [data-baseweb="tab-list"]{gap:5px;border-bottom:1px solid #e2e8f0;flex-wrap:wrap}
.stTabs [data-baseweb="tab"]{border-radius:9px 9px 0 0;padding:11px 15px;font-weight:600}
.stAlert{border-radius:10px}
[data-testid="stExpander"]{border-radius:10px}
.geo-kicker{font-size:.76rem;font-weight:750;color:#64748b;letter-spacing:.12em;text-transform:uppercase}
.geo-subtitle{color:#64748b;font-size:.95rem;margin-top:-.4rem;margin-bottom:1rem}
.geo-card{padding:16px 18px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;margin:8px 0 14px}
.geo-card strong{color:#0f172a}
.geo-note{font-size:.86rem;color:#475569;line-height:1.6}
@media(prefers-color-scheme:dark){
.geo-card{background:#17212e;border-color:#334155}
.geo-card strong{color:#f8fafc}
.geo-note{color:#cbd5e1}
[data-testid="stMetric"]{background:#17212e;border-color:#334155}
}
</style>""", unsafe_allow_html=True)

# Key-free basemaps. CartoDB Positron/Dark now require an API key in current folium, so they are avoided.
BASEMAPS = {
    "Muted grey (Esri)": "Esri.WorldGrayCanvas",
    "Satellite (Esri)": "Esri.WorldImagery",
    "OpenStreetMap": "OpenStreetMap",
}
# One-line, plain-English description of each selectable base layer.
LAYER_HELP = {
    "mag_rtp": "Airborne magnetic intensity (reduced to pole). Magnetite-rich rocks and structures stand out.",
    "mag_tilt": "Tilt derivative of magnetics: balances strong and weak anomalies to trace contacts and edges.",
    "grav": "Bouguer gravity: bulk rock density. Highs can mark dense intrusions or shallow basement.",
    "grav_hgm": "Horizontal gradient of gravity: peaks over density contrasts such as deep faults.",
    "structural_complexity": "Where CV lineaments of different orientations cross: a proxy for fault intersections.",
    "cnn_recon_error": "Where the CNN autoencoder cannot reconstruct the geophysics: unusual texture worth a look.",
    "geochem_target_anomaly": "Multi-element Cu-Au-Bi anomaly score from the ILR + Isolation-Forest/LOF model.",
}


@st.cache_resource(show_spinner="Loading exploration datasets…")
def load():
    required = [PROC / "feature_stack.nc", PROC / "shap.nc", OUT / "web" / "layers.json",
                OUT / "metrics.json", OUT / "targets.geojson", OUT / "deposits.geojson",
                OUT / "geochem_scored.geojson"]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        return None, missing
    ds = xr.open_dataset(PROC / "feature_stack.nc").load()
    sh = xr.open_dataset(PROC / "shap.nc").load()
    objs = [json.loads(p.read_text()) for p in required[2:]]
    return (ds, sh, *objs), []


loaded, missing = load()
if loaded is None:
    st.title("🧭 GeoAI-Prospector")
    st.warning("The application needs pipeline-generated outputs before the dashboard can open.")
    st.code("python -m geoai_prospector.pipeline --source synthetic\nstreamlit run app/streamlit_app.py",
            language="bash")
    with st.expander("Missing files"):
        st.code("\n".join(missing))
    st.stop()

ds, sh, LAYERS, M, TARGETS, DEPOSITS, GEOCHEM = loaded
FEATS = [str(f) for f in sh["feature"].values]
SV = sh["shap"].values
to_grid = Transformer.from_crs("EPSG:4326", ds.attrs["crs"], always_xy=True)
BOUNDS = LAYERS["prospectivity_pct"]["bounds"]          # [[south, west], [north, east]]
CENTER = [(BOUNDS[0][0] + BOUNDS[1][0]) / 2, (BOUNDS[0][1] + BOUNDS[1][1]) / 2]
ALL_TARGETS = TARGETS.get("features", [])
RAW_LAYERS = [k for k in LAYERS
              if k not in ("prospectivity_pct", "prospectivity_std") and not k.startswith("lineament")]


def overlay(m, name, opacity=0.85):
    layer = LAYERS.get(name)
    if not layer:
        return
    png = OUT / "web" / layer["png"]
    if png.exists():
        folium.raster_layers.ImageOverlay(image=str(png), bounds=layer["bounds"], opacity=opacity,
                                          name=layer["label"], interactive=False, zindex=2).add_to(m)


def frame(m):
    """Zoom the map to the data extent so it isn't a small square in a sea of tiles."""
    m.fit_bounds(BOUNDS, padding=(8, 8))


def add_points(m, top_only, include_deposits=True):
    feats = [f for f in ALL_TARGETS if f.get("properties", {}).get("rank", 99999) <= top_only]
    if feats:
        folium.GeoJson({"type": "FeatureCollection", "features": feats}, name="Ranked targets",
                       style_function=lambda f: {"color": "#00d9ed" if f["properties"].get("status") == "greenfield"
                                                 else "#ffffff", "weight": 2, "fillOpacity": .06},
                       tooltip=folium.GeoJsonTooltip(fields=["rank", "status", "peak_score"],
                                                     aliases=["Rank", "Type", "Score"], localize=True)).add_to(m)
    if include_deposits:
        group = folium.FeatureGroup(name="Known deposits")
        for f in DEPOSITS.get("features", []):
            lon, lat = f["geometry"]["coordinates"][:2]
            folium.CircleMarker([lat, lon], radius=4, color="#082f20", weight=1, fill=True,
                                fill_color="#4ade80", fill_opacity=1,
                                tooltip=str(f.get("properties", {}).get("NAME", "Known deposit"))).add_to(group)
        group.add_to(m)


def basemap_tiles(m, choice):
    folium.TileLayer(BASEMAPS[choice], name="Basemap", control=False).add_to(m)


def help_card(title, text):
    st.markdown(f'<div class="geo-card"><strong>{title}</strong><div class="geo-note">{text}</div></div>',
                unsafe_allow_html=True)


def metric_value(section, key, default=float("nan")):
    try:
        return float(M[section][key])
    except (KeyError, TypeError, ValueError):
        return default


# ========================= sidebar =========================
with st.sidebar:
    st.markdown('<div class="geo-kicker">Exploration intelligence</div>', unsafe_allow_html=True)
    st.title("🧭 GeoAI")
    st.caption("PROSPECTOR · EAST TENNANT, NT")
    st.divider()
    st.markdown("**Study area**")
    st.caption("IOCG copper–gold · GDA2020 / MGA53 · 500 m analysis grid")
    st.markdown("**Data source**")
    source = str(M.get("data_source", M.get("source", "Not recorded")))
    st.info(f"Run source: **{source}**. Confirm provenance in the model card before presenting results.")
    st.divider()
    st.markdown("**Map controls**")
    raw_choice = st.selectbox("Geophysical layer", RAW_LAYERS, format_func=lambda k: LAYERS[k]["label"])
    st.caption(LAYER_HELP.get(raw_choice, ""))
    show_lin = st.toggle("Show interpreted lineaments", value=True)
    right_choice = st.radio("Right map shows", ["prospectivity_pct", "prospectivity_std"],
                            format_func=lambda k: "Prospectivity" if k == "prospectivity_pct" else "Uncertainty",
                            horizontal=True)
    basemap = st.selectbox("Basemap", list(BASEMAPS), index=0)
    max_targets = max(1, len(ALL_TARGETS))
    n_targets = st.slider("Number of targets", min_value=1, max_value=max_targets, value=min(20, max_targets))
    opacity = st.slider("Overlay transparency", min_value=.2, max_value=1.0, value=.85, step=.05)
    st.divider()
    st.caption("Lineaments are detected geophysical edges, not confirmed faults. Target scores are relative "
               "rankings, not deposit probabilities.")
    with st.expander("Map symbol guide"):
        st.markdown("**Cyan boundaries:** greenfield targets  \n**White boundaries:** targets near known deposits  "
                    "\n**Green points:** recorded deposits  \n**Cyan/white lines:** detected geophysical lineaments")

# ========================= header =========================
st.markdown('<div class="geo-kicker">Australian geospatial AI · research demonstration</div>', unsafe_allow_html=True)
st.title("GeoAI-Prospector | Exploration Intelligence")
st.markdown('<div class="geo-subtitle">Explore geophysical evidence, compare ranked targets, and inspect model '
            'explanations in one workspace.</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Spatial CV · ROC-AUC", f"{metric_value('spatial_cv', 'roc_auc'):.3f}")
k2.metric("Spatial CV · PR-AUC", f"{metric_value('spatial_cv', 'pr_auc'):.3f}")
k3.metric("Deposits in top 10%", f"{metric_value('capture', 'deposits_in_top_10pct_area'):.0%}")
k4.metric("Ranked target areas", f"{len(ALL_TARGETS):,}")
st.caption("Metrics are calculated from the selected pipeline outputs. Synthetic results must not be presented as "
           "independently validated field performance.")

intro, maps, explain, targets_tab, validation = st.tabs([
    "◉ Overview", "▦ Exploration map", "⌖ Location analysis", "◎ Target register", "▤ Model validation"])

# ========================= overview =========================
with intro:
    st.subheader("From geological measurements to prioritised exploration areas")
    help_card("What this platform does", "The system combines gravity and magnetic grids, geochemical anomaly "
              "indicators, and geological structure features. A trained classifier ranks grid cells by their "
              "similarity to the available known mineralisation evidence.")
    a, bcol, c = st.columns(3)
    with a:
        help_card("01 · Detect geological patterns", "Image processing and a CNN-derived reconstruction signal "
                  "identify patterns in geophysical rasters. Detected edges require geological interpretation.")
    with bcol:
        help_card("02 · Combine evidence", "Geophysical, structural, and geochemical features are aligned to a "
                  "common analysis grid and passed into the prospectivity model.")
    with c:
        help_card("03 · Prioritise and explain", "The model produces relative rankings, candidate target polygons, "
                  "uncertainty indicators, and feature-level SHAP explanations.")
    st.subheader("How to present this demonstration")
    st.markdown("1. Open **Exploration map** and compare one physical evidence layer with the ranked output.\n"
                "2. Open **Location analysis** and click a target to see the supporting features.\n"
                "3. Open **Target register** to filter and export areas for further investigation.\n"
                "4. Open **Model validation** to explain spatial cross-validation and limitations.")
    st.warning("**Scientific interpretation:** No-known-deposit areas are not proven barren. Model scores indicate "
               "relative priority, not the probability of a commercially viable deposit.")

# ========================= exploration map =========================
with maps:
    st.subheader("Synchronized exploration maps")
    right_label = "ranked prospectivity" if right_choice == "prospectivity_pct" \
        else "model uncertainty (rank spread across CV folds)"
    left, right = st.columns(2)
    left.markdown(f"**Geophysical evidence:** {LAYERS[raw_choice]['label']}")
    right.markdown(f"**Model output:** {right_label}")
    dm = DualMap(location=CENTER, zoom_start=9, tiles=None, layout="horizontal")
    for side in (dm.m1, dm.m2):
        basemap_tiles(side, basemap)
    overlay(dm.m1, raw_choice, opacity)
    if show_lin:
        overlay(dm.m1, "lineament_deep", 1.0)
        overlay(dm.m1, "lineament_shallow", .65)
    overlay(dm.m2, right_choice, opacity)
    add_points(dm.m2, n_targets)
    frame(dm.m1)
    frame(dm.m2)
    folium.LayerControl(collapsed=True).add_to(dm)
    components.html(dm.get_root().render(), height=520, scrolling=False)
    with st.expander("Understanding this map", expanded=True):
        c1, c2 = st.columns(2)
        c1.markdown("**Left: observed or derived evidence**  \nSelect a geophysical layer in the sidebar. The "
                    "coloured image shows spatial variation; overlaid lines are algorithmically detected gradients, "
                    "not confirmed faults.")
        c2.markdown("**Right: relative exploration priority**  \nBright areas indicate higher model rankings. Cyan "
                    "polygons are greenfield candidates, white polygons overlap known deposits, green dots mark "
                    "recorded deposits. Switch the right map to **uncertainty** in the sidebar.")
    st.caption("Both maps pan and zoom together, framed to the analysis footprint. The surrounding basemap is "
               "contextual only.")

# ========================= location analysis =========================
with explain:
    st.subheader("Explain a selected map location")
    st.caption("Select a point inside the coloured analysis area to inspect its ranking and model drivers.")
    mcol, detailcol = st.columns([1.45, 1], gap="large")
    with mcol:
        fmap = folium.Map(location=CENTER, zoom_start=9, tiles=None)
        basemap_tiles(fmap, basemap)
        overlay(fmap, "prospectivity_pct", opacity)
        add_points(fmap, n_targets)
        frame(fmap)
        MousePosition().add_to(fmap)
        Fullscreen().add_to(fmap)
        state = st_folium(fmap, height=490, use_container_width=True, returned_objects=["last_clicked"],
                          key="explanation_map")
    with detailcol:
        click = (state or {}).get("last_clicked")
        if click:
            x, y = to_grid.transform(click["lng"], click["lat"])
            r = int(np.argmin(np.abs(ds.y.values - y)))
            c = int(np.argmin(np.abs(ds.x.values - x)))
        elif ALL_TARGETS:
            p = ALL_TARGETS[0]["properties"]
            r, c = int(p["row"]), int(p["col"])
            st.info("Showing target #1. Click the map to analyse another location.")
        else:
            r = c = 0
            st.info("No ranked targets available; showing the first grid cell.")
        if not np.isfinite(ds["prospectivity"].values[r, c]):
            st.warning("This location is outside the model's valid data footprint.")
        else:
            info = explain_pixel(ds, SV, FEATS, r, c, top=4)
            top_pct = max(.1, 100 * (1 - info["percentile"]))
            st.metric("Relative prospectivity score", f"{info['score']:.2f}", f"Top {top_pct:.1f}% of analysed area")
            st.markdown("**Interpretation**")
            st.write(info["text"])
            if "prospectivity_std" in ds:
                spread = float(ds["prospectivity_std"].values[r, c])
                st.caption(f"Across-fold variation: ±{spread:.1f} percentile points. Larger variation indicates a "
                           "less stable ranking - collect data before trusting it.")
            contrib = pd.DataFrame({"Feature": [FEATURES[f][0] for f in FEATS], "Contribution": SV[:, r, c]})
            contrib = contrib.reindex(contrib["Contribution"].abs().sort_values(ascending=False).index).head(8)
            st.markdown("**Strongest model drivers**")
            st.bar_chart(contrib.set_index("Feature")["Contribution"], horizontal=True, height=280)
            st.caption("SHAP contributions are on the model's log-odds scale. Positive values raise its prediction; "
                       "negative values lower it.")
            with st.expander("All input feature values"):
                st.dataframe(pd.DataFrame({"Feature": [FEATURES[f][0] for f in FEATS],
                                           "Value": [float(ds[f].values[r, c]) for f in FEATS]}),
                             hide_index=True, width="stretch")

# ========================= target register =========================
with targets_tab:
    st.subheader("Ranked exploration target register")
    st.caption("These are model-generated candidate areas requiring expert review and independent verification.")
    df = pd.DataFrame([f["properties"] for f in ALL_TARGETS])
    if df.empty:
        st.info("No targets generated.")
    else:
        f1, f2 = st.columns([1, 2])
        status_options = ["All"] + sorted(df["status"].dropna().unique().tolist())
        status_filter = f1.selectbox("Target type", status_options)
        if status_filter != "All":
            df = df[df["status"] == status_filter]
        search = f2.text_input("Search target rank or explanation", placeholder="e.g. 3 or gravity")
        if search:
            df = df[df.astype(str).apply(lambda x: x.str.contains(search, case=False, regex=False)).any(axis=1)]
        columns = [x for x in ["rank", "status", "peak_score", "peak_std", "area_km2", "peak_x", "peak_y",
                               "explanation"] if x in df]
        st.dataframe(df[columns], width="stretch", hide_index=True,
                     column_config={
                         "peak_score": st.column_config.ProgressColumn("Relative score", min_value=0, max_value=1),
                         "peak_std": st.column_config.NumberColumn("± rank (pts)", format="%.1f"),
                         "area_km2": st.column_config.NumberColumn("Area km²", format="%.1f"),
                         "explanation": st.column_config.TextColumn("Model explanation", width="large")})
        st.caption(f"Showing {len(df)} of {len(ALL_TARGETS)} generated target areas.")
    st.markdown("**Export results**")
    buttons = [("Target polygons · GeoJSON", "targets.geojson"), ("GIS layer · GeoPackage", "targets.gpkg"),
               ("Prospectivity raster · GeoTIFF", "prospectivity.tif"), ("Technical report · Markdown", "REPORT.md")]
    for col, (label, filename) in zip(st.columns(4), buttons, strict=True):
        file = OUT / filename
        with col:
            if file.exists():
                st.download_button(label, data=file.read_bytes(), file_name=filename, key=f"dl_{filename}",
                                   width="stretch")
            else:
                st.button(label, disabled=True, help=f"Missing generated file: {filename}", width="stretch")

# ========================= model validation =========================
with validation:
    st.subheader("Model validation and scientific transparency")
    help_card("Why spatial validation matters", "Nearby pixels share geological patterns. Random train/test splits "
              "can therefore overestimate performance. Buffered spatial cross-validation evaluates geographically "
              "separated areas and is the primary reported estimate.")
    a, bcol = st.columns(2, gap="large")
    with a:
        st.markdown("**Spatial versus random cross-validation**")
        vals = pd.DataFrame({"ROC-AUC": [metric_value("spatial_cv", "roc_auc"), metric_value("random_cv", "roc_auc")],
                             "PR-AUC": [metric_value("spatial_cv", "pr_auc"), metric_value("random_cv", "pr_auc")]},
                            index=["Buffered spatial CV", "Random CV (optimistic)"])
        st.dataframe(vals.round(3), width="stretch")
        st.caption(f"Spatial exclusion buffer: {metric_value('spatial_cv', 'buffer_m', 0) / 1000:.1f} km")
        if M.get("ablation"):
            st.markdown("**Contribution of different feature groups**")
            st.bar_chart(pd.Series({k: v.get("roc_auc", float("nan")) for k, v in M["ablation"].items()},
                                   name="ROC-AUC"))
        if M.get("cnn_loss"):
            with st.expander("CNN reconstruction training curve"):
                st.line_chart(pd.Series(M["cnn_loss"], name="Reconstruction loss"))
    with bcol:
        st.markdown("**Global model explanations**")
        if (OUT / "shap_beeswarm.png").exists():
            st.image(str(OUT / "shap_beeswarm.png"), width="stretch")
        else:
            st.info("SHAP summary image not generated.")
        if (OUT / "shap_target1.png").exists():
            with st.expander("Detailed explanation of target #1"):
                st.image(str(OUT / "shap_target1.png"), width="stretch")
    st.markdown("**Limitations and responsible use**")
    st.markdown("- **Positive-unlabelled learning:** an area without a recorded deposit is not necessarily barren.\n"
                "- **Exploration bias:** geochemical samples are unevenly distributed and may reflect historical "
                "activity. Distance-to-sample is deliberately excluded from the features.\n"
                "- **Geophysical interpretation:** detected edges are not independently verified faults; mapped "
                "faults enter as a separate feature.\n"
                "- **Mineral specificity:** this model addresses IOCG Cu-Au; lithium and cobalt require separate "
                "geological evidence and validation.\n"
                "- **Uncalibrated scores:** model scores are not probabilities of economic mineralisation.\n"
                "- **Independent verification:** fieldwork, geological interpretation and additional surveys are "
                "required before exploration decisions.")
    with st.expander("Run metadata"):
        st.json({k: v for k, v in M.items() if k not in {"ablation", "cnn_loss"}}, expanded=False)

st.divider()
st.caption("GeoAI-Prospector · Research and decision-support demonstration · Not a substitute for geological "
           "assessment, land-access checks or regulatory approvals.")
