import streamlit as st
import pandas as pd
import numpy as np
import os
import altair as alt

st.set_page_config(page_title="Waste Collector — Selection", layout="wide")

# Logo path (same idea as app.py)
LOGO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Logo.jpeg")

# Top row
col_left, col_right = st.columns([8, 2])
with col_left:
    st.title("Waste Collector — Bin Selection")
    st.markdown("Same selection panel as the forecast app (no prediction here yet).")
with col_right:
    try:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=100)
    except Exception:
        pass

# Paths (copy the robust approach from app.py)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

PROCESSED_CSV_CANDIDATES = [
    os.path.join(REPO_ROOT, "data", "processed_filllevel.csv"),
    os.path.join(REPO_ROOT, "..", "data", "processed_filllevel.csv"),
]
PROCESSED_CSV = next((p for p in PROCESSED_CSV_CANDIDATES if os.path.exists(p)), PROCESSED_CSV_CANDIDATES[0])

# Load data (same behavior as app.py)
df = None
min_date = None
max_date = None
serials = []
available_serials = []

if os.path.exists(PROCESSED_CSV):
    try:
        df = pd.read_csv(PROCESSED_CSV, parse_dates=["date"])
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        else:
            df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

        min_date = df["timestamp"].min().date()
        max_date = df["timestamp"].max().date()

        serials = sorted(df["serialNumber"].astype(str).unique().tolist())
        available_serials = serials[:]  # no scaler filtering on this page
    except Exception as e:
        st.warning(f"Failed to read processed CSV: {e}")
        df = None
else:
    st.warning(f"Processed CSV not found at {PROCESSED_CSV}.")
    df = None

# Sidebar (bin selection exactly like app.py, but with forecast_days instead of end_date)
with st.sidebar:
    st.header("Controls")

    st.markdown("---")
    st.markdown("Data range / bin selection")

    # Days instead of end-date
    forecast_days = st.number_input("Forecast horizon (days)", min_value=1, max_value=365, value=7)
    st.session_state["forecast_days"] = int(forecast_days)

    # Same selection method UI
    selection_method = st.radio("Selection method:", ["Dropdown", "Map-based"], horizontal=True)
    st.session_state["selection_method"] = selection_method

    if selection_method == "Dropdown":
        selected_bins = st.multiselect(
            "Select bin serialNumber(s)",
            options=available_serials,
            default=available_serials[:3] if len(available_serials) >= 3 else available_serials
        )
        st.session_state["selected_bins_dropdown"] = selected_bins
    else:
        selected_bins = []  # map selection happens in main panel (same as app.py)

# Main panel (map selection block copied from app.py, but no Predict button / no predictions)
if selection_method == "Map-based" and df is not None:
    st.subheader("📍 Select bins by geographic area")

    # Prepare map data
    bin_locations = df.drop_duplicates(subset=["serialNumber"])[["serialNumber", "latitude", "longitude"]].copy()
    bin_locations["serialNumber_str"] = bin_locations["serialNumber"].astype(str)
    brush = alt.selection_interval(name="brush")

    # bounds with padding
    lat_min, lat_max = bin_locations["latitude"].min(), bin_locations["latitude"].max()
    lon_min, lon_max = bin_locations["longitude"].min(), bin_locations["longitude"].max()

    lat_padding = (lat_max - lat_min) * 0.1 if lat_max != lat_min else 0.01
    lon_padding = (lon_max - lon_min) * 0.1 if lon_max != lon_min else 0.01

    lat_domain = [lat_min - lat_padding, lat_max + lat_padding]
    lon_domain = [lon_min - lon_padding, lon_max + lon_padding]

    map_chart = (
        alt.Chart(bin_locations)
        .mark_circle(size=150, color="#81b43a", opacity=0.8)
        .encode(
            x=alt.X("longitude:Q", scale=alt.Scale(domain=lon_domain)),
            y=alt.Y("latitude:Q", scale=alt.Scale(domain=lat_domain)),
            tooltip=["serialNumber:N", "latitude:Q", "longitude:Q"],
            color=alt.condition(brush, alt.value("#81b43a"), alt.value("#ccc")),
        )
        .add_params(brush)
        .properties(width=900, height=600, title="Bins Map — Drag to select region")
        .interactive()
    )

    selected_data = st.altair_chart(map_chart, use_container_width=True, on_select="rerun")

    if selected_data and "selection" in selected_data and "brush" in selected_data["selection"]:
        brush_data = selected_data["selection"]["brush"]

        mask = pd.Series([True] * len(bin_locations), index=bin_locations.index)
        for key, value in brush_data.items():
            if key == "longitude" and isinstance(value, list) and len(value) == 2:
                mask &= (bin_locations["longitude"] >= value[0]) & (bin_locations["longitude"] <= value[1])
            elif key == "latitude" and isinstance(value, list) and len(value) == 2:
                mask &= (bin_locations["latitude"] >= value[0]) & (bin_locations["latitude"] <= value[1])

        selected_bins_map = bin_locations.loc[mask, "serialNumber_str"].tolist()
        st.session_state["selected_bins_map"] = selected_bins_map

        if selected_bins_map:
            st.success(f"Selected {len(selected_bins_map)} bin(s)")
        else:
            st.session_state["selected_bins_map"] = []
            st.info("No bins in that selection.")
    elif "selected_bins_map" in st.session_state and st.session_state["selected_bins_map"]:
        st.info(f"Currently selected: {len(st.session_state['selected_bins_map'])} bin(s)")
    else:
        st.info("Drag a rectangle to select bins.")

# Summary (so you can see it’s working)
st.markdown("---")
st.subheader("Current selection")

if selection_method == "Dropdown":
    st.write("**Method:** Dropdown")
    st.write("**Selected bins:**", st.session_state.get("selected_bins_dropdown", []))
else:
    st.write("**Method:** Map-based")
    st.write("**Selected bins:**", st.session_state.get("selected_bins_map", []))

st.write("**Forecast horizon (days):**", st.session_state.get("forecast_days", 7))
