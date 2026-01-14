import os
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import tensorflow as tf
from datetime import datetime, timedelta

st.set_page_config(page_title="Waste Collector — Fill-rate ", layout="wide")

# Header + Logo
LOGO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Logo.jpeg")

col_left, col_right = st.columns([8, 2])
with col_left:
    st.title("Waste Collector — Fill-rate (Accumulated)")
    # st.markdown("Loads a saved model + X_test/y_test, predicts fill_rates, and plots accumulated fill level.")
with col_right:
    try:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=100)
    except Exception:
        pass

#paths
HERE = os.path.dirname(os.path.abspath(__file__))          # .../pages
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))         # project root

MODEL_PATH = os.path.join(REPO_ROOT, "outputs", "models", "lstm_fill_rate_model.keras")
X_PATH = os.path.join(REPO_ROOT, "data", "X_test.npy")
Y_PATH = os.path.join(REPO_ROOT, "data", "y_test.npy")
PROCESSED_CSV = os.path.join(REPO_ROOT, "data", "processed_filllevel.csv")

# Cache loaders
@st.cache_resource
def load_model_cached(path: str):
    return tf.keras.models.load_model(path)

@st.cache_data
def load_test_npy(x_path: str, y_path: str):
 
    return np.load(x_path), np.load(y_path)

# Load processed CSV for selection UI
df = None
available_serials = []
df = pd.read_csv(PROCESSED_CSV, parse_dates=["date"])
df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["timestamp"])
available_serials = sorted(df["serialNumber"].astype(str).unique().tolist())

# Sidebar: selection + days + predict button
with st.sidebar:
    st.header("Controls")

    forecast_days = st.number_input(
        "Amount of days to predict",
        min_value=1,
        max_value=365,
        value=7
    )

    selection_method = st.radio("Selection method:", ["Dropdown", "Map-based"], horizontal=True)

    if selection_method == "Dropdown":
        selected_bins = st.multiselect(
            "Select bin serialNumber(s)",
            options=available_serials,
            default=available_serials[:4] if len(available_serials) >= 4 else available_serials
        )
    else:
        selected_bins = []  # set from map

    predict_btn = st.button("Predict")

# Map selection (copied behavior from app.py)
if selection_method == "Map-based" and df is not None:
    st.subheader("Select bins by geographic area")

    bin_locations = df.drop_duplicates(subset=["serialNumber"])[["serialNumber", "latitude", "longitude"]].copy()
    bin_locations["serialNumber_str"] = bin_locations["serialNumber"].astype(str)

    brush = alt.selection_interval(name="brush")

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
        st.session_state.selected_bins_map = selected_bins_map

        if selected_bins_map:
            st.success(f"Selected {len(selected_bins_map)} bin(s)")
        else:
            st.session_state.selected_bins_map = []
            st.info("No bins in that selection.")
    elif "selected_bins_map" in st.session_state and st.session_state.selected_bins_map:
        st.info(f"Currently selected: {len(st.session_state.selected_bins_map)} bin(s)")
    else:
        st.info("Drag a rectangle to select bins. Click 'Predict' to plot.")

# Predict + Accumulate + Plot 
def _monday_of_current_week(d: datetime) -> datetime:
    return d - timedelta(days=d.weekday())

if predict_btn:
    # Determine bins selected
    if selection_method == "Map-based":
        bins_to_plot = st.session_state.get("selected_bins_map", [])
    else:
        bins_to_plot = selected_bins

    if not bins_to_plot:
        st.error("No bins selected. Select bins via dropdown or map.")
        st.stop()

    # Load model
    try:
        model = load_model_cached(MODEL_PATH)
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

    # Load data
    try:
        X_test, y_test = load_test_npy(X_PATH, Y_PATH)
    except Exception as e:
        st.error(f"Test data load failed: {e}")
        st.stop()

    # Predict fill rates
    try:
        fill_rates = model.predict(X_test, verbose=0)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    fill_rates_flat = np.array(fill_rates).reshape(-1)

    n_days = int(forecast_days)
    n_bins_requested = len(bins_to_plot)

    # We FORCE the plot to have 1 line per selected bin by taking the first (bins * days) preds
    needed = n_bins_requested * n_days

    if fill_rates_flat.size < needed:
        n_bins = max(1, fill_rates_flat.size // n_days)
        needed = n_bins * n_days
        st.warning(
            f"Not enough predictions for {n_bins_requested} bins × {n_days} days "
            f"(need {n_bins_requested*n_days}, got {fill_rates_flat.size}). "
            f"Plotting {n_bins} bin(s) instead."
        )
    else:
        n_bins = n_bins_requested

    fill_rates_2d = fill_rates_flat[:needed].reshape(n_bins, n_days)

    # Accumulate
    fill_levels = np.cumsum(fill_rates_2d, axis=1)

    labels = [str(b) for b in bins_to_plot[:n_bins]]

    # Create weekday/date axis
    start = _monday_of_current_week(datetime.now())
    dates = pd.date_range(start=start.date(), periods=n_days, freq="D")
    x_labels = [d.strftime("%A") for d in dates]


    # Long df for Altair
    rows = []
    for i in range(n_bins):
        for t in range(n_days):
            rows.append({"x": x_labels[t], "bin": labels[i], "fill_level": float(fill_levels[i, t])})
    plot_df = pd.DataFrame(rows)

    st.subheader("Predicted cummulative Fill-Rates")

    base = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            # x=alt.X("x:N", sort=x_labels, title="Weekday / Date"),
            x=alt.X("x:N", sort=x_labels, title="Weekday / Date", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("fill_level:Q", title="Fill level"),
            color=alt.Color("bin:N", title=""),
            tooltip=["bin:N", "x:N", "fill_level:Q"],
        )
    )

    full_line = (
        alt.Chart(pd.DataFrame({"y": [10]}))
        .mark_rule(color="red", strokeDash=[6, 4])
        .encode(y="y:Q")
    )

    st.altair_chart((base + full_line).properties(height=450), use_container_width=True)

    st.markdown("---")
    st.write("**Bins selected:**", bins_to_plot)
    st.write("**Plotted bins:**", labels)
    st.write("**fill_rates shape:**", np.array(fill_rates).shape)
    st.write("**Used predictions:**", needed)
    st.write("**Accumulated fill_levels shape:**", fill_levels.shape)

else:
    st.info("Pick bins (dropdown/map), set amount of days, then click Predict.")
