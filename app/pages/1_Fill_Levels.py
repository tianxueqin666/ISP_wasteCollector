import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import altair as alt
from datetime import timedelta

st.set_page_config(page_title="Waste Collector — Forecast", layout="wide")

# Logo path 
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Logo.jpeg")

# Top row
col_left, col_right = st.columns([8, 2])
with col_left:
    st.title("Waste Collector — Fill-level Forecast")
    st.markdown("Simple demo app that loads a saved Keras model + pickled scalers and runs per-bin predictions.")
with col_right:
    try:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=100)
    except Exception:
        pass

# Paths 
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

# Candidate model folders
MODEL_FOLDER_CANDIDATES = [
    os.path.join(REPO_ROOT, "outputs", "models"),
    os.path.join(REPO_ROOT, "..", "outputs", "models"),
]

MODEL_FOLDER = next((p for p in MODEL_FOLDER_CANDIDATES if os.path.isdir(p)), MODEL_FOLDER_CANDIDATES[0])
PROCESSED_CSV_CANDIDATES = [
    os.path.join(REPO_ROOT, "data", "processed_filllevel.csv"),
    os.path.join(REPO_ROOT, "..", "data", "processed_filllevel.csv"),
]
PROCESSED_CSV = next((p for p in PROCESSED_CSV_CANDIDATES if os.path.exists(p)), PROCESSED_CSV_CANDIDATES[0])

# Default model/scaler 
DEFAULT_MODEL = os.path.join(MODEL_FOLDER, "bilstm_model.keras")
DEFAULT_SCALER = os.path.join(MODEL_FOLDER, "bilstm_scalers.pkl")
 

# Sidebar
with st.sidebar:
    st.header("Controls")

    models = []
    if os.path.isdir(MODEL_FOLDER):
        for name in os.listdir(MODEL_FOLDER):
            if name.endswith(".keras"):
                models.append(os.path.join(MODEL_FOLDER, name))
    models = sorted(models)

    model_choice = st.selectbox("Pick model file", options=[DEFAULT_MODEL] + models if models else [DEFAULT_MODEL])

    scaler_choice = st.text_input("Scaler path (pickle)", value=DEFAULT_SCALER)

    n_steps = st.number_input("Sequence length (n_steps)", value=30, min_value=1, max_value=365)

    st.markdown("---")
    st.markdown("Data range / bin selection")

    df = None
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

            available_serials = serials
            try:
                if os.path.exists(scaler_choice):
                    with open(scaler_choice, "rb") as fh:
                        scalers_obj = pickle.load(fh)
                    if isinstance(scalers_obj, dict):
                        scaler_keys = set([str(k) for k in scalers_obj.keys()])
                        available_serials = [s for s in serials if str(s) in scaler_keys]
            except Exception as e:
                st.sidebar.warning(f"Could not read scaler file to filter bins: {e}")
                available_serials = serials
        except Exception as e:
            st.warning(f"Failed to read processed CSV: {e}")
            df = None
            min_date = None
            max_date = None
            serials = []
    else:
        st.warning(f"Processed CSV not found at {PROCESSED_CSV}. You can upload one or provide sequences manually.")
        min_date = None
        max_date = None
        serials = []

    # Only request end_date (model predicts next day after end_date)
    if min_date and max_date:
        end_date = st.date_input("End date (prediction baseline)", value=max_date, min_value=min_date, max_value=max_date)
    else:
        end_date = st.date_input("End date (prediction baseline)")

    # Default selection method
    selection_method = st.radio("Selection method:", ["Dropdown", "Map-based"], horizontal=True)

    if selection_method == "Dropdown":
        selected_bins = st.multiselect("Select bin serialNumber(s)", options=available_serials, default=available_serials[:3])
    else:
        selected_bins = []  

    # st.markdown("---")
    predict_btn = st.button("Predict")

@st.cache_resource
def load_keras_model(path):
    import tensorflow as tf
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    model = tf.keras.models.load_model(path)
    return model

@st.cache_data
def load_scalers(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    return obj


def plot_time_series_df(series, title=None):
    try:
        df_plot = series.reset_index()
        df_plot.columns = ["timestamp", "value"]
        chart = (
            alt.Chart(df_plot)
            .mark_line(color="#81b43a")
            .encode(x="timestamp:T", y="value:Q", tooltip=["timestamp", "value"]) 
            .interactive()
        )
        if title:
            chart = chart.properties(title=title)
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.write(f"Could not render chart: {e}")

model = None
scalers = None
model_load_error = None
scaler_missing = False

try:
    model = load_keras_model(model_choice)
except Exception as e:
    model_load_error = str(e)

scalers = load_scalers(scaler_choice)
if scalers is None:
    scaler_missing = True

# Main Panel
if selection_method == "Map-based" and df is not None:
    st.subheader("Select bins by geographic area")
    
    # Prepare map data
    bin_locations = df.drop_duplicates(subset=["serialNumber"])[["serialNumber", "latitude", "longitude"]].copy()
    bin_locations["serialNumber_str"] = bin_locations["serialNumber"].astype(str)
    brush = alt.selection_interval(name="brush")
    
    # Calculate data bounds with 10% padding for better visibility
    lat_min, lat_max = bin_locations["latitude"].min(), bin_locations["latitude"].max()
    lon_min, lon_max = bin_locations["longitude"].min(), bin_locations["longitude"].max()
    
    lat_padding = (lat_max - lat_min) * 0.1
    lon_padding = (lon_max - lon_min) * 0.1
    
    lat_domain = [lat_min - lat_padding, lat_max + lat_padding]
    lon_domain = [lon_min - lon_padding, lon_max + lon_padding]
    
    map_chart = alt.Chart(bin_locations).mark_circle(size=150, color="#81b43a", opacity=0.8).encode(
        x=alt.X("longitude:Q", scale=alt.Scale(domain=lon_domain)),
        y=alt.Y("latitude:Q", scale=alt.Scale(domain=lat_domain)),
        tooltip=["serialNumber:N", "latitude:Q", "longitude:Q"],
        color=alt.condition(brush, alt.value("#81b43a"), alt.value("#ccc"))
    ).add_params(
        brush
    ).properties(
        width=900,
        height=600,
        title="Bins Map — Drag to select region"
    ).interactive()
    
    # Display chart and capture selection
    selected_data = st.altair_chart(map_chart, use_container_width=True, on_select="rerun")
    
    if selected_data and "selection" in selected_data and "brush" in selected_data["selection"]:
        brush_data = selected_data["selection"]["brush"]
        
        mask = pd.Series([True] * len(bin_locations), index=bin_locations.index)
        for key, value in brush_data.items():
            if key == "longitude" and isinstance(value, list) and len(value) == 2:
                mask &= (bin_locations["longitude"] >= value[0]) & (bin_locations["longitude"] <= value[1])
            elif key == "latitude" and isinstance(value, list) and len(value) == 2:
                mask &= (bin_locations["latitude"] >= value[0]) & (bin_locations["latitude"] <= value[1])
        
        selected_bins = bin_locations.loc[mask, "serialNumber_str"].tolist()
        if selected_bins:
            st.session_state.selected_bins_map = selected_bins
            st.success(f"Selected {len(selected_bins)} bin(s)")
        else:
            st.session_state.selected_bins_map = []
    elif "selected_bins_map" in st.session_state and st.session_state.selected_bins_map:
        st.info(f"Currently selected: {len(st.session_state.selected_bins_map)} bin(s)")
    else:
        st.info("Drag a rectangle to select bins. Click 'Predict' to use selected bins.")


# Map predictions
if predict_btn:
    # Determine which bins to use based on selection method
    if selection_method == "Map-based":
        if "selected_bins_map" in st.session_state and st.session_state.selected_bins_map:
            bins_to_predict = st.session_state.selected_bins_map
        else:
            st.warning("No bins selected from map. Please drag to select bins on the map above.")
            bins_to_predict = []
    else:
        bins_to_predict = selected_bins
    
    if not bins_to_predict:
        st.error("No bins selected. Please select bins using the dropdown or map.")
    elif model_load_error:
        st.error(f"Model load failed: {model_load_error}")
    elif model is None:
        st.error("Model not loaded.")
    elif scaler_missing:
        st.error(f"Scalers file not found or failed to load: {scaler_choice}")
    elif df is None:
        st.error("No data available. Ensure processed CSV exists.")
    else:
        results = []
        charts_data = []  
        for bin_id in bins_to_predict:
            bin_id_orig = bin_id
            bin_mask = df["serialNumber"].astype(str) == str(bin_id_orig)
            df_bin = df[bin_mask].sort_values("timestamp")
            if df_bin.empty:
                st.warning(f"No data for bin {bin_id_orig}")
                continue

            # take last n_steps up to end_date
            cutoff = pd.to_datetime(end_date)
            df_cut = df_bin[df_bin["timestamp"] <= cutoff].copy()
            if len(df_cut) < n_steps:
                st.warning(f"Bin {bin_id_orig}: not enough data before {end_date} (have {len(df_cut)}, need {n_steps})")
                continue
            seq = df_cut["latestFullness"].values[-n_steps:]
            # get scaler for this bin 
            scaler = None
            if isinstance(scalers, dict):
                scaler = scalers.get(bin_id_orig) or scalers.get(int(bin_id_orig)) or scalers.get(str(bin_id_orig))
            else:
                scaler = scalers

            if scaler is None:
                st.warning(f"No scaler found for bin {bin_id_orig}, skipping")
                continue

            # Predict
            try:
                seq_reshaped = np.array(seq).reshape(-1, 1)
                seq_scaled = scaler.transform(seq_reshaped).reshape(1, n_steps, 1)
                y_pred_norm = model.predict(seq_scaled, verbose=0).flatten()[0]
                y_pred_orig = scaler.inverse_transform([[y_pred_norm]])[0, 0]

                pred_date = pd.to_datetime(end_date) + timedelta(days=1)
                results.append({"serialNumber": bin_id_orig, "predicted_date": pred_date.date(), "predicted_fill": float(y_pred_orig)})

                hist = df_cut.set_index("timestamp")["latestFullness"].tail(n_steps * 3)
                chart_df = pd.concat([hist, pd.Series({pd.to_datetime(pred_date): y_pred_orig})])
                charts_data.append((bin_id_orig, chart_df))
            except Exception as e:
                st.error(f"Prediction failed for bin {bin_id_orig}: {e}")

        if results:
            res_df = pd.DataFrame(results)
            st.success("Predictions ready")
            st.dataframe(res_df)

            # Export
            st.markdown("---")
            st.subheader("Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = res_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            with col2:
                json_data = res_df.to_json(orient="records", indent=2).encode("utf-8")
                st.download_button(
                    label="Download as JSON",
                    data=json_data,
                    file_name=f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col3:
                try:
                    import io
                    excel_buffer = io.BytesIO()
                    res_df.to_excel(excel_buffer, index=False, sheet_name="Predictions")
                    excel_buffer.seek(0)
                    st.download_button(
                        label="📊 Download as Excel",
                        data=excel_buffer.getvalue(),
                        file_name=f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except ImportError:
                    st.info("Install `openpyxl` for Excel export: `pip install openpyxl`")

            charts_container = st.container()
            with charts_container:
                for bin_id_orig, chart_series in charts_data:
                    st.subheader(f"Bin {bin_id_orig}")
                    plot_time_series_df(chart_series, title=f"Bin {bin_id_orig}")
else:
    st.info("Configure parameters in the sidebar and click Predict.")

st.sidebar.markdown("---")
st.sidebar.write(f"**Model file**: `{os.path.basename(model_choice)}`")
st.sidebar.write(f"**Scaler file**: `{os.path.basename(scaler_choice)}`")
if model_load_error:
    st.sidebar.error("Model failed to load — check logs")


