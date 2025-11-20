import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from datetime import timedelta

st.set_page_config(page_title="Waste Collector — Forecast", layout="wide")

st.title("Waste Collector — Fill-level Forecast")
st.markdown("Simple demo app that loads a saved Keras model + pickled scalers and runs per-bin predictions.")

# Paths (relative to repo root)
# Resolve model/data paths relative to the repo root (two levels up from this file)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# Candidate model folders (handle being run from project root or from app/)
MODEL_FOLDER_CANDIDATES = [
    os.path.join(REPO_ROOT, "outputs", "models"),
    os.path.join(REPO_ROOT, "..", "outputs", "models"),
]

# Pick the first existing model folder, falling back to the first candidate
MODEL_FOLDER = next((p for p in MODEL_FOLDER_CANDIDATES if os.path.isdir(p)), MODEL_FOLDER_CANDIDATES[0])

# Processed CSV path (same logic)
PROCESSED_CSV_CANDIDATES = [
    os.path.join(REPO_ROOT, "data", "processed_filllevel.csv"),
    os.path.join(REPO_ROOT, "..", "data", "processed_filllevel.csv"),
]
PROCESSED_CSV = next((p for p in PROCESSED_CSV_CANDIDATES if os.path.exists(p)), PROCESSED_CSV_CANDIDATES[0])

# Default model/scaler filenames
DEFAULT_MODEL = os.path.join(MODEL_FOLDER, "bilstm_model.keras")
DEFAULT_SCALER = os.path.join(MODEL_FOLDER, "bilstm_scalers.pkl")

# Sidebar: model selection and basic controls
with st.sidebar:
    st.header("Controls")

    # discover .keras models in outputs/models
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

    # Load processed CSV if available to populate bins and date range
    df = None
    if os.path.exists(PROCESSED_CSV):
        try:
            # 'infer_datetime_format' is deprecated; rely on pandas default parsing
            df = pd.read_csv(PROCESSED_CSV, parse_dates=["date"])
            # Normalize column names: 'date' may be named 'date' or 'timestamp' in different scripts
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            else:
                df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            min_date = df["timestamp"].min().date()
            max_date = df["timestamp"].max().date()

            serials = sorted(df["serialNumber"].astype(str).unique().tolist())

            # If scaler file exists, restrict selectable bins to those present in the scaler dict
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

    selected_bins = st.multiselect("Select bin serialNumber(s)", options=available_serials, default=available_serials[:3])
    if len(available_serials) < len(serials):
        st.info(f"Showing {len(available_serials)}/{len(serials)} bins — only bins with saved scalers are selectable.")

    st.markdown("---")
    predict_btn = st.button("Predict")

# Helpers: load model & scalers
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

# Try loading model and scalers (deferred until needed)
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

# Main: on predict
if predict_btn:
    if model_load_error:
        st.error(f"Model load failed: {model_load_error}")
    elif model is None:
        st.error("Model not loaded.")
    elif scaler_missing:
        st.error(f"Scalers file not found or failed to load: {scaler_choice}")
    elif df is None or len(selected_bins) == 0:
        st.error("No data available or no bins selected. Ensure processed CSV exists and bins are selected.")
    else:
        results = []
        charts_col1 = st.container()
        for bin_id in selected_bins:
            bin_id_orig = bin_id
            # serialNumber column may be int; df has ints
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
            # get scaler for this bin (scalers expected keyed by serialNumber)
            scaler = None
            if isinstance(scalers, dict):
                # try both str and int keys
                scaler = scalers.get(bin_id_orig) or scalers.get(int(bin_id_orig)) or scalers.get(str(bin_id_orig))
            else:
                scaler = scalers

            if scaler is None:
                st.warning(f"No scaler found for bin {bin_id_orig}, skipping")
                continue

            # transform sequence and predict
            try:
                seq_reshaped = np.array(seq).reshape(-1, 1)
                seq_scaled = scaler.transform(seq_reshaped).reshape(1, n_steps, 1)
                y_pred_norm = model.predict(seq_scaled, verbose=0).flatten()[0]
                y_pred_orig = scaler.inverse_transform([[y_pred_norm]])[0, 0]

                pred_date = pd.to_datetime(end_date) + timedelta(days=1)
                results.append({"serialNumber": bin_id_orig, "predicted_date": pred_date.date(), "predicted_fill": float(y_pred_orig)})

                # show chart with history and predicted point
                with charts_col1:
                    st.subheader(f"Bin {bin_id_orig}")
                    # Use tail() instead of deprecated .last() which expects an offset string
                    hist = df_cut.set_index("timestamp")["latestFullness"].tail(n_steps * 3)
                    # append predicted point to the series for plotting
                    chart_df = pd.concat([hist, pd.Series({pd.to_datetime(pred_date): y_pred_orig})])
                    st.line_chart(chart_df)
            except Exception as e:
                st.error(f"Prediction failed for bin {bin_id_orig}: {e}")

        if results:
            res_df = pd.DataFrame(results)
            st.success("Predictions ready")
            st.dataframe(res_df)

            # CSV download
            csv = res_df.to_csv(index=False).encode("utf-8")
            st.download_button(label="Download predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")
else:
    st.info("Configure parameters in the sidebar and click Predict.")

# Footer: quick diagnostics
st.sidebar.markdown("---")
st.sidebar.write(f"Model path: {model_choice}")
st.sidebar.write(f"Scaler path: {scaler_choice}")
if model_load_error:
    st.sidebar.error("Model failed to load — check logs")


