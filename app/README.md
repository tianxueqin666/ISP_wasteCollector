# Waste Collector — Streamlit App (Minimal)

This folder contains a minimal Streamlit app that demonstrates loading a saved Keras model and pickled scalers to produce per-bin predictions from the processed CSV.

Files
- `app.py` — main Streamlit application.
- `requirements.txt` — Python packages used by the app.

Quick run (from project root) using the project's venv:

```bash
# activate venv (zsh)
source ./venv/bin/activate

# run streamlit app
./venv/bin/streamlit run app/app.py
```

Notes
- The app expects to find the model and scaler under `outputs/models/` by default (it looks for `bilstm_model.keras` and `bilstm_scalers.pkl`).
- The processed CSV is expected at `data/processed_filllevel.csv`. If not present you can upload or adapt the app to take manual input sequences.
- TensorFlow/streamlit must be installed in the venv. If not, run:

```bash
./venv/bin/python -m pip install -r app/requirements.txt
```

If you want, I can:
- Add a file-upload UI to accept a processed CSV from the browser.
- Add map selection (Leaflet + streamlit-folium) to select bins visually.
- Implement server-style FastAPI backend instead of a single-streamlit file.
