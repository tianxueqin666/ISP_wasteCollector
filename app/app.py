import streamlit as st

st.set_page_config(page_title="Waste Collector", layout="wide")

pg = st.navigation([
    st.Page("pages/1_Fill_Levels.py", title="Fill Levels", icon="🟪"),
    st.Page("pages/2_Fill_Rates.py", title="Fill Rates", icon="🟨"),
], position="sidebar")

pg.run()