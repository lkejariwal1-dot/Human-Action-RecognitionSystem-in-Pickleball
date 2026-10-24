"""Rally: entry point.

Run with:  streamlit run app.py
"""

import streamlit as st

from components import ui

# Must be the first Streamlit command in the script.
st.set_page_config(
    page_title="Rally",
    page_icon=":material/sports_tennis:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# The four pages. Each one is a script in the views/ folder.
home = st.Page("views/dashboard.py", title="Home", default=True)
analyse = st.Page("views/analyse.py", title="Analyse")
history = st.Page("views/history.py", title="History")
about = st.Page("views/about.py", title="About")

# position="hidden" turns off Streamlit's default menu so we can draw our own.
page = st.navigation([home, analyse, history, about], position="hidden")

ui.load_css()

# Session-only history, filled by the Analyse page.
st.session_state.setdefault("history", [])

with st.sidebar:
    ui.sidebar_brand()
    for p in (home, analyse, history, about):
        st.page_link(p)
    ui.sidebar_status()

page.run()