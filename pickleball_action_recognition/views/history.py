"""History: session-only list of analyses."""

import pandas as pd
import streamlit as st

from components import ui

ui.page_header(
    "History",
    "Analyses from this session are listed here. Nothing is saved to a database, so the list resets when the session ends.",
)

history = st.session_state.get("history", [])

if not history:
    ui.html("""
    <div class="rl-empty">
    <h2 class="rl-h3">No analyses yet</h2>
    <p class="rl-muted">Run an analysis on the Analyse page and it will appear in this list.</p>
    </div>
    """)
else:
    st.dataframe(pd.DataFrame(history))