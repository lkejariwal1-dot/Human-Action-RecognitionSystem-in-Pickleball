"""Home: headline, the 16:9 video stage, strokes and pipeline."""

import streamlit as st

from components import ui

# ---------------- Hero ----------------
ui.html("""
<div class="rl-hero">
<h1 class="rl-display">Read the movement.</h1>
<p class="rl-lead">Rally recognises pickleball strokes from video, using pose tracking and a sequence model. Add a clip, then continue to Analyse to see the predicted stroke.</p>
</div>
""")

# ---------------- Video stage: the main focus ----------------
video = ui.video_stage("home")
if video:
    with st.container(key="cta"):
        st.page_link("views/analyse.py", label="Continue to Analyse")
    ui.replace_button("home")

# ---------------- Strokes ----------------
ui.html(f"""
<section class="rl-section">
<div>
<h2 class="rl-h2">Five strokes</h2>
<p class="rl-muted">The classes the model tells apart.</p>
</div>
{ui.strokes_list()}
</section>
""")