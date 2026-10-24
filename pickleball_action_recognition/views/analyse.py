"""Analyse: the video is the focus; results sit beside it."""

from html import escape

import streamlit as st

import service
from components import ui
from components.ui import STROKES

ui.page_header("Analyse", "Upload a clip of one stroke, then run the model to classify it.")

left, right = st.columns([2, 1], gap="large")

with left:
    video = ui.video_stage("analyse")
    if video:
        ui.replace_button("analyse")
    overlay = st.toggle("Show skeleton overlay", value=False,
                        help="Also renders an annotated video (slower on CPU).")
    run = st.button("Analyse video", type="primary", disabled=video is None)

    if run and video:
        try:
            with st.spinner("Detecting pose and classifying... (CPU can take a minute)"):
                res = service.run_analysis(video["bytes"], video["name"], overlay)
            st.session_state["result"] = {"file": video["name"], **{
                k: res[k] for k in ("action", "confidence", "probabilities", "video")}}
            st.success(f'Prediction: {res["action"].title()} ({res["confidence"] * 100:.1f}%)')
            st.session_state["history"].append({
                "File": video["name"], "Action": res["action"],
                "Confidence": f'{res["confidence"] * 100:.1f}%'})
        except Exception as e:          # no player found, missing model, bad file...
            st.session_state.pop("result", None)
            st.error(f"Analysis failed: {e}")

    result = st.session_state.get("result")
    if result and video and result["file"] == video["name"] and result["video"]:
        st.video(str(result["video"]))

with right:
    result = st.session_state.get("result")
    if not video or not result or result["file"] != video["name"]:
        result = None
    size = f'{video["name"]} ({len(video["bytes"]) / 1e6:.1f} MB)' if video else "None selected"
    action = escape(result["action"].title()) if result else "Not analysed yet"
    conf = f'{result["confidence"] * 100:.1f}%' if result else "Not analysed yet"
    breakdown = "".join(
        f'<div class="rl-result-row"><span>{name}</span><strong>'
        f'{result["probabilities"].get(name.lower(), 0) * 100:.1f}%</strong></div>' if result else
        f'<div class="rl-result-row"><span>{name}</span><strong>-</strong></div>'
        for name, _ in STROKES)
    ui.html(f"""
    <div class="rl-result">
    <h2 class="rl-h3">Result</h2>
    <div class="rl-result-row"><span>File</span><strong>{escape(size)}</strong></div>
    <div class="rl-result-row"><span>Predicted action</span><strong>{action}</strong></div>
    <div class="rl-result-row"><span>Confidence</span><strong>{conf}</strong></div>
    </div>
    <div class="rl-result rl-result-gap">
    <h2 class="rl-h3">Class breakdown</h2>
    {breakdown}
    </div>
    """)