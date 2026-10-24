"""Reusable UI helpers for Rally, so every page looks consistent."""

from html import escape
from pathlib import Path
from textwrap import dedent

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent

# File types the uploader accepts. Browsers may not preview AVI or MKV.
ALLOWED_TYPES = ["mp4", "mov", "avi", "mkv"]

# The five target classes. Shared by the Home and About pages.
STROKES = [
    ("Serve", "Starts every point. Hit underhand, diagonally into the opposite service court."),
    ("Forehand", "A groundstroke played on the racket-hand side of the body."),
    ("Backhand", "A groundstroke played across the body, on the non-racket side."),
    ("Volley", "A shot returned out of the air, before the ball bounces."),
    ("Dink", "A soft shot that lands in the opponent's non-volley zone."),
]


def load_css() -> None:
    """Read styles.css and inject it into the page."""
    css = (ROOT / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def html(markup: str) -> None:
    """Render a block of HTML.

    Streamlit's Markdown treats blank lines and lines indented by 4+ spaces
    as special, so we dedent the text and keep HTML blocks free of blank lines.
    """
    st.markdown(dedent(markup).strip(), unsafe_allow_html=True)


def sidebar_brand() -> None:
    html("""
    <div class="rl-brand">
    <p class="rl-brand-name">Rally</p>
    <p class="rl-brand-sub">Pickleball Action Recognition</p>
    </div>
    """)


def sidebar_status() -> None:
    try:
        import service
        ready = service.model_ready()
    except Exception:
        ready = False
    title, text = (("Model connected", "Trained CNN-BiLSTM on YOLOv8 pose keypoints. Upload a clip and press Analyse.")
                   if ready else
                   ("Model not found", "Train it first with 02_train.py so runs/models/best_cnn_bilstm.pt exists."))
    html(f"""
    <div class="rl-side-note">
    <p>{title}</p>
    <span>{text}</span>
    </div>
    """)


def page_header(title: str, subtitle: str) -> None:
    html(f"""
    <div class="rl-page-head">
    <h1 class="rl-title">{title}</h1>
    <p class="rl-lead">{subtitle}</p>
    </div>
    """)


def strokes_list() -> str:
    """Return the five strokes as an HTML list (a string, not rendered yet)."""
    items = "".join(
        f'<li><span class="rl-stroke-name">{name}</span>'
        f'<span class="rl-stroke-text">{text}</span></li>'
        for name, text in STROKES
    )
    return f'<ul class="rl-strokes">{items}</ul>'


def video_stage(key: str):
    """Show the 16:9 uploader, or the preview once a video is chosen.

    The video is kept in st.session_state["video"] so it carries over
    between pages. Returns that dict, or None if nothing is uploaded.
    """
    video = st.session_state.get("video")

    if video is None:
        uploaded = st.file_uploader(
            "Upload a pickleball video",
            type=ALLOWED_TYPES,
            key=f"uploader_{key}",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            st.session_state["video"] = {
                "name": uploaded.name,
                "type": uploaded.type or "video/mp4",
                "bytes": uploaded.getvalue(),
            }
            st.rerun()
        html('<p class="rl-caption">MP4, MOV, AVI or MKV, up to 200 MB. '
             'Some browsers cannot preview AVI or MKV files.</p>')
        return None

    st.video(video["bytes"], format=video["type"])
    html(f'<p class="rl-caption">{escape(video["name"])}</p>')
    return video


def replace_button(key: str) -> None:
    """A button that clears the current video so another can be chosen."""
    if st.button("Choose a different video", key=f"replace_{key}"):
        st.session_state.pop("video", None)
        st.rerun()