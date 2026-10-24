"""Bridge between the Streamlit frontend and the pose + CNN-BiLSTM backend."""
import os
import tempfile
from pathlib import Path

import streamlit as st

import config
from src.inference import analyze_video, load_trained
from src.pose import PoseExtractor
from src.utils import get_device

# Folder that holds models/best_cnn_bilstm.pt (the --project_dir you trained with).
PROJECT_DIR = str(Path(__file__).resolve().parent / "runs")


@st.cache_resource(show_spinner="Loading model...")
def _load():
    """Load model + YOLO pose once per server, not on every click."""
    paths = config.make_paths(PROJECT_DIR)
    if not paths.best.exists():
        raise FileNotFoundError(f"No trained model at {paths.best}. Run 02_train.py first.")
    device = get_device()
    model, classes = load_trained(paths.best, device)
    extractor = PoseExtractor(config.POSE_WEIGHTS, device)
    return paths, device, model, classes, extractor


def model_ready() -> bool:
    return config.make_paths(PROJECT_DIR).best.exists()


def run_analysis(video_bytes: bytes, filename: str, with_overlay: bool) -> dict:
    """Save the upload to a temp file, run the backend, return its result dict."""
    paths, device, model, classes, extractor = _load()
    suffix = Path(filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(video_bytes)
        tmp = f.name
    try:
        return analyze_video(tmp, extractor, model, classes, device,
                             paths.results, write_video=with_overlay)
    finally:
        os.unlink(tmp)
