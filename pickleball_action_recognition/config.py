from pathlib import Path
from types import SimpleNamespace

# ===================== EDIT THESE (or pass --dataset_root / --project_dir on the command line) =====================
DATASET_ROOT = "data/pickleball_dataset"   # contains serve/ forehand/ backhand/ volley/ dink/
PROJECT_DIR = "runs/pickleball"            # cache, models, plots and metrics are saved here
CLASSES = ["serve", "forehand", "backhand", "volley", "dink"]
# ===================================================================================================================

SEQ_LEN = 32                  # frames sampled uniformly from each clip
POSE_WEIGHTS = "yolov8s-pose.pt"   # yolov8n-pose.pt = faster | yolov8m-pose.pt = more accurate
POSE_CONF = 0.25              # person detection threshold
KP_THR = 0.10                 # keypoints below this confidence are treated as missing
MIN_DETECT_RATE = 0.5         # drop clips where the player is found in < 50% of sampled frames

BATCH_SIZE = 16
EPOCHS = 100
PATIENCE = 15                 # early stopping on validation macro-F1
LR = 1e-3
SEED = 42

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def add_path_args(parser):
    parser.add_argument("--dataset_root", default=DATASET_ROOT, help="folder with one sub-folder per action")
    parser.add_argument("--project_dir", default=PROJECT_DIR, help="where results are stored")
    return parser


def make_paths(project_dir):
    root = Path(project_dir)
    p = SimpleNamespace(
        root=root,
        cache=root / "pose_cache",
        models=root / "models",
        results=root / "results",
        best=root / "models" / "best_cnn_bilstm.pt",
    )
    for d in (p.cache, p.models, p.results):
        d.mkdir(parents=True, exist_ok=True)
    return p
