"""STEP 4: predict the action in ONE clip, save a skeleton video and per-frame joint angles."""
import argparse
import config
from src.inference import analyze_video, load_trained
from src.pose import PoseExtractor
from src.utils import get_device


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, help="path to a short clip of one stroke")
    ap.add_argument("--project_dir", default=config.PROJECT_DIR)
    ap.add_argument("--weights", default=config.POSE_WEIGHTS)
    ap.add_argument("--no_video", action="store_true", help="skip the annotated video (prediction + angles CSV only)")
    args = ap.parse_args()

    paths = config.make_paths(args.project_dir)
    device = get_device()
    model, classes = load_trained(paths.best, device)
    extractor = PoseExtractor(args.weights, device)
    res = analyze_video(args.video, extractor, model, classes, device, paths.results, write_video=not args.no_video)

    print(f"\nPrediction : {res['action']} ({res['confidence'] * 100:.1f}%)")
    print("Probabilities:", res["probabilities"])
    if res["video"]:
        print("Annotated video:", res["video"])
    print("Joint angles CSV:", res["angles_csv"])
    print("\nMean joint angles (deg):")
    print(res["angles"].drop(columns=["frame", "time_s"]).filter(regex="elbow|shoulder$|hip|knee").mean().round(1).to_string())


if __name__ == "__main__":
    main()
