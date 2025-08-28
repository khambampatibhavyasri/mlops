# Simple threshold-based retrain trigger (placeholder for Phase 6)
import json, argparse, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="models/metrics_baseline.json")
    ap.add_argument("--mape_threshold", type=float, default=0.28)
    args = ap.parse_args()

    with open(args.metrics, "r") as f:
        met = json.load(f)

    test_mape = met["test"]["MAPE"]
    if test_mape > args.mape_threshold:
        print(f"[Trigger] Test MAPE {test_mape:.3f} > {args.mape_threshold:.3f}. Retraining...")
        os.system("python src/models/train_baseline.py --in data/processed/features.csv --model models/baseline_linear.json --metrics models/metrics_baseline.json")
    else:
        print(f"[OK] Test MAPE {test_mape:.3f} <= {args.mape_threshold:.3f}. No retrain.")

if __name__ == "__main__":
    main()
