import argparse, json, os
import numpy as np
import pandas as pd

def psi(ref, cur, buckets=10):
    eps=1e-6
    q = np.linspace(0, 1, buckets+1)
    cuts = np.quantile(ref, q)
    ref_counts = np.histogram(ref, bins=cuts)[0] + eps
    cur_counts = np.histogram(cur, bins=cuts)[0] + eps
    ref_props = ref_counts / ref_counts.sum()
    cur_props = cur_counts / cur_counts.sum()
    val = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
    return float(val)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    ref = pd.read_csv(args.reference)
    cur = pd.read_csv(args.current)

    features_to_check = [c for c in ref.columns if c in ["price","on_promo","competitor_diff","units","rolling_mean_7","rolling_mean_14"]]
    report = {}
    for c in features_to_check:
        try:
            report[c] = {"psi": psi(ref[c].to_numpy(), cur[c].to_numpy(), buckets=10)}
        except Exception as e:
            report[c] = {"error": str(e)}

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved drift report to {args.report}")

if __name__ == "__main__":
    main()
