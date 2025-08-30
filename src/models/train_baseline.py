# src/models/train_baseline.py

import argparse
import json
import os
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input features CSV")
    ap.add_argument("--model", required=True, help="Path to save JSON model")
    ap.add_argument("--metrics", required=True, help="Path to save metrics JSON")
    ap.add_argument("--alpha", type=float, required=True, help="Ridge alpha")
    ap.add_argument("--target", required=True, help="Target column name")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)

    # Separate X/y (drop date column if present)
    cols = [c for c in df.columns if c != args.target]
    if "date" in cols:
        cols.remove("date")
    X = df[cols]
    y = df[args.target]

    model = Ridge(alpha=args.alpha)
    model.fit(X, y)

    preds = model.predict(X)
    mae = float(mean_absolute_error(y, preds))
    rmse = float(mean_squared_error(y, preds, squared=False))

    # Save a minimal JSON "model"
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    model_blob = {
        "type": "Ridge",
        "alpha": args.alpha,
        "intercept": float(model.intercept_),
        "coef": [float(c) for c in model.coef_],
        "features": cols,
        "target": args.target,
    }
    with open(args.model, "w") as f:
        json.dump(model_blob, f, indent=2)

    # Save metrics
    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
    metrics = {"train_mae": mae, "train_rmse": rmse}
    with open(args.metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({"alpha": args.alpha, **metrics}, indent=2))

if __name__ == "__main__":
    main()
