# src/models/train_baseline.py

import argparse
import json
import os
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn  # ensures the sklearn flavor is available

EXPERIMENT_NAME = "retail-sales"   # shows up as the experiment in the UI
TRACKING_URI = "file:./mlruns"     # local store in your repo

def main():
    ap = argparse.ArgumentParser(description="Train a Ridge baseline and log to MLflow.")
    ap.add_argument("--in", dest="inp", required=True, help="Input features CSV")
    ap.add_argument("--model", required=True, help="Path to save JSON model")
    ap.add_argument("--metrics", required=True, help="Path to save metrics JSON")
    ap.add_argument("--alpha", type=float, required=True, help="Ridge alpha")
    ap.add_argument("--target", required=True, help="Target column name")
    args = ap.parse_args()

    # ---- Load features
    df = pd.read_csv(args.inp)
    cols = [c for c in df.columns if c != args.target]
    if "date" in cols:
        cols.remove("date")
    X = df[cols]
    y = df[args.target]

    # ---- Set up MLflow
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        # Log key params (add more if you like)
        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("alpha", args.alpha)
        mlflow.log_param("target", args.target)
        mlflow.log_param("n_features", X.shape[1])

        # ---- Train
        model = Ridge(alpha=args.alpha)
        model.fit(X, y)
        preds = model.predict(X)

        # ---- Metrics
        mae = float(mean_absolute_error(y, preds))
        rmse = float(mean_squared_error(y, preds, squared=False))
        metrics = {"train_mae": mae, "train_rmse": rmse}

        # Save metrics file for your repo artifacts
        os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
        with open(args.metrics, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save a minimal JSON “model” so your pipeline still produces the same outputs
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

        # ---- Log to MLflow
        mlflow.log_metrics(metrics)
        # attach the JSON files as artifacts so you can open them in the UI
        mlflow.log_artifact(args.metrics)
        mlflow.log_artifact(args.model)
        # also log the sklearn model in MLflow format (optional but nice)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(json.dumps({"experiment": EXPERIMENT_NAME,
                          "run_id": run.info.run_id,
                          **metrics}, indent=2))

if __name__ == "__main__":
    main()
