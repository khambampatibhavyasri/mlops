# src/models/train_baseline.py

import argparse
import json
import os
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn  # ensure sklearn flavor is registered
import os, mlflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment("retail-sales")


EXPERIMENT_NAME = "retail-sales"
TRACKING_URI = "file:./mlruns"

def main():
    ap = argparse.ArgumentParser(description="Train a Ridge baseline and log to MLflow.")
    ap.add_argument("--in", dest="inp", required=True, help="Input features CSV")
    ap.add_argument("--model", required=True, help="Path to save JSON model")
    ap.add_argument("--metrics", required=True, help="Path to save metrics JSON")
    ap.add_argument("--alpha", type=float, required=True, help="Ridge alpha")
    ap.add_argument("--target", required=True, help="Target column name")
    args = ap.parse_args()

    # Load features
    df = pd.read_csv(args.inp)
    feature_cols = [c for c in df.columns if c != args.target]
    if "date" in feature_cols:
        feature_cols.remove("date")
    X = df[feature_cols]
    y = df[args.target]

    # MLflow setup
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        # Params
        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("alpha", args.alpha)
        mlflow.log_param("target", args.target)
        mlflow.log_param("n_features", X.shape[1])

        # Train
        model = Ridge(alpha=args.alpha)
        model.fit(X, y)
        preds = model.predict(X)

        # Metrics (compat mode: compute RMSE manually)
        mae = float(mean_absolute_error(y, preds))
        mse = float(mean_squared_error(y, preds))
        rmse = mse ** 0.5
        metrics = {"train_mae": mae, "train_rmse": rmse}

        # Save metrics file
        os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
        with open(args.metrics, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save a minimal JSON “model” (for your pipeline artifacts)
        os.makedirs(os.path.dirname(args.model), exist_ok=True)
        model_blob = {
            "type": "Ridge",
            "alpha": args.alpha,
            "intercept": float(model.intercept_),
            "coef": [float(c) for c in model.coef_],
            "features": feature_cols,
            "target": args.target,
        }
        with open(args.model, "w") as f:
            json.dump(model_blob, f, indent=2)

        # Log to MLflow
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(args.metrics)
        mlflow.log_artifact(args.model)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(json.dumps({"experiment": EXPERIMENT_NAME,
                          "run_id": run.info.run_id,
                          **metrics}, indent=2))

if __name__ == "__main__":
    main()
