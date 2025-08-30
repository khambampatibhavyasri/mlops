# src/automl/pycaret_regression.py

import argparse, os, json
import pandas as pd
import mlflow

# PyCaret regression API
from pycaret.regression import setup, compare_models, pull, predict_model, save_model

def main():
    ap = argparse.ArgumentParser(description="Run PyCaret EDA + AutoML for regression and log to MLflow.")
    ap.add_argument("--in", dest="inp", required=True, help="Input features CSV (from features stage)")
    ap.add_argument("--target", required=True, help="Target column name (e.g., 'units')")
    ap.add_argument("--leaderboard", required=True, help="Path to write PyCaret leaderboard CSV")
    ap.add_argument("--pred-sample", required=True, help="Path to write sample predictions CSV")
    ap.add_argument("--model-dir", required=True, help="Directory base name for saving best model (PyCaret will create files)")
    ap.add_argument("--metrics", required=True, help="Path to write top-model metrics JSON")
    ap.add_argument("--experiment-name", default="pycaret-regression", help="MLflow experiment name")
    ap.add_argument("--tracking-uri", default="file:./mlruns", help="MLflow tracking URI")
    args = ap.parse_args()

    # Ensure consistent tracking with your baseline
    mlflow.set_tracking_uri(args.tracking_uri)

    # Load data
    df = pd.read_csv(args.inp)

    # Drop 'date' if present (not useful for most tabular models)
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    # ---- PyCaret setup ----
    # What it does:
    # - infers types, handles missing values (by default), sets a random seed
    # - optionally logs everything to MLflow under experiment_name
    s = setup(
        data=df,
        target=args.target,
        session_id=42,
        log_experiment=True,
        experiment_name=args.experiment_name,
        verbose=False
        # You can add: normalize=True, remove_multicollinearity=True, etc.
        # depending on your needs and PyCaret version.
    )

    # ---- Model comparison ----
    # What it does:
    # - trains a suite of algorithms with CV
    # - returns the best model according to default metric (RMSE for regression)
    best = compare_models()

    # Grab the comparison leaderboard (last "pull()" after compare_models())
    lb = pull()
    os.makedirs(os.path.dirname(args.leaderboard), exist_ok=True)
    lb.to_csv(args.leaderboard, index=False)

    # ---- Holdout predictions of the best model ----
    preds = predict_model(best)  # uses PyCaret's holdout split
    os.makedirs(os.path.dirname(args.pred_sample), exist_ok=True)
    preds.head(200).to_csv(args.pred_sample, index=False)

    # ---- Save top-model metrics ----
    # Leaderboard usually has columns like: Model, MAE, MSE, RMSE, R2 ...
    first = lb.iloc[0].to_dict()
    metrics = {}
    for k in ["MAE", "MSE", "RMSE", "R2"]:
        if k in first:
            metrics[k.lower()] = float(first[k])
    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
    with open(args.metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- Persist the best model ----
    # PyCaret saves as "<base_name>.pkl" (and possibly related files).
    # We'll pass a base path like models/pycaret_best (no extension).
    os.makedirs(os.path.dirname(args.model_dir), exist_ok=True)
    save_model(best, args.model_dir)

    print(f"Saved leaderboard -> {args.leaderboard}")
    print(f"Saved sample predictions -> {args.pred_sample}")
    print(f"Saved metrics -> {args.metrics}")
    print(f"Saved best model base -> {args.model_dir} (PyCaret adds .pkl)")

if __name__ == "__main__":
    main()
