# src/automl/pycaret_regression.py

import argparse, os, json
import pandas as pd
import mlflow
from pycaret.regression import setup, compare_models, pull, predict_model, save_model

def main():
    ap = argparse.ArgumentParser(description="Run PyCaret EDA + AutoML for regression and log to MLflow.")
    ap.add_argument("--in", dest="inp", required=True, help="Input features CSV (from features stage)")
    ap.add_argument("--target", required=True, help="Target column name (e.g., 'units')")
    ap.add_argument("--leaderboard", required=True, help="Path to write PyCaret leaderboard CSV")
    ap.add_argument("--pred-sample", required=True, help="Path to write sample predictions CSV")
    ap.add_argument("--model-dir", required=True, help="Base path for saving best model (PyCaret adds .pkl)")
    ap.add_argument("--metrics", required=True, help="Path to write top-model metrics JSON")
    ap.add_argument("--experiment-name", default="pycaret-regression", help="MLflow experiment name")
    ap.add_argument("--tracking-uri", default="file:./mlruns", help="MLflow tracking URI")
    args = ap.parse_args()

    # Ensure we use the same tracking store as the rest of the project
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # Load data
    df = pd.read_csv(args.inp)
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    # -------- PyCaret setup (disable its internal MLflow logger!) --------
    setup(
        data=df,
        target=args.target,
        session_id=42,
        log_experiment=False,     # <<< important: avoid PyCaret's MLflow integration
        verbose=False
    )

    # Compare models and get leaderboard
    best = compare_models()
    lb = pull()

    os.makedirs(os.path.dirname(args.leaderboard), exist_ok=True)
    lb.to_csv(args.leaderboard, index=False)

    # Holdout predictions (sample)
    preds = predict_model(best)
    os.makedirs(os.path.dirname(args.pred_sample), exist_ok=True)
    preds.head(200).to_csv(args.pred_sample, index=False)

    # Extract top metrics from leaderboard
    top = lb.iloc[0].to_dict()
    metrics = {}
    for k in ["MAE", "MSE", "RMSE", "R2"]:
        if k in top and pd.notna(top[k]):
            metrics[k.lower()] = float(top[k])

    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
    with open(args.metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save the best model (PyCaret adds .pkl)
    os.makedirs(os.path.dirname(args.model_dir), exist_ok=True)
    model_path_base = save_model(best, args.model_dir)  # returns base path without extension

    # -------- Manual MLflow logging (works with any MLflow version) -------
    with mlflow.start_run() as run:
        # log params you care about
        mlflow.log_param("automl_library", "pycaret")
        mlflow.log_param("target", args.target)
        if "Model" in top:
            mlflow.log_param("best_model_name", str(top["Model"]))

        # log metrics
        if metrics:
            mlflow.log_metrics(metrics)

        # log artifacts
        mlflow.log_artifact(args.leaderboard)
        mlflow.log_artifact(args.pred_sample)
        mlflow.log_artifact(args.metrics)

        # also log the PyCaret model file (ends with .pkl)
        pkl_path = model_path_base + ".pkl"
        if os.path.exists(pkl_path):
            mlflow.log_artifact(pkl_path)

        print(json.dumps({
            "experiment": args.experiment_name,
            "run_id": run.info.run_id,
            "best_model": str(top.get("Model", "N/A")),
            **metrics
        }, indent=2))

if __name__ == "__main__":
    main()
