# src/train/train.py
import sys, json, pathlib, joblib
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_train(path):
    df = pd.read_parquet(path)
    y = df["sales"]
    X = df.drop(columns=["sales"])
    return X, y

def main(train_path: str, model_out: str, metrics_out: str,
         n_estimators: int, max_depth: int, random_state: int):
    X, y = load_train(train_path)

    mlflow.set_tracking_uri("file:./mlruns")          # local store in ./mlruns
    mlflow.set_experiment("retail-sales")
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X, y)

        preds = model.predict(X)
        mae = mean_absolute_error(y, preds)
        rmse = mean_squared_error(y, preds, squared=False)

        metrics = {"train_mae": float(mae), "train_rmse": float(rmse)}
        pathlib.Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_out, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save model file
        pathlib.Path(model_out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_out)

        # Log to MLflow
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(metrics_out)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Saved model -> {model_out}")
        print(f"Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train_path  = sys.argv[1]        # data/processed/train.parquet
    model_out   = sys.argv[2]        # models/model.pkl
    metrics_out = sys.argv[3]        # reports/train_metrics.json
    n_estimators = int(sys.argv[4])
    max_depth    = int(sys.argv[5])
    random_state = int(sys.argv[6])
    main(train_path, model_out, metrics_out, n_estimators, max_depth, random_state)
