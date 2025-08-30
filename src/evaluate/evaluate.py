# src/evaluate/evaluate.py
import sys, json, pathlib, joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_test(path):
    df = pd.read_parquet(path)
    y = df["sales"]
    X = df.drop(columns=["sales"])
    return X, y

def main(model_path: str, test_path: str, metrics_out: str):
    X, y = load_test(test_path)
    model = joblib.load(model_path)
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = mean_squared_error(y, preds, squared=False)
    metrics = {"test_mae": float(mae), "test_rmse": float(rmse)}

    pathlib.Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    model_path  = sys.argv[1]        # models/model.pkl
    test_path   = sys.argv[2]        # data/processed/test.parquet
    metrics_out = sys.argv[3]        # reports/test_metrics.json
    main(model_path, test_path, metrics_out)
