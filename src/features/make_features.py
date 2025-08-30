# src/features/make_features.py
import sys, json, pathlib
import pandas as pd
from sklearn.model_selection import train_test_split

def main(in_parquet: str, out_train: str, out_test: str, test_size: float, random_state: int):
    df = pd.read_parquet(in_parquet)

    # --- EDIT THIS for your actual target & features ---
    # Assume there's a column 'sales' to predict; everything else is feature.
    target_col = "sales"
    assert target_col in df.columns, f"Expected column '{target_col}' in dataset."

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pathlib.Path(out_train).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_test).parent.mkdir(parents=True, exist_ok=True)

    train_df = pd.concat([X_train, y_train.rename(target_col)], axis=1)
    test_df  = pd.concat([X_test,  y_test.rename(target_col)], axis=1)

    train_df.to_parquet(out_train, index=False)
    test_df.to_parquet(out_test, index=False)

if __name__ == "__main__":
    in_parquet = sys.argv[1]         # data/interim/ingested.parquet
    out_train  = sys.argv[2]         # data/processed/train.parquet
    out_test   = sys.argv[3]         # data/processed/test.parquet
    test_size  = float(sys.argv[4])  # from params
    random_state = int(sys.argv[5])  # from params
    main(in_parquet, out_train, out_test, test_size, random_state)
