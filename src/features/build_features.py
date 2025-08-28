import argparse, os
import pandas as pd

def build_features(df):
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id","product_id","date"])

    # Lags & rolling
    df["sales_lag_7"]  = df.groupby(["store_id","product_id"])["units"].shift(7)
    df["sales_lag_14"] = df.groupby(["store_id","product_id"])["units"].shift(14)
    df["rolling_mean_7"]  = df.groupby(["store_id","product_id"])["units"].rolling(7, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    df["rolling_mean_14"] = df.groupby(["store_id","product_id"])["units"].rolling(14, min_periods=1).mean().reset_index(level=[0,1], drop=True)

    # Competitor delta
    df["competitor_diff"] = df["competitor_price"] - df["price"]

    # Categorical one-hots
    df = pd.get_dummies(df, columns=["store_id","product_id","day_of_week","month"],
                        prefix=["store","prod","dow","mon"], dtype=int)

    for c in ["sales_lag_7","sales_lag_14","rolling_mean_7","rolling_mean_14"]:
        df[c] = df[c].fillna(0.0)

    base_cols = ["price","on_promo","is_holiday","competitor_diff",
                 "sales_lag_7","sales_lag_14","rolling_mean_7","rolling_mean_14"]
    cats = [c for c in df.columns if c.startswith(("store_","prod_","dow_","mon_"))]
    cols = ["date","units"] + base_cols + [c for c in cats if c not in base_cols]
    return df[cols]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    out = build_features(df)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"FEATURE_ROWS={len(out)};OUT={args.out}")

if __name__ == "__main__":
    main()
