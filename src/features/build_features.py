# src/features/build_features.py

import argparse
import os
import pandas as pd

def as_bool(s: str) -> bool:
    return str(s).lower() in {"1", "true", "yes", "y"}

def build_features(
    df: pd.DataFrame,
    target: str,
    use_one_hot: bool,
    include_price: bool,
    include_promo: bool,
    include_holiday: bool,
    include_competitor_diff: bool,
) -> pd.DataFrame:
    # parse & sort for deterministic lags/rollings
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "product_id", "date"])

    # lags/rollings computed over the TARGET (keeps your original names)
    df["sales_lag_7"]  = df.groupby(["store_id","product_id"])[target].shift(7)
    df["sales_lag_14"] = df.groupby(["store_id","product_id"])[target].shift(14)
    df["rolling_mean_7"]  = (
        df.groupby(["store_id","product_id"])[target]
          .rolling(7, min_periods=1).mean()
          .reset_index(level=[0,1], drop=True)
    )
    df["rolling_mean_14"] = (
        df.groupby(["store_id","product_id"])[target]
          .rolling(14, min_periods=1).mean()
          .reset_index(level=[0,1], drop=True)
    )

    # competitor diff if requested
    if include_competitor_diff:
        df["competitor_diff"] = df["competitor_price"] - df["price"]

    # fill NA from lags/rollings
    for c in ["sales_lag_7","sales_lag_14","rolling_mean_7","rolling_mean_14"]:
        df[c] = df[c].fillna(0.0)

    # base columns by flags
    base_cols = []
    if include_price:   base_cols.append("price")
    if include_promo:   base_cols.append("on_promo")
    if include_holiday: base_cols.append("is_holiday")
    if include_competitor_diff: base_cols.append("competitor_diff")
    base_cols += ["sales_lag_7","sales_lag_14","rolling_mean_7","rolling_mean_14"]

    # categorical handling
    cat_cols = ["store_id","product_id","day_of_week","month"]
    if use_one_hot:
        df = pd.get_dummies(
            df,
            columns=cat_cols,
            prefix=["store","prod","dow","mon"],
            dtype=int
        )
        cat_out = [c for c in df.columns if c.startswith(("store_","prod_","dow_","mon_"))]
    else:
        cat_out = cat_cols

    # final order
    cols = ["date", target] + base_cols + [c for c in cat_out if c not in base_cols]
    out = df[cols].copy()
    out["date"] = out["date"].dt.date.astype(str)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",  required=True)
    ap.add_argument("--out", dest="out",  required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--use-one-hot",         default="true")
    ap.add_argument("--include-price",       default="true")
    ap.add_argument("--include-promo",       default="true")
    ap.add_argument("--include-holiday",     default="true")
    ap.add_argument("--include-competitor-diff", default="true")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    out = build_features(
        df=df,
        target=args.target,
        use_one_hot=as_bool(args.use_one_hot),
        include_price=as_bool(args.include_price),
        include_promo=as_bool(args.include_promo),
        include_holiday=as_bool(args.include_holiday),
        include_competitor_diff=as_bool(args.include_competitor_diff),
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"FEATURE_ROWS={len(out)};OUT={args.out}")

if __name__ == "__main__":
    main()
