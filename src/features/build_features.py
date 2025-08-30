# src/features/build_features.py

import argparse
import os
import pandas as pd

def as_bool(s: str) -> bool:
    """Convert various truthy strings to bool."""
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
    # Normalize date and sort for lag/rolling
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "product_id", "date"])

    # Lag and rolling stats on the target column (keep original feature names)
    df["sales_lag_7"] = df.groupby(["store_id", "product_id"])[target].shift(7)
    df["sales_lag_14"] = df.groupby(["store_id", "product_id"])[target].shift(14)
    df["rolling_mean_7"] = (
        df.groupby(["store_id", "product_id"])[target]
          .rolling(7, min_periods=1)
          .mean()
          .reset_index(level=[0, 1], drop=True)
    )
    df["rolling_mean_14"] = (
        df.groupby(["store_id", "product_id"])[target]
          .rolling(14, min_periods=1)
          .mean()
          .reset_index(level=[0, 1], drop=True)
    )

    # Competitor price difference if requested
    if include_competitor_diff:
        df["competitor_diff"] = df["competitor_price"] - df["price"]

    # One-hot encode categorical columns if requested
    cat_cols = ["store_id", "product_id", "day_of_week", "month"]
    if use_one_hot:
        df = pd.get_dummies(
            df,
            columns=cat_cols,
            prefix=["store", "prod", "dow", "mon"],
            dtype=int,
        )
        cats = [c for c in df.columns if c.startswith(("store_", "prod_", "dow_", "mon_"))]
    else:
        cats = cat_cols

    # Fill missing values in lag/rolling columns
    for c in ["sales_lag_7", "sales_lag_14", "rolling_mean_7", "rolling_mean_14"]:
        df[c] = df[c].fillna(0.0)

    # Select base features depending on flags
    base_cols = []
    if include_price:
        base_cols.append("price")
    if include_promo:
        base_cols.append("on_promo")
    if include_holiday:
        base_cols.append("is_holiday")
    if include_competitor_diff:
        base_cols.append("competitor_diff")
    # Always include the lag/rolling features
    base_cols += ["sales_lag_7", "sales_lag_14", "rolling_mean_7", "rolling_mean_14"]

    # Final column order: date, target, selected base features, then categorical features
    cols = ["date", target] + base_cols + [c for c in cats if c not in base_cols]

    # Keep date as a string for CSV output
    out_df = df[cols].copy()
    out_df["date"] = out_df["date"].dt.date.astype(str)
    return out_df

def main():
    parser = argparse.ArgumentParser(description="Build engineered features")
    parser.add_argument("--in", dest="inp", required=True, help="Path to the input CSV")
    parser.add_argument("--out", dest="out", required=True, help="Path to the output CSV")
    parser.add_argument("--target", required=True, help="Name of the target column")
    parser.add_argument("--use-one-hot", dest="use_one_hot", type=str, default="true")
    parser.add_argument("--include-price", dest="include_price", type=str, default="true")
    parser.add_argument("--include-promo", dest="include_promo", type=str, default="true")
    parser.add_argument("--include-holiday", dest="include_holiday", type=str, default="true")
    parser.add_argument("--include-competitor-diff", dest="include_competitor_diff", type=str, default="true")
    args = parser.parse_args()

    df = pd.read_csv(args.inp)
    out_df = build_features(
        df=df,
        target=args.target,
        use_one_hot=as_bool(args.use_one_hot),
        include_price=as_bool(args.include_price),
        include_promo=as_bool(args.include_promo),
        include_holiday=as_bool(args.include_holiday),
        include_competitor_diff=as_bool(args.include_competitor_diff),
    )

    # Write to output location
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"FEATURE_ROWS={len(out_df)};OUT={args.out}")

if __name__ == "__main__":
    main()
