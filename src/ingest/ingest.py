import argparse, os
import pandas as pd

REQUIRED_COLS = [
    "date","store_id","product_id","day_of_week","month","on_promo",
    "base_price","price","competitor_price","stock","is_holiday","units","revenue"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", help="Path to raw sales.csv")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Basic data quality gates
    if (df["price"] <= 0).any():
        raise ValueError("Found non-positive prices.")
    if df.isna().any().any():
        raise ValueError("Found missing values.")

    # Normalize date
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"INGESTED_ROWS={len(df)};OUT={args.out}")

if __name__ == "__main__":
    main()
