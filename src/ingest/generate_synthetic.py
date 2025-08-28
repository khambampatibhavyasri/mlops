import argparse, os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def weekly_seasonality(dt):
    dow = dt.weekday()
    return 0.0 if dow < 4 else (0.15 if dow == 4 else 0.35)  # Fri +15%, Sat/Sun +35%

def monthly_seasonality(dt):
    return {1:-0.05, 2:-0.12, 3:0.02, 4:0.10, 5:0.12, 6:0.06, 7:0.04}.get(dt.month, 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output CSV path (e.g., data/raw/sales.csv)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2025-07-31")
    ap.add_argument("--stores", type=int, default=8)
    ap.add_argument("--products", type=int, default=30)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    start_date = datetime.fromisoformat(args.start)
    end_date = datetime.fromisoformat(args.end)

    dates = pd.date_range(start_date, end_date, freq="D")
    store_ids = [f"S{i:02d}" for i in range(1, args.stores+1)]
    product_ids = [f"P{i:03d}" for i in range(1, args.products+1)]

    base_demand_prod = rng.normal(40, 10, size=args.products).clip(5, 120)
    base_price_prod  = rng.normal(300, 80, size=args.products).clip(60, 800)
    price_elast      = rng.uniform(0.15, 0.6, size=args.products)
    promo_lift_prod  = rng.uniform(0.10, 0.60, size=args.products)
    store_effect     = rng.normal(1.0, 0.15, size=args.stores).clip(0.6, 1.5)

    holiday_dates = set([
        datetime(2025,1,26),  # Republic Day
        datetime(2025,3,14),  # Holi (approx)
        datetime(2025,5,1),   # Labour Day
        datetime(2025,4,14),  # Ambedkar Jayanti (approx)
    ])

    def holiday_boost(dt): return 0.40 if dt in holiday_dates else 0.0
    def trend_factor(dt):
        total_days = (end_date - start_date).days + 1
        pos = (dt - start_date).days / max(total_days-1, 1)
        return 0.08 * pos  # +8% by end

    rows = []
    for dt in dates:
        dow = dt.weekday()
        mon = dt.month
        week_seas = weekly_seasonality(dt)
        month_seas = monthly_seasonality(dt)
        hol = 1 if dt.to_pydatetime() in holiday_dates else 0

        for si, s in enumerate(store_ids):
            s_eff = store_effect[si]
            for pi, p in enumerate(product_ids):
                base_d = base_demand_prod[pi]
                base_price = base_price_prod[pi]

                promo_prob = 0.18 + (0.12 if dow >= 5 else 0) + (0.20 if hol else 0)
                on_promo = 1 if rng.random() < min(0.7, promo_prob) else 0

                discount = rng.uniform(0.05, 0.25) if on_promo else rng.uniform(-0.05, 0.08)
                price = max(5.0, base_price * (1.0 - discount))
                comp_price = (base_price * rng.normal(1.0, 0.05)) * (1.0 - rng.uniform(-0.06, 0.12))

                stock = int(rng.integers(0, 300))
                if rng.random() < 0.02:
                    stock = 0

                rel_price_change = (price - base_price) / max(base_price, 1e-6)
                price_impact = (1.0 - price_elast[pi] * rel_price_change)
                promo_lift = 1.0 + (promo_lift_prod[pi] if on_promo else 0.0)
                seasonal = 1.0 + week_seas + month_seas + holiday_boost(dt.to_pydatetime()) + trend_factor(dt.to_pydatetime())
                noise = rng.normal(1.0, 0.15)

                expected = base_d * s_eff * price_impact * promo_lift * seasonal * noise
                units = max(0.0, expected)
                if stock == 0:
                    units = 0.0
                else:
                    units = min(units, stock)

                revenue = units * price
                rows.append((dt.strftime("%Y-%m-%d"), s, p, int(dow), mon, on_promo, round(base_price,2),
                             round(price,2), round(comp_price,2), stock, hol, round(units,2), round(revenue,2)))

    columns = ["date","store_id","product_id","day_of_week","month","on_promo","base_price","price",
               "competitor_price","stock","is_holiday","units","revenue"]
    df = pd.DataFrame(rows, columns=columns)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"WROTE {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
