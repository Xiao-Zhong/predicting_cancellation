"""
Score policies with the any-cancellation model (cancel_3m.json): each policy's chance of cancelling, with any type,
within the 3 months after a month-end.

  in : a features file (CSV or parquet), one row per policy-month: POLICY_HASH, INFORMATION_DATE and the 74 features
       of feature_list.csv. Only the month-end scored is needed.
  out: CSV with POLICY_HASH, INFORMATION_DATE, p_cancel_3m; most likely to cancel first

  python predict_cancel_3m.py examples/example_features.csv
  python predict_cancel_3m.py features.parquet --month 2026-08 --out cancel_3m_2026-08.csv
"""
import argparse

import xgboost as xgb

import common as C

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("features", help="the features file, CSV or parquet")
p.add_argument("--month", help="month-end to score, YYYY-MM; default the latest in the file; 'all' scores every row")
p.add_argument("--model", default=str(C.HERE / "cancel_3m.json"))
p.add_argument("--out", help="output CSV; default predictions_cancel_3m_<YYYY-MM>.csv")
a = p.parse_args()

m = xgb.Booster(model_file=a.model)
feats = m.feature_names
df, M = C.read_features(a.features, feats, a.month)
C.set_categories(df)

# Training went on 100 rounds past the best one before it stopped; only the trees up to the best round are used.
best = int(m.attr("best_iteration"))
df["p_cancel_3m"] = m.predict(xgb.DMatrix(df[feats], enable_categorical=True), iteration_range=(0, best + 1))

out = df[C.ID_COLS + ["p_cancel_3m"]].sort_values("p_cancel_3m", ascending=False)
path = a.out or f"predictions_cancel_3m_{M or 'all'}.csv"
out.to_csv(path, index=False, float_format="%.6f", date_format="%Y-%m-%d")
print(f"{len(out):,} policy-months scored ({M or 'every month'}) with {best + 1} of {m.num_boosted_rounds()} trees; "
      f"mean p_cancel_3m {out.p_cancel_3m.mean():.4f}; written to {path}")
