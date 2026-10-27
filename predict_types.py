"""
Score policies with the four cancellation-type models (xgb_<type>.json): each policy's chance of leaving within the
3 months after a month-end M, by type: Switcher, System Lapsed, Leave PHI or Non-Recoverable.

Each model reads a 3-month window: the policy's 74 features at M-2, M-1 and M side by side, as <feature>_m1, _m2 and
_m3 (224 columns with present_m1 and present_m2). present_m1 / present_m2 is 0 when the policy has no row at M-2 / M-1;
that month's features are then blank, as in training.

  in : a features file (CSV or parquet) as for predict_cancel_3m.py, with the two month-ends before the one scored
  out: CSV, one row per policy at M; most likely to leave first:
         p_<type>, p_Stay   the four types and Stay as one distribution that adds up to 1
         p_leave            1 - p_Stay: leaves with one of the four types within the 3 months
         score_<type>       each model's own score, P(leaves with that type | leaves with that type or stays);
                            rank by it to pick a type's top x %. The four do not add up to 1.

  python predict_types.py examples/example_features.csv
  python predict_types.py features.parquet --month 2026-08 --out types_2026-08.csv
"""
import argparse

import numpy as np
import pandas as pd
import xgboost as xgb

import common as C

TYPES = ["Switcher", "System Lapsed", "Leave PHI", "Non-Recoverable"]
WIN = 3                                     # months per window: M-2, M-1 and M

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("features", help="the features file, CSV or parquet")
p.add_argument("--month", help="month-end to score, YYYY-MM; default the latest in the file")
p.add_argument("--models", default=str(C.HERE), help="folder holding xgb_<type>.json")
p.add_argument("--out", help="output CSV; default predictions_types_<YYYY-MM>.csv")
a = p.parse_args()
if a.month == "all":
    raise SystemExit("--month all is not available here: each month-end needs the two before it")

models = {}
for t in TYPES:
    models[t] = xgb.Booster(model_file=f"{a.models}/xgb_{t.replace(' ', '_')}.json")
feats = models[TYPES[0]].feature_names
assert all(m.feature_names == feats for m in models.values()), "the four models read different columns"
base = [f.removesuffix(f"_m{WIN}") for f in feats if f.endswith(f"_m{WIN}")]       # the 74 monthly features

df, M = C.read_features(a.features, base, a.month, back=WIN - 1)
C.set_categories(df)
mo = df.INFORMATION_DATE.dt.to_period("M")
now = df[mo == M].set_index("POLICY_HASH")
parts = []
for k in range(1, WIN + 1):                 # k = 1, 2, 3 holds M-2, M-1, M
    rows = df[mo == M - (WIN - k)].set_index("POLICY_HASH")[base]
    if k < WIN:
        parts.append(pd.Series(now.index.isin(rows.index), index=now.index, name=f"present_m{k}").astype("int8"))
    parts.append(rows.reindex(now.index).add_suffix(f"_m{k}"))
wx = pd.concat(parts, axis=1)[feats]
print(f"{M}: {len(wx):,} policies; with a row at M-1 {wx.present_m2.mean():.0%}, at M-2 {wx.present_m1.mean():.0%}")

dm = xgb.DMatrix(wx, enable_categorical=True)
raw = np.column_stack([m.predict(dm, iteration_range=(0, int(m.attr("best_iteration")) + 1))   # trees to the best round
                       for m in models.values()])

# The four scores as one distribution with Stay: odds_t = s_t / (1 - s_t), p_Stay = 1 / (1 + sum of odds),
# p_<type> = odds_t * p_Stay. Then p_<type> / (p_<type> + p_Stay) = s_t, each model's own score.
odds = np.clip(raw.astype(np.float64), 1e-9, 1 - 1e-9)
odds = odds / (1 - odds)
p_stay = 1 / (1 + odds.sum(axis=1))
out = pd.DataFrame({"POLICY_HASH": wx.index, "INFORMATION_DATE": now.INFORMATION_DATE.to_numpy()})
for i, t in enumerate(TYPES):
    out[f"p_{t}"] = odds[:, i] * p_stay
out["p_Stay"] = p_stay
out["p_leave"] = 1 - p_stay
for i, t in enumerate(TYPES):
    out[f"score_{t}"] = raw[:, i]
out = out.sort_values("p_leave", ascending=False)
path = a.out or f"predictions_types_{M}.csv"
out.to_csv(path, index=False, float_format="%.6f", date_format="%Y-%m-%d")
print("trees used: " + ", ".join(f"{t} {int(m.attr('best_iteration')) + 1} of {m.num_boosted_rounds()}"
                                 for t, m in models.items()))
print(f"expected leavers, % of the {len(out):,} policies: "
      + ", ".join(f"{t} {100 * out[f'p_{t}'].mean():.2f}" for t in TYPES)
      + f"; any of the four {100 * out.p_leave.mean():.2f}; written to {path}")
