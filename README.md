# Cancellation models: scoring package

Five XGBoost models that estimate, for each policy active at a month-end, the chance it cancels within the next three
months. Two scripts score a features file with them.

| Model | Files | Estimates | Input rows per policy |
|:---|:---|:---|:---|
| Any cancellation | `cancel_3m.json` | the chance of cancelling, any type | the month-end scored |
| Cancellation type | `xgb_Switcher.json`, `xgb_System_Lapsed.json`, `xgb_Leave_PHI.json`, `xgb_Non-Recoverable.json` | the chance of leaving with each type, and of staying | the month-end scored and the two before it |

## Files

| File | What it is |
|:---|:---|
| `predict_cancel_3m.py` | scores with the any-cancellation model |
| `predict_types.py` | scores with the four type models |
| `common.py` | reading the features file and coding its categories, used by both scripts |
| `categories.json` | the values of the 12 categorical features, in the order the models learnt them |
| `feature_list.csv` | the 74 input features: name, number or category, meaning |
| `examples/example_features.csv` | made-up input: six invented policies, Jun to Aug 2026 |
| `examples/expected_*.csv` | what the two scripts should output for the example |
| `requirements.txt` | the Python packages, at the versions tested |
| `SHA256SUMS` | checksums of the model files and `categories.json` |

## Quick start

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
sha256sum -c SHA256SUMS                                     # the files arrived unchanged
.venv/bin/python predict_cancel_3m.py examples/example_features.csv
.venv/bin/python predict_types.py examples/example_features.csv
```

This writes `predictions_cancel_3m_2026-08.csv` and `predictions_types_2026-08.csv`, which should match
`examples/expected_cancel_3m.csv` and `examples/expected_types.csv`. Other package versions or hardware can change the
last decimal place. Both scripts take `--month YYYY-MM` to score another month-end than the latest in the file, and
`--out` for the output file; `--help` shows the options.

## The input

- **One row per policy-month** with `POLICY_HASH`, `INFORMATION_DATE` (the month-end, `YYYY-MM-DD`) and the 74
  features of `feature_list.csv`. Other columns are ignored. CSV or parquet; for a large file use parquet, from which
  the scripts read only the months they need.

## The output

`predict_cancel_3m.py`, one row per policy, most likely to cancel first:

| Column | Meaning |
|:---|:---|
| `p_cancel_3m` | the chance the policy cancels, with any type, within the 3 months after `INFORMATION_DATE` |

`predict_types.py`, one row per policy, most likely to leave first:

| Column | Meaning |
|:---|:---|
| `p_Switcher`, `p_System Lapsed`, `p_Leave PHI`, `p_Non-Recoverable`, `p_Stay` | the four types and staying as one distribution; the five add up to 1 |
| `p_leave` | `1 - p_Stay`: leaves with one of the four types within the 3 months |
| `score_<type>` | each model's own score, P(leaves with that type \| leaves with that type or stays). Rank by it to pick a type's top x %. The four do not add up to 1. |

The `p_` columns are probabilities: their sum over a group of policies is the number of cancellations expected in it.
The two scripts use separate models, so `p_cancel_3m` and `p_leave` are two estimates of nearly the same thing and
will differ; use one or the other.

The type probabilities come from the four scores as `odds_t = score_t / (1 - score_t)`,
`p_Stay = 1 / (1 + sum of the odds)`, `p_<type> = odds_t * p_Stay`.

## The examples

The six policies in `examples/example_features.csv` are made up. 

| Policy | Invented profile | `p_cancel_3m` | `p_leave` | Most likely type |
|:---|:---|---:|---:|:---|
| EX01 | family, Gold hospital and extras, held 10–15 years, pays monthly by direct debit, regular small claims | 0.8 % | 0.5 % | Switcher |
| EX02 | single, 25–34, Basic hospital, pays quarterly, in arrears in Jul and Aug 2026 | 4.4 % | 2.8 % | System Lapsed |
| EX03 | couple, 55–64, moved from Gold to Silver with a higher excess in Jul 2026, after a complaint and an NPS of 3 in Jun | 0.1 % | 0.3 % | Switcher |
| EX04 | single, 75+, hospital only, pays quarterly at a branch, large hospital claims May to Jul 2026 | 0.8 % | 0.5 % | Non-Recoverable |
| EX05 | single, 35–44, NSW, extras only, sold by a broker, pays yearly, no claims since Jun 2025 | 5.2 % | 2.8 % | Switcher |
| EX06 | couple, 25–34, joined 10 Jul 2026, so it has rows for Jul and Aug only | 1.3 % | 1.2 % | Switcher |

## About the models

- **Data.** the policy data: month-ends Sep 2024 to May 2026, the ones whose 3-month outcome is known. Policies were
  split 80 / 20; the models were trained on the 80 % and checked on the 20 % they never saw. 12 % of the training
  policies were held back to decide when training stopped.
- **Any-cancellation model.** XGBoost on the 74 features of the month-end scored; positives are policies that cancelled
  with any type within 3 months. It uses 713 of its 813 trees.
- **Type models.** One XGBoost per type, on the 74 features of M-2, M-1 and M side by side (224 columns, with
  `present_m1` and `present_m2`). Each is trained on the policies that stayed and those that left with its type.
  They use the first 344 of 444 (Switcher), 559 of 659 (System Lapsed), 268 of 368 (Leave PHI) and 212 of 312
  (Non-Recoverable) trees.
- **Model files.** XGBoost 2.1.4 JSON. They hold trees only, no policy data.

## Tested

- `predict_cancel_3m.py` reproduces the scores saved when the model was evaluated, for all 109,322 held-out policies at
  31 May 2026, to within 1e-6.
- `predict_types.py` reproduces the saved raw scores and the dashboard probabilities for all 108,412 held-out policies
  at 31 Mar 2026, to within the 6-decimal rounding of the output.
- The example gives the same scores from CSV as from the parquet it was written from.
- Versions: Python 3.14.7, xgboost 2.1.4, pandas 3.0.6, numpy 2.5.3, pyarrow 25.0.1, on Linux.
