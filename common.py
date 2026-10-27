"""
Shared by predict_cancel_3m.py and predict_types.py: reading the features file and coding its categorical columns.

The features file has one row per policy-month: POLICY_HASH, INFORMATION_DATE (the month-end) and the 74 features of
feature_list.csv, as build_features.py makes them. CSV or parquet.
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ID_COLS = ["POLICY_HASH", "INFORMATION_DATE"]
# Each categorical feature's values in the order the models learnt them. A model knows a value only by its position in
# this list, so the lists must not be rebuilt from new data.
CATEGORIES = json.loads((HERE / "categories.json").read_text())


def read_features(path, columns, month=None, back=0):
    """
    The rows of month-end `month` (YYYY-MM; default the latest in the file; "all" for every row) and of the `back`
    month-ends before it, with ID_COLS and `columns`. Returns (rows, the month scored as a pandas Period, or None for all).
    """
    path = Path(path)
    parquet = path.suffix == ".parquet"
    if parquet:
        import pyarrow.parquet as pq
        have = pq.read_schema(path).names
    else:
        have = pd.read_csv(path, nrows=0).columns.tolist()
    cols = ID_COLS + [c for c in columns if c not in ID_COLS]
    missing = [c for c in cols if c not in have]
    if missing:
        raise SystemExit(f"{path}: missing column(s) {', '.join(missing)}  (feature_list.csv lists them)")

    if parquet:   # read only the months needed: a full features file has millions of rows
        dates = pd.to_datetime(pd.read_parquet(path, columns=["INFORMATION_DATE"]).INFORMATION_DATE)
        M = None if month == "all" else pd.Period(month, "M") if month else dates.max().to_period("M")
        filters = None if M is None else [("INFORMATION_DATE", ">=", (M - back).start_time.date()),
                                          ("INFORMATION_DATE", "<=", M.end_time.date())]
        df = pd.read_parquet(path, columns=cols, filters=filters)
    else:
        df = pd.read_csv(path, usecols=cols, dtype={c: "string" for c in ["POLICY_HASH", *CATEGORIES] if c in cols})
    df["INFORMATION_DATE"] = pd.to_datetime(df.INFORMATION_DATE)
    df["POLICY_HASH"] = df.POLICY_HASH.astype(str)
    mo = df.INFORMATION_DATE.dt.to_period("M")
    if not parquet:
        M = None if month == "all" else pd.Period(month, "M") if month else mo.max()
        if M is not None:
            keep = (mo >= M - back) & (mo <= M)
            df, mo = df[keep], mo[keep]
    if M is not None and not (mo == M).any():
        raise SystemExit(f"{path}: no rows for {M}")
    if (dup := pd.DataFrame({"p": df.POLICY_HASH, "m": mo}).duplicated()).any():
        raise SystemExit(f"{path}: {dup.sum():,} policies appear twice in a month, e.g. {df.POLICY_HASH[dup].iloc[0]}")
    return df.reset_index(drop=True), M


def set_categories(df):
    """Code each categorical column with its training list. A value the models never saw is left blank (missing), as
    in training; the count is printed."""
    for c, values in CATEGORIES.items():
        if c not in df:
            continue
        new = df[c].notna() & ~df[c].isin(values)
        if new.any():
            print(f"{c}: {new.sum():,} rows with values the models never saw, scored as blank: "
                  f"{sorted(df.loc[new, c].unique())[:5]}")
        df[c] = pd.Categorical(df[c].where(~new), categories=values)
    return df
