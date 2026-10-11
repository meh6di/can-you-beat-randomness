"""Keep the needed columns, validate, and write data_clean/*.csv."""
import glob
import os

import numpy as np
import pandas as pd

import config as C

KEEP_COLUMNS = [
    "Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
    "AvgH", "AvgD", "AvgA",          # market-average odds (pre-closing snapshot)
    "PSH", "PSD", "PSA",             # Pinnacle (pre-closing)
    "PSCH", "PSCD", "PSCA",          # Pinnacle CLOSING odds  (CLV)
    "AvgCH", "AvgCD", "AvgCA",       # market-average closing odds
    "MaxH", "MaxD", "MaxA",          # best available odds
]
ODDS_COLS = [c for c in KEEP_COLUMNS if c[:2] in ("Av", "PS", "Ma")]


def read_csv_robust(path):
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path}")


def clean_one(df):
    cols = [c for c in KEEP_COLUMNS if c in df.columns]
    d = df[cols].copy()
    d = d.dropna(how="all")
    d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, format="mixed", errors="coerce")
    d = d.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR", "AvgH", "AvgD", "AvgA"])
    n0 = len(d)
    d = d[d["FTR"].isin(["H", "D", "A"])]
    for c in ODDS_COLS:                      # decimal odds must be > 1, else treat as missing
        if c in d:
            d.loc[d[c] <= 1.0, c] = np.nan
    d = d.dropna(subset=["AvgH", "AvgD", "AvgA"])
    d = d.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
    return d.sort_values("Date", kind="stable").reset_index(drop=True), n0 - len(d)


def clean_datasets():
    os.makedirs(C.CLEAN_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(C.RAW_DIR, "*.csv")))
    if not files:
        print(f"No CSV files in '{C.RAW_DIR}'. Run fetch_data.py first.")
        return
    for fp in files:
        name = os.path.basename(fp)
        d, dropped = clean_one(read_csv_robust(fp))
        d.to_csv(os.path.join(C.CLEAN_DIR, name), index=False)
        miss = {c: f"{d[c].isna().mean():.0%}" for c in ("PSCH", "MaxH") if c in d and d[c].isna().any()}
        extra = f" | missing: {miss}" if miss else ""
        print(f"{name}: {len(d)} matches ({dropped} invalid dropped){extra}")
    print(f"\nCleaned files saved to '{C.CLEAN_DIR}'.")


if __name__ == "__main__":
    clean_datasets()
