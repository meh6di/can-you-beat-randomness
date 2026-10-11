"""Walk-forward (expanding window) predictions for the test seasons.

For every calendar month M of the test seasons:
    train      = labelled matches before (M start - VAL_DAYS)
    validation = the VAL_DAYS before M     (tunes C, blend weight, calibration)
    test       = matches in M              (predicted, never seen)
So models are retrained monthly as new results arrive, hyper-parameters are chosen
causally, and nothing after M ever influences M's predictions.
Output: results/wf_predictions.csv, results/wf_meta.csv
Usage:  python walkforward.py
"""
import os

import pandas as pd

import config as C
from common import OUTCOMES, load_features
from models import fit_predict_fold

MODEL_NAMES = ["lr", "blend", "xgb", "xgb_cal", "clim"]


def labelled(df):
    return df[~df["Season"].isin(C.FEATURE_ONLY_SEASONS)].sort_values("Date", kind="stable")


def make_folds(lab):
    """Yield (period, train, val, test) with train < val < test strictly in time."""
    test_all = lab[lab["Season"].isin(C.TEST_SEASONS)]
    for period in sorted(test_all["Date"].dt.to_period("M").unique()):
        start, end = period.start_time, period.end_time
        val_start = start - pd.Timedelta(days=C.VAL_DAYS)
        test = test_all[(test_all["Date"] >= start) & (test_all["Date"] <= end)]
        val = lab[(lab["Date"] >= val_start) & (lab["Date"] < start)]
        train = lab[lab["Date"] < val_start]
        yield period, train, val, test


def main():
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    lab = labelled(load_features())
    keep = ["league", "Season", "Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG",
            "market_prob_h", "market_prob_d", "market_prob_a"]
    keep += [c for c in lab.columns if c[:2] in ("Av", "PS", "Ma") and c[-1] in "HDA"]
    frames, metas = [], []
    for period, train, val, test in make_folds(lab):
        if test.empty:
            continue
        if len(train) < C.MIN_TRAIN_MATCHES:
            print(f"  {period}: only {len(train)} training matches - skipped")
            continue
        assert train["Date"].max() < val["Date"].min() <= val["Date"].max() < test["Date"].min()
        preds, meta = fit_predict_fold(train, val, test)
        f = test[keep].copy()
        for name in MODEL_NAMES:
            for k, o in enumerate(OUTCOMES):
                f[f"p_{name}_{o}"] = preds[name][:, k]
        f["split"] = f["Season"].map(lambda s: "holdout" if s in C.HOLDOUT_SEASONS else "dev")
        frames.append(f)
        metas.append(dict(period=str(period), n_train=len(train), n_val=len(val), n_test=len(test), **meta))
        print(f"  {period}: train {len(train)} / val {len(val)} / test {len(test)} | "
              f"C={meta['lr_C']} blend_w={meta['blend_w']:.2f}")
    if not frames:
        raise SystemExit("No folds produced - check seasons in config.py and the data.")
    pd.concat(frames, ignore_index=True).to_csv(f"{C.RESULTS_DIR}/wf_predictions.csv", index=False)
    pd.DataFrame(metas).to_csv(f"{C.RESULTS_DIR}/wf_meta.csv", index=False)
    print(f"\nSaved {C.RESULTS_DIR}/wf_predictions.csv and wf_meta.csv")


if __name__ == "__main__":
    main()
