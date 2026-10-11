"""Shared, dependency-light utilities used by every analysis script."""
import glob
import os
import sys

import numpy as np
import pandas as pd

import config as C
from config import (DEVIG_METHOD, FEATURES_DIR, LEAGUES, RESULTS_DIR, SEED)  # re-exported

OUTCOMES = ["H", "D", "A"]
OUTCOME_IDX = {"H": 0, "D": 1, "A": 2}
MODELS = ["lr", "blend", "xgb", "xgb_cal"]
LABELS = {
    "lr": "Logistic regression (scaled, tuned)",
    "blend": "Market x strength-model blend",
    "xgb": "XGBoost (raw)",
    "xgb_cal": "XGBoost (calibrated)",
    "clim": "Climatology (class freq.)",
    "market": "Market (Avg, de-vigged)",
    "pinnacle": "Pinnacle (de-vigged)",
    "pinnacle_close": "Pinnacle close (de-vigged)",
}


# ------------------------------------------------------------------------ odds
def devig(odds, method=None):
    """Remove the bookmaker margin from an (n, 3) array of decimal odds.

    'proportional': p_k = (1/o_k) / sum(1/o)
    'power'       : p_k = (1/o_k)^k with k>=1 chosen so that sum p = 1. Takes more margin
                    from longshots than proportional, i.e. it partly corrects the
                    favourite-longshot bias.
    """
    method = method or DEVIG_METHOD
    inv = 1.0 / np.asarray(odds, float)
    if method == "proportional":
        return inv / inv.sum(axis=1, keepdims=True)
    if method != "power":
        raise ValueError(method)
    lo, hi = np.ones(len(inv)), np.full(len(inv), 20.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        too_big = (inv ** mid[:, None]).sum(axis=1) > 1.0
        lo, hi = np.where(too_big, mid, lo), np.where(too_big, hi, mid)
    p = inv ** ((lo + hi) / 2)[:, None]
    return p / p.sum(axis=1, keepdims=True)


def odds_array(df, prefix="Avg"):
    return df[[f"{prefix}H", f"{prefix}D", f"{prefix}A"]].to_numpy(float)


def model_probs(df, model):
    return df[[f"p_{model}_{o}" for o in OUTCOMES]].to_numpy(float)


def actual_index(df):
    return df["FTR"].map(OUTCOME_IDX).to_numpy()


# ------------------------------------------------------------------------ data
def load_features():
    """All leagues' engineered features (every season), chronological."""
    frames = [pd.read_csv(p, parse_dates=["Date"])
              for p in sorted(glob.glob(os.path.join(FEATURES_DIR, "*.csv")))]
    if not frames:
        raise SystemExit(f"No feature files in '{FEATURES_DIR}/'. Run engineer_features.py.")
    return pd.concat(frames, ignore_index=True).sort_values("Date", kind="stable").reset_index(drop=True)


def split_from_argv(default="holdout"):
    """Read `--split dev|holdout|all` from the command line."""
    if "--split" in sys.argv:
        return sys.argv[sys.argv.index("--split") + 1]
    return default


def get_predictions(split="all"):
    """Walk-forward predictions (results/wf_predictions.csv), optionally one split."""
    path = os.path.join(RESULTS_DIR, "wf_predictions.csv")
    if not os.path.exists(path):
        raise SystemExit("results/wf_predictions.csv not found. Run walkforward.py first.")
    df = pd.read_csv(path, parse_dates=["Date"])
    if split != "all":
        df = df[df["split"] == split]
    if df.empty:
        raise SystemExit(f"No rows for split '{split}'.")
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------- betting
def pick_bets(df, model, rule="edge", threshold=0.06, price="Avg"):
    """One bet per match on the best outcome if its score clears the threshold.

    rule 'edge': score = p_model - p_market_fair   (probability points)
    rule 'ev'  : score = p_model * odds - 1        (expected value per unit; accounts for vig)
    Stakes 1 unit at the `price` odds ('Avg', 'Max' = best available, 'PS' = Pinnacle).
    Returns a frame indexed like df: league, Date, pick, odds, fair_p, p_model, won, profit.
    """
    p = model_probs(df, model)
    odds = odds_array(df, price)
    fair = df[["market_prob_h", "market_prob_d", "market_prob_a"]].to_numpy(float)
    score = (p - fair) if rule == "edge" else (p * odds - 1.0)
    score = np.nan_to_num(score, nan=-np.inf)
    pick = score.argmax(axis=1)
    rows = np.arange(len(df))
    best = score[rows, pick]
    mask = (best >= threshold) & (best > 0)
    pk = pick[mask]
    bets = df.loc[mask, ["league", "Date"]].copy()
    bets["pick"] = pk
    bets["odds"] = odds[mask, pk]
    bets["fair_p"] = fair[mask, pk]
    bets["p_model"] = p[mask, pk]
    bets["won"] = actual_index(df)[mask] == pk
    bets["profit"] = np.where(bets["won"], bets["odds"] - 1.0, -1.0)
    return bets


def full_profit_vector(df, bets):
    s = np.zeros(len(df))
    s[df.index.get_indexer(bets.index)] = bets["profit"].to_numpy()
    return s


def favorite_profit_vector(df, price="Avg"):
    odds = odds_array(df, price)
    pick = odds.argmin(axis=1)
    rows = np.arange(len(df))
    return np.where(actual_index(df) == pick, odds[rows, pick] - 1.0, -1.0)


def random_expected_profit_vector(df, price="Avg"):
    odds = odds_array(df, price)
    y = np.zeros_like(odds)
    y[np.arange(len(df)), actual_index(df)] = 1.0
    return (y * odds - 1.0).mean(axis=1)


def clv_of_bets(df, bets, price="Avg"):
    """CLV of each bet vs Pinnacle's de-vigged closing price, and 'pick skill'
    (CLV minus the average CLV of all three outcomes of the same match)."""
    close = devig(odds_array(df, "PSC"))
    clv_all = odds_array(df, price) * close - 1.0
    idx = df.index.get_indexer(bets.index)
    pk = bets["pick"].to_numpy()
    clv = clv_all[idx, pk]
    return clv, clv - clv_all[idx].mean(axis=1)


def market_null(bets, n_sim=20_000, rng=None, chunk=2000):
    """ROI distribution of these exact bets if outcomes ~ de-vigged market probabilities."""
    rng = rng or np.random.default_rng(SEED)
    p, odds = bets["fair_p"].to_numpy(), bets["odds"].to_numpy()
    n = len(p)
    out = np.empty(n_sim)
    for i in range(0, n_sim, chunk):
        k = min(chunk, n_sim - i)
        out[i:i + k] = (rng.random((k, n)) < p) @ odds / n - 1.0
    return out


# ---------------------------------------------------------------- bootstrapping
def bootstrap_means(x, n_boot=10_000, rng=None, chunk=500):
    rng = rng or np.random.default_rng(SEED)
    x = np.asarray(x, float)
    n = len(x)
    out = np.empty(n_boot)
    for i in range(0, n_boot, chunk):
        k = min(chunk, n_boot - i)
        out[i:i + k] = x[rng.integers(0, n, (k, n))].mean(axis=1)
    return out


def mean_ci(x, n_boot=10_000, alpha=0.05, rng=None):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    lo, hi = np.quantile(bootstrap_means(x, n_boot, rng), [alpha / 2, 1 - alpha / 2])
    return x.mean(), lo, hi


def fmt_pct(x, d=2):
    return "-" if pd.isna(x) else f"{100 * x:.{d}f}%"
