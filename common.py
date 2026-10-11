"""Shared utilities for the evaluation scripts (stats_tests, prob_eval, clv_analysis).

Reproduces the models and betting rule of the original project (per-league models
trained on 2019-24, tested on 2024-26, 6% probability-edge rule, one bet per match),
but trains everything once and caches the test-set predictions in
results/predictions.csv so every analysis uses exactly the same numbers.
"""
import os
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

FEATURES_DIR = "data_features"
RESULTS_DIR = "results"
# Chronological split: train on older seasons, test on the newest two (pooled).
TRAIN_SEASONS = ["2019-2020", "2021-2022", "2022-2023", "2023-2024"]
TEST_SEASONS = ["2024-2025", "2025-2026"]
LEAGUES = ["premier_league", "la_liga", "serie_a", "bundesliga", "ligue_1"]
SEED = 42

FEATURE_COLUMNS = [
    "home_roll_points_5", "away_roll_points_5",
    "home_roll_gf_5", "away_roll_ga_5",
    "home_roll_sot_for_5", "away_roll_sot_ag_5",
    "home_roll_corners_for_5", "away_roll_corners_ag_5",
    "home_roll_sot_acc_5", "away_roll_sot_acc_5",
    "home_roll_conv_5", "away_roll_conv_5",
    "diff_points_5", "diff_sot_5", "diff_corners_5", "diff_rest_days",
    "market_prob_h", "market_prob_d", "market_prob_a",
]

# Outcome order used for every (n, 3) array in this code base.
OUTCOMES = ["H", "D", "A"]
OUTCOME_IDX = {"H": 0, "D": 1, "A": 2}
MODELS = ["logreg", "xgb", "xgb_cal"]
LABELS = {
    "logreg": "Logistic regression",
    "xgb": "XGBoost (raw)",
    "xgb_cal": "XGBoost (Platt-calibrated)",
    "clim": "Climatology (class freq.)",
    "market": "Market (Avg, de-vigged)",
    "pinnacle": "Pinnacle (de-vigged)",
    "pinnacle_close": "Pinnacle close (de-vigged)",
}

_XGB_LABEL = {"A": 0, "D": 1, "H": 2}


# --------------------------------------------------------------------------- data
def load_league_data(league):
    train = []
    for s in TRAIN_SEASONS:
        path = os.path.join(FEATURES_DIR, f"{league}_{s}.csv")
        if os.path.exists(path):
            train.append(pd.read_csv(path))
    test = []
    for s in TEST_SEASONS:
        path = os.path.join(FEATURES_DIR, f"{league}_{s}.csv")
        if os.path.exists(path):
            test.append(pd.read_csv(path))
    if not train or not test:
        return None, None
    return pd.concat(train, ignore_index=True), pd.concat(test, ignore_index=True)


def _xgb():
    return XGBClassifier(
        n_estimators=75, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=SEED, eval_metric="mlogloss",
    )


def fit_predict(train_df, test_df):
    """Return {model_name: (n, 3) probabilities in H, D, A order}."""
    X_tr, X_te = train_df[FEATURE_COLUMNS], test_df[FEATURE_COLUMNS]
    y = train_df["FTR"]
    out = {}

    lr = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_tr, y)
    classes = list(lr.classes_)
    out["logreg"] = lr.predict_proba(X_te)[:, [classes.index(o) for o in OUTCOMES]]

    y_num = y.map(_XGB_LABEL)  # A=0, D=1, H=2  ->  reorder columns to H, D, A
    out["xgb"] = _xgb().fit(X_tr, y_num).predict_proba(X_te)[:, [2, 1, 0]]

    cal = CalibratedClassifierCV(estimator=_xgb(), method="sigmoid", cv=3).fit(X_tr, y_num)
    out["xgb_cal"] = cal.predict_proba(X_te)[:, [2, 1, 0]]

    freq = y.value_counts(normalize=True)
    out["clim"] = np.tile([freq.get(o, 0.0) for o in OUTCOMES], (len(test_df), 1))
    return out


def get_predictions(refresh=False):
    """All test-season matches (5 leagues, sorted by date) with model probabilities.

    Columns added: league, p_{model}_{H,D,A} for model in MODELS + ['clim'].
    Cached in results/predictions.csv; pass refresh=True (or --refresh) to rebuild.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cache = os.path.join(RESULTS_DIR, "predictions.csv")
    if os.path.exists(cache) and not refresh:
        return pd.read_csv(cache, parse_dates=["Date"])

    frames = []
    for league in LEAGUES:
        train_df, test_df = load_league_data(league)
        if train_df is None:
            print(f"Skipping {league}: missing files in {FEATURES_DIR}/")
            continue
        preds = fit_predict(train_df, test_df)
        df = test_df.copy()
        df["league"] = league
        for model, arr in preds.items():
            for k, o in enumerate(OUTCOMES):
                df[f"p_{model}_{o}"] = arr[:, k]
        frames.append(df)
        print(f"  trained {league}: {len(train_df)} train / {len(test_df)} test matches")
    if not frames:
        raise SystemExit(f"No data found in {FEATURES_DIR}/. Run the data pipeline first.")
    full = pd.concat(frames, ignore_index=True)
    full["Date"] = pd.to_datetime(full["Date"])
    full = full.sort_values("Date", kind="stable").reset_index(drop=True)
    full.to_csv(cache, index=False)
    return full


# ----------------------------------------------------------------- array helpers
def odds_array(df, prefix="Avg"):
    return df[[f"{prefix}H", f"{prefix}D", f"{prefix}A"]].to_numpy(float)


def devig(odds):
    """Proportional (multiplicative) de-vigging of an (n, 3) odds array."""
    inv = 1.0 / odds
    return inv / inv.sum(axis=1, keepdims=True)


def model_probs(df, model):
    return df[[f"p_{model}_{o}" for o in OUTCOMES]].to_numpy(float)


def actual_index(df):
    return df["FTR"].map(OUTCOME_IDX).to_numpy()


# --------------------------------------------------------------------- betting
def pick_bets(df, model, rule="edge", threshold=0.06):
    """One bet per match on the best outcome, if its score clears the threshold.

    rule='edge': score = p_model - p_market_fair   (the original project's rule)
    rule='ev'  : score = p_model * odds - 1        (expected value per unit staked)
    Bets are placed at the market-average odds (AvgH/D/A), as in the original code.
    Returns a DataFrame indexed like `df` with: league, Date, pick (0/1/2), odds,
    fair_p (market fair prob. of the pick), won, profit.
    """
    p = model_probs(df, model)
    odds = odds_array(df, "Avg")
    fair = df[["market_prob_h", "market_prob_d", "market_prob_a"]].to_numpy(float)
    score = (p - fair) if rule == "edge" else (p * odds - 1.0)
    pick = score.argmax(axis=1)
    rows = np.arange(len(df))
    mask = (score[rows, pick] >= threshold) & (score[rows, pick] > 0)

    bets = df.loc[mask, ["league", "Date"]].copy()
    pk = pick[mask]
    bets["pick"] = pk
    bets["odds"] = odds[mask, pk]
    bets["fair_p"] = fair[mask, pk]
    bets["won"] = actual_index(df)[mask] == pk
    bets["profit"] = np.where(bets["won"], bets["odds"] - 1.0, -1.0)
    return bets


def full_profit_vector(df, bets):
    """Per-match profit for every match in df (0 where no bet was placed)."""
    s = pd.Series(0.0, index=df.index)
    s.loc[bets.index] = bets["profit"].to_numpy()
    return s.to_numpy()


def favorite_profit_vector(df):
    odds = odds_array(df, "Avg")
    pick = odds.argmin(axis=1)
    rows = np.arange(len(df))
    return np.where(actual_index(df) == pick, odds[rows, pick] - 1.0, -1.0)


def random_expected_profit_vector(df):
    """Expected per-match profit of a uniformly random pick (no sampling noise)."""
    odds = odds_array(df, "Avg")
    y = np.zeros_like(odds)
    y[np.arange(len(df)), actual_index(df)] = 1.0
    return (y * odds - 1.0).mean(axis=1)


# ----------------------------------------------------------------- bootstrapping
def bootstrap_means(x, n_boot=10000, rng=None, chunk=500):
    """Bootstrap distribution of the mean of x (resampling observations)."""
    rng = rng or np.random.default_rng(SEED)
    x = np.asarray(x, float)
    n = len(x)
    out = np.empty(n_boot)
    for i in range(0, n_boot, chunk):
        k = min(chunk, n_boot - i)
        out[i:i + k] = x[rng.integers(0, n, (k, n))].mean(axis=1)
    return out


def mean_ci(x, n_boot=10000, alpha=0.05, rng=None):
    """(mean, lo, hi) with a percentile bootstrap CI. NaNs if x is empty."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    b = bootstrap_means(x, n_boot, rng)
    lo, hi = np.quantile(b, [alpha / 2, 1 - alpha / 2])
    return x.mean(), lo, hi


def fmt_pct(x, d=2):
    return "-" if pd.isna(x) else f"{100 * x:.{d}f}%"
