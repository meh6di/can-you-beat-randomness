"""Item 2 - Probabilistic evaluation against the market.

Does the model add information beyond the market's own probabilities?
  * Log loss, Brier score and Ranked Probability Score (RPS), per source,
    with paired (per-match) differences vs the de-vigged market and 95% CIs.
  * Calibration: reliability diagrams and ECE for every source, with an
    ECE "noise floor" (what a perfectly calibrated forecaster would score by chance).

Sources: de-vigged market average, Pinnacle (if columns present), climatology,
logistic regression, raw XGBoost, Platt-calibrated XGBoost.
Lower is better for every score. Negative delta = better than the market.

Usage:  python prob_eval.py [--split holdout|dev|all]  (default: holdout)
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

from common import (
    MODELS, LABELS, OUTCOMES, RESULTS_DIR, SEED, get_predictions, split_from_argv, odds_array, devig,
    model_probs, actual_index,
)

N_BINS = 10
N_FLOOR_SIMS = 300
rng = np.random.default_rng(SEED)


# -------------------------------------------------------------------------- scores
def per_match_scores(p, y_idx):
    n = len(p)
    Y = np.zeros_like(p)
    Y[np.arange(n), y_idx] = 1.0
    logloss = -np.log(np.clip(p[np.arange(n), y_idx], 1e-12, 1.0))
    brier = ((p - Y) ** 2).sum(axis=1)
    # RPS over the ordered outcomes H < D < A
    rps = (((np.cumsum(p, axis=1) - np.cumsum(Y, axis=1))[:, :2]) ** 2).sum(axis=1) / 2.0
    return {"log_loss": logloss, "brier": brier, "rps": rps}


def build_sources(df):
    src = {"market": devig(odds_array(df, "Avg"))}
    if {"PSH", "PSD", "PSA"} <= set(df.columns):
        src["pinnacle"] = devig(odds_array(df, "PS"))
    if {"PSCH", "PSCD", "PSCA"} <= set(df.columns):
        src["pinnacle_close"] = devig(odds_array(df, "PSC"))
    for m in MODELS + ["clim"]:
        src[m] = model_probs(df, m)
    return src


# ------------------------------------------------------------------- calibration
def reliability(p, y, n_bins=N_BINS):
    """Equal-mass bins on predicted prob p (1-D) vs outcome indicator y (0/1)."""
    bins = np.array_split(np.argsort(p, kind="stable"), n_bins)
    mp = np.array([p[b].mean() for b in bins])
    fr = np.array([y[b].mean() for b in bins])
    cnt = np.array([len(b) for b in bins])
    se = np.sqrt(np.clip(fr * (1 - fr), 1e-9, None) / cnt)
    ece = (cnt * np.abs(mp - fr)).sum() / cnt.sum()
    return mp, fr, se, ece


def onehot(y_idx):
    Y = np.zeros((len(y_idx), 3))
    Y[np.arange(len(y_idx)), y_idx] = 1.0
    return Y


def ece_pooled(p, Y):
    return reliability(p.ravel(), Y.ravel())[3]


def ece_noise_floor(p, n_sims=N_FLOOR_SIMS):
    """ECE distribution if outcomes were drawn from p itself (perfect calibration)."""
    cum = np.cumsum(p, axis=1)
    out = np.empty(n_sims)
    for s in range(n_sims):
        u = rng.random(len(p))[:, None]
        idx = np.minimum((u > cum).sum(axis=1), 2)
        out[s] = ece_pooled(p, onehot(idx))
    return out


def main():
    split = split_from_argv("holdout")
    print(f"Split: {split}")
    df = get_predictions(split)
    src = build_sources(df)

    ok = np.ones(len(df), bool)
    for p in src.values():
        ok &= np.isfinite(p).all(axis=1)
    if (~ok).any():
        print(f"Dropping {(~ok).sum()} matches with missing probabilities in any source "
              f"(all sources are compared on the same {ok.sum()} matches).")
    df = df[ok].reset_index(drop=True)
    src = {k: v[ok] for k, v in src.items()}
    y_idx = actual_index(df)
    n = len(df)
    print(f"Evaluating {n} matches\n")

    # ------------------------------------------------------------ proper scores
    scores = {k: per_match_scores(p, y_idx) for k, p in src.items()}
    rows = []
    for name in src:
        for metric in ("log_loss", "brier", "rps"):
            s = scores[name][metric]
            d = s - scores["market"][metric]
            se = d.std(ddof=1) / np.sqrt(n)
            mean_d = d.mean()
            rows.append(dict(
                source=LABELS[name], metric=metric, mean=s.mean(),
                delta_vs_market=mean_d, ci_lo=mean_d - 1.96 * se, ci_hi=mean_d + 1.96 * se,
                p_value=(np.nan if name == "market" else 2 * (1 - norm.cdf(abs(mean_d / se)))) if se > 0 else np.nan,
            ))
    sc = pd.DataFrame(rows)
    sc.to_csv(f"{RESULTS_DIR}/prob_scores.csv", index=False)
    for metric in ("log_loss", "brier", "rps"):
        t = sc[sc.metric == metric].drop(columns="metric").round(4)
        print(f"=== {metric} (lower = better; delta < 0 means better than market) ===")
        print(t.to_string(index=False), "\n")

    # per-league log-loss delta vs market (is any league different?)
    rows = []
    for league, idx in df.groupby("league").groups.items():
        pos = df.index.get_indexer(idx)
        for name in src:
            if name == "market":
                continue
            d = scores[name]["log_loss"][pos] - scores["market"]["log_loss"][pos]
            se = d.std(ddof=1) / np.sqrt(len(d))
            rows.append(dict(league=league, source=LABELS[name], n=len(d),
                             delta_log_loss=d.mean(), ci_lo=d.mean() - 1.96 * se,
                             ci_hi=d.mean() + 1.96 * se))
    pl = pd.DataFrame(rows)
    pl.to_csv(f"{RESULTS_DIR}/prob_scores_per_league.csv", index=False)
    print("=== Log-loss delta vs market, per league (saved to prob_scores_per_league.csv) ===")
    print(pl.pivot(index="source", columns="league", values="delta_log_loss").round(4).to_string(), "\n")

    # --------------------------------------------------------------- calibration
    Y = onehot(y_idx)
    rows = []
    for name, p in src.items():
        floor = ece_noise_floor(p)
        e = ece_pooled(p, Y)
        row = dict(source=LABELS[name], ece=e, ece_noise_floor=floor.mean(),
                   p_vs_perfect=(1 + (floor >= e).sum()) / (len(floor) + 1))
        for k, o in enumerate(OUTCOMES):
            row[f"ece_{o}"] = reliability(p[:, k], Y[:, k])[3]
        rows.append(row)
    cal = pd.DataFrame(rows)
    cal.to_csv(f"{RESULTS_DIR}/calibration.csv", index=False)
    print(f"=== Calibration (ECE, {N_BINS} equal-mass bins, pooled over H/D/A) ===")
    print(cal.round(4).to_string(index=False))
    print("ece_noise_floor = ECE expected from sampling noise alone if the forecaster were perfectly\n"
          "calibrated; only ECE clearly above it (small p_vs_perfect) indicates miscalibration.\n")

    # ------------------------------------------------------------------- figure
    plot_names = [k for k in ["market", "pinnacle"] + MODELS if k in src]
    ncols = 3
    nrows = int(np.ceil(len(plot_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, plot_names):
        mp, fr, se, ece = reliability(src[name].ravel(), Y.ravel())
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.errorbar(mp, fr, yerr=1.96 * se, fmt="o-", color="royalblue", capsize=3)
        ax.set_title(f"{LABELS[name]}\nECE = {ece:.3f}", fontsize=10)
        ax.set_xlabel("Predicted probability (bin mean)")
        ax.set_ylabel("Observed frequency")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(True, ls=":", alpha=0.5)
    for ax in axes[len(plot_names):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/reliability.png", dpi=200)
    print(f"Saved tables and {RESULTS_DIR}/reliability.png")


if __name__ == "__main__":
    main()
