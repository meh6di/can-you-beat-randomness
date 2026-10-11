"""Item 1 - Statistical testing of the betting results.

1. Bootstrap CIs on ROI (overall and per league).
2. Market-simulation null: if the de-vigged market probabilities were the TRUE
   probabilities, what ROI would the model's exact bets earn by chance? Gives a
   p-value for the observed ROI (one-sided: "better than chance under an efficient market").
3. Paired bootstrap on per-match profit between strategies (model vs model,
   model vs favorite, model vs random).
4. Power analysis: smallest ROI this sample could detect, and bets needed.

Usage:  python stats_tests.py [--refresh]
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

from common import (
    MODELS, LABELS, RESULTS_DIR, SEED, get_predictions, pick_bets, mean_ci,
    bootstrap_means, full_profit_vector, favorite_profit_vector,
    random_expected_profit_vector, fmt_pct,
)

THRESHOLDS = [0.03, 0.06]   # 0.06 is the main setting; 0.03 shows threshold sensitivity
MAIN_THR = 0.06
N_BOOT = 10_000
N_SIM = 20_000
rng = np.random.default_rng(SEED)


# ------------------------------------------------------------ market-simulation null
def market_null(bets, n_sim=N_SIM):
    """ROI distribution of these exact bets if outcomes ~ de-vigged market probs.

    Each bet wins with probability fair_p; profit = win * odds - 1.
    Bets are on different matches (one per match), so they are independent.
    """
    p = bets["fair_p"].to_numpy()
    odds = bets["odds"].to_numpy()
    n = len(p)
    rois = np.empty(n_sim)
    chunk = 2000
    for i in range(0, n_sim, chunk):
        k = min(chunk, n_sim - i)
        wins = rng.random((k, n)) < p
        rois[i:i + k] = wins @ odds / n - 1.0
    return rois


def summarize(bets):
    n = len(bets)
    if n == 0:
        return dict(bets=0), np.array([])
    roi, lo, hi = mean_ci(bets["profit"], N_BOOT, rng=rng)
    null = market_null(bets)
    return dict(
        bets=n, roi=roi, ci_lo=lo, ci_hi=hi,
        null_mean=null.mean(), null_sd=null.std(),
        z_vs_null=(roi - null.mean()) / null.std(),
        p_vs_null=(1 + (null >= roi).sum()) / (len(null) + 1),
        ci_excludes_0=bool(lo > 0 or hi < 0),
    ), null


def print_table(title, df, pct_cols):
    out = df.copy()
    for c in pct_cols:
        if c in out:
            out[c] = out[c].map(fmt_pct)
    for c in ("z_vs_null", "p_vs_null"):
        if c in out:
            out[c] = out[c].map(lambda v: f"{v:.3f}")
    print(f"\n=== {title} ===")
    print(out.to_string(index=False))


def main():
    df = get_predictions(refresh="--refresh" in sys.argv)
    n_matches = len(df)
    print(f"{n_matches} test matches across {df['league'].nunique()} leagues")

    # ---------------------------------------------------------------- 1+2 overall
    rows, nulls = [], {}
    for thr in THRESHOLDS:
        for m in MODELS:
            bets = pick_bets(df, m, "edge", thr)
            res, null = summarize(bets)
            res.update(model=LABELS[m], threshold=thr)
            rows.append(res)
            if thr == MAIN_THR:
                nulls[m] = (res["roi"], null)
    overall = pd.DataFrame(rows)[["model", "threshold", "bets", "roi", "ci_lo", "ci_hi",
                                  "null_mean", "z_vs_null", "p_vs_null", "ci_excludes_0"]]
    print_table("Overall ROI: bootstrap CI and market-simulation null "
                "(1 bet/match, edge rule)", overall, ["roi", "ci_lo", "ci_hi", "null_mean"])
    print("\nHow to read: the CI says whether ROI differs from 0 (profitable?). p_vs_null says whether "
          "ROI beats what chance\nalone would give if the market were efficient (null_mean is about "
          "-overround, not 0). Both matter.")
    print(f"\n{len(overall)} model/threshold combinations were tested on one season: "
          f"treat any single p<0.05 with caution (multiple comparisons).")
    overall.to_csv(f"{RESULTS_DIR}/stats_overall.csv", index=False)

    # --------------------------------------------------------------- per league
    rows = []
    for m in MODELS:
        bets = pick_bets(df, m, "edge", MAIN_THR)
        for league, g in bets.groupby("league"):
            res, _ = summarize(g)
            res.update(model=LABELS[m], league=league)
            rows.append(res)
    per_league = pd.DataFrame(rows)[["model", "league", "bets", "roi", "ci_lo", "ci_hi",
                                     "p_vs_null", "ci_excludes_0"]]
    print_table(f"Per-league ROI at {MAIN_THR:.0%} threshold (note the CI widths)",
                per_league, ["roi", "ci_lo", "ci_hi"])
    per_league.to_csv(f"{RESULTS_DIR}/stats_per_league.csv", index=False)

    # ------------------------------------------------------------ paired comparison
    strat = {LABELS[m]: full_profit_vector(df, pick_bets(df, m, "edge", MAIN_THR)) for m in MODELS}
    strat["Favorite (all matches)"] = favorite_profit_vector(df)
    strat["Random (expected)"] = random_expected_profit_vector(df)
    names = list(strat)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = strat[names[i]] - strat[names[j]]
            boots = bootstrap_means(d, N_BOOT, rng) * n_matches   # difference in TOTAL profit
            rows.append(dict(
                A=names[i], B=names[j], diff_units=d.sum(),
                ci_lo=np.quantile(boots, 0.025), ci_hi=np.quantile(boots, 0.975),
                p_two_sided=min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())),
            ))
    paired = pd.DataFrame(rows)
    print(f"\n=== Paired bootstrap on per-match profit (A - B, total units, "
          f"{MAIN_THR:.0%} threshold) ===")
    print(paired.round(3).to_string(index=False))
    print("(Unbet matches count as 0 profit, so strategies with different bet counts are comparable.)")
    paired.to_csv(f"{RESULTS_DIR}/stats_paired.csv", index=False)

    # -------------------------------------------------------------------- power
    z_a, z_b = norm.ppf(0.95), norm.ppf(0.80)   # one-sided alpha=5%, power=80%
    rows = []
    for m in MODELS:
        p = pick_bets(df, m, "edge", MAIN_THR)["profit"].to_numpy()
        sd, n = p.std(ddof=1), len(p)
        row = dict(model=LABELS[m], bets=n, sd_per_bet=sd, se_roi=sd / np.sqrt(n),
                   min_detectable_roi=(z_a + z_b) * sd / np.sqrt(n))
        for e in (0.02, 0.03, 0.05):
            row[f"bets_for_{int(e * 100)}pct"] = int(np.ceil(((z_a + z_b) * sd / e) ** 2))
        rows.append(row)
    power = pd.DataFrame(rows)
    print("\n=== Power analysis (H1: true ROI > 0; alpha 5% one-sided, power 80%) ===")
    disp = power.copy()
    for c in ("se_roi", "min_detectable_roi"):
        disp[c] = disp[c].map(fmt_pct)
    disp["sd_per_bet"] = disp["sd_per_bet"].map(lambda v: f"{v:.2f}")
    print(disp.to_string(index=False))
    print("min_detectable_roi = smallest true ROI this sample would detect 80% of the time.")
    power.to_csv(f"{RESULTS_DIR}/stats_power.csv", index=False)

    # --------------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 3.8), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), MODELS):
        roi, null = nulls[m]
        ax.hist(100 * null, bins=60, color="lightgray", edgecolor="white")
        ax.axvline(100 * roi, color="royalblue", lw=2.2, label=f"observed {100 * roi:.1f}%")
        ax.axvline(0, color="black", ls=":", lw=1)
        p = (1 + (null >= roi).sum()) / (len(null) + 1)
        ax.set_title(f"{LABELS[m]}\np = {p:.3f}", fontsize=10)
        ax.set_xlabel("ROI under efficient-market null (%)")
        ax.legend(fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("Simulations")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/market_null.png", dpi=200)
    print(f"\nSaved tables and {RESULTS_DIR}/market_null.png")


if __name__ == "__main__":
    main()
