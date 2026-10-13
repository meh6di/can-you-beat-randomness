"""Item 3 - Closing Line Value (CLV) against Pinnacle's closing odds.

CLV asks: did we get a better price than the sharpest market's final (closing) fair price?
    CLV = odds_taken * p_close_fair - 1        (expected profit per unit if the closing
                                                 de-vigged Pinnacle probability is the truth)
It is far lower-variance than realised ROI because it doesn't depend on who won.

Reported per strategy:
  * mean CLV (with bootstrap CI) and share of bets with CLV > 0
  * share of bets that beat Pinnacle's *raw* closing odds (stricter: includes their margin)
  * "pick skill": CLV of the bet minus the average CLV of all 3 outcomes of that match.
    Soft-book odds carry a margin, so every blind bet has negative CLV; pick skill
    isolates whether the model chose the better side of the match.
  * ROI of the same bets at the odds actually taken (average, or best available odds), and the
    SE of mean CLV vs SE of mean ROI (the variance reduction)
Baselines: favorite agent and uniformly random pick over all matches.

Requires closing-odds columns (PSCH/PSCD/PSCA, optionally MaxH/D/A). Add them to
KEEP_COLUMNS in clean_data.py, then re-run clean_data.py and engineer_features.py
(see the patched clean_data.py shipped with this script).

Usage:  python clv_analysis.py [--split holdout|dev|all]  (default: holdout)
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (
    MODELS, LABELS, RESULTS_DIR, SEED, get_predictions, split_from_argv, pick_bets, odds_array, devig,
    mean_ci, bootstrap_means, fmt_pct,
)

THRESHOLD = 0.06
N_BOOT = 10_000
rng = np.random.default_rng(SEED)


def row_for(label, odds_label, clv, profit=None, skill=None, beat_raw=None):
    r = dict(strategy=label, odds_used=odds_label, bets=len(clv))
    m, lo, hi = mean_ci(clv, N_BOOT, rng=rng)
    r.update(clv=m, clv_lo=lo, clv_hi=hi, pct_clv_pos=(clv > 0).mean(),
             se_clv=clv.std(ddof=1) / np.sqrt(len(clv)))
    if beat_raw is not None:
        r["pct_beat_raw_close"] = beat_raw.mean()
    if profit is not None:
        r["roi"] = profit.mean()
        r["se_roi"] = profit.std(ddof=1) / np.sqrt(len(profit))
        r["se_ratio_roi_over_clv"] = r["se_roi"] / r["se_clv"]
    if skill is not None:
        b = bootstrap_means(skill, N_BOOT, rng)
        r["pick_skill"] = skill.mean()
        r["pick_skill_lo"], r["pick_skill_hi"] = np.quantile(b, [0.025, 0.975])
        r["pick_skill_p"] = min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))
    return r


def main():
    split = split_from_argv("holdout")
    print(f"Split: {split}")
    df = get_predictions(split)
    needed = ["PSCH", "PSCD", "PSCA"]
    if not set(needed) <= set(df.columns):
        raise SystemExit(
            "Closing odds (PSCH/PSCD/PSCA) not found in the feature files.\n"
            "Run: python clean_data.py && python engineer_features.py && python walkforward.py"
        )

    close_raw = odds_array(df, "PSC")
    ok = np.isfinite(close_raw).all(axis=1) & np.isfinite(odds_array(df, "Avg")).all(axis=1)
    if (~ok).any():
        print(f"Dropping {(~ok).sum()} matches without Pinnacle closing odds.")
    df = df[ok].reset_index(drop=True)
    close_raw = close_raw[ok]
    fair_close = devig(close_raw)
    n_all = len(df)
    print(f"{n_all} matches with Pinnacle closing odds\n")

    price_sets = [("Avg", "Average market odds (as in the backtests)")]
    if {"MaxH", "MaxD", "MaxA"} <= set(df.columns) and np.isfinite(odds_array(df, "Max")).all():
        price_sets.append(("Max", "Best available odds (line-shopping, optimistic)"))

    rows, curves = [], {}
    for prefix, plabel in price_sets:
        taken = odds_array(df, prefix)
        clv_all = taken * fair_close - 1.0               # (n, 3): CLV of every possible bet
        rows_idx = np.arange(n_all)

        # baselines over all matches
        fav = odds_array(df, "Avg").argmin(axis=1)
        rows.append(row_for("Favorite (all matches)", plabel, clv_all[rows_idx, fav],
                            beat_raw=taken[rows_idx, fav] > close_raw[rows_idx, fav]))
        rows.append(row_for("Random pick (expected, all matches)", plabel, clv_all.mean(axis=1)))

        for m in MODELS:
            bets = pick_bets(df, m, "edge", THRESHOLD)
            if len(bets) < 2:
                print(f"  (no bets for {LABELS[m]} at this threshold - skipped)")
                continue
            idx = df.index.get_indexer(bets.index)
            pk = bets["pick"].to_numpy()
            clv = clv_all[idx, pk]
            skill = clv - clv_all[idx].mean(axis=1)      # vs random side of the same match
            beat = taken[idx, pk] > close_raw[idx, pk]
            # profit at the odds actually taken for this price set (pick_bets prices at Avg only,
            # which made the best-odds rows repeat the average-odds ROI)
            profit = np.where(bets["won"].to_numpy(), taken[idx, pk] - 1.0, -1.0)
            rows.append(row_for(f"{LABELS[m]} ({THRESHOLD:.0%} edge)", plabel, clv,
                                profit=profit, skill=skill, beat_raw=beat))
            if prefix == "Avg":
                curves[m] = (bets["Date"].to_numpy(), bets["profit"].cumsum().to_numpy(),
                             pd.Series(clv).cumsum().to_numpy())

    res = pd.DataFrame(rows)
    res.to_csv(f"{RESULTS_DIR}/clv_summary.csv", index=False)

    pct_cols = ["clv", "clv_lo", "clv_hi", "pct_clv_pos", "pct_beat_raw_close", "roi",
                "se_clv", "se_roi", "pick_skill", "pick_skill_lo", "pick_skill_hi"]
    for prefix, plabel in price_sets:
        t = res[res.odds_used == plabel].drop(columns="odds_used").copy()
        for c in pct_cols:
            if c in t:
                t[c] = t[c].map(fmt_pct)
        for c in ("se_ratio_roi_over_clv", "pick_skill_p"):
            if c in t:
                t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
        print(f"=== CLV vs Pinnacle close | prices: {plabel} ===")
        print(t.to_string(index=False), "\n")
    print("Reading guide: blind bets at soft-book prices have negative CLV (the margin).\n"
          "pick_skill > 0 (CI excluding 0) means the model chose sides that the sharp closing\n"
          "line later agreed with. se_ratio = how many times noisier ROI is than CLV.\n"
          "Caveat: Avg odds are a pre-closing snapshot; timing differences vs Pinnacle's close\n"
          "add noise, and de-vigging uses config.DEVIG_METHOD ('power'; Shin is an easy upgrade).")

    # ------------------------------------------------------------------- figure
    plot_models = [m for m in MODELS if m in curves]
    fig, axes = plt.subplots(1, len(plot_models), figsize=(5 * len(plot_models), 3.8), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), plot_models):
        _, cum_profit, cum_clv = curves[m]
        ax.plot(cum_profit, color="royalblue", label="Realised profit")
        ax.plot(cum_clv, color="darkorange", ls="--", label="Cumulative CLV (expected vs close)")
        ax.axhline(0, color="gray", ls=":", alpha=0.7)
        ax.set_title(LABELS[m], fontsize=10)
        ax.set_xlabel("Bets (chronological)")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("Units")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/clv_curves.png", dpi=200)
    print(f"\nSaved {RESULTS_DIR}/clv_summary.csv and {RESULTS_DIR}/clv_curves.png")


if __name__ == "__main__":
    main()
