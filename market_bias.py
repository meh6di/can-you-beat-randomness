"""Item 7 - Favourite-longshot bias: study the market itself (no model involved).

Uses every match in the feature files (all seasons, 5 leagues). For each (match, outcome)
pair we ask what blind betting at various odds would have returned.
  * ROI by odds bucket at average, best (Max) and Pinnacle prices, with bootstrap CIs
  * Observed win frequency vs the market's de-vigged probability, per bucket
  * Slope of per-bet profit on log(odds) with a match-clustered bootstrap CI
    (negative slope = longshots lose more per unit staked = favourite-longshot bias)
  * Simple strategies (favourite, underdog, home, draw, away) with CIs and per-season ROI
Usage: python market_bias.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (RESULTS_DIR, SEED, actual_index, devig, fmt_pct, load_features, mean_ci,
                    odds_array)

EDGES = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 1e9]
LABELS = ["<1.5", "1.5-2", "2-3", "3-4", "4-6", "6-10", "10+"]
N_BOOT = 2000
rng = np.random.default_rng(SEED)


def main():
    df = load_features()
    n = len(df)
    y = actual_index(df)
    won = np.zeros((n, 3), bool)
    won[np.arange(n), y] = True
    prices = {"Avg": odds_array(df, "Avg")}
    for p in ("PS", "Max"):
        if f"{p}H" in df and df[f"{p}H"].notna().any():
            prices[p] = odds_array(df, p)
    fair = devig(prices["Avg"])
    print(f"{n} matches, {df['Season'].nunique()} seasons, prices: {list(prices)}")

    # ------------------------------------------------------------- odds buckets
    avg = prices["Avg"]
    bucket = np.digitize(avg, EDGES[1:-1])          # (n,3) bucket id
    rows = []
    for b, name in enumerate(LABELS):
        m = bucket == b
        if m.sum() < 30:
            continue
        row = dict(bucket=name, n=int(m.sum()), implied_raw=(1 / avg[m]).mean(),
                   fair_prob=fair[m].mean(), win_freq=won[m].mean())
        row["win_freq_se"] = np.sqrt(row["win_freq"] * (1 - row["win_freq"]) / row["n"])
        for pname, o in prices.items():
            mm = m & np.isfinite(o)
            roi, lo, hi = mean_ci(won[mm] * o[mm] - 1.0, N_BOOT, rng=rng)
            row.update({f"roi_{pname}": roi, f"roi_{pname}_lo": lo, f"roi_{pname}_hi": hi})
        rows.append(row)
    buckets = pd.DataFrame(rows)
    buckets.to_csv(f"{RESULTS_DIR}/market_bias_buckets.csv", index=False)
    show = buckets.copy()
    for c in show.columns:
        if c not in ("bucket", "n"):
            show[c] = show[c].map(lambda v: fmt_pct(v, 2))
    print("\n=== Blind betting by odds bucket (bucketed on average odds) ===")
    print(show[["bucket", "n", "implied_raw", "fair_prob", "win_freq"] + [c for c in show if c.startswith("roi_") and not c.endswith(("lo", "hi"))]].to_string(index=False))

    # ------------------------------------------------------ slope (clustered boot)
    srows = []
    for pname, o in prices.items():
        ok = np.isfinite(o).all(axis=1)
        lo_, pr_ = np.log(o[ok]), won[ok] * o[ok] - 1.0
        m = ok.sum()

        def slope(idx):
            x, yv = lo_[idx].ravel(), pr_[idx].ravel()
            return ((x - x.mean()) * (yv - yv.mean())).mean() / x.var()

        s0 = slope(np.arange(m))
        boots = np.array([slope(rng.integers(0, m, m)) for _ in range(1000)])
        srows.append(dict(price=pname, slope_profit_per_log_odds=s0,
                          ci_lo=np.quantile(boots, 0.025), ci_hi=np.quantile(boots, 0.975),
                          p_two_sided=min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))))
    slopes = pd.DataFrame(srows)
    slopes.to_csv(f"{RESULTS_DIR}/market_bias_slope.csv", index=False)
    print("\n=== Slope of per-bet profit on log(odds) (negative = favourite-longshot bias) ===")
    print(slopes.round(4).to_string(index=False))

    # ---------------------------------------------------------------- strategies
    picks = {
        "Favourite": avg.argmin(axis=1), "Underdog": avg.argmax(axis=1),
        "Always home": np.zeros(n, int), "Always draw": np.ones(n, int), "Always away": np.full(n, 2),
    }
    srows, season_rows = [], []
    for sname, pk in picks.items():
        for pname, o in prices.items():
            ok = np.isfinite(o).all(axis=1)
            prof = np.where(y == pk, o[np.arange(n), pk] - 1.0, -1.0)
            roi, lo, hi = mean_ci(prof[ok], N_BOOT, rng=rng)
            srows.append(dict(strategy=sname, price=pname, n=int(ok.sum()), roi=roi, ci_lo=lo, ci_hi=hi))
            if sname == "Favourite" and pname in ("Avg", "PS"):
                for season, g in pd.Series(prof[ok]).groupby(df.loc[ok, "Season"].to_numpy()):
                    season_rows.append(dict(price=pname, season=season, n=len(g), roi=g.mean()))
    strat = pd.DataFrame(srows)
    strat.to_csv(f"{RESULTS_DIR}/market_bias_strategies.csv", index=False)
    pd.DataFrame(season_rows).to_csv(f"{RESULTS_DIR}/market_bias_favourite_by_season.csv", index=False)
    show = strat.copy()
    for c in ("roi", "ci_lo", "ci_hi"):
        show[c] = show[c].map(fmt_pct)
    print("\n=== Blind strategies over all matches ===")
    print(show.to_string(index=False))
    print("\nFavourite ROI by season (stability check):")
    print(pd.DataFrame(season_rows).pivot(index="season", columns="price", values="roi").map(fmt_pct).to_string())

    # -------------------------------------------------------------------- figure
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(buckets))
    a1.errorbar(x, buckets["win_freq"], yerr=1.96 * buckets["win_freq_se"], fmt="o", color="royalblue",
                label="observed win frequency", capsize=3)
    a1.plot(x, buckets["fair_prob"], "s--", color="darkorange", label="market fair probability")
    a1.set_xticks(x, buckets["bucket"]); a1.set_xlabel("Average odds bucket"); a1.set_ylabel("Probability")
    a1.set_title("Observed vs implied probability"); a1.legend(); a1.grid(True, ls=":", alpha=0.5)
    for k, (pname, col) in enumerate(zip(prices, ["gray", "royalblue", "crimson"])):
        r = buckets[f"roi_{pname}"] * 100
        err = np.vstack([r - buckets[f"roi_{pname}_lo"] * 100, buckets[f"roi_{pname}_hi"] * 100 - r])
        a2.errorbar(x + 0.12 * (k - 1), r, yerr=err, fmt="o-", color=col, capsize=3, label=f"price: {pname}")
    a2.axhline(0, color="black", ls=":")
    a2.set_xticks(x, buckets["bucket"]); a2.set_xlabel("Average odds bucket"); a2.set_ylabel("Blind ROI (%)")
    a2.set_title("ROI of betting every outcome in the bucket (95% CI)"); a2.legend(); a2.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/market_bias.png", dpi=200)
    print(f"\nSaved tables and {RESULTS_DIR}/market_bias.png")


if __name__ == "__main__":
    main()
