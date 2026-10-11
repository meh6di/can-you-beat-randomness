"""Item 4 - Choose the betting rule on DEV (2024-25), report it once on HOLDOUT (2025-26).

Grid: model x rule {edge, ev} x threshold x price {Avg, Max, PS}.
Selection criterion = mean CLV vs Pinnacle's closing price on DEV (low variance, unlike ROI),
requiring >= MIN_BETS bets. The winner per price source is then evaluated ONCE on the
holdout: ROI + bootstrap CI + market-null p-value + CLV. The whole grid is also scored on
the holdout *for diagnostics only* (rank correlation dev vs holdout = how much of the dev
result was winner's-curse noise); it is never used to choose anything.

Usage: python strategy_select.py
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import (LABELS, MODELS, RESULTS_DIR, SEED, clv_of_bets, get_predictions, market_null,
                    mean_ci, odds_array, fmt_pct, pick_bets)

GRID = {"edge": [0.02, 0.04, 0.06, 0.08, 0.10], "ev": [0.0, 0.03, 0.06, 0.10]}
PRICES = ["Avg", "Max", "PS"]
MIN_BETS = 150
rng = np.random.default_rng(SEED)


def usable(df, price):
    ok = np.isfinite(odds_array(df, "PSC")).all(1) & np.isfinite(odds_array(df, price)).all(1)
    return df[ok].reset_index(drop=True)


def evaluate(df, model, rule, thr, price):
    bets = pick_bets(df, model, rule, thr, price)
    if len(bets) < 2:
        return None, bets
    clv, skill = clv_of_bets(df, bets, price)
    return dict(bets=len(bets), roi=bets["profit"].mean(), clv=clv.mean(), pick_skill=skill.mean()), bets


def main():
    dev_all, hold_all = get_predictions("dev"), get_predictions("holdout")
    grid_rows, chosen = [], {}
    for price in PRICES:
        if f"{price}H" not in dev_all or dev_all[f"{price}H"].isna().all():
            print(f"(price source {price} unavailable - skipped)")
            continue
        dev, hold = usable(dev_all, price), usable(hold_all, price)
        for model in MODELS:
            for rule, thrs in GRID.items():
                for thr in thrs:
                    d, _ = evaluate(dev, model, rule, thr, price)
                    h, _ = evaluate(hold, model, rule, thr, price)
                    row = dict(price=price, model=model, rule=rule, threshold=thr)
                    row.update({f"dev_{k}": v for k, v in (d or {}).items()})
                    row.update({f"hold_{k}": v for k, v in (h or {}).items()})
                    grid_rows.append(row)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(f"{RESULTS_DIR}/select_grid.csv", index=False)
    print(f"Grid size: {len(grid)} configurations scored on DEV ({int(len(grid) / len(PRICES))} per price).")

    out_rows = []
    for price, g in grid.groupby("price"):
        cand = g[g["dev_bets"] >= MIN_BETS].dropna(subset=["dev_clv"])
        if cand.empty:
            print(f"[{price}] no configuration reaches {MIN_BETS} bets on dev")
            continue
        best = cand.loc[cand["dev_clv"].idxmax()]
        hold = usable(hold_all, price)
        bets = pick_bets(hold, best["model"], best["rule"], best["threshold"], price)
        if len(bets) < 2:
            continue
        clv, skill = clv_of_bets(hold, bets, price)
        roi, lo, hi = mean_ci(bets["profit"], 10_000, rng=rng)
        null = market_null(bets, 20_000, rng)
        c_m, c_lo, c_hi = mean_ci(clv, 10_000, rng=rng)
        rho = spearmanr(g["dev_clv"], g["hold_clv"], nan_policy="omit")[0]
        out_rows.append(dict(
            price=price, model=LABELS[best["model"]], rule=best["rule"], threshold=best["threshold"],
            dev_bets=int(best["dev_bets"]), dev_clv=best["dev_clv"],
            hold_bets=len(bets), hold_roi=roi, roi_lo=lo, roi_hi=hi,
            null_mean=null.mean(), p_vs_null=(1 + (null >= roi).sum()) / (len(null) + 1),
            hold_clv=c_m, clv_lo=c_lo, clv_hi=c_hi, hold_pick_skill=skill.mean(),
            rank_corr_dev_vs_hold_clv=rho))
        chosen[price] = dict(model=best["model"], rule=best["rule"], threshold=float(best["threshold"]))
    res = pd.DataFrame(out_rows)
    res.to_csv(f"{RESULTS_DIR}/select_holdout.csv", index=False)
    with open(f"{RESULTS_DIR}/selected_strategy.json", "w") as f:
        json.dump(chosen, f, indent=2)

    disp = res.copy()
    for c in ("dev_clv", "hold_roi", "roi_lo", "roi_hi", "null_mean", "hold_clv", "clv_lo", "clv_hi", "hold_pick_skill"):
        disp[c] = disp[c].map(fmt_pct)
    for c in ("p_vs_null", "rank_corr_dev_vs_hold_clv"):
        disp[c] = disp[c].map(lambda v: f"{v:.3f}")
    print("\n=== Strategy chosen on DEV (by CLV), evaluated once on HOLDOUT ===")
    print(disp.to_string(index=False))
    print("\nrank_corr_dev_vs_hold_clv near 0 means the dev ranking of configurations did not carry over\n"
          "(selection noise). The CI/p-value apply to ONE pre-chosen strategy, so no multiple-testing\n"
          "penalty is needed on the holdout itself; the penalty was paid on dev.")


if __name__ == "__main__":
    main()
