"""Item 8 - Risk analysis by simulation (replaces plot_gambling_risks.py / deep_analysis.py).

Takes the bets of one strategy (default: the Avg-price strategy chosen by strategy_select.py,
else logistic regression / 6% edge) in chronological order and asks:

 A. Realised equity curve vs the 5-95% band of an *efficient-market null*
    (outcomes redrawn from de-vigged market probabilities; flat 1-unit stakes).
 B. Distribution of max drawdown and probability of ruin for several bankroll sizes under
    two scenarios: 'observed' (bets resampled from the realised ones: keeps the empirical
    edge) and 'null' (efficient market).
 C. Staking schemes from a 100-unit bankroll: flat 1 unit, 1% of bankroll, quarter-Kelly
    (capped at 3%) -> median/5th/95th final bankroll, P(loss), P(drawdown > 50%),
    P(bankroll < 20, i.e. an 80% drawdown).

Usage: python risk_analysis.py [--model lr --rule edge --thr 0.06 --price Avg --split holdout]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import LABELS, RESULTS_DIR, SEED, get_predictions, pick_bets

N_SIM = 3000
BANKROLLS = [10, 20, 50, 100, 200]
BANK0 = 100.0
rng = np.random.default_rng(SEED)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--rule"); ap.add_argument("--thr", type=float)
    ap.add_argument("--price"); ap.add_argument("--split", default="holdout")
    a = ap.parse_args()
    sel = f"{RESULTS_DIR}/selected_strategy.json"
    if a.model is None and os.path.exists(sel):
        s = json.load(open(sel)).get("Avg")
        if s:
            a.model, a.rule, a.thr, a.price = s["model"], s["rule"], s["threshold"], "Avg"
    a.model = a.model or "lr"; a.rule = a.rule or "edge"
    a.thr = 0.06 if a.thr is None else a.thr; a.price = a.price or "Avg"
    return a


def draw(scenario, odds, fair_p, p_model, won, S):
    n = len(odds)
    if scenario == "observed":              # resample bets (iid): keeps the realised edge
        j = rng.integers(0, n, (S, n))
        return won[j], odds[j], p_model[j]
    j = np.tile(np.arange(n), (S, 1))      # efficient-market null: same bets, outcomes redrawn
    return rng.random((S, n)) < fair_p[j], odds[j], p_model[j]


def run_staking(w, o, pm, scheme):
    S, n = w.shape
    bank = np.full(S, BANK0); peak = bank.copy(); mdd = np.zeros(S); lowest = bank.copy()
    for t in range(n):
        if scheme == "flat 1 unit":
            stake = np.minimum(1.0, bank)
        elif scheme == "1% of bankroll":
            stake = 0.01 * bank
        else:                                # quarter Kelly with the model's probability, cap 3%
            kelly = np.clip((pm[:, t] * o[:, t] - 1) / (o[:, t] - 1), 0, None)
            stake = np.minimum(0.25 * kelly, 0.03) * bank
        bank = bank + stake * np.where(w[:, t], o[:, t] - 1.0, -1.0)
        peak = np.maximum(peak, bank)
        mdd = np.maximum(mdd, (peak - bank) / peak)
        lowest = np.minimum(lowest, bank)
    return bank, mdd, lowest


def main():
    a = parse()
    df = get_predictions(a.split)
    bets = pick_bets(df, a.model, a.rule, a.thr, a.price).sort_values("Date", kind="stable")
    n = len(bets)
    if n < 20:
        raise SystemExit(f"Only {n} bets for this strategy - nothing meaningful to simulate.")
    name = f"{LABELS[a.model]} | {a.rule} {a.thr} | price {a.price} | {a.split}"
    print(f"Strategy: {name}  ({n} bets)")
    odds, fair_p = bets["odds"].to_numpy(), bets["fair_p"].to_numpy()
    p_model, won, profit = bets["p_model"].to_numpy(), bets["won"].to_numpy(), bets["profit"].to_numpy()

    # ------------------------------------------------------- A: equity vs null band
    w, o, _ = draw("null", odds, fair_p, p_model, won, N_SIM)
    null_cum = np.cumsum(np.where(w, o - 1.0, -1.0), axis=1)
    band = np.percentile(null_cum, [5, 50, 95], axis=0)
    actual_cum = np.cumsum(profit)

    # ------------------------------------------- B: drawdown and ruin (flat 1 unit)
    dd_rows, ruin_rows = [], []
    for scen in ("observed", "null"):
        w, o, pm = draw(scen, odds, fair_p, p_model, won, N_SIM)
        cum = np.cumsum(np.where(w, o - 1.0, -1.0), axis=1)
        aug = np.hstack([np.zeros((N_SIM, 1)), cum])
        mdd = (np.maximum.accumulate(aug, axis=1) - aug).max(axis=1)
        dd_rows.append(dict(scenario=scen, final_median=np.median(cum[:, -1]),
                            final_p5=np.percentile(cum[:, -1], 5), final_p95=np.percentile(cum[:, -1], 95),
                            max_dd_median=np.median(mdd), max_dd_p5=np.percentile(mdd, 5),
                            max_dd_p95=np.percentile(mdd, 95), p_final_negative=(cum[:, -1] < 0).mean()))
        low = cum.min(axis=1)
        for B in BANKROLLS:
            ruin_rows.append(dict(scenario=scen, bankroll_units=B, p_ruin=(low <= -B).mean()))
    dd, ruin = pd.DataFrame(dd_rows), pd.DataFrame(ruin_rows)
    real_mdd = (np.maximum.accumulate(np.r_[0, actual_cum]) - np.r_[0, actual_cum]).max()
    dd.to_csv(f"{RESULTS_DIR}/risk_drawdown.csv", index=False)
    ruin.to_csv(f"{RESULTS_DIR}/risk_ruin.csv", index=False)
    print(f"\nRealised: final {actual_cum[-1]:+.1f} units, max drawdown {real_mdd:.1f} units")
    print("\n=== Flat 1-unit staking: outcome and drawdown distributions (units) ===")
    print(dd.round(3).to_string(index=False))
    print("\n=== P(ruin) = P(cumulative loss reaches the whole bankroll), flat 1 unit ===")
    print(ruin.pivot(index="bankroll_units", columns="scenario", values="p_ruin").round(3).to_string())

    # ------------------------------------------------------------ C: staking schemes
    st_rows = []
    for scen in ("observed", "null"):
        w, o, pm = draw(scen, odds, fair_p, p_model, won, N_SIM)
        for scheme in ("flat 1 unit", "1% of bankroll", "quarter Kelly (cap 3%)"):
            bank, mdd, low = run_staking(w, o, pm, scheme)
            st_rows.append(dict(scenario=scen, scheme=scheme, final_median=np.median(bank),
                                final_p5=np.percentile(bank, 5), final_p95=np.percentile(bank, 95),
                                p_loss=(bank < BANK0).mean(), p_dd_over_50pct=(mdd > 0.5).mean(),
                                p_below_20=(low < 0.2 * BANK0).mean()))
    st = pd.DataFrame(st_rows)
    st.to_csv(f"{RESULTS_DIR}/risk_staking.csv", index=False)
    print(f"\n=== Staking schemes from a {BANK0:.0f}-unit bankroll ===")
    print(st.round(3).to_string(index=False))
    print("Kelly uses the model's own probabilities: if they are over-confident, Kelly over-bets.\n")

    # ---------------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    x = np.arange(n + 1)
    ax[0].fill_between(x[1:], band[0], band[2], color="lightgray", label="efficient-market null (5-95%)")
    ax[0].plot(x[1:], band[1], color="firebrick", ls="--", label="null median")
    ax[0].plot(x[1:], actual_cum, color="royalblue", lw=2, label="realised")
    ax[0].axhline(0, color="black", ls=":")
    ax[0].set_title("Realised equity vs luck-only band"); ax[0].set_xlabel("Bets"); ax[0].set_ylabel("Units")
    ax[0].legend(fontsize=8); ax[0].grid(True, ls=":", alpha=0.5)
    for scen, c in (("observed", "royalblue"), ("null", "firebrick")):
        r = ruin[ruin.scenario == scen]
        ax[1].plot(r["bankroll_units"], r["p_ruin"], "o-", color=c, label=scen)
    ax[1].set_xscale("log"); ax[1].set_ylim(0, 1.02)
    ax[1].set_title("P(ruin) vs bankroll (flat 1-unit stakes)"); ax[1].set_xlabel("Bankroll (units)")
    ax[1].set_ylabel("Probability"); ax[1].legend(); ax[1].grid(True, ls=":", alpha=0.5)
    schemes = list(st["scheme"].unique())
    for k, (scen, c) in enumerate((("observed", "royalblue"), ("null", "firebrick"))):
        r = st[st.scenario == scen].set_index("scheme").loc[schemes]
        xs = np.arange(len(schemes)) + 0.18 * (k - 0.5)
        ax[2].errorbar(xs, r["final_median"], yerr=[r["final_median"] - r["final_p5"], r["final_p95"] - r["final_median"]],
                       fmt="o", color=c, capsize=4, label=scen)
    ax[2].axhline(BANK0, color="black", ls=":")
    ax[2].set_xticks(range(len(schemes)), [s.replace(" (", "\n(") for s in schemes], fontsize=8)
    ax[2].set_title("Final bankroll (median, 5-95%)"); ax[2].legend(); ax[2].grid(True, ls=":", alpha=0.5)
    plt.suptitle(name, fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/risk_analysis.png", dpi=200)
    print(f"Saved tables and {RESULTS_DIR}/risk_analysis.png")


if __name__ == "__main__":
    main()
