# Can You Beat Randomness?

> *"You can't watch a match without a betting ad popping up, so I built a model to find out if anyone actually beats the market."*

An empirical investigation of football betting-market efficiency: walk-forward predictive models evaluated against de-vigged market prices and Pinnacle closing lines across the top five European leagues, and a test of whether observed out-of-sample returns reflect real inefficiencies or just sample variance under an efficient-market null.

> **Disclaimer:** Solely for academic, educational, and research purposes. Not financial or gambling advice.

---

## Core Conclusion: Can You Beat Randomness?

**No, not with statistical support.** Data cover seven seasons (2019–20 to 2025–26) and five leagues. Models are evaluated walk-forward on 2024–25 (development) and 2025–26 (holdout).

1. **The market is the best forecaster.** De-vigged average odds achieve the best log loss (0.961), Brier score (0.571) and RPS (0.192). Only calibrated XGBoost is *significantly* worse (Δ log loss = +0.010, 95% CI [+0.003, +0.017], p = 0.006); logistic regression and raw XGBoost are directionally worse with CIs that include zero. The optimised market–strength blend weight collapses to w ≈ 0: Elo and form features add no information beyond market prices.
2. **Failure to reject the efficient-market null.** Every holdout ROI is consistent with 20,000 outcome redraws from de-vigged market probabilities (p-values from 0.08 to 0.85 across model–threshold combinations). The smallest p-value (raw XGBoost, 3% threshold, ROI −1.0%, p = 0.084) is above even the uncorrected 5% level; the Bonferroni threshold for the 8 variants tested would be about 0.006. ROIs must be compared with the null mean (about −6% to −9%), not with zero.
3. **Small pick skill, but no profit.** The selected logistic strategy (6% edge, 77 holdout bets) shows a statistically significant pick skill of **+2.18%** (95% CI [+1.29%, +3.16%]). Two caveats: (a) simply backing the favourite already earns about +2.0 points at average odds (CLV −4.85% vs −6.88% for a random pick), so most of this skill at average odds is what a favourite lean would give (about +0.65 points at best odds, vs +2.15% for the model); (b) positive pick skill does not make CLV positive. Mean CLV vs Pinnacle's close is −4.12% at average odds (−0.46%, CI [−1.46%, +0.55%] at best odds), and only 32.5% of wagers beat the raw closing odds at average prices. The realised ROI of +7.5% (CI [−25.9%, +43.7%], p = 0.19) is a lucky outcome of 77 bets, not an edge. Raw and calibrated XGBoost show no significant pick skill.
4. **Structural power deficit.** With 139–226 bets at the 6% threshold and a per-bet standard deviation of about 1.3–1.6 units, the smallest true ROI detectable with 80% power at α = 0.05 (one-sided) is **22%–33%** (about 45% for the 77-bet selected strategy; about 13% for the ~850-bet 3% threshold runs, assuming σ ≈ 1.5). Detecting a 2% edge needs roughly **27,700–38,400 bets**, about 30–45 times the bets placed at the 3% threshold in the holdout.
5. **Selection on development data does not carry over cleanly.** The strategy chosen on 2024–25 by maximum mean CLV (logistic regression, 6% edge) saw CLV fall from −2.5% to −4.1% at average odds and from +1.4% to −0.5% at best odds, but stay roughly flat at Pinnacle prices (−1.5% to −1.3%). The development-vs-holdout rank correlation across the parameter grid is ρ = 0.74, so the relative ordering of settings is fairly stable and the evidence for a strong winner's curse is limited. What does not carry over is the level of CLV.
6. **Bankroll risk without edge.** Under the efficient-market null, flat 1-unit staking from a 100-unit bankroll has a median final P&L of −11.3 units and a 73.2% chance of ending negative. Ruin probability is 74.5% for a 10-unit bankroll and 44.8% for 20 units. Quarter-Kelly using model probabilities raises the probability of a >50% drawdown to 14.5% (vs 2.0% for flat staking). A plausible cause is that Kelly stakes scale with estimated edge and the largest estimated edges are mostly noise; this was not tested.
7. **Favourite–longshot bias at average odds.** Pooling 37,374 outcome-level bets, ROI falls by 5.7 percentage points per unit of log-odds (95% CI [−0.103, −0.009], p = 0.022). The loss is concentrated in the 6–10 odds bucket (−17.4% at average odds). The slopes at Pinnacle closing prices (−0.043, p = 0.090) and maximum odds (−0.011, p = 0.684) are smaller but lie inside the average-odds interval and were not tested against each other, so the bias is not shown to be absent at better prices. The 10+ bucket is too sparse to interpret.

---

## What This Project Does

1. **Data**: Match results and multi-bookmaker odds from `football-data.co.uk` for the Premier League, La Liga, Serie A, Bundesliga and Ligue 1, 2019–20 to 2025–26. The 2020–21 season (mostly played without crowds) is used only to build ratings and form, never to train or test.
2. **Features (leakage-controlled)**: Elo ratings carried across seasons, exponentially weighted form (points, goals for/against, shots on target, corners; half-life of 5 matches, shrunk to the league mean for teams with few games), games played, rest days, and power-method de-vigged market log-ratios.
3. **Walk-forward models**: Refit every calendar month of 2024–25 and 2025–26:
   - L2-regularised logistic regression (standardised features, league dummies, validation-tuned C).
   - Market × strength log-linear blend (p ∝ p_mkt^(1−w) · p_strength^w, w tuned on validation).
   - XGBoost, raw and recalibrated with multinomial vector scaling on a validation window.
4. **Betting rules**: One unit per match on the best outcome if its edge (p̂ − p_mkt) or expected value (p̂·o − 1) clears a threshold (3% or 6% in the reported runs), at average, best-available or Pinnacle prices.
5. **Statistical inference**
   - Bootstrap ROI intervals and an efficient-market null (20,000 simulations per strategy) in `stats_tests.py`.
   - Log loss, multiclass Brier, RPS and ECE against its sampling-noise floor in `prob_eval.py`.
   - Closing-line value and pick skill in `clv_analysis.py`.
   - Dev/holdout strategy selection in `strategy_select.py`.
   - Favourite–longshot analysis in `market_bias.py`.
   - Monte Carlo bankroll risk in `risk_analysis.py`.

All numbers live in `results/` (CSV + PNG). `results/summary.md` and `results/tex/` are generated by `make_report_tables.py`.

---

## Probabilistic Quality (lower is better)

Paired differences are per-match versus the market (average odds, de-vigged).

| Model / Benchmark | Log Loss | Brier | RPS | Δ Log Loss vs Market (95% CI) | ECE (noise floor ≈ 0.020) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Market (Avg odds, de-vigged)** | **0.961** | **0.571** | **0.192** | reference | 0.021 |
| Pinnacle (de-vigged) | 0.962 | 0.572 | 0.192 | +0.001 [−0.000, +0.002] | 0.026 |
| Pinnacle close (de-vigged) | 0.963 | 0.573 | 0.193 | +0.002 [−0.002, +0.007] | 0.022 |
| Market × strength blend (w ≈ 0) | 0.961 | 0.571 | 0.192 | +0.000 | 0.021 |
| Logistic regression | 0.964 | 0.573 | 0.193 | +0.004 [−0.003, +0.010] | 0.012 |
| XGBoost (raw) | 0.966 | 0.574 | 0.194 | +0.005 [−0.001, +0.012] | 0.023 |
| XGBoost (calibrated) | 0.970 | 0.577 | 0.195 | **+0.010 [+0.003, +0.017]** | 0.023 |
| Climatology (class frequencies) | 1.070 | 0.647 | 0.228 | +0.109 [+0.085, +0.134] | 0.023 |

All sources, including the market, have ECE within or close to the sampling-noise floor, so ECE does not separate the models.

---

## Holdout Betting Results (1 bet per match)

| Model | Threshold | Bets | ROI | 95% CI | p vs null |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic regression | 3% | 860 | −2.02% | [−11.9%, +8.1%] | 0.146 |
| XGBoost (raw) | 3% | 825 | −1.04% | [−13.0%, +11.4%] | 0.084 |
| XGBoost (calibrated) | 3% | 832 | −7.12% | [−16.4%, +2.2%] | 0.509 |
| Logistic regression | 6% | 139 | +0.32% | [−25.3%, +26.6%] | 0.261 |
| XGBoost (raw) | 6% | 178 | −22.85% | [−45.4%, +0.8%] | 0.854 |
| XGBoost (calibrated) | 6% | 226 | −10.58% | [−27.2%, +7.5%] | 0.679 |

The blend places no bets at either threshold because it stays too close to the market. Under the null, expected ROI is about −6% to −9% (minus the margin), not zero.

---

## Key Visualisations

- **Observed ROI vs efficient-market null** (`results/market_null.png`): holdout ROIs inside the simulated null distribution.
- **Realised profit vs cumulative CLV** (`results/clv_curves.png`): realised P&L wanders randomly while cumulative CLV drifts smoothly. CLV's standard error is about 34× smaller than ROI's for the logistic strategy (16–20× for XGBoost).
- **Favourite–longshot bias** (`results/market_bias.png`): blind ROI by odds bucket. At average odds, favourites (o < 1.5) are near break-even (−0.6%) and the 6–10 bucket loses 17.4%.
- **Equity vs luck-only band** (`results/`): realised equity against the efficient-market null band, ruin probability by bankroll, and staking comparison.

---

## Quickstart

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_all.py                       # full pipeline
python -m pytest -q                     # leakage and metric tests
```

Flags:

* `python run_all.py --skip-fetch`: reuse cached data in `data_raw/`.
* `python run_all.py --from walkforward`: restart at model training.
* `python run_all.py --only market_bias`: run a single module.

---

## Project Layout

```
config.py              Paths, season splits, league parameters, hyperparameters
fetch_data.py          Download with retry logic from football-data.co.uk
clean_data.py          Validation, deduplication, odds sanity checks
engineer_features.py   Elo, EWMA form, power-method de-vigging
models.py              Logistic regression, vector-scaled XGBoost, blend
walkforward.py         Monthly walk-forward training and inference
common.py              De-vigging, bet selection, bootstrap, CLV routines
stats_tests.py         Bootstrap ROI, efficient-market null, power calculations
prob_eval.py           Log loss, Brier, RPS, paired tests, ECE noise floors
clv_analysis.py        Closing-line value, pick skill, variance ratio
strategy_select.py     Grid search on dev (2024–25), single holdout run (2025–26)
market_bias.py         Favourite–longshot bias and log-odds regressions
risk_analysis.py       Monte Carlo drawdown, ruin and staking comparison
make_report_tables.py  Generates results/summary.md and results/tex/
run_all.py             Pipeline orchestrator
tests/                 Leakage and invariant tests
report.tex             Report source
docs/                  Compiled report PDF
```

---

## Methodology

* **Chronological isolation**: For each test month M, training uses matches before M − 365 days; the validation window is the 365 days before M (tunes C, the blend weight w and the calibrator); month M is evaluated out-of-fold. 2024–25 is the development season and 2025–26 the holdout.
* **Leakage prevention**: A match's features use only strictly earlier days. A unit test perturbs one match result and verifies that no feature at or before that match changes.
* **Power de-vigging**: Margin is removed with p_k = (1/o_k)^c such that Σ p_k = 1, with c ≥ 1, which takes proportionally more margin from longshots. Shin's method is a natural alternative not tested here.
* **Selection rule**: Rule, threshold and price source are chosen on the development season by maximum mean CLV (at least 150 bets) and evaluated once on the holdout.
* **Efficient-market null**: Holding the placed bets fixed, outcomes are redrawn 20,000 times from de-vigged market probabilities. The p-value is the share of simulations with ROI at least as high as observed.
* **Pick skill**: CLV = o_taken · p_Pinnacle-close − 1, minus the average CLV of all three outcomes of the same match. This separates choosing the better side from the margin every blind bet pays.
* **Power**: MDE = (z₀.₉₅ + z₀.₈₀) · σ/√n, and the bets needed for a target edge e is n = ((z₀.₉₅ + z₀.₈₀) · σ / e)².

---

## Limitations

* Small holdout bet counts at the 6% threshold (139–226; 77 for the selected strategy) cannot separate a 2–5% edge from luck.
* Pick skill should be read against a favourite-backing benchmark (about +2 points at average odds).
* Favourite–longshot slopes at different price sources were not formally compared.
* Features exclude lineups, injuries and expected goals; rest days ignore European and cup fixtures.
* Best-odds results assume the best quoted price is obtainable; average odds are a pre-closing snapshot, so CLV contains timing noise.
* Rules were selected on one season, so the size of the development grid should be kept in mind.

---

## Academic Report

Full derivations, methodology and tables: **[`docs/report_randomness.pdf`](https://github.com/meh6di/can-you-beat-randomness/blob/main/docs/report_randomness.pdf)**

---

## License & Contact

Distributed under the MIT License (see `LICENSE`).

**Mohamed El Mehdi El Allaoui**, École Centrale Casablanca

GitHub: [@meh6di](https://github.com/meh6di)
