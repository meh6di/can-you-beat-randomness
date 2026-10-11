# Can You Beat Randomness?

> *"You can't watch a match without a betting ad popping up, so I built a model to find out if anyone actually beats market randomness."*

An empirical investigation of football betting-market efficiency: evaluating walk-forward predictive models against Pinnacle closing lines across the top five European leagues, and determining whether observed out-of-sample returns reflect structural market inefficiencies or sample variance under an efficient-market null hypothesis.

> **Disclaimer:** Solely for academic, educational, and research purposes. Not financial or gambling advice.

---

### Core Conclusion: Can You Beat Randomness?

**No.** Across seven seasons and five leagues evaluated under strict walk-forward out-of-sample discipline:

1. **Information Supremacy of the Market**: De-vigged closing lines achieve strictly lower cross-entropy (log loss), multiclass Brier scores, and Ranked Probability Scores (RPS) than all models ($\Delta \text{LogLoss} > 0$, 95% CI bounded strictly away from zero). In the market-strength blend ($p \propto p_{\text{mkt}}^{1-w} p_{\text{str}}^w$), the optimized weight collapses to $w < 0.05$, confirming that historical Elo ratings and form metrics provide zero marginal predictive information once closing prices are set.
2. **Failure to Reject the Efficient-Market Null**: Observed holdout returns across all strategy thresholds remain fully consistent with $20{,}000$ counterfactual draws from the de-vigged market null distribution ($p \in [0.15, 0.78]$); no strategy survives multiple-testing correction.
3. **Absence of Closing-Line Value (CLV)**: The 95% bootstrap confidence interval for pick skill—isolating directional selection from the underlying overround—spans zero ($[-0.75\%, +0.55\%]$). Strategies beat closing odds on $49.8\%$ of wagers, indistinguishable from a fair coin toss.
4. **Structural Statistical Power Deficit**: With holdout sample sizes of $n \approx 300\text{--}500$ bets and a per-bet return standard deviation of $\sigma \approx 1.6$, the minimum true ROI detectable with 80% power at $\alpha = 0.05$ is **$12\%\text{--}15\%$**. Detecting a plausible $2\%$ edge requires $n > 7{,}500$ independent bets; single-season track records are mathematically underpowered to distinguish skill from luck.
5. **Winner's Curse**: Parameter selection maximizing mean CLV on the 2024–25 development season completely regresses on the 2025–26 holdout season (rank correlation $\rho = -0.05$), demonstrating that threshold optimization captures sample-specific noise rather than persistent market mispricings.
6. **Bankroll Ruin**: Model probability miscalibration in distributional tails causes Kelly and fractional Kelly staking to catastrophically over-stake, accelerating the probability of ruin past $30\%$ for standard bankroll sizes.

---

## What This Project Does

1. **Data** — Match results and multi-bookmaker odds from `football-data.co.uk` across five leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1) from 2019–20 through 2025–26.
2. **Features (Leakage-Controlled)** — Inter-season Elo ratings, exponentially weighted moving average (EWMA) form metrics (points, attack/defense goals, shots on target, corners), rest days, and power-method de-vigged market probability log-ratios.
3. **Walk-Forward Models** — Monthly refitting using strictly preceding calendar windows:
   - $L_2$-regularized Logistic Regression (standardized features, league dummies, validation-tuned $C$).
   - Market $\times$ Strength Log-Linear Blend ($p \propto p_{\text{mkt}}^{1-w} p_{\text{strength}}^w$, $w \in [0, 1]$ tuned on validation).
   - XGBoost (raw and recalibrated via multiclass vector scaling on a validation window).
4. **Statistical Inference & Decision Framework**
   - Percentile bootstrap confidence intervals and an *efficient-market null* ($20{,}000$ Monte Carlo simulations per strategy) for every ROI (`stats_tests.py`).
   - Log loss, multiclass Brier score, and Ranked Probability Score (RPS) evaluated against market consensus, with Expected Calibration Error (ECE) tested against empirical sampling-noise floors (`prob_eval.py`).
   - Closing-Line Value ($\mathrm{CLV} = o_{\text{taken}} p^{\text{Pinnacle close}} - 1$) and match-level pick skill, isolating directional selection variance from match outcome variance (`clv_analysis.py`).
   - Strategy and threshold selection on 2024–25 (development), evaluated once on untouched 2025–26 (holdout) (`strategy_select.py`).
   - Econometric analysis of favourite–longshot bias across odds buckets and log-odds regressions (`market_bias.py`).
   - Monte Carlo capital risk analysis: drawdown distributions, ruin probabilities, and comparison of flat, proportional, and Kelly staking under miscalibrated probabilities (`risk_analysis.py`).

All numbers live in `results/` (CSV + PNG); `results/summary.md` and `results/tex/` are generated directly by `make_report_tables.py`, eliminating manual reporting errors.

---

## Statistical Evaluation

### Probabilistic Quality (Holdout Season 2025–26)

| Model / Benchmark | Log Loss | Brier Score | RPS | Paired $\Delta \text{LogLoss}$ vs Market (95% CI) | ECE (Equal-Mass) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Market (Pinnacle De-vigged)** | **0.968** | **0.574** | **0.188** | *Reference Benchmark* | **0.016** (Noise Floor: 0.015) |
| **Blend ($w < 0.05$)** | 0.970 | 0.575 | 0.189 | $+0.002$ $[+0.001, +0.004]$ | 0.018 |
| **Logistic Regression** | 0.974 | 0.579 | 0.191 | $+0.006$ $[+0.003, +0.009]$ | 0.021 |
| **Calibrated XGBoost** | 0.979 | 0.583 | 0.193 | $+0.011$ $[+0.007, +0.016]$ | 0.024 |
| **Raw XGBoost** | 0.998 | 0.592 | 0.198 | $+0.030$ $[+0.022, +0.039]$ | 0.048 |

---

## Key Visualizations

- **Realised PnL vs Cumulative CLV** (`results/clv_curves.png`): Demonstrates the $\sim 18\times$ variance reduction of CLV over realized return, showing how cumulative binary outcomes drift randomly away from price-efficiency metrics.
- **Observed ROI vs Efficient-Market Null** (`results/market_null.png`): Situates holdout returns inside the empirical null distribution generated by sampling outcomes from de-vigged market probabilities.
- **Favourite–Longshot Bias** (`results/market_bias.png`): Displays empirical return as a function of log-odds, showing structural underperformance on longshots ($o > 6.00$, $\text{ROI} < -18\%$) versus favourites ($o < 1.40$, $\text{ROI} \approx -2.5\%$).

---

## Quickstart & Pipeline Execution

```sh
# Setup environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run complete pipeline sequentially
python run_all.py

# Run test suite
python -m pytest -q

```

Execution flags:

* `python run_all.py --skip-fetch` — Reuses existing cached data in `data_raw/`.
* `python run_all.py --from walkforward` — Re-runs starting at model training.
* `python run_all.py --only market_bias` — Executes a single target module.

---

## Project Layout

```
config.py              Configuration paths, season splits, league parameters, hyperparameters
fetch_data.py          Download validation with retry logic from football-data.co.uk
clean_data.py          Data integrity, deduplication, schema validation, odds boundary checks
engineer_features.py   Inter-season Elo, EWMA form metrics, power-method market de-vigging
models.py              Estimators (Regularized Logistic Regression, Vector-Scaled XGBoost, Blend)
walkforward.py         Monthly expanding-window training and out-of-fold inference
common.py              De-vigging algorithms, betting selection logic, bootstrap engines, CLV routines
stats_tests.py         Bootstrap ROI intervals, efficient-market null simulation, power calculations
prob_eval.py           Log loss, multiclass Brier score, RPS, paired tests, ECE noise floors
clv_analysis.py        Pinnacle closing-line value, pick-skill derivation, variance ratio tests
strategy_select.py     Parameter grid search on dev (2024–25), single-shot holdout evaluation (2025–26)
market_bias.py         Favourite–longshot bias quantification and log-odds regressions
risk_analysis.py       Monte Carlo drawdown simulations, ruin probabilities, staking comparisons
make_report_tables.py  Generates results/summary.md and LaTeX tables in results/tex/
run_all.py             Pipeline orchestrator
tests/                 Unit test suite for leakage verification and mathematical invariants
report.tex             Full academic report source
docs/                  Pre-compiled academic report PDF

```

---

## Methodological Protocols

* **Chronological Isolation**: For each test month $M$, training uses matches strictly preceding $M - 365\text{ days}$. The validation window covers the 365 days immediately preceding $M$ (used to tune $C$, blend weight $w$, and calibration maps). Month $M$ is evaluated strictly out-of-fold.
* **Leakage Prevention**: Features for any match are computed using match data concluding strictly prior to that match date. An automated test perturbs past match outcomes and verifies that no subsequent or current features are altered.
* **Power De-vigging**: Bookmaker margin is removed via the power formulation:

$$p_k = (1/o_k)^c \quad \text{such that} \quad \sum_{k \in \{H,D,A\}} p_k = 1$$



where $c \ge 1$, correcting for longshot over-taxing.
* **Selection Rule & Holdout Quarantine**: Strategies are chosen on the 2024–25 development season by maximizing mean CLV (subject to $n \ge 150$) and tested exactly once on the 2025–26 holdout season.
* **Formal Power Derivation**: Smallest detectable edge at $80\%$ power and $\alpha = 0.05$ (one-sided):

$$\text{MDE} = (z_{0.95} + z_{0.80}) \frac{\sigma}{\sqrt{n}}$$



---

## Academic Report

The complete academic paper with derivations, methodology, and LaTeX-compiled tables is available at:
👉 **[`docs/report_randomness.pdf`](https://github.com/meh6di/can-you-beat-randomness/blob/main/docs/report_randomness.pdf)**

---

## License & Contact

Distributed under the MIT License (see `LICENSE`).

**Mohamed El Mehdi El Allaoui** — École Centrale Casablanca

GitHub: [@meh6di](https://www.google.com/search?q=https://github.com/meh6di)
