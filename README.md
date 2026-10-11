# Can You Beat Randomness?

An educational study of football betting-market efficiency: can statistical models beat the
closing odds of the top-5 European leagues, and how much of any apparent edge is just luck?

> **Disclaimer:** for educational and research purposes only. Not betting or financial advice.

## What this project does

1. **Data** - football-data.co.uk results and odds, 5 leagues, 2019-20 to 2025-26.
2. **Features** (leakage-tested) - Elo ratings and exponentially weighted form for both teams,
   carried across seasons; de-vigged market probabilities (power method).
3. **Walk-forward models** - retrained every month with a time-ordered validation window:
   tuned logistic regression, a market x strength-model blend, raw and calibrated XGBoost.
4. **Evidence, not anecdotes**
   - bootstrap CIs and an *efficient-market null* p-value for every ROI (`stats_tests.py`)
   - log loss / Brier / RPS versus the market and calibration with noise floors (`prob_eval.py`)
   - closing-line value against Pinnacle, far less noisy than ROI (`clv_analysis.py`)
   - rule/threshold chosen on 2024-25, reported once on the 2025-26 holdout (`strategy_select.py`)
   - the market itself: favourite-longshot bias by odds bucket (`market_bias.py`)
   - bankroll risk by simulation: drawdowns, ruin, flat vs % vs Kelly staking (`risk_analysis.py`)

All numbers live in `results/` (CSV + PNG); `results/summary.md` and `results/tex/` are generated,
so the README and report never contain hand-copied figures. **Run the pipeline to see results.**

## Quick start

```sh
python -m venv .venv && source .venv/bin/activate     # Python >= 3.11
pip install -r requirements.txt
python run_all.py            # fetch -> clean -> features -> walk-forward -> analyses -> tables
python -m pytest -q          # leakage, de-vigging, scoring and fold-chronology tests
```

Useful variants: `python run_all.py --skip-fetch`, `--from walkforward`, `--only market_bias`.
Analyses accept `--split holdout|dev|all` (default `holdout`).

## Project layout

```
config.py              paths, seasons, dev/holdout split, hyper-parameters
fetch_data.py          download (validated, retried)      clean_data.py     validate + keep columns
engineer_features.py   Elo, EWMA form, market features    models.py         model zoo per fold
walkforward.py         monthly expanding-window predictions
common.py              devig, bet selection, CLV, bootstrap helpers
stats_tests.py  prob_eval.py  clv_analysis.py  strategy_select.py  market_bias.py  risk_analysis.py
make_report_tables.py  results/*.csv -> results/summary.md + results/tex/*.tex
run_all.py             one entry point              tests/    pytest suite
report.tex             the write-up (inputs results/tex/*)   legacy/   original 2021-24 scripts
```

## Method notes

- **Chronology everywhere.** For each test month: train = matches before (month - 365 days),
  validation = the 365 days before the month (tunes C, blend weight, calibration), test = the month.
  Seasons 2024-25 are *dev* (choose the rule), 2025-26 is the *holdout*. 2020-21 feeds ratings only.
- **No look-ahead.** Features at a match use only earlier days; `tests/` perturbs a result and
  checks that no feature up to that match changes.
- **Market de-vigging.** Power method by default (partly corrects favourite-longshot bias).
- **Betting rules.** `edge` (p_model - p_market) or `ev` (p_model x odds - 1), at average, best
  (`Max`) or Pinnacle prices. Selection criterion on dev: mean CLV vs Pinnacle close.
- **Known limits.** Domestic-only rest days; no lineups/injuries/xG (not in football-data); one
  static team-strength family (Elo); best-odds results assume you can actually get the best price;
  `Avg*` is a pre-closing snapshot, so CLV has some timing noise.

## License and contact

MIT (see `LICENSE`). Mohamed El Mehdi El Allaoui - Ecole Centrale Casablanca -
<https://github.com/meh6di/can-you-beat-randomness>
