"""Central configuration: paths, seasons, splits and hyper-parameters.

Chronology (never shuffled):
  train      : every labelled match before the validation window
  validation : the VAL_DAYS before each test month (tunes C, blend weight, calibrators)
  DEV        : 2024-25  -> used to CHOOSE the betting rule/threshold (strategy_select.py)
  HOLDOUT    : 2025-26  -> touched once, to report the chosen strategy
"""
RAW_DIR = "data_raw"
CLEAN_DIR = "data_clean"
FEATURES_DIR = "data_features"
RESULTS_DIR = "results"

LEAGUE_CODES = {          # football-data.co.uk division codes
    "premier_league": "E0", "la_liga": "SP1", "serie_a": "I1",
    "bundesliga": "D1", "ligue_1": "F1",
}
LEAGUES = list(LEAGUE_CODES)

SEASONS = {               # football-data URL code -> label
    "1920": "2019-2020", "2021": "2020-2021", "2122": "2021-2022",
    "2223": "2022-2023", "2324": "2023-2024", "2425": "2024-2025", "2526": "2025-2026",
}
# Used to build Elo / form (so ratings carry across seasons) but never to train or test
# (mostly played without crowds).
FEATURE_ONLY_SEASONS = ["2020-2021"]

DEV_SEASONS = ["2024-2025"]
HOLDOUT_SEASONS = ["2025-2026"]
TEST_SEASONS = DEV_SEASONS + HOLDOUT_SEASONS

SEED = 42
DEVIG_METHOD = "power"    # "power" or "proportional"

ELO = dict(k=20.0, home_adv=60.0, start=1500.0, new_team=1450.0, season_regress=0.25)
EWM_HALFLIFE = 5          # matches
SHRINK_K = 3.0            # form is shrunk to the league mean with weight n/(n+K)

VAL_DAYS = 365            # validation window before every walk-forward fold
MIN_TRAIN_MATCHES = 1500

# ----------------------------------------------------------------- feature lists
FORM_METRICS = ["points", "gf", "ga", "sot_for", "sot_ag", "corners_for", "corners_ag"]
STRENGTH_FEATURES = (
    ["elo_diff", "elo_sum"]
    + [f"home_f_{m}" for m in FORM_METRICS]
    + [f"away_f_{m}" for m in FORM_METRICS]
    + ["home_n_games", "away_n_games", "diff_points", "diff_sot", "diff_corners",
       "diff_goal_diff", "home_rest", "away_rest"]
)
MARKET_FEATURES = ["mkt_lr_ha", "mkt_lr_d"]     # log-ratios (no collinearity)
