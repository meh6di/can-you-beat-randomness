"""Leakage-free features, built per league ACROSS seasons (state carries over).

  * Elo ratings (home advantage, goal-difference multiplier, regression to the mean each
    season; newly promoted teams start lower)
  * Exponentially weighted form for BOTH teams: points, attack (goals for), defence
    (goals against), shots on target for/against, corners for/against. Shrunk towards the
    league mean while a team has few games (n_games), so promoted teams don't produce noise.
  * Rest days, differentials, de-vigged market probabilities and market log-ratios.
Everything at row t only uses matches strictly before row t's date.
Output: data_features/<league>.csv  (all seasons, with a `Season` column)
"""
import glob
import os

import numpy as np
import pandas as pd

import config as C
from common import devig


def add_market_features(df):
    odds = df[["AvgH", "AvgD", "AvgA"]].to_numpy(float)
    p = devig(odds)
    df["market_prob_h"], df["market_prob_d"], df["market_prob_a"] = p.T
    df["overround"] = (1.0 / odds).sum(axis=1)
    df["mkt_lr_ha"] = np.log(p[:, 0] / p[:, 2])
    df["mkt_lr_d"] = np.log(p[:, 1] / (p[:, 0] + p[:, 2]))
    return df


def compute_elo(df, cfg=None):
    """Pre-match Elo ratings (arrays) for matches sorted by date."""
    cfg = cfg or C.ELO
    rating, cur, first = {}, None, df["Season"].iloc[0]
    eh, ea = np.empty(len(df)), np.empty(len(df))
    goals_h = np.nan_to_num(df["FTHG"].to_numpy(float))
    goals_a = np.nan_to_num(df["FTAG"].to_numpy(float))
    for i, (season, h, a, res) in enumerate(zip(df["Season"], df["HomeTeam"], df["AwayTeam"], df["FTR"])):
        if season != cur:
            if cur is not None:      # regress every rating towards the mean between seasons
                for t in rating:
                    rating[t] = (1 - cfg["season_regress"]) * rating[t] + cfg["season_regress"] * cfg["start"]
            cur = season
        for t in (h, a):
            if t not in rating:
                rating[t] = cfg["start"] if season == first else cfg["new_team"]
        rh, ra = rating[h], rating[a]
        eh[i], ea[i] = rh, ra                                   # recorded BEFORE the update
        exp_h = 1.0 / (1.0 + 10 ** (-(rh + cfg["home_adv"] - ra) / 400.0))
        s = {"H": 1.0, "D": 0.5, "A": 0.0}[res]
        delta = cfg["k"] * (1.0 + np.log1p(abs(goals_h[i] - goals_a[i]))) * (s - exp_h)
        rating[h] += delta
        rating[a] -= delta
    return eh, ea


def compute_form(df):
    """EWMA form + n_games + rest days, per team, carried across seasons."""
    d = df.reset_index(drop=True).copy()
    d["mid"] = np.arange(len(d))
    pts = {"H": {"H": 3, "D": 1, "A": 0}, "A": {"H": 0, "D": 1, "A": 3}}

    def side(team, ishome):
        s = "H" if ishome else "A"
        f, g = ("FTHG", "FTAG") if ishome else ("FTAG", "FTHG")
        sf, sa = ("HST", "AST") if ishome else ("AST", "HST")
        cf, ca = ("HC", "AC") if ishome else ("AC", "HC")
        return pd.DataFrame({
            "mid": d["mid"], "Date": d["Date"], "Team": d[team], "IsHome": int(ishome),
            "points": d["FTR"].map(pts[s]), "gf": d[f], "ga": d[g],
            "sot_for": d.get(sf, np.nan), "sot_ag": d.get(sa, np.nan),
            "corners_for": d.get(cf, np.nan), "corners_ag": d.get(ca, np.nan),
        })

    long = pd.concat([side("HomeTeam", True), side("AwayTeam", False)])
    long = long.sort_values(["Team", "Date", "mid"], kind="stable").reset_index(drop=True)
    g = long.groupby("Team", sort=False)
    long["n_games"] = g.cumcount()                                  # prior matches in dataset
    long["rest"] = g["Date"].diff().dt.days.fillna(7).clip(2, 14)
    w = long["n_games"] / (long["n_games"] + C.SHRINK_K)
    for m in C.FORM_METRICS:
        # league-wide prior = mean over strictly EARLIER DAYS (no same-day leakage)
        daily = long.groupby("Date")[m].agg(["sum", "count"]).sort_index()
        prior_day = daily["sum"].cumsum().shift(1) / daily["count"].cumsum().shift(1)
        prior = long["Date"].map(prior_day)
        ewm = g[m].transform(lambda s: s.shift(1).ewm(halflife=C.EWM_HALFLIFE, min_periods=1).mean())
        long[f"f_{m}"] = (w * ewm.fillna(prior) + (1 - w) * prior).fillna(0.0)
    H = long[long["IsHome"] == 1].set_index("mid")
    A = long[long["IsHome"] == 0].set_index("mid")
    out = d
    for m in C.FORM_METRICS:
        out[f"home_f_{m}"] = out["mid"].map(H[f"f_{m}"])
        out[f"away_f_{m}"] = out["mid"].map(A[f"f_{m}"])
    for col, src in (("n_games", "n_games"), ("rest", "rest")):
        out[f"home_{col}"] = out["mid"].map(H[src])
        out[f"away_{col}"] = out["mid"].map(A[src])
    return out.drop(columns="mid")


def build_league_features(df):
    """df: one league, all seasons, with `Season`. Returns df + features (sorted by date)."""
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    df = add_market_features(df)
    df["elo_home"], df["elo_away"] = compute_elo(df)
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["elo_sum"] = df["elo_home"] + df["elo_away"] - 2 * C.ELO["start"]
    df = compute_form(df)
    df["diff_points"] = df["home_f_points"] - df["away_f_points"]
    df["diff_sot"] = df["home_f_sot_for"] - df["away_f_sot_for"]
    df["diff_corners"] = df["home_f_corners_for"] - df["away_f_corners_for"]
    df["diff_goal_diff"] = (df["home_f_gf"] - df["home_f_ga"]) - (df["away_f_gf"] - df["away_f_ga"])
    return df


def process_all():
    os.makedirs(C.FEATURES_DIR, exist_ok=True)
    for league in C.LEAGUES:
        frames = []
        for label in C.SEASONS.values():
            p = os.path.join(C.CLEAN_DIR, f"{league}_{label}.csv")
            if os.path.exists(p):
                f = pd.read_csv(p, parse_dates=["Date"])
                f["Season"] = label
                frames.append(f)
        if not frames:
            print(f"Skipping {league}: no cleaned files")
            continue
        out = build_league_features(pd.concat(frames, ignore_index=True))
        out["league"] = league
        out.to_csv(os.path.join(C.FEATURES_DIR, f"{league}.csv"), index=False)
        print(f"{league}: {len(out)} matches, {out['Season'].nunique()} seasons")
    print(f"\nFeatures saved to '{C.FEATURES_DIR}/'.")


if __name__ == "__main__":
    process_all()
