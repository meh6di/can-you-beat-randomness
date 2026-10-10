import os
import glob
import pandas as pd

CLEAN_DIR = "data_clean"
FEATURES_DIR = "data_features"

def build_rich_features(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 1. Market implied fair probabilities (removing bookmaker overround)
    raw_h = 1.0 / df["AvgH"]
    raw_d = 1.0 / df["AvgD"]
    raw_a = 1.0 / df["AvgA"]
    overround = raw_h + raw_d + raw_a
    df["market_prob_h"] = raw_h / overround
    df["market_prob_d"] = raw_d / overround
    df["market_prob_a"] = raw_a / overround

    # 2. Reshape match data to team-match perspective
    h = df[["Date", "HomeTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR", "FTR"]].copy()
    h.columns = ["Date", "Team", "GF", "GA", "Shots_For", "Shots_Ag", "SoT_For", "SoT_Ag", "Corners_For", "Corners_Ag", "Fouls_For", "Fouls_Ag", "YC", "YC_Ag", "RC", "RC_Ag", "FTR"]
    h["Points"] = h["FTR"].apply(lambda x: 3 if x == "H" else (1 if x == "D" else 0))
    h["IsHome"] = 1

    a = df[["Date", "AwayTeam", "FTAG", "FTHG", "AS", "HS", "AST", "HST", "AC", "HC", "AF", "HF", "AY", "HY", "AR", "HR", "FTR"]].copy()
    a.columns = ["Date", "Team", "GF", "GA", "Shots_For", "Shots_Ag", "SoT_For", "SoT_Ag", "Corners_For", "Corners_Ag", "Fouls_For", "Fouls_Ag", "YC", "YC_Ag", "RC", "RC_Ag", "FTR"]
    a["Points"] = a["FTR"].apply(lambda x: 3 if x == "A" else (1 if x == "D" else 0))
    a["IsHome"] = 0

    team_matches = pd.concat([h, a]).sort_values(["Team", "Date"]).reset_index(drop=True)

    # 3. Rest Days (fatigue / schedule congestion)
    team_matches["rest_days"] = team_matches.groupby("Team")["Date"].diff().dt.days.fillna(7).clip(upper=14)

    # 4. Rolling 5-Game Averages (shifted by 1 to prevent data leakage)
    metrics = [
        "Points", "GF", "GA", "Shots_For", "Shots_Ag",
        "SoT_For", "SoT_Ag", "Corners_For", "Corners_Ag",
        "Fouls_For", "YC", "RC"
    ]
    for m in metrics:
        team_matches[f"roll_{m.lower()}_5"] = (
            team_matches.groupby("Team")[m]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )

    # 5. Quality ratios
    team_matches["roll_sot_acc_5"] = team_matches["roll_sot_for_5"] / (team_matches["roll_shots_for_5"] + 1e-5)
    team_matches["roll_conv_5"] = team_matches["roll_gf_5"] / (team_matches["roll_sot_for_5"] + 1e-5)

    # 6. Separate back into Home and Away sets
    feat_cols = [c for c in team_matches.columns if c.startswith("roll_")] + ["rest_days"]

    h_feats = team_matches[team_matches["IsHome"] == 1][["Date", "Team"] + feat_cols].copy()
    h_feats.columns = ["Date", "HomeTeam"] + [f"home_{c}" for c in feat_cols]

    a_feats = team_matches[team_matches["IsHome"] == 0][["Date", "Team"] + feat_cols].copy()
    a_feats.columns = ["Date", "AwayTeam"] + [f"away_{c}" for c in feat_cols]

    # Merge features back into match dataframe
    out = df.merge(h_feats, on=["Date", "HomeTeam"], how="left")
    out = out.merge(a_feats, on=["Date", "AwayTeam"], how="left")

    # 7. Differentials (Home advantage / strength gap)
    out["diff_points_5"] = out["home_roll_points_5"] - out["away_roll_points_5"]
    out["diff_sot_5"] = out["home_roll_sot_for_5"] - out["away_roll_sot_for_5"]
    out["diff_corners_5"] = out["home_roll_corners_for_5"] - out["away_roll_corners_for_5"]
    out["diff_rest_days"] = out["home_rest_days"] - out["away_rest_days"]

    return out

def process_all_files():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    files = glob.glob(os.path.join(CLEAN_DIR, "*.csv"))

    if not files:
        print(f"No files found in '{CLEAN_DIR}'. Run clean_data.py first.")
        return

    for file_path in files:
        filename = os.path.basename(file_path)
        df = pd.read_csv(file_path)
        df_rich = build_rich_features(df)

        # Drop matchday 1 where teams have 0 prior matches in the dataset
        df_rich = df_rich.dropna(subset=["home_roll_points_5", "away_roll_points_5"]).reset_index(drop=True)

        out_path = os.path.join(FEATURES_DIR, filename)
        df_rich.to_csv(out_path, index=False)
        print(f"Engineered rich features for {filename} ({len(df_rich)} matches)")

    print(f"\nAll files saved with rich features to '{FEATURES_DIR}'.")

if __name__ == "__main__":
    process_all_files()