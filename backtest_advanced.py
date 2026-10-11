import os
import random
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

FEATURES_DIR = "data_features"

# Require at least a 6% probability edge over fair market odds
EDGE_BUFFER = 0.06

FEATURE_COLUMNS = [
    "home_roll_points_5", "away_roll_points_5",
    "home_roll_gf_5", "away_roll_ga_5",
    "home_roll_sot_for_5", "away_roll_sot_ag_5",
    "home_roll_corners_for_5", "away_roll_corners_ag_5",
    "home_roll_sot_acc_5", "away_roll_sot_acc_5",
    "home_roll_conv_5", "away_roll_conv_5",
    "diff_points_5", "diff_sot_5", "diff_corners_5", "diff_rest_days",
    "market_prob_h", "market_prob_d", "market_prob_a"
]

LEAGUES = [
    "premier_league",
    "la_liga",
    "serie_a",
    "bundesliga",
    "ligue_1"
]

# Map string labels to numeric for XGBoost
LABEL_MAP = {"A": 0, "D": 1, "H": 2}
INV_LABEL_MAP = {0: "A", 1: "D", 2: "H"}

def load_league_data(league_name):
    train_seasons = ["2021-2022", "2022-2023"]
    test_season = "2023-2024"

    train_dfs = []
    for s in train_seasons:
        path = os.path.join(FEATURES_DIR, f"{league_name}_{s}.csv")
        if os.path.exists(path):
            train_dfs.append(pd.read_csv(path))

    test_path = os.path.join(FEATURES_DIR, f"{league_name}_{test_season}.csv")
    if not train_dfs or not os.path.exists(test_path):
        return None, None

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def evaluate_predictions(test_df, probs_dict, model_name):
    bets = 0
    profit = 0.0

    for idx, row in test_df.iterrows():
        actual = row["FTR"]
        odds = {"H": row["AvgH"], "D": row["AvgD"], "A": row["AvgA"]}
        fair_probs = {"H": row["market_prob_h"], "D": row["market_prob_d"], "A": row["market_prob_a"]}

        # Find the outcome with the single highest positive edge
        best_outcome = None
        max_edge = 0.0

        for outcome in ["H", "D", "A"]:
            edge = probs_dict[outcome][idx] - fair_probs[outcome]
            if edge > max_edge:
                max_edge = edge
                best_outcome = outcome

        # Place at most 1 bet if edge clears the buffer
        if best_outcome and max_edge >= EDGE_BUFFER:
            bets += 1
            profit += (odds[best_outcome] - 1.0) if best_outcome == actual else -1.0

    roi = (profit / bets * 100) if bets > 0 else 0.0
    print(f"[{model_name:<10}] Bets: {bets:<4} | Profit: {profit:>7.2f} units | ROI: {roi:>6.2f}%")

def run_league(league_name):
    train_df, test_df = load_league_data(league_name)
    if train_df is None or test_df is None:
        print(f"Skipping {league_name}: missing files.")
        return

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["FTR"]
    X_test = test_df[FEATURE_COLUMNS]

    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_probs_raw = lr.predict_proba(X_test)
    lr_classes = list(lr.classes_)
    lr_probs = {c: lr_probs_raw[:, i] for i, c in enumerate(lr_classes)}

    # 2. XGBoost
    y_train_num = y_train.map(LABEL_MAP)
    xgb = XGBClassifier(
        n_estimators=75,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss"
    )
    xgb.fit(X_train, y_train_num)
    xgb_probs_raw = xgb.predict_proba(X_test)
    xgb_probs = {INV_LABEL_MAP[i]: xgb_probs_raw[:, i] for i in range(3)}

    print(f"\n================ {league_name.upper()} (Threshold: {EDGE_BUFFER*100:.0f}%, Max 1 Bet/Match) ================")
    evaluate_predictions(test_df, lr_probs, "LOG_REG")
    evaluate_predictions(test_df, xgb_probs, "XGBOOST")

def main():
    for league in LEAGUES:
        run_league(league)

if __name__ == "__main__":
    main()