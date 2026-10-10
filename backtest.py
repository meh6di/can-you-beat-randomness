import os
import random
import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURES_DIR = "data_features"

# Value threshold buffer (e.g. 0.03 = bet only if model has a 3% edge over the market)
EDGE_BUFFER = 0.03

# Features fed into the model
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

def run_backtest_for_league(league_name):
    train_df, test_df = load_league_data(league_name)
    if train_df is None or test_df is None:
        print(f"Skipping {league_name}: Missing train or test CSV files in {FEATURES_DIR}/")
        return

    # Prepare features and target
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["FTR"]  # 'H', 'D', 'A'

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["FTR"]

    # Fit Logistic Regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Class order is typically ['A', 'D', 'H'] alphabetically
    classes = list(model.classes_)
    probs = model.predict_proba(X_test)

    # Simulation trackers
    results = {
        "random": {"bets": 0, "profit": 0.0},
        "favorite": {"bets": 0, "profit": 0.0},
        "model": {"bets": 0, "profit": 0.0}
    }

    random.seed(42)

    for idx, row in test_df.iterrows():
        actual = row["FTR"]
        odds = {"H": row["AvgH"], "D": row["AvgD"], "A": row["AvgA"]}
        fair_probs = {"H": row["market_prob_h"], "D": row["market_prob_d"], "A": row["market_prob_a"]}

        # Model predicted probabilities
        model_probs = {classes[i]: probs[idx][i] for i in range(len(classes))}

        # 1. Random Agent
        rand_pick = random.choice(["H", "D", "A"])
        results["random"]["bets"] += 1
        results["random"]["profit"] += (odds[rand_pick] - 1.0) if rand_pick == actual else -1.0

        # 2. Favorite Agent (lowest odds)
        fav_pick = min(odds, key=odds.get)
        results["favorite"]["bets"] += 1
        results["favorite"]["profit"] += (odds[fav_pick] - 1.0) if fav_pick == actual else -1.0

        # 3. Model Agent (Value betting)
        for outcome in ["H", "D", "A"]:
            if model_probs[outcome] > fair_probs[outcome] + EDGE_BUFFER:
                results["model"]["bets"] += 1
                results["model"]["profit"] += (odds[outcome] - 1.0) if outcome == actual else -1.0

    print(f"\n================ League: {league_name.upper()} (2023-2024 Test) ================")
    for agent, stats in results.items():
        bets = stats["bets"]
        profit = stats["profit"]
        roi = (profit / bets * 100) if bets > 0 else 0.0
        print(f"[{agent.upper():<8}] Total Bets: {bets:<4} | Profit: {profit:>7.2f} units | ROI: {roi:>6.2f}%")

def main():
    for league in LEAGUES:
        run_backtest_for_league(league)

if __name__ == "__main__":
    main()