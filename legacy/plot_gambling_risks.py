import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

FEATURES_DIR = "data_features"
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

LEAGUES = ["premier_league", "la_liga", "serie_a", "bundesliga", "ligue_1"]

def get_league_predictions(league_name):
    train_seasons = ["2021-2022", "2022-2023"]
    test_season = "2023-2024"

    train_dfs = [
        pd.read_csv(os.path.join(FEATURES_DIR, f"{league_name}_{s}.csv"))
        for s in train_seasons
        if os.path.exists(os.path.join(FEATURES_DIR, f"{league_name}_{s}.csv"))
    ]
    test_path = os.path.join(FEATURES_DIR, f"{league_name}_{test_season}.csv")

    if not train_dfs or not os.path.exists(test_path):
        return None

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.read_csv(test_path)

    # Train league-specific model
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(train_df[FEATURE_COLUMNS], train_df["FTR"])

    probs = lr.predict_proba(test_df[FEATURE_COLUMNS])
    classes = list(lr.classes_)

    test_df = test_df.copy()
    for i, c in enumerate(classes):
        test_df[f"prob_{c}"] = probs[:, i]

    return test_df

def collect_chronological_bets():
    all_tests = []
    for league in LEAGUES:
        pred_df = get_league_predictions(league)
        if pred_df is not None:
            all_tests.append(pred_df)

    # Combine all 5 leagues and sort by date as a real bettor would experience them
    full_df = pd.concat(all_tests, ignore_index=True)
    full_df["Date"] = pd.to_datetime(full_df["Date"])
    full_df = full_df.sort_values("Date").reset_index(drop=True)

    bets = []
    for _, row in full_df.iterrows():
        actual = row["FTR"]
        odds = {"H": row["AvgH"], "D": row["AvgD"], "A": row["AvgA"]}
        fair = {"H": row["market_prob_h"], "D": row["market_prob_d"], "A": row["market_prob_a"]}

        best_pick = None
        max_edge = 0.0
        for out in ["H", "D", "A"]:
            edge = row[f"prob_{out}"] - fair[out]
            if edge > max_edge:
                max_edge = edge
                best_pick = out

        # Only bet if edge exceeds the buffer
        if best_pick and max_edge >= EDGE_BUFFER:
            pnl = (odds[best_pick] - 1.0) if best_pick == actual else -1.0
            bets.append({
                "Date": row["Date"],
                "Pick": best_pick,
                "Actual": actual,
                "Odds": odds,
                "Model_PnL": pnl
            })

    return pd.DataFrame(bets)

def main():
    bets_df = collect_chronological_bets()
    n_bets = len(bets_df)
    print(f"Total model bets across Europe: {n_bets}")

    # 1. Calculate Model Path
    model_cum = np.cumsum(bets_df["Model_PnL"].values)
    model_cum = np.insert(model_cum, 0, 0.0)

    # 2. Simulate 100 Random Bettors (The "Chance Cloud")
    random.seed(42)
    mc_runs = 100
    random_paths = np.zeros((mc_runs, n_bets + 1))

    for run in range(mc_runs):
        run_pnl = []
        for _, row in bets_df.iterrows():
            rand_pick = random.choice(["H", "D", "A"])
            pnl = (row["Odds"][rand_pick] - 1.0) if rand_pick == row["Actual"] else -1.0
            run_pnl.append(pnl)
        random_paths[run, 1:] = np.cumsum(run_pnl)

    # 3. Calculate Drawdown (Peak-to-trough decline)
    peak = np.maximum.accumulate(model_cum)
    drawdown = model_cum - peak
    max_drawdown = np.min(drawdown)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Top Panel: Model vs Chance Cloud
    for i in range(mc_runs):
        ax1.plot(random_paths[i], color="lightgray", alpha=0.3, linewidth=1)
    # Average of random bettors (The House Margin gravity line)
    mean_random = np.mean(random_paths, axis=0)
    ax1.plot(mean_random, color="firebrick", linestyle="--", linewidth=2, label="House Edge Baseline (Avg Random Bettor)")
    ax1.plot(model_cum, color="royalblue", linewidth=2.5, label="Logistic Regression Model")
    ax1.axhline(0, color="black", linestyle=":", alpha=0.6)

    ax1.set_title("The Illusion of Skill: Model vs. 100 Random Bettors", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Profit / Loss (Units)", fontsize=11)
    ax1.legend(loc="lower left", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Bottom Panel: Underwater Plot (Drawdown)
    ax2.fill_between(range(len(drawdown)), drawdown, 0, color="crimson", alpha=0.3)
    ax2.plot(drawdown, color="crimson", linewidth=1.8, label="Model Drawdown (Loss from Peak)")
    # Hypothetical bust threshold (e.g. 20-unit bankroll)
    ax2.axhline(-20, color="black", linestyle="--", linewidth=1.5, label="20-Unit Bankroll Bust Threshold")

    ax2.set_title(f"The Risk of Ruin: Underwater Plot (Max Drawdown: {max_drawdown:.1f} Units)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Bets Placed Throughout 2023-2024 Season", fontsize=11)
    ax2.set_ylabel("Deficit from Peak (Units)", fontsize=11)
    ax2.legend(loc="lower left", framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    output_img = "gambling_risk_analysis.png"
    plt.savefig(output_img, dpi=300)
    print(f"Chart saved successfully as '{output_img}'.")

if __name__ == "__main__":
    main()