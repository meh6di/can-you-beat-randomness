import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

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

LEAGUES = [
    "premier_league",
    "la_liga",
    "serie_a",
    "bundesliga",
    "ligue_1"
]

LABEL_MAP = {"A": 0, "D": 1, "H": 2}
INV_LABEL_MAP = {0: "A", 1: "D", 2: "H"}

def load_league_data(league_name):
    train_seasons = ["2021-2022", "2022-2023"]
    test_season = "2023-2024"

    train_dfs = [
        pd.read_csv(os.path.join(FEATURES_DIR, f"{league_name}_{s}.csv"))
        for s in train_seasons
        if os.path.exists(os.path.join(FEATURES_DIR, f"{league_name}_{s}.csv"))
    ]
    test_path = os.path.join(FEATURES_DIR, f"{league_name}_{test_season}.csv")

    if not train_dfs or not os.path.exists(test_path):
        return None, None

    return pd.concat(train_dfs, ignore_index=True), pd.read_csv(test_path)

def simulate_strategy(test_df, probs_dict):
    history = []
    breakdown = {
        "H": {"bets": 0, "wins": 0, "profit": 0.0},
        "D": {"bets": 0, "wins": 0, "profit": 0.0},
        "A": {"bets": 0, "wins": 0, "profit": 0.0}
    }

    cumulative_pnl = 0.0
    bankroll_curve = [0.0]

    for idx, row in test_df.iterrows():
        actual = row["FTR"]
        odds = {"H": row["AvgH"], "D": row["AvgD"], "A": row["AvgA"]}
        fair = {"H": row["market_prob_h"], "D": row["market_prob_d"], "A": row["market_prob_a"]}

        best_pick, max_edge = None, 0.0
        for out in ["H", "D", "A"]:
            edge = probs_dict[out][idx] - fair[out]
            if edge > max_edge:
                max_edge = edge
                best_pick = out

        # Place at most 1 bet if edge satisfies buffer
        if best_pick and max_edge >= EDGE_BUFFER:
            won = (best_pick == actual)
            pnl = (odds[best_pick] - 1.0) if won else -1.0

            cumulative_pnl += pnl
            breakdown[best_pick]["bets"] += 1
            if won:
                breakdown[best_pick]["wins"] += 1
            breakdown[best_pick]["profit"] += pnl

        bankroll_curve.append(cumulative_pnl)

    return breakdown, bankroll_curve

def print_breakdown(name, bd):
    total_bets = sum(bd[o]["bets"] for o in ["H", "D", "A"])
    total_pnl = sum(bd[o]["profit"] for o in ["H", "D", "A"])
    total_roi = (total_pnl / total_bets * 100) if total_bets > 0 else 0.0

    print(f"\n--- {name} Breakdown ---")
    for o in ["H", "D", "A"]:
        b = bd[o]["bets"]
        w = bd[o]["wins"]
        p = bd[o]["profit"]
        win_rate = (w / b * 100) if b > 0 else 0.0
        roi = (p / b * 100) if b > 0 else 0.0
        print(f"  Outcome [{o}]: Bets: {b:<4} | WinRate: {win_rate:>5.1f}% | Profit: {p:>7.2f}u | ROI: {roi:>6.2f}%")
    print(f"  TOTAL: Bets: {total_bets:<4} | Net Profit: {total_pnl:>7.2f}u | Total ROI: {total_roi:>6.2f}%")

def main():
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, league in enumerate(LEAGUES):
        train_df, test_df = load_league_data(league)
        if train_df is None or test_df is None:
            print(f"Skipping {league}: missing data.")
            continue

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["FTR"]
        X_test = test_df[FEATURE_COLUMNS]

        # 1. Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        lr_raw = lr.predict_proba(X_test)
        lr_classes = list(lr.classes_)
        lr_probs = {c: lr_raw[:, idx] for idx, c in enumerate(lr_classes)}

        # 2. Calibrated XGBoost (Platt Scaling)
        xgb_base = XGBClassifier(
            n_estimators=75,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss"
        )
        xgb_calibrated = CalibratedClassifierCV(estimator=xgb_base, method="sigmoid", cv=3)
        xgb_calibrated.fit(X_train, y_train.map(LABEL_MAP))
        xgb_raw = xgb_calibrated.predict_proba(X_test)
        xgb_probs = {INV_LABEL_MAP[k]: xgb_raw[:, k] for k in range(3)}

        print(f"\n{'='*25} LEAGUE: {league.upper()} {'='*25}")
        bd_lr, curve_lr = simulate_strategy(test_df, lr_probs)
        bd_xgb, curve_xgb = simulate_strategy(test_df, xgb_probs)

        print_breakdown("Logistic Regression", bd_lr)
        print_breakdown("Calibrated XGBoost", bd_xgb)

        # Plot curves
        ax = axes[i]
        ax.plot(curve_lr, label="LogReg", color="royalblue", linewidth=1.8)
        ax.plot(curve_xgb, label="Calibrated XGBoost", color="darkorange", linewidth=1.8)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
        ax.set_title(league.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Matches Evaluated")
        ax.set_ylabel("Profit / Loss (Units)")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.5)

    # Hide unused 6th subplot
    axes[-1].axis("off")

    plt.tight_layout()
    chart_filename = "bankroll_curves.png"
    plt.savefig(chart_filename, dpi=300)
    print(f"\nChart saved successfully as '{chart_filename}'.")

if __name__ == "__main__":
    main()