import os
import glob
import pandas as pd

RAW_DIR = "data_raw"
CLEAN_DIR = "data_clean"

# Core columns required for features, targets, and market benchmarking

KEEP_COLUMNS = [
    "Div", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HS", "AS", "HST", "AST",       # Shots
    "HC", "AC",                     # Corners
    "HF", "AF",                     # Fouls
    "HY", "AY", "HR", "AR",         # Cards
    "AvgH", "AvgD", "AvgA",         # Average Market Odds
    "PSH", "PSD", "PSA"             # Pinnacle Benchmark Odds
]
def clean_datasets():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    raw_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))

    if not raw_files:
        print(f"No CSV files found in '{RAW_DIR}'. Run your fetch script first.")
        return

    for file_path in raw_files:
        filename = os.path.basename(file_path)
        df = pd.read_csv(file_path)

        # Retain only columns that exist in the file
        available_cols = [col for col in KEEP_COLUMNS if col in df.columns]
        df_clean = df[available_cols].copy()

        # Parse date and sort chronologically
        df_clean["Date"] = pd.to_datetime(df_clean["Date"], dayfirst=True)
        df_clean = df_clean.sort_values(by="Date").reset_index(drop=True)

        # Drop rows where match results or basic odds are missing
        df_clean = df_clean.dropna(subset=["FTR", "AvgH", "AvgD", "AvgA"])

        out_path = os.path.join(CLEAN_DIR, filename)
        df_clean.to_csv(out_path, index=False)
        print(f"Cleaned {filename}: {df.shape[1]} cols -> {df_clean.shape[1]} cols ({len(df_clean)} rows)")

    print(f"\nAll files cleaned and saved to '{CLEAN_DIR}'.")

if __name__ == "__main__":
    clean_datasets()