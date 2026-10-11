import os
import urllib.request

# Define top 5 leagues and their Football-Data codes
LEAGUES = {
    "premier_league": "E0",
    "la_liga": "SP1",
    "serie_a": "I1",
    "bundesliga": "D1",
    "ligue_1": "F1"
}

# Football-Data URL codes mapped to readable names
SEASONS = {
    "1920": "2019-2020",   # extra training data
    # "2021": "2020-2021",   # COVID season (mostly no crowds): left out on purpose
    "2122": "2021-2022",
    "2223": "2022-2023",
    "2324": "2023-2024",
    "2425": "2024-2025",   # fresh holdout
    "2526": "2025-2026"    # fresh holdout
}

OUTPUT_DIR = "data_raw"

def fetch_datasets():
    # Create the folder if it doesn't already exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_url = "https://www.football-data.co.uk/mmz4281"

    for league_name, league_code in LEAGUES.items():
        for season_code, season_label in SEASONS.items():
            url = f"{base_url}/{season_code}/{league_code}.csv"
            filename = f"{league_name}_{season_label}.csv"
            filepath = os.path.join(OUTPUT_DIR, filename)

            # Skip download if the file is already there
            if os.path.exists(filepath):
                print(f"Skipping {filename} (already exists)")
                continue

            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
            except Exception as e:
                print(f"Failed to fetch {url}: {e}")

    print("\nDownload complete. Files are ready in the 'data_raw' folder.")

if __name__ == "__main__":
    fetch_datasets()