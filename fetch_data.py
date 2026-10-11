"""Download football-data.co.uk CSVs (all leagues x seasons in config.py)."""
import os
import time
import urllib.request

import config as C

BASE_URL = "https://www.football-data.co.uk/mmz4281"


def looks_like_csv(path):
    """Cheap sanity check: header must contain the Div/Date/HomeTeam columns."""
    try:
        with open(path, "r", encoding="latin-1") as f:
            head = f.readline()
        return all(c in head for c in ("Div", "Date", "HomeTeam"))
    except OSError:
        return False


def download(url, path, retries=3):
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, path)
            if looks_like_csv(path):
                return True
            print(f"  not a valid CSV (attempt {attempt}): {url}")
        except Exception as e:  # network errors, 404 ...
            print(f"  failed (attempt {attempt}): {url}: {e}")
        time.sleep(1.5 * attempt)
    if os.path.exists(path):
        os.remove(path)     # never leave a corrupt file behind
    return False


def fetch_datasets():
    os.makedirs(C.RAW_DIR, exist_ok=True)
    failed = []
    for league, code in C.LEAGUE_CODES.items():
        for season_code, label in C.SEASONS.items():
            path = os.path.join(C.RAW_DIR, f"{league}_{label}.csv")
            if os.path.exists(path) and looks_like_csv(path):
                print(f"Skipping {league}_{label}.csv (already exists)")
                continue
            print(f"Downloading {league}_{label}.csv ...")
            if not download(f"{BASE_URL}/{season_code}/{code}.csv", path):
                failed.append(f"{league}_{label}")
    print(f"\nDone. Files in '{C.RAW_DIR}/'.")
    if failed:
        print("FAILED:", ", ".join(failed))


if __name__ == "__main__":
    fetch_datasets()
