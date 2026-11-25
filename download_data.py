
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Start with smaller downloads to test reliability.
ROWS_PER_MONTH = 1000
MONTHS = range(1, 11)

COLUMNS = (
    "unique_key,created_date,closed_date,"
    "agency,agency_name,complaint_type,"
    "descriptor,borough,incident_zip,"
    "status,latitude,longitude"
)

month_files = []

for month in MONTHS:

    file_path = DATA_DIR / f"nyc_311_2025_{month:02d}.csv"
    month_files.append(file_path)

    # Don't download a completed month twice.
    if file_path.exists():
        print(f"Month {month}: Already downloaded. Skipping.")
        continue

    start = f"2025-{month:02d}-01T00:00:00"
    end = f"2025-{month + 1:02d}-01T00:00:00"

    params = {
        "$select": COLUMNS,
        "$where": (
            f"created_date >= '{start}' "
            f"AND created_date < '{end}'"
        ),
        "$limit": ROWS_PER_MONTH
    }

    for attempt in range(1, 5):

        try:
            print(f"Downloading month {month}, attempt {attempt}...")

            response = requests.get(
                URL,
                params=params,
                timeout=(15, 45)
            )

            response.raise_for_status()

            data = response.json()
            df = pd.DataFrame(data)

            if df.empty:
                raise ValueError("No records returned.")

            df.to_csv(file_path, index=False)

            print(f"Month {month}: Saved {len(df)} records")
            break

        except (requests.RequestException, ValueError) as error:

            print(f"Request failed: {error}")

            if attempt == 4:
                raise RuntimeError(
                    f"Month {month} failed after 4 attempts. "
                    "Previously saved months are safe."
                ) from error

            time.sleep(5 * attempt)

# Combine the downloaded monthly files.
all_data = [
    pd.read_csv(file, dtype=str)
    for file in month_files
]

final_df = pd.concat(all_data, ignore_index=True)

final_df.drop_duplicates(
    subset=["unique_key"], inplace=True
)

output = DATA_DIR / "nyc_311_2025_sample.csv"
final_df.to_csv(output, index=False)

print("\nDOWNLOAD COMPLETE!")
print("Total records:", len(final_df))
print("Total columns:", len(final_df.columns))
print("Saved to:", output)

print("\nTop complaint types:")
print(final_df["complaint_type"].value_counts().head(10))
