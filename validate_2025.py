
import pandas as pd
from pathlib import Path

folder = Path("data/processed")

months = {
    "April": ("nyc_311_april_7_13.csv", 4),
    "July": ("nyc_311_july_7_13.csv", 7),
    "October": ("nyc_311_october_7_13.csv", 10)
}

for month, (filename, number) in months.items():
    print(f"\n{'=' * 40}")
    print(f"{month.upper()} 2025")
    print("=" * 40)

    df = pd.read_csv(folder / filename, low_memory=False)

    # Normalize column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    dates = pd.to_datetime(
        df["created_date"],
        format="mixed",
        errors="coerce"
    )

    start = pd.Timestamp(2025, number, 7)
    end = pd.Timestamp(2025, number, 13)

    valid = dates.dt.normalize().between(start, end)

    print("Total records:", len(df))
    print("Total columns:", len(df.columns))
    print("Invalid/out-of-range dates:", (~valid).sum())

    if "unique_key" in df.columns:
        print("Duplicate IDs:", df["unique_key"].duplicated().sum())
        print("Missing IDs:", df["unique_key"].isna().sum())

    print("\nRECORDS BY DATE:")
    print(dates.dt.date.value_counts().sort_index())

    print("\nMISSING EXPECTED DATES:")
    expected = pd.date_range(start, end)
    observed = pd.DatetimeIndex(
        dates.dropna().dt.normalize().unique()
    )
    print(list(expected.difference(observed).date))
