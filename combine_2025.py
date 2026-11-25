"""Combine four NYC 311 sampled weeks into one raw, validated 2025 CSV.

Run from your NYC project directory:
    python combine_2025.py

This script does not overwrite your existing cleaned dataset.
"""

from pathlib import Path
import re
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
PROCESSED = PROJECT_DIR / "data" / "processed"
OUTPUT = PROCESSED / "nyc_311_2025_combined.csv"

WEEKS = [
    ("January", "nyc_311_2025_jan_7_13.csv", "2025-01-07", "2025-01-14"),
    ("April", "nyc_311_april_7_13.csv", "2025-04-07", "2025-04-14"),
    ("July", "nyc_311_july_7_13.csv", "2025-07-07", "2025-07-14"),
    ("October", "nyc_311_october_7_13.csv", "2025-10-07", "2025-10-14"),
]


def load_week(month, filename, start, end):
    path = PROCESSED / filename
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find {month} file: {path}")

    # Read as text to preserve unique IDs and leading zeros in ZIP codes.
    df = pd.read_csv(path, dtype=str, low_memory=False)
    # NYC exports use both "Unique Key" / "Created Date" and
    # "unique_key" / "created_date". Standardize every column name.
    original_columns = list(df.columns)
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(col).replace("\ufeff", "").strip().lower()).strip("_")
        for col in df.columns
    ]
    if not df.columns.is_unique:
        repeated = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(
            f"{month}: multiple source columns map to the same standardized name: {repeated}. "
            f"Original columns: {original_columns}"
        )

    required = {"unique_key", "created_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{month} is missing required columns after header cleanup: {sorted(missing)}\n"
            f"Original headers (first 15): {original_columns[:15]}\n"
            f"Standardized headers (first 15): {df.columns.tolist()[:15]}"
        )

    # January's older 311 export uses 'complaint_type'; newer exports may use
    # 'problem_formerly_complaint_type'. Populate the latter consistently.
    if "complaint_type" in df.columns:
        if "problem_formerly_complaint_type" not in df.columns:
            df["problem_formerly_complaint_type"] = df["complaint_type"]
        else:
            df["problem_formerly_complaint_type"] = (
                df["problem_formerly_complaint_type"]
                .replace("", pd.NA)
                .fillna(df["complaint_type"])
            )

    # Drop a prior calculated field, if present. The cleaning script should
    # compute resolution times consistently across all four months.
    df = df.drop(columns=["resolution_hours"], errors="ignore")

    # Handles both '01/07/2025 02:36:31 AM' and ISO-format dates.
    dates = pd.to_datetime(df["created_date"], format="mixed", errors="coerce")
    invalid_dates = dates.isna().sum()
    if invalid_dates:
        raise ValueError(f"{month}: {invalid_dates:,} unparseable created_date values")

    in_range = (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))
    excluded = (~in_range).sum()
    if excluded:
        print(f"  Note: excluding {excluded:,} {month} rows outside {start} to {end} (exclusive)")

    df = df.loc[in_range].copy()
    dates = dates.loc[in_range]
    df["created_date"] = dates.dt.strftime("%Y-%m-%d %H:%M:%S")
    df["sample_month"] = month

    if df.empty:
        raise ValueError(f"{month}: no records remain in the requested date range")
    if df["unique_key"].isna().any() or df["unique_key"].str.strip().eq("").any():
        raise ValueError(f"{month}: some service requests have missing unique_key values")

    daily = dates.dt.date.value_counts().sort_index()
    expected_dates = pd.date_range(start, periods=7).date
    missing_days = [str(d) for d in expected_dates if d not in daily.index]
    if missing_days:
        raise ValueError(f"{month}: missing requested days: {', '.join(missing_days)}")

    print(f"\n{month.upper()}: {len(df):,} requests")
    print(daily.to_string())
    return df


def main():
    frames = [load_week(*week) for week in WEEKS]
    combined = pd.concat(frames, ignore_index=True, sort=False)

    duplicates = combined["unique_key"].duplicated(keep=False)
    if duplicates.any():
        samples = combined.loc[duplicates, ["unique_key", "sample_month"]].head(10)
        raise ValueError(
            f"Found {duplicates.sum():,} rows with duplicate IDs. "
            f"No output saved. Examples:\n{samples.to_string(index=False)}"
        )

    combined.to_csv(OUTPUT, index=False)

    print("\n" + "=" * 45)
    print("COMBINED DATA VALIDATION")
    print(f"TOTAL RECORDS: {len(combined):,}")
    print(f"TOTAL COLUMNS: {len(combined.columns)}")
    print(f"DUPLICATE IDS: {combined['unique_key'].duplicated().sum():,}")
    print("WEEKS INCLUDED: January, April, July, October (7 days each)")
    print("\nRECORDS BY MONTH:")
    print(combined["sample_month"].value_counts().reindex(["January", "April", "July", "October"]).to_string())
    print(f"\nSUCCESS! Saved combined data to:\n{OUTPUT}")
    print("Existing *_clean.csv files have NOT been overwritten.")


if __name__ == "__main__":
    main()
