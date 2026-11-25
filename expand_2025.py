
import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

# ==========================================
# NYC 311 DATA EXPANSION — PHASE 1
# ==========================================

API_URL = (
    "https://data.cityofnewyork.us/resource/"
    "erm2-nwe9.json"
)

START_DATE = date(2025, 4, 7)
NUMBER_OF_DAYS = 7
BATCH_SIZE = 3000

COLUMNS = [
    "unique_key",
    "created_date",
    "closed_date",
    "agency",
    "agency_name",
    "complaint_type",
    "descriptor",
    "borough",
    "incident_zip",
    "incident_address",
    "status",
    "open_data_channel_type",
    "latitude",
    "longitude",
]

BASE = Path("data/raw/phase1")
PARTS = BASE / "parts"
VERIFIED = BASE / "verified"

PARTS.mkdir(parents=True, exist_ok=True)
VERIFIED.mkdir(parents=True, exist_ok=True)

# Read token from PowerShell environment variable.
token = os.getenv("SOCRATA_APP_TOKEN")

if not token:
    raise SystemExit(
        "Missing SOCRATA_APP_TOKEN. "
        "Set it in PowerShell before running."
    )

session = requests.Session()
session.headers.update({
    "X-App-Token": token
})


# ==========================================
# 1. API REQUESTS WITH RETRIES
# ==========================================

def request_data(params):

    for attempt in range(1, 4):
        try:
            response = session.get(
                API_URL,
                params=params,
                timeout=(15, 90)
            )

            if response.status_code == 429:
                raise RuntimeError(
                    "API rate limit reached (429). "
                    "Stop and resume later. "
                    "Previously downloaded pages are saved."
                )

            response.raise_for_status()
            return response.json()

        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.HTTPError
        ) as error:

            print(
                f"Request failed, attempt {attempt}: {error}"
            )

            # Don't repeatedly retry invalid queries
            # or authorization errors.
            if isinstance(error, requests.HTTPError):
                status = error.response.status_code

                if status < 500:
                    raise

            if attempt == 3:
                raise

            time.sleep(5 * attempt)


# ==========================================
# 2. DOWNLOAD ONE COMPLETE DAY
# ==========================================

def download_day(day):

    next_day = day + timedelta(days=1)
    day_string = day.isoformat()

    final_file = VERIFIED / f"nyc_311_{day_string}.csv"

    if final_file.exists():
        print(f"{day_string}: Already verified. Skipping.")
        return

    part_folder = PARTS / day_string
    part_folder.mkdir(parents=True, exist_ok=True)

    start = f"{day_string}T00:00:00"
    end = f"{next_day.isoformat()}T00:00:00"

    base_filter = (
        f"created_date >= '{start}' "
        f"AND created_date < '{end}'"
    )

    # Resume from existing downloaded pages.
    saved_pages = sorted(
        part_folder.glob("page_*.csv")
    )

    last_date = None
    last_key = None

    for index, file in enumerate(saved_pages):
        expected_name = f"page_{index:04d}.csv"

        if file.name != expected_name:
            raise ValueError(
                f"Unexpected saved page: {file.name}"
            )

        previous = pd.read_csv(
            file, dtype=str
        )

        if previous.empty:
            raise ValueError("Empty saved page.")

        last_date = previous["created_date"].iloc[-1]
        last_key = previous["unique_key"].iloc[-1]

    page_number = len(saved_pages)

    print(f"\nDownloading {day_string}")
    print(f"Previously saved pages: {page_number}")

    while True:

        # Keyset pagination avoids large offsets.
        where = base_filter

        if last_date is not None:
            where += (
                f" AND (created_date > '{last_date}' "
                f"OR (created_date = '{last_date}' "
                f"AND unique_key > '{last_key}'))"
            )

        params = {
            "$select": ",".join(COLUMNS),
            "$where": where,
            "$order": (
                "created_date ASC, unique_key ASC"
            ),
            "$limit": BATCH_SIZE
        }

        records = request_data(params)

        if not records:
            break

        batch = pd.DataFrame(records).reindex(
            columns=COLUMNS
        )

        if batch["unique_key"].isna().any():
            raise ValueError("Missing request ID.")

        if batch["unique_key"].duplicated().any():
            raise ValueError("Duplicate IDs in batch.")

        first_cursor = (
            batch["created_date"].iloc[0],
            batch["unique_key"].iloc[0]
        )

        if last_date is not None:
            if first_cursor <= (last_date, last_key):
                raise ValueError(
                    "Pagination did not advance."
                )

        # Save each page immediately.
        filename = f"page_{page_number:04d}.csv"

        temporary_file = part_folder / (
            f"page_{page_number:04d}.tmp"
        )
        saved_file = part_folder / filename

        batch.to_csv(temporary_file, index=False)
        temporary_file.replace(saved_file)

        last_date = batch["created_date"].iloc[-1]
        last_key = batch["unique_key"].iloc[-1]

        page_number += 1

        print(
            f"Page {page_number}: "
            f"{len(batch):,} records saved"
        )

        if len(batch) < BATCH_SIZE:
            break

        time.sleep(1)

    # ======================================
    # 3. VALIDATE THE DOWNLOADED DAY
    # ======================================

    files = sorted(part_folder.glob("page_*.csv"))

    if not files:
        raise ValueError(
            f"No records downloaded for {day_string}"
        )

    df = pd.concat(
        [
            pd.read_csv(f, dtype=str)
            for f in files
        ],
        ignore_index=True
    )

    duplicates = df["unique_key"].duplicated().sum()

    dates = pd.to_datetime(
        df["created_date"],
        errors="coerce"
    )

    invalid_dates = (
        dates.isna()
        | (dates < pd.Timestamp(day))
        | (dates >= pd.Timestamp(next_day))
    ).sum()

    print("\nVALIDATION")
    print("Downloaded records:", len(df))
    print("Duplicate IDs:", duplicates)
    print("Invalid dates:", invalid_dates)

    if duplicates or invalid_dates:
        raise ValueError(
            "Data-quality validation failed."
        )

    # Check official record count.
    count_result = request_data({
        "$select": "count(*) AS total",
        "$where": base_filter
    })

    official_count = int(
        count_result[0]["total"]
    )

    print("Official API count:", official_count)

    if len(df) != official_count:
        raise ValueError(
            f"Record-count mismatch for {day_string}. "
            "Saved pages remain available. "
            "Investigate before treating this day "
            "as complete."
        )

    # Save only after validation succeeds.
    temporary_final = VERIFIED / (
        f"nyc_311_{day_string}.tmp"
    )

    df.to_csv(temporary_final, index=False)
    temporary_final.replace(final_file)

    print(f"VERIFIED: {day_string}")


# ==========================================
# 4. DOWNLOAD APRIL 7-13, 2025
# ==========================================

def main():

    print("=" * 45)
    print("NYC 311 DATA EXPANSION")
    print("=" * 45)

    for offset in range(NUMBER_OF_DAYS):

        current_day = (
            START_DATE + timedelta(days=offset)
        )

        download_day(current_day)

    # ======================================
    # 5. FINAL SUMMARY
    # ======================================

    verified_files = sorted(
        VERIFIED.glob("nyc_311_*.csv")
    )

    total_records = 0

    for file in verified_files:

        data = pd.read_csv(
            file,
            usecols=["unique_key"],
            dtype=str
        )

        total_records += len(data)

    print("\n" + "=" * 45)
    print("DOWNLOAD COMPLETE")
    print("=" * 45)

    print("Verified days:", len(verified_files))
    print("Total records:", f"{total_records:,}")
    print("Saved to:", VERIFIED)


if __name__ == "__main__":
    main()
