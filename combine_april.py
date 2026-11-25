
import pandas as pd
from pathlib import Path

# =====================================
# 1. LOAD BOTH APRIL DATASETS
# =====================================

april7 = pd.read_csv(
    "data/raw/phase1/verified/nyc_311_2025-04-07.csv",
    dtype=str
)

april8_13 = pd.read_csv(
    "data/raw/phase1/manual/nyc_311_apr_8_to_13.csv",
    dtype=str,
    low_memory=False
)

print("APRIL 7 RECORDS:", len(april7))
print("APRIL 8-13 RAW RECORDS:", len(april8_13))

# =====================================
# 2. STANDARDIZE COLUMN NAMES
# =====================================

rename = {
    "Unique Key": "unique_key",
    "Created Date": "created_date",
    "Closed Date": "closed_date",
    "Agency": "agency",
    "Agency Name": "agency_name",
    "Problem (formerly Complaint Type)": "complaint_type",
    "Problem Detail (formerly Descriptor)": "descriptor",
    "Borough": "borough",
    "Incident Zip": "incident_zip",
    "Incident Address": "incident_address",
    "Status": "status",
    "Open Data Channel Type": "open_data_channel_type",
    "Latitude": "latitude",
    "Longitude": "longitude"
}

april8_13 = april8_13.rename(columns=rename)

columns = list(rename.values())

for name, frame in [
    ("April 7", april7),
    ("April 8-13", april8_13)
]:
    missing = [c for c in columns if c not in frame.columns]

    if missing:
        raise ValueError(
            f"{name} missing columns: {missing}"
        )

april7 = april7[columns].copy()
april8_13 = april8_13[columns].copy()

# =====================================
# 3. CONVERT DATE FORMATS
# =====================================

date_format = "%m/%d/%Y %I:%M:%S %p"

for column in ["created_date", "closed_date"]:
    april7[column] = pd.to_datetime(
        april7[column], errors="coerce"
    )

    april8_13[column] = pd.to_datetime(
        april8_13[column],
        format=date_format,
        errors="coerce"
    )

# Reject unparseable creation dates
for name, frame in [
    ("April 7", april7),
    ("April 8-13", april8_13)
]:
    if frame["created_date"].isna().any():
        raise ValueError(
            f"{name} has invalid creation dates."
        )

# =====================================
# 4. REMOVE APRIL 14 RECORDS
# =====================================

start = pd.Timestamp("2025-04-08")
end = pd.Timestamp("2025-04-14")

valid_dates = (
    (april8_13["created_date"] >= start) &
    (april8_13["created_date"] < end)
)

excluded = (~valid_dates).sum()

april8_13 = april8_13.loc[valid_dates].copy()

print("\nRECORDS OUTSIDE DATE RANGE:", excluded)
print("VALID APRIL 8-13 RECORDS:", len(april8_13))

# =====================================
# 5. COMBINE DATASETS
# =====================================

combined = pd.concat(
    [april7, april8_13],
    ignore_index=True
)

# =====================================
# 6. VALIDATE THE COMBINED DATA
# =====================================

duplicates = combined["unique_key"].duplicated().sum()

invalid_dates = (
    (combined["created_date"] < pd.Timestamp("2025-04-07")) |
    (combined["created_date"] >= pd.Timestamp("2025-04-14"))
).sum()

missing_ids = combined["unique_key"].isna().sum()

print("\nCOMBINED DATA VALIDATION")
print("Total records:", len(combined))
print("Duplicate IDs:", duplicates)
print("Missing IDs:", missing_ids)
print("Records outside April 7-13:", invalid_dates)

if duplicates or missing_ids or invalid_dates:
    raise ValueError("Combined dataset failed validation.")

# =====================================
# 7. ANALYZE APRIL DATA
# =====================================

print("\nREQUESTS BY DATE")
print(
    combined["created_date"]
    .dt.date.value_counts()
    .sort_index()
)

print("\nREQUESTS BY BOROUGH")
print(combined["borough"].value_counts())

print("\nTOP 10 COMPLAINT TYPES")
print(combined["complaint_type"].value_counts().head(10))

# =====================================
# 8. SAVE THE DATA
# =====================================

Path("data/processed").mkdir(
    parents=True, exist_ok=True
)

output = "data/processed/nyc_311_april_7_13.csv"

combined.to_csv(output, index=False)

print("\nAPRIL DATASET SAVED!")
print("File:", output)
print("Total rows:", len(combined))
