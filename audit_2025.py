
import pandas as pd
from pathlib import Path

folder = Path("data/processed")

files = [
    "nyc_311_april_7_13.csv",
    "nyc_311_july_7_13.csv",
    "nyc_311_october_7_13.csv"
]

frames = []

for filename in files:
    df = pd.read_csv(folder / filename, low_memory=False)
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    frames.append(df)

# Verify column consistency
for df in frames[1:]:
    assert set(df.columns) == set(frames[0].columns), \
        "Column mismatch!"

data = pd.concat(frames, ignore_index=True)

print("TOTAL RECORDS:", len(data))
print("TOTAL COLUMNS:", len(data.columns))
print("DUPLICATE REQUEST IDs:",
      data["unique_key"].duplicated().sum())

print("\nMISSING VALUES (%)")
missing = data.isna().mean().mul(100)
print(missing.sort_values(ascending=False).to_string())

print("\nBOROUGH VALUES")
print(data["borough"].value_counts(dropna=False))

print("\nSTATUS VALUES")
print(data["status"].value_counts(dropna=False))

print("\nTOP 20 COMPLAINT TYPES")
print(data["problem_formerly_complaint_type"].value_counts(dropna=False).head(20))

# Validate resolution times
created = pd.to_datetime(
    data["created_date"], format="mixed", errors="coerce"
)
closed = pd.to_datetime(
    data["closed_date"], format="mixed", errors="coerce"
)

negative = (closed < created).sum()

print("\nINVALID CREATED DATES:", created.isna().sum())
print("UNPARSEABLE NONEMPTY CLOSED DATES:",
      (data["closed_date"].notna() & closed.isna()).sum())
print("CLOSED BEFORE CREATED:", negative)

print("\nAUDIT COMPLETE")
