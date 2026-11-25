
import pandas as pd
import requests

df = pd.read_csv(
    "data/processed/nyc_311_week1_flagged.csv",
    usecols=["created_date"]
)

df["created_date"] = pd.to_datetime(
    df["created_date"], errors="coerce"
)

start = df["created_date"].min()
end = df["created_date"].max()

# Include the last second in the comparison.
end_exclusive = end + pd.Timedelta(seconds=1)

print("LOCAL DATASET")
print("Rows:", len(df))
print("First request:", start)
print("Last request:", end)

url = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

params = {
    "$select": "count(*) AS total",
    "$where": (
        f"created_date >= '{start:%Y-%m-%dT%H:%M:%S}' "
        f"AND created_date < "
        f"'{end_exclusive:%Y-%m-%dT%H:%M:%S}'"
    )
}

try:
    response = requests.get(
        url, params=params, timeout=(15, 90)
    )
    response.raise_for_status()

    official_count = int(response.json()[0]["total"])

    print("\nOFFICIAL API")
    print("Records in date range:", official_count)

    difference = official_count - len(df)

    print("\nCOMPARISON")
    print("Local records:", len(df))
    print("API records:", official_count)
    print("Difference:", difference)

    if difference == 0:
        print("Counts match for this date range.")
    else:
        print("Counts differ. Investigate before scaling.")

except requests.RequestException as error:
    print("\nAPI request failed:", error)
    print("Keep existing files; retry later if rate-limited.")
