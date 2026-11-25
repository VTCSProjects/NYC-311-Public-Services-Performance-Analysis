
import pandas as pd

df = pd.read_csv(
    "data/processed/nyc_311_week1_clean.csv",
    dtype={"Incident Zip": "string", "unique_key": "string"},
    low_memory=False
)

df["created_date"] = pd.to_datetime(
    df["created_date"], errors="coerce"
)

df["date"] = df["created_date"].dt.date

# Focus on residential noise complaints
noise = df[
    df["complaint_type"] == "Noise - Residential"
].copy()

bronx = noise[
    noise["borough"] == "BRONX"
].copy()

print("\n1. RESIDENTIAL NOISE BY BOROUGH")
print(noise["borough"].value_counts())

# Investigate daily patterns across boroughs
print("\n2. DAILY RESIDENTIAL NOISE REQUESTS")
print(
    pd.crosstab(noise["date"], noise["borough"])
    .to_string()
)

# Which Bronx ZIP codes contribute most?
print("\n3. TOP BRONX ZIP CODES")
print(
    bronx["Incident Zip"]
    .value_counts(dropna=False)
    .head(15)
)

# Look for specific types of residential noise
print("\n4. NOISE DESCRIPTIONS")
print(
    bronx["descriptor"]
    .value_counts(dropna=False)
    .head(15)
)

# Determine how complaints were reported
print("\n5. REPORTING CHANNELS")
print(
    bronx["Open Data Channel Type"]
    .value_counts(dropna=False)
)

# Check for repeated incident locations
addresses = (
    bronx["Incident Address"]
    .dropna()
    .str.strip()
)

addresses = addresses[addresses != ""]

address_counts = addresses.value_counts()

print("\n6. REPEATED INCIDENT LOCATIONS")
print("Total Bronx noise requests:", len(bronx))
print("Requests with an address:", len(addresses))
print("Distinct recorded addresses:", len(address_counts))

if len(address_counts) > 0:
    print(
        "Requests at 10 most frequent addresses:",
        address_counts.head(10).sum()
    )
    print(
        "Largest count at one address:",
        address_counts.iloc[0]
    )

print("\nINVESTIGATION COMPLETE")
