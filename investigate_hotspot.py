
import pandas as pd

df = pd.read_csv(
    "data/processed/nyc_311_week1_clean.csv",
    dtype={"unique_key": "string", "Incident Zip": "string"},
    low_memory=False
)

df["created_date"] = pd.to_datetime(
    df["created_date"], errors="coerce"
)

# Normalize addresses to avoid minor spacing/case differences.
df["address_key"] = (
    df["Incident Address"]
    .astype("string")
    .str.upper()
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

# Focus on Bronx residential noise requests.
bronx_noise = df[
    (df["borough"] == "BRONX") &
    (df["complaint_type"] == "Noise - Residential")
].copy()

# Find most frequently reported address.
address_counts = bronx_noise["address_key"].value_counts()

top_address = address_counts.index[0]

hotspot = bronx_noise[
    bronx_noise["address_key"] == top_address
].copy()

print("\n1. HOTSPOT SUMMARY")
print("Total requests:", len(hotspot))
print("Unique request IDs:", hotspot["unique_key"].nunique())
print("Share of all NYC sample records:",
      round(len(hotspot) / len(df) * 100, 2), "%")

print("\n2. REQUEST TIMESTAMPS")
print("First report:", hotspot["created_date"].min())
print("Last report:", hotspot["created_date"].max())
print("Distinct timestamps:",
      hotspot["created_date"].nunique())

print("\n3. REPORTS BY DAY")
print(
    hotspot.groupby(
        hotspot["created_date"].dt.date
    ).size().to_string()
)

print("\n4. BUSIEST REPORTING HOURS")
print(
    hotspot.groupby(
        hotspot["created_date"].dt.floor("h")
    ).size().nlargest(10).to_string()
)

print("\n5. REQUESTS AT IDENTICAL TIMESTAMPS")
print(
    hotspot["created_date"].value_counts().head(10)
)

print("\n6. REPORTING CHANNEL")
print(
    hotspot["Open Data Channel Type"]
    .value_counts(dropna=False)
)

print("\n7. COMPLAINT DETAILS")
print(
    hotspot["descriptor"]
    .value_counts(dropna=False)
)

print("\n8. ZIP CODE DISTRIBUTION")
print(
    hotspot["Incident Zip"]
    .value_counts(dropna=False)
)

# Sensitivity analysis: remove only residential
# noise records at the dominant address.
without_hotspot = df.drop(index=hotspot.index)

original = df["borough"].value_counts()
adjusted = without_hotspot["borough"].value_counts()

comparison = pd.DataFrame({
    "Original": original,
    "Without hotspot": adjusted
}).fillna(0).astype(int)

print("\n9. BOROUGH COMPARISON")
print(comparison.to_string())

print("\nINVESTIGATION COMPLETE")
