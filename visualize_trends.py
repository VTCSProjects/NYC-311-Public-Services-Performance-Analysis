
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the aggregated daily results
df = pd.read_csv(
    "data/processed/nyc_311_daily_trends.csv",
    parse_dates=["day"]
)

# Exclude partial boundary days
partial = (
    df["partial_boundary_day"]
    .astype(str).str.lower().eq("true")
)

df = df.loc[~partial].copy()

Path("reports/figures").mkdir(
    parents=True, exist_ok=True
)

# CHART 1: Original vs hotspot-adjusted requests
fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(
    df["day"],
    df["total_requests"],
    marker="o",
    linewidth=2,
    label="All Requests"
)

ax.plot(
    df["day"],
    df["without_hotspot"],
    marker="o",
    linewidth=2,
    label="Excluding Hotspot"
)

ax.set_title(
    "NYC 311 Daily Service Requests",
    fontsize=16
)
ax.set_xlabel("Date")
ax.set_ylabel("Number of Requests")
ax.legend()
ax.grid(alpha=0.3)

fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(
    "reports/figures/daily_requests.png",
    dpi=300
)
plt.show()
plt.close(fig)


# CHART 2: Hotspot percentage by day
fig, ax = plt.subplots(figsize=(11, 6))

ax.bar(
    df["day"].dt.strftime("%b %d"),
    df["hotspot_pct"]
)

ax.set_title(
    "Percentage of NYC 311 Requests From One Hotspot",
    fontsize=15
)
ax.set_xlabel("Date")
ax.set_ylabel("Share of Requests (%)")
ax.set_ylim(0, 45)

for i, pct in enumerate(df["hotspot_pct"]):
    ax.text(
        i, pct + 0.5,
        f"{pct:.1f}%",
        ha="center"
    )

fig.tight_layout()
fig.savefig(
    "reports/figures/hotspot_percentage.png",
    dpi=300
)
plt.show()
plt.close(fig)

print("\nVisualizations saved successfully!")
