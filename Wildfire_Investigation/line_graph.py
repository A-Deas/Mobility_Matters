import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# -----------------------------
# Paths / constants
# -----------------------------
PM_GLOB = "Data/PM2.5_FullYear/*/*.parquet"
UT_HEXES = "Data/utah_hexes_res8.parquet"
FINAL_HEXES_CSV = "Wildfire_Investigation/final_wildfire_hexes3.csv"

YEAR = 2016
MONTH_MIN = 1
MONTH_MAX = 12
DAY_MIN = 1
DAY_MAX = 31

STUDY_WEEK_START = datetime(YEAR, 7, 25)
STUDY_WEEK_END   = datetime(YEAR, 7, 31)

# OUT_PATH = "Plots/Heat Maps/wildfire_plume_daily_pm25_line.png"

# -----------------------------
# Query daily average PM2.5
# -----------------------------
con = duckdb.connect()

daily_df = con.execute(f"""
WITH plume_hexes AS (
    SELECT hex_id
    FROM read_csv_auto('{FINAL_HEXES_CSV}')
),

pm_in_plume AS (
    SELECT
        pm.h3_polyfill AS hex_id,
        pm.month,
        pm.day,
        pm.value AS pm25
    FROM read_parquet('{PM_GLOB}') AS pm
    JOIN read_parquet('{UT_HEXES}') AS ut
      ON pm.h3_polyfill = ut.hex_id
    JOIN plume_hexes AS ph
      ON pm.h3_polyfill = ph.hex_id
)
SELECT
    month,
    day,
    avg(pm25) AS avg_pm25
FROM pm_in_plume
GROUP BY month, day
ORDER BY month, day
""").fetchdf()

# --- I mean....we don't want to filter anything if we are looking at the whole year!
    # WHERE pm.month BETWEEN {MONTH_MIN} AND {MONTH_MAX}
    #   AND pm.day BETWEEN {DAY_MIN} AND {DAY_MAX}

# Build a real date column
daily_df["date"] = pd.to_datetime(
    {
        "year": YEAR,
        "month": daily_df["month"],
        "day": daily_df["day"]
    }
)

print(daily_df)

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(daily_df["date"], daily_df["avg_pm25"], linewidth=2)

# Shade study week
ax.axvspan(STUDY_WEEK_START, STUDY_WEEK_END, alpha=0.2)

ax.set_title("Average Daily PM2.5 in Wildfire-Affected Hexes", fontsize=14, weight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Average PM2.5 (µg/m³)")

ax.grid(True, alpha=0.3)
fig.autofmt_xdate()
plt.tight_layout()

# plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

# print(f"Saved: {OUT_PATH}")

result = con.execute(f"""
WITH plume_hexes AS (
    SELECT hex_id
    FROM read_csv_auto('{FINAL_HEXES_CSV}')
),

pm_in_plume AS (
    SELECT
        pm.month,
        pm.day,
        pm.value AS pm25
    FROM read_parquet('{PM_GLOB}') AS pm
    JOIN read_parquet('{UT_HEXES}') AS ut
      ON pm.h3_polyfill = ut.hex_id
    JOIN plume_hexes AS ph
      ON pm.h3_polyfill = ph.hex_id
),

summary AS (
    SELECT
        'full_year' AS period,
        avg(pm25) AS avg_pm25
    FROM pm_in_plume

    UNION ALL

    SELECT
        'study_week' AS period,
        avg(pm25) AS avg_pm25
    FROM pm_in_plume
    WHERE month = 7 AND day BETWEEN 25 AND 31
)

SELECT * FROM summary;
""").fetchdf()

print(result)

# Optional: compute difference nicely
year_avg = result.loc[result["period"] == "full_year", "avg_pm25"].iloc[0]
week_avg = result.loc[result["period"] == "study_week", "avg_pm25"].iloc[0]

print(f"\nFull year average:  {year_avg:.3f}")
print(f"Study week average: {week_avg:.3f}")
print(f"Difference:         {week_avg - year_avg:.3f}")