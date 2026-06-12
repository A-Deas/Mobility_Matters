import duckdb

PM_GLOB = "Data/PM2.5/*.parquet"
UT_HEXES = "Data/utah_hexes_res8.parquet"
FINAL_HEXES_CSV = "Wildfire_Investigation/final_wildfire_hexes3.csv"

con = duckdb.connect()

result = con.execute(f"""
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
    WHERE pm.month = 7
      AND pm.day BETWEEN 1 AND 31
),

summary AS (
    SELECT
        CASE
            WHEN day BETWEEN 1 AND 24 THEN 'pre_wildfire'
            WHEN day BETWEEN 25 AND 31 THEN 'wildfire_week'
        END AS period,
        count(*) AS n_records,
        avg(pm25) AS avg_pm25,
        min(pm25) AS min_pm25,
        max(pm25) AS max_pm25
    FROM pm_in_plume
    GROUP BY 1
)

SELECT *
FROM summary
ORDER BY period;
""").fetchdf()

print(result)

pre_avg = result.loc[result["period"] == "pre_wildfire", "avg_pm25"].iloc[0]
fire_avg = result.loc[result["period"] == "wildfire_week", "avg_pm25"].iloc[0]

print(f"\nAverage PM2.5 in plume hexes, July 1–24:  {pre_avg:.3f}")
print(f"Average PM2.5 in plume hexes, July 25–31: {fire_avg:.3f}")
print(f"Difference: {fire_avg - pre_avg:.3f}")