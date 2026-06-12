from pathlib import Path
import duckdb

# -------------------------
# Paths / constants
# -------------------------
BASE_DIR = Path("Output") # Output/<place>/<place>_final_run1.parquet
RUN_TAG = "run1"

PM_PARQUET_GLOB = "Data/PM2.5/*.parquet"
UTAH_HEXES_PATH = "Data/utah_hexes_res8.parquet"

MONTH = 7 # july
DAY_MIN = 25 # mon
DAY_MAX = 31 # sun

OUT_TAG = "SWAPPED" # mark the new files

def swap_pm_data_duckdb():
    con = duckdb.connect() # DuckDB is KING

    # 1) Create a VIEW for the new PM2.5 dataset over the desired week
    con.execute(f"""
        CREATE OR REPLACE VIEW pm_conus AS
        SELECT
            pm.day AS day,
            CAST(pm.h3_polyfill AS VARCHAR) AS hex_id,
            CAST(pm.value AS DOUBLE) AS pm_value
        FROM read_parquet('{PM_PARQUET_GLOB}') AS pm
        WHERE pm.month = {MONTH}
          AND pm.day BETWEEN {DAY_MIN} AND {DAY_MAX}
    """)

    print("Number of rows in pm_conus:", con.execute("SELECT COUNT(*) FROM pm_conus").fetchone()[0])

    # 2) Loop through place folders
    for place_dir in sorted(p for p in BASE_DIR.iterdir() if p.is_dir()):
        safe_name = place_dir.name
        in_file  = place_dir / f"{safe_name}_final_{RUN_TAG}.parquet"
        out_file = place_dir / f"{safe_name}_final_{RUN_TAG}_{OUT_TAG}.parquet"

        print(f"Swapping PM for: {in_file}")

        # 3) Read in the files and swap the PM2.5 values, everything else stays exactly the same
        con.execute(f"""
            COPY (
                WITH base AS (
                    SELECT
                        *,
                        ({DAY_MIN} + CAST(FLOOR(tick / 24) AS INTEGER)) AS pm_day -- 25 + FLOOR(23/24) = 25 + 0 = 25, 25 + FLOOR(47/24) = 25 + 1 = 26, etc.
                    FROM read_parquet('{in_file.as_posix()}')
                )
                SELECT
                    -- keep all original columns, but overwrite pm2.5
                    b.p_id,
                    b.sex,
                    b.age,
                    b.place,
                    b.home_hex,
                    b.agent_type,
                    b.tick,
                    b.day,
                    b.hour,
                    b.activity,
                    b.h3_act_loc,
                    COALESCE(pm.pm_value, -999) AS "pm2.5" -- COALESCE is used as a safety check to make sure the PM2.5 values are swapping correctly!
                FROM base as b
                LEFT JOIN pm_conus as pm -- keep every row from the table on the left (base), even if table on the right (pm_conus) has no matching row. If no match, we get null pm2.5 value/entry (handled by coalesce above).
                  ON pm.day = b.pm_day -- THIS MAKES SURE WE TAKE THE PM2.5 VALUES FROM THE RIGHT DAY!!
                 AND pm.hex_id = CAST(b.h3_act_loc AS VARCHAR)
            )
            TO '{out_file.as_posix()}'
            (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)

        # Check for any PM2.5 values that may not have been swapped properly
        missing_ct = con.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{out_file.as_posix()}')
            WHERE "pm2.5" = -999
        """).fetchone()[0]
        total_ct = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_file.as_posix()}')").fetchone()[0]

        print(f"  Wrote: {out_file.name} | Rows = {total_ct:,} | Missing hex values = {missing_ct:,}")

    con.close()

if __name__ == "__main__":
    swap_pm_data_duckdb()
