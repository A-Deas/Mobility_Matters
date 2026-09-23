import duckdb
import pandas as pd
import h3
import time
from datetime import datetime
from pathlib import Path


TABLE = "DifferenceMetrics"
BATCH_SIZE = 25000
TEST_MODE = False

SIMULATION_DB_PATH = "Data/DuckDB_Files/SimulationResults.db"
SIMULATION_TABLE = "SimulationResults"

OUTPUT_DIR = Path("Data/DuckDB_Files")
DB_PATH = OUTPUT_DIR / f"{TABLE}.db"

HEAD_DIR = Path("Dataframe_Heads")
HEAD_FILE = HEAD_DIR / "difference_metrics_head.csv"

TEMP_DIR = Path("Data/DuckDB_Temp")

con = duckdb.connect(str(DB_PATH))

con.execute("SET threads=1")
con.execute("SET preserve_insertion_order=false")
con.execute("SET memory_limit='10GB'")
con.execute(f"SET temp_directory='{TEMP_DIR}'")

con.execute(f"ATTACH '{SIMULATION_DB_PATH}' AS simulation")

start_time = time.time()

print(f"\nDifference metric calculation started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================
# Create H3 center lookup
# ============================================================
print("\nCreating H3 center lookup...")

hexes = con.execute(f"""
    SELECT DISTINCT h3_act
    FROM simulation.{SIMULATION_TABLE}
    WHERE h3_act IS NOT NULL
""").fetchall()

hex_records = []

for (hex_id,) in hexes:
    lat, lon = h3.cell_to_latlng(hex_id)
    hex_records.append({"h3_cell": hex_id, "lat": lat, "lon": lon})

hex_centers_df = pd.DataFrame(hex_records)

con.register("hex_centers_df", hex_centers_df)

con.execute("""
    CREATE OR REPLACE TABLE HexCenters AS
    SELECT *
    FROM hex_centers_df
""")

con.unregister("hex_centers_df")

print(f"H3 centers created for {len(hex_records):,} unique hexes.")


# ============================================================
# Create agent index for batching
# ============================================================
print("\nCreating agent index...")

con.execute(f"""
    CREATE OR REPLACE TABLE AgentIds AS
    SELECT
        p_id,
        ROW_NUMBER() OVER () AS agent_index -- row numbers start at 1
    FROM (
        SELECT DISTINCT p_id
        FROM simulation.{SIMULATION_TABLE}
    )
""")

num_agents = con.execute("""
    SELECT COUNT(*)
    FROM AgentIds
""").fetchone()[0]

print(f"Unique agents: {num_agents:,}")


# ============================================================
# Check for exact zero static exposures
# ============================================================
zero_static = con.execute(f"""
    SELECT COUNT(*)
    FROM simulation.{SIMULATION_TABLE}
    WHERE static_exposure = 0
""").fetchone()[0]

print("\n--- Static Exposure Denominator Check ---")
print(f"Rows with static exposure exactly equal to zero: {zero_static:,}")


# ============================================================
# Create empty DifferenceMetrics table
# ============================================================
print(f"\nCreating empty {TABLE} table...")

con.execute(f"""
    CREATE OR REPLACE TABLE {TABLE} (
        -- Agent demographics:

        p_id VARCHAR,
        agent_type VARCHAR,
        sex VARCHAR,
        age DOUBLE,

        -- Static exposure metrics to use downstream

        min_static_exposure DOUBLE,
        mean_static_exposure DOUBLE,

        -- Exposure difference metrics:

        mean_raw_diff DOUBLE,
        mean_abs_diff DOUBLE,
        max_abs_diff DOUBLE,
        mpd DOUBLE,
        mapd DOUBLE,

        -- Mobility metrics:

        non_home_hours BIGINT,
        unique_nonhome_nontravel_hexes BIGINT, -- these count the number activity locations away from home (because travel is going from one location to another)
        num_location_changes BIGINT,
        min_transition_distance_km DOUBLE,
        avg_transition_distance_km DOUBLE,
        max_transition_distance_km DOUBLE,
        travel_hours BIGINT
    )
""")

print(f"{TABLE} structure successfully created.")


# ============================================================
# Compute agent-level metrics in batches
# ============================================================
print(f"\nProcessing agents in batches of {BATCH_SIZE:,}...")

for batch_start in range(1, num_agents + 1, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE - 1, num_agents)
    batch_start_time = time.time()

    con.execute(f"""
        INSERT INTO {TABLE}

        WITH batch_agents AS (
            SELECT p_id
            FROM AgentIds
            WHERE agent_index BETWEEN {batch_start} AND {batch_end}
        ),

        observed_locations AS (
            SELECT
                s.p_id,
                s.tick,
                s.h3_act,
                LAG(s.h3_act) OVER (PARTITION BY s.p_id ORDER BY s.tick) AS previous_h3_act
            FROM simulation.{SIMULATION_TABLE} AS s

            JOIN batch_agents AS b     -- Keep only rows whose p_id appears in the current batch (plain JOIN and INNER JOIN are equivalent)
                ON s.p_id = b.p_id 

            WHERE s.h3_act IS NOT NULL -- Retain only known modeled locations; travel has no assigned H3 (travel means going from one location to another anyway)
                                       -- AND you can't observe a location where h3_act is null anyway LOL (this just always happens to be travel here)
        ),

        schedule_endpoints AS (
            SELECT
                p_id,
                ARG_MIN(h3_act, tick) AS first_h3_act,  -- ARG_MIN(h3_act, tick) means: Give me the h3_act belonging to the smallest tick.
                ARG_MAX(h3_act, tick) AS last_h3_act    -- ARG_MAX(h3_act, tick) means: Give me the h3_act belonging to the largest tick.
            FROM observed_locations
            GROUP BY p_id
        ),

        transition_pairs AS (

            -- Normal transitions within the represented week

            SELECT
                p_id,
                previous_h3_act,
                h3_act
            FROM observed_locations
            WHERE previous_h3_act IS NOT NULL   -- Previous_h3_act could be NULL from the LAG above (like at tick 0), so remove these
            AND h3_act != previous_h3_act       -- Make sure we only take the instances where locations actually transitioned!

            UNION ALL

            -- Transition across the cyclic week boundary

            SELECT
                p_id,
                last_h3_act AS previous_h3_act,
                first_h3_act AS h3_act
            FROM schedule_endpoints
            WHERE last_h3_act != first_h3_act   -- Catching the edges cases here that stem from our reordering
        ),

        location_transitions AS (
            SELECT
                t.p_id,
                t.previous_h3_act,
                t.h3_act,

                -- Haversine function for distance between two (lat, lon) points:

                6371.0088 * -- multiply by Earth's radius to get it km
                    2 * ASIN(
                    SQRT(
                    POWER(SIN(RADIANS(c2.lat - c1.lat) / 2), 2) +
                    COS(RADIANS(c1.lat)) * 
                    COS(RADIANS(c2.lat)) * 
                    POWER(SIN(RADIANS(c2.lon - c1.lon) / 2), 2)
                )) AS transition_distance_km

            FROM transition_pairs AS t

            -- We have two different H3 cell IDs now so we need to join hex centers twice: Once for the previous hex cell, and again for the current hex cell:

            JOIN HexCenters AS c1
                ON t.previous_h3_act = c1.h3_cell

            JOIN HexCenters AS c2
                ON t.h3_act = c2.h3_cell
        ),

        mobility_metrics AS (
            SELECT
                p_id,
                COUNT(*) AS num_location_changes,
                MIN(transition_distance_km) AS min_transition_distance_km,
                AVG(transition_distance_km) AS avg_transition_distance_km,
                MAX(transition_distance_km) AS max_transition_distance_km
            FROM location_transitions
            GROUP BY p_id
        ),

        agent_metrics AS (
            SELECT
                s.p_id,
                MAX(s.agent_type) AS agent_type,
                MAX(s.sex) AS sex,
                MAX(s.age) AS age,

                MIN(s.static_exposure) AS min_static_exposure,
                AVG(s.static_exposure) AS mean_static_exposure,

                AVG(s.dynamic_exposure - s.static_exposure) AS mean_raw_diff,
                AVG(ABS(s.dynamic_exposure - s.static_exposure)) AS mean_abs_diff,
                MAX(ABS(s.dynamic_exposure - s.static_exposure)) AS max_abs_diff,

                AVG(CASE WHEN s.static_exposure != 0 THEN (s.dynamic_exposure - s.static_exposure) / s.static_exposure END) * 100 AS mpd,
                AVG(CASE WHEN s.static_exposure != 0 THEN ABS((s.dynamic_exposure - s.static_exposure) / s.static_exposure) END) * 100 AS mapd,

                SUM(CASE WHEN s.h3_act IS DISTINCT FROM s.h3_home THEN 1 ELSE 0 END) AS non_home_hours,
                    -- IS DISTINCT FROM allows NULL travel locations to be treated as different from the known home H3.
                    -- This also prevents activities occurring at the home H3 (e.g., work_home) from being counted as non-home.

                COUNT(DISTINCT CASE
                    WHEN s.h3_act IS NOT NULL    -- Skip instances of travel (no hexes to count anyway)
                    AND s.h3_act != s.h3_home
                    THEN s.h3_act
                END) AS unique_nonhome_nontravel_hexes,

                SUM(CASE WHEN s.activity = 'travel' THEN 1 ELSE 0 END) AS travel_hours

            FROM simulation.{SIMULATION_TABLE} AS s
            JOIN batch_agents AS b
                ON s.p_id = b.p_id
            GROUP BY s.p_id
        )

        SELECT
            a.p_id,
            a.agent_type,
            a.sex,
            a.age,
            a.min_static_exposure,
            a.mean_static_exposure,
            a.mean_raw_diff,
            a.mean_abs_diff,
            a.max_abs_diff,
            a.mpd,
            a.mapd,
            a.non_home_hours,
            a.unique_nonhome_nontravel_hexes,
            COALESCE(m.num_location_changes, 0) AS num_location_changes,
            COALESCE(m.min_transition_distance_km, 0) AS min_transition_distance_km,
            COALESCE(m.avg_transition_distance_km, 0) AS avg_transition_distance_km,
            COALESCE(m.max_transition_distance_km, 0) AS max_transition_distance_km,
            a.travel_hours

        FROM agent_metrics AS a

        LEFT JOIN mobility_metrics AS m
            ON a.p_id = m.p_id
    """)

    batch_elapsed = time.time() - batch_start_time
    batch_formatted = time.strftime("%H:%M:%S", time.gmtime(batch_elapsed))

    print(f"Processed agents {batch_start:,} - {batch_end:,} of {num_agents:,} | Batch runtime: {batch_formatted}", flush=True)

    if TEST_MODE:
        print("\nTEST MODE: Stopping after one batch.")
        break

print(f"\n{TABLE} successfully created.")


# ============================================================
# Save a preview
# ============================================================
print("\nSaving a preview...")

preview = con.execute(f"""
    SELECT *
    FROM {TABLE}
    ORDER BY p_id
    LIMIT 1000
""").df()

preview.to_csv(HEAD_FILE, index=False)

print("Saved dataframe head for preview.")


# ============================================================
# Check agent counts
# ============================================================
metrics_agents = con.execute(f"""
    SELECT COUNT(*)
    FROM {TABLE}
""").fetchone()[0]

if TEST_MODE:
    expected_agents = min(BATCH_SIZE, num_agents)

    print("\n--- Test Batch Agent Count Check ---")
    print(f"Expected agents: {expected_agents:,}")
    print(f"Agents in DifferenceMetrics: {metrics_agents:,}")
    print(f"Agent counts match: {expected_agents == metrics_agents}")
else:
    print("\n--- Agent Count Check ---")
    print(f"Agents in SimulationResults: {num_agents:,}")
    print(f"Agents in DifferenceMetrics: {metrics_agents:,}")
    print(f"Agent counts match: {num_agents == metrics_agents}")


# ============================================================
# Check for missing difference metrics
# ============================================================
missing_metrics = con.execute(f"""
    SELECT COUNT(*)
    FROM {TABLE}
    WHERE min_static_exposure is NULL
       OR mean_static_exposure is NULL
       OR mean_raw_diff IS NULL
       OR mean_abs_diff IS NULL
       OR max_abs_diff IS NULL
       OR mpd IS NULL
       OR mapd IS NULL
""").fetchone()[0]

print("\n--- Difference Metric Check ---")
print(f"Agents with missing difference metrics: {missing_metrics:,}")
print(f"All agents have difference metrics: {missing_metrics == 0}")


# ============================================================
# Preview difference metric and mobility metric summaries
# ============================================================
difference_summary = con.execute(f"""
    SELECT
        MIN(min_static_exposure) AS min_min_static_exposure,
        AVG(min_static_exposure) AS avg_min_static_exposure,
        MAX(min_static_exposure) AS max_min_static_exposure,

        MIN(mean_static_exposure) AS min_mean_static_exposure,
        AVG(mean_static_exposure) AS avg_mean_static_exposure,
        MAX(mean_static_exposure) AS max_mean_static_exposure,

        MIN(mean_raw_diff) AS min_mean_raw_diff,
        AVG(mean_raw_diff) AS avg_mean_raw_diff,
        MAX(mean_raw_diff) AS max_mean_raw_diff,

        MIN(mean_abs_diff) AS min_mean_abs_diff,
        AVG(mean_abs_diff) AS avg_mean_abs_diff,
        MAX(mean_abs_diff) AS max_mean_abs_diff,

        MIN(max_abs_diff) AS min_max_abs_diff,
        AVG(max_abs_diff) AS avg_max_abs_diff,
        MAX(max_abs_diff) AS max_max_abs_diff,

        MIN(mpd) AS min_mpd,
        AVG(mpd) AS avg_mpd,
        MAX(mpd) AS max_mpd,

        MIN(mapd) AS min_mapd,
        AVG(mapd) AS avg_mapd,
        MAX(mapd) AS max_mapd
    FROM {TABLE}
""").df()

mobility_summary = con.execute(f"""
    SELECT
        MIN(non_home_hours) AS min_non_home_hours,
        AVG(non_home_hours) AS avg_non_home_hours,
        MAX(non_home_hours) AS max_non_home_hours,

        MIN(unique_nonhome_nontravel_hexes) AS min_unique_nonhome_nontravel_hexes,
        AVG(unique_nonhome_nontravel_hexes) AS avg_unique_nonhome_nontravel_hexes,
        MAX(unique_nonhome_nontravel_hexes) AS max_unique_nonhome_nontravel_hexes,

        MIN(num_location_changes) AS min_location_changes,
        AVG(num_location_changes) AS avg_location_changes,
        MAX(num_location_changes) AS max_location_changes,

        MIN(min_transition_distance_km) AS min_transition_distance_km,
        AVG(avg_transition_distance_km) AS avg_mean_transition_distance_km,
        MAX(max_transition_distance_km) AS max_transition_distance_km,

        MIN(travel_hours) AS min_travel_hours,
        AVG(travel_hours) AS avg_travel_hours,
        MAX(travel_hours) AS max_travel_hours
    FROM {TABLE}
""").df()

print("\n--- Difference Metric Summary ---")

for i in range(0, len(difference_summary.columns), 3):
    group = difference_summary.iloc[:, i:i+3].T
    print(group.to_string(header=False))
    print()

print("\n--- Mobility Metric Summary ---")

for i in range(0, len(mobility_summary.columns), 3):
    group = mobility_summary.iloc[:, i:i+3].T
    print(group.to_string(header=False))
    print()


# ============================================================
# Check schema
# ============================================================
schema = con.execute(f"""
    DESCRIBE {TABLE}
""").df()

print("\n--- DifferenceMetrics Schema ---")
print(schema)

con.close()

elapsed = time.time() - start_time
formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed))

print(f"\nFinished!")
print(f"Database saved to: {DB_PATH}")
print(f"Preview saved to: {HEAD_FILE}")

print(f"\nTotal difference metric runtime: {formatted}")
print(f"Difference metric calculation finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")