import duckdb
import time
from datetime import datetime
from pathlib import Path


TABLE = "SimulationResults"
BATCH_SIZE = 5000

SIMULATION_DB_PATH = "Data/DuckDB_Files/SimulationResults.db"
SIMULATION_TABLE = "SimulationResults"

HEAD_DIR = Path("Dataframe_Heads")
HEAD_FILE = HEAD_DIR / "simulation_results_imputed_travel_head.csv"

TEMP_DIR = Path("Data/DuckDB_Temp")


# ================================================================================================================================================
# Connect directly to SimulationResults
#
# We are updating SimulationResults IN PLACE.
# The only rows that will be modified are travel rows where dynamic_exposure is currently NULL.
# ================================================================================================================================================
con = duckdb.connect(SIMULATION_DB_PATH)

con.execute("SET threads=1")
con.execute("SET preserve_insertion_order=false")
con.execute("SET memory_limit='10GB'")
con.execute(f"SET temp_directory='{TEMP_DIR}'")

start_time = time.time()

print(f"\nTravel exposure interpolation started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# Find ONLY agents that actually need travel interpolation
# ============================================================
print("\nCreating travel-agent index...")

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE TravelAgentIds AS

    SELECT
        p_id,
        ROW_NUMBER() OVER (ORDER BY p_id) AS agent_index

    FROM (
        SELECT DISTINCT p_id
        FROM {SIMULATION_TABLE}
        WHERE activity = 'travel'
          AND dynamic_exposure IS NULL
    )
""")

num_travel_agents = con.execute("""
    SELECT COUNT(*)
    FROM TravelAgentIds
""").fetchone()[0]

total_agents = con.execute(f"""
    SELECT COUNT(DISTINCT p_id)
    FROM {SIMULATION_TABLE}
""").fetchone()[0]

print(f"Total agents: {total_agents:,}")
print(f"Agents requiring travel interpolation: {num_travel_agents:,}")
print(f"Agents requiring no interpolation: {total_agents - num_travel_agents:,}")

# ============================================================
# Process ONLY agents containing missing travel exposures
# ============================================================
print(f"\nProcessing travel agents in batches of {BATCH_SIZE:,}...")

for batch_start in range(1, num_travel_agents + 1, BATCH_SIZE):

    batch_end = min(batch_start + BATCH_SIZE - 1, num_travel_agents)

    batch_start_time = time.time()

    # ================================================================================================================
    # Create interpolation values for this batch
    #
    # IMPORTANT: We still load ALL 168 rows for each selected agent.
    # We need the non-travel rows because those provide the origin and destination exposures for interpolation.
    # ================================================================================================================
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE BatchTravelImputations AS

        WITH batch_agents AS (
            SELECT p_id
            FROM TravelAgentIds
            WHERE agent_index BETWEEN {batch_start} AND {batch_end}
        ),

        exposure_bounds AS (

            SELECT
                s.*,

                -- ====================================================
                -- Nearest known exposure looking BACKWARD:
                --
                -- BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW creates a BACKWARD-looking window that allows SQL to look back to the beginning of this agent's schedule (tick 0).
                -- Within that backward-looking window, LAST_VALUE(...) gives the last non-null value.
                -- That's why LAST_VALUE gives us the origin data; it's the last known exposure looking backward from the current travel row.
                -- In plainer English: Give me the nearest known value when looking BACKWARD.
                -- ====================================================

                LAST_VALUE(s.dynamic_exposure IGNORE NULLS) OVER (
                    PARTITION BY s.p_id
                    ORDER BY s.tick
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS previous_exposure,

                LAST_VALUE(CASE WHEN s.dynamic_exposure IS NOT NULL THEN s.tick END IGNORE NULLS) OVER (
                    PARTITION BY s.p_id
                    ORDER BY s.tick
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS previous_tick,

                -- ====================================================
                -- Nearest known exposure looking FORWARD
                --
                -- BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING creates a FORWARD-looking window that allows SQL to look forward to the end of this agent's schedule (tick 167).
                -- Within that forward-looking window, FIRST_VALUE(...) gives the first non-null value.
                -- That's why FIRST_VALUE gives us the destination data; it's the first known exposure looking forward from the current travel row.
                -- In plainer English: Give me the nearest known value when looking FORWARD.
                -- ====================================================

                FIRST_VALUE(s.dynamic_exposure IGNORE NULLS) OVER (
                    PARTITION BY s.p_id
                    ORDER BY s.tick
                    ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
                ) AS next_exposure,

                FIRST_VALUE(CASE WHEN s.dynamic_exposure IS NOT NULL THEN s.tick END IGNORE NULLS) OVER (
                    PARTITION BY s.p_id
                    ORDER BY s.tick
                    ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
                ) AS next_tick,

                -- ====================================================
                -- Cyclic exposure bounds:
                --
                -- Due to the necessary reordering of the activity schedules, there are some edge cases where SCHEDULES begin (ticks 0-3) or end (ticks 164-167) with travel.
                -- BUT the weekly activity schedules are CYCLIC; so to get the proper endpoints for interpolation, all we need do is cross the tick 167 -> tick 0 or tick 0 -> tick 167 boundaries.
                -- It's annoying but there is no methodological problem here at all; these are the original endpoints (before reordering) anyway!
                -- ====================================================

                -- This is the very first non-null exposure in the entire simulation
                FIRST_VALUE(s.dynamic_exposure IGNORE NULLS) OVER (
                    PARTITION BY s.p_id
                    ORDER BY s.tick
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) AS first_exposure,

                -- This is the very first TICK with non-null exposure in the entire simulation
                FIRST_VALUE(CASE WHEN s.dynamic_exposure IS NOT NULL THEN s.tick END IGNORE NULLS) OVER (
                    PARTITION BY s.p_id
                    ORDER BY s.tick
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) AS first_tick,

                -- This is the very last non-null exposure in the entire simulation
                LAST_VALUE(s.dynamic_exposure IGNORE NULLS) OVER (
                    PARTITION BY s.p_id
                    ORDER BY s.tick
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) AS last_exposure,

                -- This is the very last TICK with non-null exposure in the entire simulation
                LAST_VALUE(CASE WHEN s.dynamic_exposure IS NOT NULL THEN s.tick END IGNORE NULLS) OVER (
                    PARTITION BY s.p_id
                    ORDER BY s.tick
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) AS last_tick

            FROM {SIMULATION_TABLE} AS s
            JOIN batch_agents AS b
                ON s.p_id = b.p_id
        ),

        cyclic_bounds AS (

            SELECT
                *,

                -- For normal interior travel, use the ordinary previous exposure.
                -- If travel occurs at the beginning of the schedule (ticks 0-3) and there is no previous exposure, wrap BACKWARD to the last known exposure at the END of the weekly schedule.

                COALESCE(previous_exposure, last_exposure) AS interpolation_origin_exposure,

                -- For the TICKS we need to keep things LINEAR, increasing by increments of 1!
                -- interpolation_origin_tick takes values in: -4, -3, -2, -1 (leading up to the beginning ticks: 0, 1, 2, 3)

                CASE
                    WHEN previous_exposure IS NOT NULL THEN previous_tick
                    ELSE last_tick - 168
                END AS interpolation_origin_tick,

                -- For normal interior travel, use the ordinary next exposure.
                -- If travel occurs at the end of the schedule and there is no next exposure, wrap FORWARD to the first known exposure at the BEGINNING of the weekly schedule.

                COALESCE(next_exposure, first_exposure) AS interpolation_destination_exposure,

                -- For the TICKS we need to keep things LINEAR, increasing by increments of 1!
                -- interpolation_destination_tick takes values in: 168, 169, 170, 171 (extending the ending ticks: 164, 165, 166, 167)

                CASE
                    WHEN next_exposure IS NOT NULL THEN next_tick
                    ELSE first_tick + 168
                END AS interpolation_destination_tick

            FROM exposure_bounds
        )

        SELECT
            p_id,
            agent_type,
            sex,
            age,
            day,
            date,
            day_hour,
            tick,
            activity,
            h3_act,

            -- Linear interpolation equation.
            -- The way we've handled things above allows the same interpolation equation to operate across the cyclic weekly boundary:

            CASE
                WHEN activity = 'travel' AND dynamic_exposure IS NULL THEN
                
                    interpolation_origin_exposure +                                                                                 -- starting point
                    ((tick - interpolation_origin_tick) * 1.0 / (interpolation_destination_tick - interpolation_origin_tick)) *     -- increment
                    (interpolation_destination_exposure - interpolation_origin_exposure)                                            -- how much we need to add or subtract

                ELSE dynamic_exposure
            END AS imputed_dynamic_exposure,

            h3_home,
            static_exposure

        FROM cyclic_bounds

        -- ========================================================
        -- We ONLY need to save the rows that actually need repair.
        -- All non-travel dynamic exposures are already correct.
        -- ========================================================

        WHERE activity = 'travel'
          AND dynamic_exposure IS NULL
    """)

    # ========================================================
    # Update ONLY those travel rows in SimulationResults
    #
    # Existing non-travel exposures are untouched.
    # Existing non-null dynamic exposures are untouched.
    # ========================================================
    con.execute(f"""
        UPDATE {TABLE} AS output

        SET dynamic_exposure = imputed.imputed_dynamic_exposure

        FROM BatchTravelImputations AS imputed
        WHERE output.p_id = imputed.p_id
          AND output.tick = imputed.tick
          AND output.activity = 'travel'
          AND output.dynamic_exposure IS NULL
    """)

    batch_elapsed = time.time() - batch_start_time

    batch_formatted = time.strftime("%H:%M:%S", time.gmtime(batch_elapsed))

    print(f"Processed travel agents {batch_start:,} - {batch_end:,} of {num_travel_agents:,} | Batch runtime: {batch_formatted}", flush=True)

print("\nTravel interpolation successfully completed.")

# ============================================================
# Save a preview
# ============================================================
print("\nSaving a preview...")

preview = con.execute(f"""
    SELECT *
    FROM {TABLE}
    ORDER BY p_id, tick
    LIMIT 1680
""").df()

preview.to_csv(HEAD_FILE, index=False)

print("Saved dataframe head for preview.")

# ============================================================
# Check travel interpolation
# ============================================================
travel_rows = con.execute(f"""
    SELECT COUNT(*)
    FROM {TABLE}
    WHERE activity = 'travel'
""").fetchone()[0]

missing_travel_exposure = con.execute(f"""
    SELECT COUNT(*)
    FROM {TABLE}
    WHERE activity = 'travel'
      AND dynamic_exposure IS NULL
""").fetchone()[0]

missing_nontravel_exposure = con.execute(f"""
    SELECT COUNT(*)
    FROM {TABLE}
    WHERE activity != 'travel'
      AND dynamic_exposure IS NULL
""").fetchone()[0]

missing_dynamic = con.execute(f"""
    SELECT COUNT(*)
    FROM {TABLE}
    WHERE dynamic_exposure IS NULL
""").fetchone()[0]

print("\n--- Travel Exposure Check ---")
print(f"Travel rows: {travel_rows:,}")
print(f"Travel rows still missing dynamic exposure: {missing_travel_exposure:,}")
print(f"Non-travel rows still missing dynamic exposure: {missing_nontravel_exposure:,}")
print(f"Total rows missing dynamic exposure: {missing_dynamic:,}")
print(f"All travel exposures successfully interpolated: {missing_travel_exposure == 0}")
print(f"All non-travel exposures complete: {missing_nontravel_exposure == 0}")
print(f"All dynamic exposures complete: {missing_dynamic == 0}")

# =========================================================================================
# Check interpolation bounds
#
# Only agents that required travel interpolation are checked.
# There is no reason to run these window functions over all 3+ million agents.
# =========================================================================================
bad_interpolations = con.execute(f"""
    WITH travel_bounds AS (

        SELECT
            s.*,

            LAST_VALUE(CASE WHEN activity != 'travel' THEN dynamic_exposure END IGNORE NULLS) OVER (
                PARTITION BY s.p_id
                ORDER BY tick
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS previous_exposure,

            FIRST_VALUE(CASE WHEN activity != 'travel' THEN dynamic_exposure END IGNORE NULLS) OVER (
                PARTITION BY s.p_id
                ORDER BY tick
                ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
            ) AS next_exposure,

            FIRST_VALUE(CASE WHEN activity != 'travel' THEN dynamic_exposure END IGNORE NULLS) OVER (
                PARTITION BY s.p_id
                ORDER BY tick
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            ) AS first_exposure,

            LAST_VALUE(CASE WHEN activity != 'travel' THEN dynamic_exposure END IGNORE NULLS) OVER (
                PARTITION BY s.p_id
                ORDER BY tick
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            ) AS last_exposure

        FROM {TABLE} AS s

        JOIN TravelAgentIds AS t
            ON s.p_id = t.p_id
    ),

    cyclic_bounds AS (

        SELECT
            *,
            COALESCE(previous_exposure, last_exposure) AS interpolation_origin_exposure,
            COALESCE(next_exposure, first_exposure) AS interpolation_destination_exposure
        FROM travel_bounds
    )

    SELECT COUNT(*)
    FROM cyclic_bounds
    WHERE activity = 'travel'
      AND dynamic_exposure NOT BETWEEN
          LEAST(interpolation_origin_exposure, interpolation_destination_exposure)
          AND
          GREATEST(interpolation_origin_exposure, interpolation_destination_exposure)
""").fetchone()[0]

print("\n--- Interpolation Bounds Check ---")
print(f"Travel exposures outside origin/destination exposure range: {bad_interpolations:,}")
print(f"All interpolated exposures fall between their endpoints: {bad_interpolations == 0}")

# ============================================================
# Check agent schedules
# ============================================================
bad_schedule_count = con.execute(f"""
    SELECT COUNT(*)

    FROM (
        SELECT
            p_id,
            COUNT(*) AS n

        FROM {TABLE}
        GROUP BY p_id
        HAVING COUNT(*) != 168
    )
""").fetchone()[0]

print("\n--- Agent Schedule Check ---")
print(f"Agents without exactly 168 rows: {bad_schedule_count:,}")
print(f"All agents retain 168 rows: {bad_schedule_count == 0}")

# ====================================================================================================================================
# Check static exposures
#
# Nothing in this program updates static_exposure, so this is simply checking that none are unexpectedly missing.
# ====================================================================================================================================
missing_static_exposure = con.execute(f"""
    SELECT COUNT(*)
    FROM {TABLE}
    WHERE static_exposure IS NULL
""").fetchone()[0]

print("\n--- Static Exposure Check ---")
print(f"Rows missing static exposure: {missing_static_exposure:,}")
print(f"All static exposures complete: {missing_static_exposure == 0}")

# ============================================================
# Check schema
# ============================================================
schema = con.execute(f"""
    DESCRIBE {TABLE}
""").df()

print("\n--- SimulationResults Schema ---")
print(schema)

con.close()

# ============================================================
# Runtime
# ============================================================
elapsed = time.time() - start_time

formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed))

print(f"\nFinished! Database updated in place: {SIMULATION_DB_PATH}")
print(f"Preview saved to: {HEAD_FILE}")

print(f"\nTotal interpolation runtime: {formatted}")
print(f"Travel exposure interpolation finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")