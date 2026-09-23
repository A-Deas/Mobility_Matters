import duckdb
import time
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("Logs")
LOG_FILE = LOG_DIR / "simulation_results.log"

TABLE = "SimulationResults"
BATCH_SIZE = 25000
TEST_MODE = False

SYNTHPOP_DB_PATH = "Data/DuckDB_Files/SynthPop_Cleaned.db"
SYNTHPOP_TABLE = "SynthPop_Cleaned"

PM25_DB_PATH = "Data/DuckDB_Files/PM25_Utah.db"
PM25_TABLE = "PM25_Utah"

OUTPUT_DIR = Path("Data/DuckDB_Files")
DB_PATH = OUTPUT_DIR / f"{TABLE}.db"

HEAD_DIR = Path("Dataframe_Heads")
HEAD_FILE = HEAD_DIR / "simulation_results_head.csv"

con = duckdb.connect(str(DB_PATH))

con.execute(f"ATTACH '{SYNTHPOP_DB_PATH}' AS synthpop")
con.execute(f"ATTACH '{PM25_DB_PATH}' AS pm25")

# ============================================================
# Create one home location per agent
# ============================================================
start_time = time.time()

print(f"\nSimulation started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\nCreating home location lookup...")

con.execute(f"""
    CREATE OR REPLACE TABLE HomeLocations AS
    SELECT
        p_id,
        MAX(h3_cell) AS h3_home
    FROM synthpop.{SYNTHPOP_TABLE}
    WHERE act = 'home'
    GROUP BY p_id
""")

print("Home location lookup successfully created.")

# ============================================================
# Check home locations
# ============================================================
multiple_home_agents = con.execute(f"""
    SELECT COUNT(*)
    FROM (
        SELECT p_id
        FROM synthpop.{SYNTHPOP_TABLE}
        WHERE act = 'home'
        GROUP BY p_id
        HAVING COUNT(DISTINCT h3_cell) > 1
    )
""").fetchone()[0]

missing_home_agents = con.execute(f"""
    SELECT COUNT(*)
    FROM (
        SELECT DISTINCT p_id
        FROM synthpop.{SYNTHPOP_TABLE}
    ) AS agents
    LEFT JOIN HomeLocations AS homes USING (p_id)
    WHERE homes.h3_home IS NULL
""").fetchone()[0]

print("\n--- Home Location Check ---")
print(f"Agents with multiple home hexes: {multiple_home_agents:,}")
print(f"Agents missing a home hex: {missing_home_agents:,}")

# ============================================================
# Create agent index for batching
# ============================================================
print("\nCreating agent index...")

con.execute(f"""
    CREATE OR REPLACE TABLE AgentIds AS
    SELECT
        p_id,
        ROW_NUMBER() OVER () AS agent_index
    FROM (
        SELECT DISTINCT p_id
        FROM synthpop.{SYNTHPOP_TABLE}
    )
""")

num_agents = con.execute("""
    SELECT COUNT(*)
    FROM AgentIds
""").fetchone()[0]

print(f"Unique agents: {num_agents:,}")

# ============================================================
# Create empty SimulationResults table
# ============================================================
print(f"\nCreating empty {TABLE} table...")

con.execute(f"""
    CREATE OR REPLACE TABLE {TABLE} (
        p_id VARCHAR,
        agent_type VARCHAR,
        sex VARCHAR,
        age DOUBLE,
        day VARCHAR,
        date DATE,
        day_hour VARCHAR,
        tick BIGINT,
        activity VARCHAR,
        h3_act VARCHAR,
        dynamic_exposure DOUBLE,
        h3_home VARCHAR,
        static_exposure DOUBLE
    )
""")

print(f"{TABLE} structure successfully created.")

# ============================================================
# Construct SimulationResults in batches of agents
# ============================================================
print(f"\nProcessing agents in batches of {BATCH_SIZE:,}...")

for batch_start in range(1, num_agents + 1, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE - 1, num_agents)
    batch_start_time = time.time()

    con.execute(f"""
        INSERT INTO {TABLE}

        SELECT
            s.p_id,
            s.agent_type,
            s.sex,
            s.age,
            s.day,
            DATE '2016-07-25' + CAST(FLOOR(s.tick / 24) AS INTEGER) AS date,
            s.day_hour,
            s.tick,
            s.act AS activity,
            s.h3_cell AS h3_act,
            dynamic_pm.value AS dynamic_exposure,
            h.h3_home,
            static_pm.value AS static_exposure

        FROM synthpop.{SYNTHPOP_TABLE} AS s

        JOIN AgentIds AS a
            ON s.p_id = a.p_id

        JOIN HomeLocations AS h
            ON s.p_id = h.p_id

        LEFT JOIN pm25.{PM25_TABLE} AS dynamic_pm
            ON s.h3_cell = dynamic_pm.h3_polyfill
            AND DATE '2016-07-25' + CAST(FLOOR(s.tick / 24) AS INTEGER) = dynamic_pm.date

        LEFT JOIN pm25.{PM25_TABLE} AS static_pm
            ON h.h3_home = static_pm.h3_polyfill
            AND DATE '2016-07-25' + CAST(FLOOR(s.tick / 24) AS INTEGER) = static_pm.date

        WHERE a.agent_index BETWEEN {batch_start} AND {batch_end}
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
    ORDER BY p_id, tick
    LIMIT 1680
""").df()

preview.to_csv(HEAD_FILE, index=False)
print("Saved dataframe head for preview.")


# ============================================================
# Compare row counts
# ============================================================
print("\nPerforming checks...")

synthpop_rows = con.execute(f"""
    SELECT COUNT(*)
    FROM synthpop.{SYNTHPOP_TABLE}
""").fetchone()[0]

simulation_rows = con.execute(f"""
    SELECT COUNT(*)
    FROM {TABLE}
""").fetchone()[0]

if TEST_MODE:
    expected_rows = min(BATCH_SIZE, num_agents) * 168
    print("\n--- Test Batch Row Count Check ---")
    print(f"Expected test rows: {expected_rows:,}")
    print(f"SimulationResults rows: {simulation_rows:,}")
    print(f"Test row counts match: {expected_rows == simulation_rows}")
else:
    print("\n--- Row Count Check ---")
    print(f"SynthPop_Cleaned rows: {synthpop_rows:,}")
    print(f"SimulationResults rows: {simulation_rows:,}")
    print(f"Row counts match: {synthpop_rows == simulation_rows}")


# ============================================================
# Check agent schedules
# ============================================================
num_agents = con.execute(f"""
    SELECT COUNT(DISTINCT p_id)
    FROM {TABLE}
""").fetchone()[0]

bad_schedule_count = con.execute(f"""
    SELECT COUNT(*)
    FROM (
        SELECT p_id, COUNT(*) AS n
        FROM {TABLE}
        GROUP BY p_id
        HAVING COUNT(*) != 168
    )
""").fetchone()[0]

print("\n--- Agent Schedule Check ---")
print(f"Unique agents: {num_agents:,}")
print(f"Agents without exactly 168 rows: {bad_schedule_count:,}")
print(f"All agents have 168 rows: {bad_schedule_count == 0}")

# ============================================================
# Check exposure joins
# ============================================================
travel_rows = con.execute(f"""
    SELECT COUNT(*)
    FROM {TABLE}
    WHERE activity = 'travel'
""").fetchone()[0]

missing_dynamic = con.execute(f"""
    SELECT COUNT(*)
    FROM {TABLE}
    WHERE dynamic_exposure IS NULL
""").fetchone()[0]

missing_dynamic_travel = con.execute(f"""
    SELECT COUNT(*)
    FROM {TABLE}
    WHERE dynamic_exposure IS NULL
    AND activity = 'travel'
""").fetchone()[0]

missing_dynamic_nontravel = con.execute(f"""
    SELECT COUNT(*)
    FROM {TABLE}
    WHERE dynamic_exposure IS NULL
    AND activity != 'travel'
""").fetchone()[0]

missing_static = con.execute(f"""
    SELECT COUNT(*)
    FROM {TABLE}
    WHERE static_exposure IS NULL
""").fetchone()[0]

print("\n--- Exposure Check ---")
print(f"Total travel rows: {travel_rows:,}")
print(f"Missing dynamic exposures: {missing_dynamic:,}")
print(f"Missing dynamic exposures during travel: {missing_dynamic_travel:,}")
print(f"Missing dynamic exposures outside travel: {missing_dynamic_nontravel:,}")
print(f"All non-travel dynamic exposures matched: {missing_dynamic_nontravel == 0}")

print(f"\nMissing static exposures: {missing_static:,}")
print(f"All static exposures matched: {missing_static == 0}")


# ============================================================
# Check date coverage
# ============================================================
date_check = con.execute(f"""
    SELECT
        MIN(date) AS first_date,
        MAX(date) AS last_date,
        COUNT(DISTINCT date) AS num_dates
    FROM {TABLE}
""").df()

print("\n--- Date Coverage Check ---")
print(date_check)


# ============================================================
# Check schema
# ============================================================
schema = con.execute(f"""
    DESCRIBE {TABLE}
""").df()

print("\n--- SimulationResults Schema ---")
print(schema)

con.close()

print(f"\nFinished! Database saved to: {DB_PATH}")

elapsed = time.time() - start_time
formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed))

print(f"\nTotal simulation runtime: {formatted}")
print(f"Simulation finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
