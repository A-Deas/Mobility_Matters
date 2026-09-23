import duckdb
import pandas as pd
from pathlib import Path


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DIFFERENCE_DB_PATH = "Data/DuckDB_Files/DifferenceMetrics.db"
DIFFERENCE_TABLE = "DifferenceMetrics"

SIMULATION_DB_PATH = "Data/DuckDB_Files/SimulationResults.db"
SIMULATION_TABLE = "SimulationResults"

OUTPUT_DIR = Path("Health_Impact_Calculations/Results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMP_DIR = "Data/DuckDB_Temp"

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Health outcomes and RR10 values.
#
# The morbidity RR10 values are the LOG-LINEAR estimates reported by Ru et al. (2023).
# Mortality uses the short-term all-cause mortality estimate from Yu et al. (2024).
#
# Each RR10 represents the relative risk associated with a 10 ug/m3 increase in daily mean PM2.5.
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
HEALTH_OUTCOMES = [
    (1, "Asthma emergency room visits", 1.0430),
    (2, "Asthma hospital admissions", 1.0140),
    (3, "Cardiovascular hospital admissions", 1.0100),
    (4, "Respiratory hospital admissions", 1.0135),
    (5, "Overall mortality", 1.0065)
]


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Connect to DuckDB
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
con = duckdb.connect()

con.execute("SET threads=1")
con.execute("SET preserve_insertion_order=false")
con.execute("SET memory_limit='10GB'")
con.execute(f"SET temp_directory='{TEMP_DIR}'")


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Attach databases
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
con.execute(f"ATTACH '{DIFFERENCE_DB_PATH}' AS diff_db (READ_ONLY)")
con.execute(f"ATTACH '{SIMULATION_DB_PATH}' AS sim_db (READ_ONLY)")


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Define the high-difference group.
#
# High-difference agents must meet BOTH of the following population-wide thresholds:
#
#   1. MAPD >= MAPD 98th percentile
#   2. MAD  >= MAD  98th percentile
#
# MAD is stored in DifferenceMetrics as mean_abs_diff.
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
MAPD_P98, MAD_P98 = con.execute(f"""
    SELECT
        QUANTILE_CONT(mapd, 0.98),
        QUANTILE_CONT(mean_abs_diff, 0.98)
    FROM diff_db.{DIFFERENCE_TABLE}
    WHERE mapd IS NOT NULL
      AND mean_abs_diff IS NOT NULL
""").fetchone() # returns the one row as a tuple; MAPD_98 takes the first value of this row/tuple, MAD_P98 takes the second

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE HighDifferenceAgents AS

    SELECT p_id
    FROM diff_db.{DIFFERENCE_TABLE}
    WHERE mapd >= {MAPD_P98}
      AND mean_abs_diff >= {MAD_P98}
""")

high_difference_agents = con.execute("""
    SELECT COUNT(*)
    FROM HighDifferenceAgents
""").fetchone()[0]

print("\n" + "=" * 100)
print("HIGH-DIFFERENCE GROUP")
print("=" * 100)

print(f"\nMAPD P98 threshold: {MAPD_P98:.6f}%")
print(f"MAD P98 threshold: {MAD_P98:.6f} ug/m3")
print(f"High-difference agents: {high_difference_agents:,}")


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calculate DAILY mean static and dynamic exposure for every agent:
#
#     x_bar(i,d,m) = daily mean exposure for agent i on day d under method m
#
# Each agent should contribute seven daily means, one for each day of the simulation week.
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
con.execute(f"""
    CREATE OR REPLACE TEMP TABLE DailyExposure AS

    SELECT
        p_id,
        date,
        COUNT(*) AS hourly_rows,
        AVG(static_exposure) AS daily_static_exposure,
        AVG(dynamic_exposure) AS daily_dynamic_exposure

    FROM sim_db.{SIMULATION_TABLE}

    GROUP BY
        p_id,
        date
""")

bad_daily_row_counts = con.execute("""
    SELECT COUNT(*)
    FROM DailyExposure
    WHERE hourly_rows != 24
""").fetchone()[0]

bad_agent_day_counts = con.execute("""
    SELECT COUNT(*)
    FROM (
        SELECT p_id
        FROM DailyExposure
        GROUP BY p_id
        HAVING COUNT(*) != 7
    )
""").fetchone()[0]

print("\n" + "=" * 100)
print("DAILY EXPOSURE CHECKS")
print("=" * 100)

print(f"\nAgent-days without exactly 24 hourly rows: {bad_daily_row_counts:,}")
print(f"Agents without exactly 7 daily exposure values: {bad_agent_day_counts:,}")


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create group membership.
#
# Every agent contributes to the Full Population group.
# High-difference agents also contribute to the High-Difference group.
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
con.execute(f"""
    CREATE OR REPLACE TEMP TABLE AnalysisGroups AS

    SELECT DISTINCT
        p_id,
        'Full Population' AS agent_group, -- varchar column
        1 AS group_order
    FROM sim_db.{SIMULATION_TABLE}

    UNION ALL

    SELECT
        p_id,
        'High-Difference' AS agent_group, -- varchar column
        2 AS group_order
    FROM HighDifferenceAgents
""")


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create the health-outcome lookup table.
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
con.execute("""
    CREATE OR REPLACE TEMP TABLE HealthOutcomes (
        outcome_order INTEGER,
        health_outcome VARCHAR,
        rr10 DOUBLE
    )
""")

con.executemany(
    "INSERT INTO HealthOutcomes VALUES (?, ?, ?)",
    HEALTH_OUTCOMES
)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calculate modeled attributable fractions.
#
# For each AGENT-DAY and health outcome:
#
#     AF(x) = 1 - RR10^(-x/10)
#
# We calculate AF separately using the daily static exposure and daily dynamic exposure.
# We then average those agent-day AF values within each group:
#
#     bar(AF(G,m)) = (1 / 7N_G) * sum_i sum_d AF(x_bar(i,d,m))
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
results_df = con.execute("""
    WITH agent_day_af AS (

        SELECT
            g.agent_group,
            g.group_order,
            d.p_id,
            d.date,
            h.outcome_order,
            h.health_outcome,
            h.rr10,

            1 - POWER(h.rr10, -d.daily_static_exposure / 10.0) AS af_static,
            1 - POWER(h.rr10, -d.daily_dynamic_exposure / 10.0) AS af_dynamic

        FROM DailyExposure AS d

        JOIN AnalysisGroups AS g
            ON d.p_id = g.p_id

        CROSS JOIN HealthOutcomes AS h
    ),

    group_mean_af AS (

        SELECT
            agent_group,
            group_order,
            outcome_order,
            health_outcome,
            rr10,
            COUNT(DISTINCT p_id) AS num_agents,
            COUNT(*) AS num_agent_days,
            AVG(af_static) AS mean_af_static,
            AVG(af_dynamic) AS mean_af_dynamic

        FROM agent_day_af

        GROUP BY
            agent_group,
            group_order,
            outcome_order,
            health_outcome,
            rr10
    )

    SELECT
        health_outcome,
        rr10,
        agent_group,
        num_agents,
        num_agent_days,
        mean_af_dynamic,
        mean_af_static,
        mean_af_dynamic / mean_af_static AS af_ratio_dynamic_static,
        100 * ((mean_af_dynamic / mean_af_static) - 1) AS percent_difference_vs_static

    FROM group_mean_af

    ORDER BY
        outcome_order,
        group_order
""").df()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create a compact group summary for useful checks and manuscript context.
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
group_summary_df = con.execute("""
    SELECT
        g.agent_group,
        g.group_order,
        COUNT(DISTINCT d.p_id) AS num_agents,
        COUNT(*) AS num_agent_days,
        AVG(d.daily_static_exposure) AS mean_daily_static_exposure,
        AVG(d.daily_dynamic_exposure) AS mean_daily_dynamic_exposure,
        AVG(d.daily_dynamic_exposure - d.daily_static_exposure) AS mean_daily_difference

    FROM DailyExposure AS d

    JOIN AnalysisGroups AS g
        ON d.p_id = g.p_id

    GROUP BY
        g.agent_group,
        g.group_order

    ORDER BY
        g.group_order
""").df()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Save results
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
results_file = OUTPUT_DIR / "modeled_attributable_fractions_results.csv"
group_summary_file = OUTPUT_DIR / "modeled_attributable_fractions_group_summary.csv"

results_df.to_csv(results_file, index=False)
group_summary_df.to_csv(group_summary_file, index=False)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Print results
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("\n" + "=" * 100)
print("GROUP SUMMARY")
print("=" * 100)

print(group_summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print("\n" + "=" * 100)
print("MODELED ATTRIBUTABLE FRACTION RESULTS")
print("=" * 100)

print(results_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Finished
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
con.close()

print("\n" + "=" * 100)
print("FINISHED")
print("=" * 100)

print(f"\nResults saved to:")
print(results_file)
print(group_summary_file)
