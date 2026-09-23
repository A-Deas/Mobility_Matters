import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

DIFFERENCE_DB_PATH = "Data/DuckDB_Files/DifferenceMetrics.db"
DIFFERENCE_TABLE = "DifferenceMetrics"

OUTPUT_DIR = Path("Mobility_Agent_Type_Comparison/Results")
PLOT_FILE = OUTPUT_DIR / "mobility_agent_type_comparison.pdf"
SUMMARY_FILE = OUTPUT_DIR / "mobility_agent_type_summary.csv"

MAPD_PERCENTILE = 0.98
MAD_PERCENTILE = 0.98

con = duckdb.connect(DIFFERENCE_DB_PATH, read_only=True)

con.execute("SET threads=1")
con.execute("SET preserve_insertion_order=false")
con.execute("SET memory_limit='10GB'")

MAPD_P98 = con.execute(f"""
    SELECT QUANTILE_CONT(mapd, {MAPD_PERCENTILE})
    FROM {DIFFERENCE_TABLE}
    WHERE ISFINITE(mapd) AND ISFINITE(mean_abs_diff)
""").fetchone()[0]

MAD_P98 = con.execute(f"""
    SELECT QUANTILE_CONT(mean_abs_diff, {MAD_PERCENTILE})
    FROM {DIFFERENCE_TABLE}
    WHERE ISFINITE(mapd) AND ISFINITE(mean_abs_diff)
""").fetchone()[0]

TOTAL_AGENTS = con.execute(f"""
    SELECT COUNT(*)
    FROM {DIFFERENCE_TABLE}
""").fetchone()[0]

HIGH_DIFFERENCE_AGENTS = con.execute(f"""
    SELECT COUNT(*)
    FROM {DIFFERENCE_TABLE}
    WHERE mapd >= {MAPD_P98} AND mean_abs_diff >= {MAD_P98}
""").fetchone()[0]

DIFF_METRICS = ["mpd", "mapd", "mean_abs_diff"]
MOBILITY_METRICS = [
    "non_home_hours",
    "avg_transition_distance_km",
    # "unique_nonhome_nontravel_hexes",
    # "num_location_changes",  # Optional fourth mobility panel
]
PLOT_METRICS = DIFF_METRICS + MOBILITY_METRICS

# ============================================================
# Sanity check
# ============================================================
print("\n" + "=" * 100)
print("POPULATION")
print("=" * 100)

print(f"\nTotal agents: {TOTAL_AGENTS:,}")
print(f"MAPD P98 threshold: {MAPD_P98:.6f}%")
print(f"MAD P98 threshold: {MAD_P98:.6f} µg/m³")
print(f"High-difference agents: {HIGH_DIFFERENCE_AGENTS:,}")

# ============================================================
# Plot labels and colors
# ============================================================
metric_labels = {
    "mpd": "MPD (%)",
    "mapd": "MAPD (%)",
    "mean_abs_diff": r"MAD ($\mu$g/m$^3$)",
    "non_home_hours": "Total weekly\nnon-home hours",
    "avg_transition_distance_km": "Average transition distance between\nconsecutive activity locations (km)",
    "unique_nonhome_nontravel_hexes": "Unique non-home\nH3 cells (count/week)",
    "num_location_changes": "Location changes\n(count/week)",
}

AGENT_TYPE_MAP = {
    "worker_only": "Worker",
    "worker_student": "Working student",
    "student__age_under05": "Student (aged under 5)",
    "student__age05_09": "Student (aged 5 to 9)",
    "student__age10_15": "Student (aged 10 to 15)",
    "student__lf_elig": "Student (aged 16+)",
    "childcare": "Childcare attendee",
    "youth__age_under05": "Youth (aged under 5)",
    "youth__age05_09": "Youth (aged 5 to 9)",
    "youth__age10_15": "Youth (aged 10 to 15)",
    "unemp__seeking_work": "Unemployed",
    "retired": "Retired",
    "homemaker": "Homemaker",
    "other": "Other"
}


def format_agent_type(agent_type):
    return AGENT_TYPE_MAP.get(
        agent_type,
        str(agent_type).replace("__", ": ").replace("_", " ").title()
    )

# Colors describe ROWS, not metric columns.
# All agent types are purple; the pooled high-difference row is green.
IQR_COLOR = "#A78BC1"
MEDIAN_COLOR = "#5D4777"
HIGH_IQR_COLOR = "#9DB79A"
HIGH_MEDIAN_COLOR = "#3F6549"

# =============================================================================================
# Summarize difference and mobility metrics by agent type
#
# Also compute the percentage of the high-difference group made up each agent type
# =============================================================================================
print("\nSummarizing exposure differences by agent type...")

summary_df = con.execute(f"""
    SELECT
        agent_type,
        COUNT(*) AS agent_count,
        COUNT(*) FILTER (WHERE mapd >= {MAPD_P98} AND mean_abs_diff >= {MAD_P98}) AS high_difference_count,

        QUANTILE_CONT(mpd, 0.05) AS mpd_p05,
        QUANTILE_CONT(mpd, 0.25) AS mpd_p25,
        QUANTILE_CONT(mpd, 0.50) AS mpd_median,
        QUANTILE_CONT(mpd, 0.75) AS mpd_p75,
        QUANTILE_CONT(mpd, 0.95) AS mpd_p95,

        QUANTILE_CONT(mapd, 0.05) AS mapd_p05,
        QUANTILE_CONT(mapd, 0.25) AS mapd_p25,
        QUANTILE_CONT(mapd, 0.50) AS mapd_median,
        QUANTILE_CONT(mapd, 0.75) AS mapd_p75,
        QUANTILE_CONT(mapd, 0.95) AS mapd_p95,

        QUANTILE_CONT(mean_abs_diff, 0.05) AS mean_abs_diff_p05,
        QUANTILE_CONT(mean_abs_diff, 0.25) AS mean_abs_diff_p25,
        QUANTILE_CONT(mean_abs_diff, 0.50) AS mean_abs_diff_median,
        QUANTILE_CONT(mean_abs_diff, 0.75) AS mean_abs_diff_p75,
        QUANTILE_CONT(mean_abs_diff, 0.95) AS mean_abs_diff_p95,


        QUANTILE_CONT(non_home_hours, 0.05) AS non_home_hours_p05,
        QUANTILE_CONT(non_home_hours, 0.25) AS non_home_hours_p25,
        QUANTILE_CONT(non_home_hours, 0.50) AS non_home_hours_median,
        QUANTILE_CONT(non_home_hours, 0.75) AS non_home_hours_p75,
        QUANTILE_CONT(non_home_hours, 0.95) AS non_home_hours_p95,

        QUANTILE_CONT(avg_transition_distance_km, 0.05) AS avg_transition_distance_km_p05,
        QUANTILE_CONT(avg_transition_distance_km, 0.25) AS avg_transition_distance_km_p25,
        QUANTILE_CONT(avg_transition_distance_km, 0.50) AS avg_transition_distance_km_median,
        QUANTILE_CONT(avg_transition_distance_km, 0.75) AS avg_transition_distance_km_p75,
        QUANTILE_CONT(avg_transition_distance_km, 0.95) AS avg_transition_distance_km_p95,

        QUANTILE_CONT(unique_nonhome_nontravel_hexes, 0.05) AS unique_nonhome_nontravel_hexes_p05,
        QUANTILE_CONT(unique_nonhome_nontravel_hexes, 0.25) AS unique_nonhome_nontravel_hexes_p25,
        QUANTILE_CONT(unique_nonhome_nontravel_hexes, 0.50) AS unique_nonhome_nontravel_hexes_median,
        QUANTILE_CONT(unique_nonhome_nontravel_hexes, 0.75) AS unique_nonhome_nontravel_hexes_p75,
        QUANTILE_CONT(unique_nonhome_nontravel_hexes, 0.95) AS unique_nonhome_nontravel_hexes_p95,

        QUANTILE_CONT(num_location_changes, 0.05) AS num_location_changes_p05,
        QUANTILE_CONT(num_location_changes, 0.25) AS num_location_changes_p25,
        QUANTILE_CONT(num_location_changes, 0.50) AS num_location_changes_median,
        QUANTILE_CONT(num_location_changes, 0.75) AS num_location_changes_p75,
        QUANTILE_CONT(num_location_changes, 0.95) AS num_location_changes_p95
    FROM {DIFFERENCE_TABLE}

    -- Retain all categories for accounting; remove gq_inst only from the figure.
    -- Each quantile ignores NULL values in its own metric.
    GROUP BY agent_type
""").df()

# ============================================================
# Summarize the pooled high-difference group
# ============================================================
high_difference_df = con.execute(f"""
    SELECT
        'high_difference' AS agent_type,
        COUNT(*) AS agent_count,
        COUNT(*) AS high_difference_count,
        QUANTILE_CONT(mpd, 0.05) AS mpd_p05,
        QUANTILE_CONT(mpd, 0.25) AS mpd_p25,
        QUANTILE_CONT(mpd, 0.50) AS mpd_median,
        QUANTILE_CONT(mpd, 0.75) AS mpd_p75,
        QUANTILE_CONT(mpd, 0.95) AS mpd_p95,

        QUANTILE_CONT(mapd, 0.05) AS mapd_p05,
        QUANTILE_CONT(mapd, 0.25) AS mapd_p25,
        QUANTILE_CONT(mapd, 0.50) AS mapd_median,
        QUANTILE_CONT(mapd, 0.75) AS mapd_p75,
        QUANTILE_CONT(mapd, 0.95) AS mapd_p95,

        QUANTILE_CONT(mean_abs_diff, 0.05) AS mean_abs_diff_p05,
        QUANTILE_CONT(mean_abs_diff, 0.25) AS mean_abs_diff_p25,
        QUANTILE_CONT(mean_abs_diff, 0.50) AS mean_abs_diff_median,
        QUANTILE_CONT(mean_abs_diff, 0.75) AS mean_abs_diff_p75,
        QUANTILE_CONT(mean_abs_diff, 0.95) AS mean_abs_diff_p95,


        QUANTILE_CONT(non_home_hours, 0.05) AS non_home_hours_p05,
        QUANTILE_CONT(non_home_hours, 0.25) AS non_home_hours_p25,
        QUANTILE_CONT(non_home_hours, 0.50) AS non_home_hours_median,
        QUANTILE_CONT(non_home_hours, 0.75) AS non_home_hours_p75,
        QUANTILE_CONT(non_home_hours, 0.95) AS non_home_hours_p95,

        QUANTILE_CONT(avg_transition_distance_km, 0.05) AS avg_transition_distance_km_p05,
        QUANTILE_CONT(avg_transition_distance_km, 0.25) AS avg_transition_distance_km_p25,
        QUANTILE_CONT(avg_transition_distance_km, 0.50) AS avg_transition_distance_km_median,
        QUANTILE_CONT(avg_transition_distance_km, 0.75) AS avg_transition_distance_km_p75,
        QUANTILE_CONT(avg_transition_distance_km, 0.95) AS avg_transition_distance_km_p95,

        QUANTILE_CONT(unique_nonhome_nontravel_hexes, 0.05) AS unique_nonhome_nontravel_hexes_p05,
        QUANTILE_CONT(unique_nonhome_nontravel_hexes, 0.25) AS unique_nonhome_nontravel_hexes_p25,
        QUANTILE_CONT(unique_nonhome_nontravel_hexes, 0.50) AS unique_nonhome_nontravel_hexes_median,
        QUANTILE_CONT(unique_nonhome_nontravel_hexes, 0.75) AS unique_nonhome_nontravel_hexes_p75,
        QUANTILE_CONT(unique_nonhome_nontravel_hexes, 0.95) AS unique_nonhome_nontravel_hexes_p95,

        QUANTILE_CONT(num_location_changes, 0.05) AS num_location_changes_p05,
        QUANTILE_CONT(num_location_changes, 0.25) AS num_location_changes_p25,
        QUANTILE_CONT(num_location_changes, 0.50) AS num_location_changes_median,
        QUANTILE_CONT(num_location_changes, 0.75) AS num_location_changes_p75,
        QUANTILE_CONT(num_location_changes, 0.95) AS num_location_changes_p95
    FROM {DIFFERENCE_TABLE}
    WHERE mapd >= {MAPD_P98} AND mean_abs_diff >= {MAD_P98}
""").df()

# gq_inst remains in the population-wide thresholds and pooled group.
omitted_df = summary_df[~summary_df["agent_type"].isin(AGENT_TYPE_MAP)].copy()
print("\nCategories omitted from the 14 agent-type rows:")
print(omitted_df[["agent_type", "agent_count", "high_difference_count"]].to_string(index=False))

assert summary_df["agent_count"].sum() == TOTAL_AGENTS
assert summary_df["high_difference_count"].sum() == HIGH_DIFFERENCE_AGENTS
assert HIGH_DIFFERENCE_AGENTS > 0

summary_df["share_of_high_difference_group_pct"] = (100 * summary_df["high_difference_count"] / HIGH_DIFFERENCE_AGENTS)
summary_df["high_difference_within_type_pct"] = (100 * summary_df["high_difference_count"] / summary_df["agent_count"])

summary_df = summary_df[summary_df["agent_type"].isin(AGENT_TYPE_MAP)].copy()
summary_df["type_order"] = summary_df["agent_type"].map({agent_type: i for i, agent_type in enumerate(AGENT_TYPE_MAP)})
summary_df = summary_df.sort_values(["share_of_high_difference_group_pct", "type_order"],ascending=[False, True]).reset_index(drop=True)
assert len(summary_df) == 14

num_agent_types = len(summary_df)
summary_df = pd.concat([summary_df, high_difference_df], ignore_index=True)
summary_df.to_csv(SUMMARY_FILE, index=False)

# ============================================================
# Median-and-IQR plotting function
# Horizontal line: Q1 to Q3. Vertical marker: median.
# ============================================================
def plot_metric_panel(ax, metric, y_positions, panel_df, scale_df):
    for row_number, row in panel_df.iterrows():
        p25 = row[f"{metric}_p25"]
        median = row[f"{metric}_median"]
        p75 = row[f"{metric}_p75"]
        y = y_positions[row_number]

        if row["agent_type"] == "high_difference":
            iqr_color = HIGH_IQR_COLOR
            median_color = HIGH_MEDIAN_COLOR
        else:
            iqr_color = IQR_COLOR
            median_color = MEDIAN_COLOR

        ax.hlines(y, p25, p75, color=iqr_color, linewidth=5, zorder=2)
        ax.vlines(median, y - 0.20, y + 0.20, color=median_color, linewidth=1.5, zorder=3)

    # Exposure scales use the current section; mobility scales use all rows.
    x_min = min(0, scale_df[f"{metric}_p25"].min())
    x_max = max(0, scale_df[f"{metric}_p75"].max())
    x_range = max(x_max - x_min, 0.1)
    ax.set_xlim(x_min - 0.07 * x_range, x_max + 0.09 * x_range)

    if metric == "mpd":
        ax.axvline(0, color="#B4ACBB", linewidth=0.8, linestyle=":", zorder=0)

    ax.set_title(metric_labels[metric], fontsize=10, pad=12)
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#C9C4CC")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", length=0)

# ============================================================
# Create label column, composition column, and metric panels
# ============================================================
print("\nCreating combined comparison figure...")

agent_type_df = summary_df[summary_df["agent_type"] != "high_difference"].reset_index(drop=True)
high_group_df = summary_df[summary_df["agent_type"] == "high_difference"].reset_index(drop=True)
y_positions = np.arange(len(agent_type_df))

fig, panel_axes = plt.subplots(
    2, 2 + len(PLOT_METRICS), figsize=(17.5, 9.5),
    gridspec_kw={
        "width_ratios": [1, 0.75] + [1.0] * len(DIFF_METRICS) + [1.1] * len(MOBILITY_METRICS),
        "height_ratios": [14, 1.4],
        "wspace": 0.18,
        "hspace": 0.12
    }
)
fig.subplots_adjust(left=0.025, right=0.985, top=0.84, bottom=0.23)
axes = panel_axes[0]             # Purple agent-type section
high_axes = panel_axes[1]        # Green high-difference section

for ax in axes:
    ax.set_ylim(num_agent_types - 0.35, -0.65)
    ax.set_yticks([])
    for row_number in range(num_agent_types):
        ax.axhspan(row_number - 0.48, row_number + 0.48, color="#F0EAF5", zorder=-3)
        # if row_number % 2 == 0:
        #     ax.axhspan(row_number - 0.48, row_number + 0.48, color="#F0EAF5", zorder=-3)

for ax in high_axes:
    ax.set_ylim(0.65, -0.65)
    ax.set_yticks([])
    ax.axhspan(-0.65, 0.65, color="#F0F5EE", zorder=-3)

# Agent names and composition percentages are text-only columns.
for ax in list(axes[:2]) + list(high_axes[:2]):
    ax.set_xlim(0, 1)
    ax.axis("off")

axes[0].set_title("Agent type", loc="left", fontsize=11, fontweight="bold", color=MEDIAN_COLOR, pad=17)
axes[1].set_title("Percentage of high-\ndifference group", fontsize=10, pad=12)

for row_number, row in agent_type_df.iterrows():
    axes[0].text(0.08, row_number, format_agent_type(row["agent_type"]), va="center", fontsize=10, color="#3C3048")
    axes[1].text(0.5, row_number, f"{row['share_of_high_difference_group_pct']:.2f}% (n = {int(row['high_difference_count']):,})", ha="center", va="center", fontsize=9, color=MEDIAN_COLOR)

high_axes[0].text(0.08, 0, "High-difference agents", va="center", fontsize=10, color=HIGH_MEDIAN_COLOR, fontweight="bold")
high_axes[1].text(0.5, 0, f"n = {HIGH_DIFFERENCE_AGENTS:,}", ha="center", va="center", fontsize=9, color=HIGH_MEDIAN_COLOR)

for ax, high_ax, metric in zip(axes[2:], high_axes[2:], PLOT_METRICS):
    plot_metric_panel(
        ax, metric, y_positions,
        agent_type_df, agent_type_df
    )

    plot_metric_panel(
        high_ax, metric, np.array([0]),
        high_group_df, high_group_df
    )

    high_ax.set_title("")

# ============================================================
# Group headers and separator
# ============================================================
first_mobility_panel = 2 + len(DIFF_METRICS)

for heading, first_panel, last_panel in [
    ("Difference metrics", 2, first_mobility_panel - 1),
    ("Mobility measures", first_mobility_panel, len(axes) - 1)
]:
    left = axes[first_panel].get_position().x0
    right = axes[last_panel].get_position().x1
    fig.text((left + right) / 2, 0.917, heading, ha="center", fontsize=13, fontweight="bold", color="#40374A")
    fig.add_artist(Line2D([left, right], [0.90, 0.90], transform=fig.transFigure, color="#CCC5D3", linewidth=1))

separator_x = (axes[first_mobility_panel - 1].get_position().x1 + axes[first_mobility_panel].get_position().x0) / 2

fig.add_artist(Line2D([separator_x, separator_x], [0.215, 0.90], transform=fig.transFigure, color="#CCC5D3", linewidth=1))

# fig.suptitle("Exposure differences and mobility across agent types", fontsize=17, fontweight="bold", y=0.982)

# ============================================================
# Save and finish
# ============================================================
fig.savefig(PLOT_FILE, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
con.close()
print(f"\nPlot saved to: {PLOT_FILE}")
print(f"Summary saved to: {SUMMARY_FILE}")
