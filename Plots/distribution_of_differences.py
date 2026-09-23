import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


# ========================================================================================
# Config
# ========================================================================================
DIFFERENCE_DB_PATH = "Data/DuckDB_Files/DifferenceMetrics.db"
DIFFERENCE_TABLE = "DifferenceMetrics"

HEAD_DIR = Path("Dataframe_Heads")
TOP_HEAD_FILE = HEAD_DIR / "top_mapd_agents.csv"
LOW_HEAD_FILE = HEAD_DIR / "low_mapd_agents.csv"

PLOT_DIR = Path("Plots/Difference_Metric_Distributions")

PLOT_MAX = 25
MAD_PLOT_MAX = 3


# ========================================================================================
# Load agent metrics
# ========================================================================================
def load_agent_metrics():
    con = duckdb.connect(DIFFERENCE_DB_PATH, read_only=True)

    agent_df = con.execute(f"""
        SELECT *
        FROM {DIFFERENCE_TABLE}
    """).df()

    con.close()

    print(f"\nTotal agents: {len(agent_df):,}")

    return agent_df


# ================================================================================================================================================================================
# Save the highest-MAPD agents
# ================================================================================================================================================================================
def save_top_mapd_agents(agent_df):
    top_agents = agent_df.sort_values("mapd", ascending=False).head(150)

    columns = [
        "p_id",
        "agent_type",
        "sex",
        "age",
        "mean_raw_diff",
        "mean_abs_diff",
        "max_abs_diff",
        "mpd",
        "mapd",
        "non_home_hours",
        "unique_nonhome_nontravel_hexes",
        "num_location_changes",
        "min_transition_distance_km",
        "avg_transition_distance_km",
        "max_transition_distance_km",
        "travel_hours"
    ]

    top_agents.to_csv(TOP_HEAD_FILE, index=False)

    print("\nSaved data for agents with highest 150 highest MAPD scores.")

# ========================================================================================
# Save the lowest-MAPD agents
# ========================================================================================
def save_low_mapd_agents(agent_df):
    top_agents = agent_df.sort_values("mapd", ascending=True).head(150)

    columns = [
        "p_id",
        "agent_type",
        "sex",
        "age",
        "mean_raw_diff",
        "mean_abs_diff",
        "max_abs_diff",
        "mpd",
        "mapd",
        "non_home_hours",
        "unique_nonhome_nontravel_hexes",
        "num_location_changes",
        "min_transition_distance_km",
        "avg_transition_distance_km",
        "max_transition_distance_km",
        "travel_hours"
    ]

    top_agents.to_csv(LOW_HEAD_FILE, index=False)

    print("\nSaved data for agents with lowest 150 MAPD scores.")


# ========================================================================================
# Plot MPD distribution
# ========================================================================================
def plot_mpd_distribution(agent_df):
    num_nans = agent_df["mpd"].isna().sum()
    data = agent_df["mpd"].dropna().values

    # Compute summary statistics using ALL agents
    min_val = np.min(data)
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    p98 = np.percentile(data, 98)
    max_val = np.max(data)
    mean_val = np.mean(data)
    iqr = q3 - q1

    n_total = len(data)
    n_negative = np.sum(data < 0)
    n_zero = np.sum(data == 0)
    n_positive = np.sum(data > 0)

    # Count agents above the 98th percentile
    n_total = len(data)
    n_gt_p98 = np.sum(data > p98)
    n_ge_p98 = np.sum(data >= p98)

    print("\n--- MPD ---")
    print(f"Min: {min_val:.3f}")
    print(f"Q1 (25th percentile): {q1:.3f}")
    print(f"Median (Q2): {q2:.3f}")
    print(f"Mean: {mean_val:.3f}")
    print(f"Q3 (75th percentile): {q3:.3f}")
    print(f"P98 (98th percentile): {p98:.3f}")
    print(f"Max: {max_val:.3f}")
    print(f"IQR: {iqr:.3f}")
    print(f"Negative MPD: {n_negative:,} ({n_negative/n_total:.2%})")
    print(f"Zero MPD: {n_zero:,} ({n_zero/n_total:.2%})")
    print(f"Positive MPD: {n_positive:,} ({n_positive/n_total:.2%})")
    print(f"NaN values: {num_nans:,}")
    print(f"\nAgents >= P98: {n_ge_p98:,} ({n_ge_p98/n_total:.2%})")

    # Limit only the plotted distribution, not the summary statistics
    plot_data = data[(data >= -PLOT_MAX) & (data <= PLOT_MAX)]
    weights = np.ones(len(plot_data)) / 1_000_000

    plt.figure(figsize=(10, 6))

    plt.hist(plot_data, bins=100, weights=weights, color="skyblue", edgecolor="black", alpha=0.6)

    plt.axvline(q1, color="blue", linestyle="None", label=f"Q1 = {q1:.2f}%")
    # plt.axvline(q2, color="salmon", linestyle="None", label=f"Median = {q2:.2f}%")
    plt.axvline(q3, color="orange", linestyle="None", label=f"Q3 = {q3:.2f}%")
    # plt.axvline(p98, color="red", linestyle="None", label=f"P98 = {p98:.2f}%")
    plt.axvline(0, color="grey", linestyle="None")

    # plt.axvspan(-PLOT_MAX, 0, color="blue", alpha=0.1, label="Negative differences")
    # plt.axvspan(0, PLOT_MAX, color="darkorange", alpha=0.1, label="Positive differences")

    plt.xlabel("Mean percentage difference (%)", fontsize=16)
    plt.ylabel("Number of agents (millions)", fontsize=16)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    min_handle = Line2D([0], [0], color="green", linestyle="None", label=f"Min = {min_val:.2f}%")
    max_handle = Line2D([0], [0], color="darkred", linestyle="None", label=f"Max = {max_val:.2f}%")

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.insert(0, min_handle)
    handles.append(max_handle)

    plt.legend(
        handles=handles,
        loc="upper right",
        fontsize=14,
        handlelength=0,
        handletextpad=0,
        borderpad=0.4,
        labelspacing=0.35
    )

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mpd_histogram.png", dpi=300, bbox_inches="tight")
    plt.show()


# ========================================================================================
# Plot MAPD distribution
# ========================================================================================
def plot_mapd_distribution(agent_df):
    num_nans = agent_df["mapd"].isna().sum()
    data = agent_df["mapd"].dropna().values

    # Compute summary statistics using ALL agents
    min_val = np.min(data)
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    p98 = np.percentile(data, 98)
    max_val = np.max(data)
    mean_val = np.mean(data)
    iqr = q3 - q1

    # Count agents above the 98th percentile
    n_total = len(data)
    n_gt_p98 = np.sum(data > p98)
    n_ge_p98 = np.sum(data >= p98)

    print("\n--- MAPD ---")
    print(f"Min: {min_val:.3f}")
    print(f"Q1 (25th percentile): {q1:.3f}")
    print(f"Median (Q2): {q2:.3f}")
    print(f"Mean: {mean_val:.3f}")
    print(f"Q3 (75th percentile): {q3:.3f}")
    print(f"P98 (98th percentile): {p98:.3f}")
    print(f"Max: {max_val:.3f}")
    print(f"IQR: {iqr:.3f}")
    print(f"NaN values: {num_nans:,}")

    # Limit only the plotted distribution, not the summary statistics
    plot_data = data[data <= PLOT_MAX]
    weights = np.ones(len(plot_data)) / 1_000_000

    plt.figure(figsize=(10, 6))

    plt.hist(plot_data, bins=100, weights=weights, color="skyblue", edgecolor="black", alpha=0.6)

    plt.axvline(q1, color="blue", linestyle="None", label=f"Q1 = {q1:.2f}%")
    # plt.axvline(q2, color="salmon", linestyle="None", label=f"Median = {q2:.2f}%")
    plt.axvline(q3, color="orange", linestyle="None", label=f"Q3 = {q3:.2f}%")
    plt.axvline(p98, color="red", linestyle="--", label=f"P98 = {p98:.2f}%")

    # plt.axvspan(p98, PLOT_MAX, color="red", alpha=0.1, label="High differences")

    plt.xlabel("Mean absolute percentage difference (%)", fontsize=16)
    plt.ylabel("Number of agents (millions)", fontsize=16)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # min_handle = Line2D([0], [0], color="green", linestyle="None", label=f"Min = {min_val:.2f}%")
    max_handle = Line2D([0], [0], color="darkred", linestyle="None", label=f"Max = {max_val:.2f}%")

    handles, labels = plt.gca().get_legend_handles_labels()
    # handles.insert(0, min_handle)
    handles.insert(3, max_handle)

    plt.legend(
        handles=handles,
        loc="upper right",
        # bbox_to_anchor=(0.82, 0.99),
        fontsize=14
    )

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mapd_histogram.png", dpi=300, bbox_inches="tight")
    plt.show()

# ========================================================================================
# Plot MAD distribution
# ========================================================================================
def plot_mad_distribution(agent_df):
    num_nans = agent_df["mean_abs_diff"].isna().sum()
    data = agent_df["mean_abs_diff"].dropna().values

    # Compute summary statistics using ALL agents
    min_val = np.min(data)
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    p98 = np.percentile(data, 98)
    max_val = np.max(data)
    mean_val = np.mean(data)
    iqr = q3 - q1

    # Count agents above the 98th percentile
    n_total = len(data)
    n_gt_p98 = np.sum(data > p98)
    n_ge_p98 = np.sum(data >= p98)

    print("\n--- MAD ---")
    print(f"Min: {min_val:.3f}")
    print(f"Q1 (25th percentile): {q1:.3f}")
    print(f"Median (Q2): {q2:.3f}")
    print(f"Mean: {mean_val:.3f}")
    print(f"Q3 (75th percentile): {q3:.3f}")
    print(f"P98 (98th percentile): {p98:.3f}")
    print(f"Max: {max_val:.3f}")
    print(f"IQR: {iqr:.3f}")
    print(f"NaN values: {num_nans:,}")
    print(f"Agents >= P98: {n_ge_p98:,} ({n_ge_p98/n_total:.2%})")

    plot_data = data[data <= MAD_PLOT_MAX]
    weights = np.ones(len(plot_data)) / 1_000_000

    plt.figure(figsize=(10, 6))

    plt.hist(plot_data, bins=100, weights=weights, color="skyblue", edgecolor="black", alpha=0.6)

    plt.axvline(q1, color="blue", linestyle="None", label=f"Q1 = {q1:.2f} µg/m³")
    # plt.axvline(q2, color="salmon", linestyle="None", label=f"Median = {q2:.2f} µg/m³")
    plt.axvline(q3, color="orange", linestyle="None", label=f"Q3 = {q3:.2f} µg/m³")

    plt.axvline(p98, color="red", linestyle="--", label=f"P98 = {p98:.2f} µg/m³")
    # plt.axvspan(p98, MAD_PLOT_MAX, color="red", alpha=0.1, label="High differences")

    plt.xlabel("Mean absolute difference (µg/m³)", fontsize=16)
    plt.ylabel("Number of agents (millions)", fontsize=16)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # min_handle = Line2D([0], [0], color="green", linestyle="None", label=f"Min = {min_val:.2f} µg/m³")
    max_handle = Line2D([0], [0], color="darkred", linestyle="None", label=f"Max = {max_val:.2f} µg/m³")

    handles, labels = plt.gca().get_legend_handles_labels()
    # handles.insert(0, min_handle)
    handles.insert(3, max_handle)

    plt.legend(
        handles=handles,
        loc="upper right",
        # bbox_to_anchor=(0.95, 0.99),
        fontsize=14
    )

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mad_histogram.png", dpi=300, bbox_inches="tight")
    plt.show()

def print_difference_percentiles(agent_df):
    metrics = [
        "mean_raw_diff",
        "mean_abs_diff",
        "max_abs_diff",
        "mpd",
        "mapd"
    ]

    print("\n--- Difference Metric Percentiles ---")

    for metric in metrics:
        data = agent_df[metric].dropna().values

        print(f"\n{metric}")
        print(f"Min:    {np.min(data):.3f}")
        print(f"P25:    {np.percentile(data, 25):.3f}")
        print(f"Median: {np.percentile(data, 50):.3f}")
        print(f"Mean:   {np.mean(data):.3f}")
        print(f"P75:    {np.percentile(data, 75):.3f}")
        print(f"P90:    {np.percentile(data, 90):.3f}")
        print(f"P95:    {np.percentile(data, 95):.3f}")
        print(f"P98:    {np.percentile(data, 98):.3f}")
        print(f"P99:    {np.percentile(data, 99):.3f}")
        print(f"Max:    {np.max(data):.3f}")

# ========================================================================================
# Count agents above both the MPD and MD 98th percentiles
# ========================================================================================
def count_high_difference_agents(agent_df):
    valid_df = agent_df.dropna(subset=["mapd", "mean_abs_diff"]).copy()

    mapd_p98 = valid_df["mapd"].quantile(0.98)
    mad_p98 = valid_df["mean_abs_diff"].quantile(0.98)

    joint_mask = ((valid_df["mapd"] >= mapd_p98) & (valid_df["mean_abs_diff"] >= mad_p98))

    agent_intersection = valid_df.loc[joint_mask].copy()

    joint_count = len(agent_intersection)
    joint_percent = 100 * joint_count / len(valid_df)

    print("\n--- High-difference Agents ---")
    print(f"MAPD P98 threshold: {mapd_p98:.3f}%")
    print(f"MAD P98 threshold: {mad_p98:.3f} µg/m³")
    print(f"Agents with both MAPD >= P98 and MAD >= P98: {joint_count:,} ({joint_percent:.3f}%)")

    return agent_intersection

# ========================================================================================
# Main
# ========================================================================================
def main():
    agent_df = load_agent_metrics()

    # save_top_mapd_agents(agent_df)
    # save_low_mapd_agents(agent_df)

    plot_mpd_distribution(agent_df)
    plot_mapd_distribution(agent_df)
    plot_mad_distribution(agent_df)
    # print_difference_percentiles(agent_df)

    high_difference_agents = count_high_difference_agents(agent_df)
    high_difference_agents.to_csv(HEAD_DIR / "high_difference_agents.csv", index=False)


if __name__ == "__main__":
    main()