import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

def load_agent_metrics():
    file = 'Difference Metrics/agent_difference_metrics.parquet'
    agent_df = pd.read_parquet(file)
    print(len(agent_df))

    # Find the outlier agent(s)
    out = agent_df.sort_values("agent_mape", ascending=False).head(5)
    print(out[["p_id","place","agent_type","agent_mpe","agent_mape","non_home_hours"]])

    return agent_df

def plot_mpd_distribution(agent_df):
    num_nans = agent_df["agent_mpe"].isna().sum()
    data = agent_df["agent_mpe"].dropna().values

    # Compute summary stats
    min_val = np.min(data)
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    p98 = np.percentile(data, 98)
    max_val = np.max(data)
    iqr = q3 - q1

    # Count number of agents between values
    n_total = len(data)
    n_min_to_q1 = np.sum((data >= min_val) & (data <= q1))
    n_q3_to_max = np.sum((data >= q3) & (data <= max_val))

    print(f"\n--- MPD ---")
    print(f"Min: {min_val:.3f}")
    print(f"Q1 (25th percentile): {q1:.3f}   | Count [Min, Q1]: {n_min_to_q1:,} ({n_min_to_q1/n_total:.2%})")
    print(f"Median (Q2): {q2:.3f}")
    print(f"Q3 (75th percentile): {q3:.3f}   | Count [Q3, Max]: {n_q3_to_max:,} ({n_q3_to_max/n_total:.2%})")
    print(f"Max: {max_val:.3f}")
    print(f"IQR: {iqr:.3f}")
    print(f"The number of NaN values is: {num_nans:,}")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    data = data[data <= 250]
    plot_max = np.max(data)

    plt.hist(data, bins=100, color='skyblue', edgecolor='black', alpha=0.6, density=False)

    # Add summary lines with value labels
    plt.axvline(min_val, color='green', linestyle='--', label=f"Min = {min_val:.2f}")
    plt.axvline(q1, color='blue', linestyle='--', label=f"Q1 = {q1:.2f}")
    plt.axvline(q2, color='salmon', linestyle='--', label=f"Median = 0.00")
    plt.axvline(q3, color='orange', linestyle='--', label=f"Q3 = {q3:.2f}")
    # plt.axvline(max_val, color='darkred', linestyle='--', label=f"Max = {max_val:.2f}")

    plt.axvspan(q1, min_val, color='blue', alpha=0.1, label='Negative differences')
    plt.axvspan(q3, plot_max, color='darkorange', alpha=0.1, label='Positive differences')

    # Labeling
    # plt.title(f"Distribution of MPD", fontsize=16, weight='bold')
    plt.xlabel("Mean percentage difference (μg/m³)", fontsize=16)
    plt.ylabel("Number of agents (measured in millions)", fontsize=16)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    max_handle = Line2D([0], [0], color='darkred', linestyle='None', label=f"Max = {max_val:.2f}")
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.insert(4, max_handle) # Insert max right after the P98 line (which is currently index 4)
    plt.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.95, 1.0), fontsize=16)

    plt.tight_layout()

    # Save
    plt.savefig(f"Plots/Difference Metric Distributions/mpd_histogram.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_mapd_distribution(agent_df):
    num_nans = agent_df["agent_mape"].isna().sum()
    data = agent_df["agent_mape"].dropna().values

    # Compute summary stats
    min_val = np.min(data)
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    p98 = np.percentile(data, 98)
    max_val = np.max(data)
    iqr = q3 - q1

    # Count number of agents
    n_total = len(data)
    n_ge_p98 = np.sum(data >= p98)   # or use > p98 if you want strict

    print(f"\n--- MAPD ---")
    print(f"Min: {min_val:.3f}")
    print(f"Q1 (25th percentile): {q1:.3f}")
    print(f"Median (Q2): {q2:.3f}")
    print(f"Q3 (75th percentile): {q3:.3f}")
    print(f"P98 (98th percentile): {p98:.2f}   | Count >= P98: {n_ge_p98:,} ({n_ge_p98/n_total:.2%})")
    print(f"Max: {max_val:.3f}")
    print(f"IQR: {iqr:.3f}")
    print(f"The number of NaN values is: {num_nans:,}")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    data = data[data <= 250]
    plot_max = np.max(data)
    
    plt.hist(data, bins=100, color='skyblue', edgecolor='black', alpha=0.6, density=True)

    # Add summary lines with value labels
    plt.axvline(min_val, color='green', linestyle='--', label=f"Min = {min_val:.2f}")
    plt.axvline(q1, color='blue', linestyle='--', label=f"Q1 = {q1:.2f}")
    plt.axvline(q2, color='salmon', linestyle='--', label=f"Median = {q2:.2f}")
    plt.axvline(q3, color='orange', linestyle='--', label=f"Q3 = {q3:.2f}")
    plt.axvline(p98, color='red', linestyle='--', label=f"P98 = {p98:.2f}")
    # plt.axvline(plot_max, color='darkred', linestyle='--', label=f"Max = {max_val:.2f}")

    # Gently shade the range from 98th percentile to max
    plt.axvspan(p98, plot_max, color='red', alpha=0.1, label='High differences')

    # Labeling
    # plt.title(f"Distribution of MAPD", fontsize=16, weight='bold')
    plt.xlabel("Mean absolute percentage difference (μg/m³)", fontsize=16)
    plt.ylabel("Number of agents (measured in millions)", fontsize=16)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    max_handle = Line2D([0], [0], color='darkred', linestyle='None', label=f"Max = {max_val:.2f}")
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.insert(5, max_handle) # Insert max right after the P98 line (which is currently index 4)
    plt.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.95, 1.0), fontsize=16)
    plt.tight_layout()

    # Place one more tick at the top of the yaxis
    ax = plt.gca()
    ax.set_ylim(top=.17)

    # Save
    plt.savefig(f"Plots/Difference Metric Distributions/mapd_histogram.png", dpi=300, bbox_inches="tight")
    plt.show()

def main():
    agent_df = load_agent_metrics()
    
    plot_mpd_distribution(agent_df)
    plot_mapd_distribution(agent_df)

if __name__ == "__main__":
    main()
