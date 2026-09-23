import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import duckdb
import numpy as np

# ======================
# Config
# ======================
SIMULATION_DB_PATH = "Data/DuckDB_Files/SimulationResults.db"
SIMULATION_TABLE = "SimulationResults"

PLOT_DIR = Path("Plots/")

SELECTED_AGENT_IDS = [
    "2016000563846-105602",   # Worker
    "2015000410724-3549601",  # Retired
    "2019HU0645171-5402301",  # Working student
    "2015001242138-4323305",  # Youth
]

# Activity name mapping
ACTIVITY_NAME_MAP = {
    "home": "Home",
    "work_home": "Work from home",
    "work_emp": "Work",
    "work_related": "Work related",
    "school": "School",
    "meals": "Meals",
    "errands": "Errands",
    "retail": "Retail",
    "exercise": "Exercise",
    "leisure": "Leisure",
    "childcare": "Child care",
    "adult_care": "Adult care",
    "services": "Services",
    "dropoff_pickup": "Dropoff or pickup",
    "volunteer": "Volunteer",
    "visit_friends_relatives": "Visit friends or relatives",
    "religious_community": "Religious community",
    "medical": "Medical",
    "travel": "Travel",
    "change_transp": "Change of transportation",
    "other": "Other",
    "unknown": "Unknown"
}

# Raw activities (for ordering) + nice names
RAW_ACTIVITIES = list(ACTIVITY_NAME_MAP.keys())
NICE_ACTIVITIES = [ACTIVITY_NAME_MAP[a] for a in RAW_ACTIVITIES]

# Colors
base_colors = sns.color_palette("tab20")
indigo = (75/255, 0, 130/255)
black = (0, 0, 0)
custom_colors = base_colors + [indigo] + [black]

PALETTE_DICT = dict(zip(NICE_ACTIVITIES, custom_colors))

# ======================
# Helpers
# ======================
def load_agent(agent_id):
    con = duckdb.connect(SIMULATION_DB_PATH, read_only=True)

    agent_df = con.execute(f"""
        SELECT *
        FROM {SIMULATION_TABLE}
        WHERE p_id = ?
        ORDER BY tick
    """, [agent_id]).df()

    con.close()

    if agent_df.empty:
        return agent_df

    agent_df["hour_num"] = agent_df["day_hour"].str.slice(0, 2).astype(int)
    agent_df["activity_nice"] = agent_df["activity"].map(ACTIVITY_NAME_MAP)

    return agent_df


# ======================
# Plotting
# ======================
def plot_hourly_exposure(agent_df, agent_id, agent_type):
    fig, axes = plt.subplots(3, 3, figsize=(15, 8), sharey=True)
    axes = axes.flatten()

    valid_days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    agent_df = agent_df[agent_df["day"].isin(valid_days)].copy()
    agent_df["day"] = pd.Categorical(agent_df["day"], ordered=True, categories=valid_days)

    max_exposure = agent_df[["dynamic_exposure", "static_exposure"]].max().max()
    y_max = max_exposure * 1.05 if pd.notna(max_exposure) and max_exposure > 0 else 1.0

    for i, day in enumerate(agent_df["day"].cat.categories):
        ax = axes[i]
        day_data = agent_df[agent_df["day"] == day]

        sns.barplot(
            data=day_data,
            x="hour_num",
            y="dynamic_exposure",
            hue="activity_nice",
            hue_order=NICE_ACTIVITIES,
            dodge=False,
            ax=ax,
            palette=PALETTE_DICT
        )

        static_values = day_data["static_exposure"].dropna()

        static_exposure = static_values.iloc[0]
        ax.axhline(static_exposure, color="black", linestyle="--", linewidth=1.3, zorder=5)

        ax.set_title(day.capitalize(), fontsize=16, weight='bold')
        ax.set_xlabel("Hour of Day", fontsize=16)
        ax.set_ylabel("PM2.5 (µg/m³)", fontsize=16)
        ax.tick_params(axis="both", labelsize=16)
        ax.set_ylim(0, y_max)

        if agent_type == "worker_student":
            ax.set_ylim(0, max(30, y_max))
            ax.set_yticks(np.arange(0, 31, 10))

        ticks = list(range(0, 24, 3)) + [23]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{h:2d}" for h in ticks])

        if ax.legend_ is not None:
            ax.legend_.remove()

    # Drop unused subplots
    for j in range(len(agent_df["day"].cat.categories), len(axes)):
        fig.delaxes(axes[j])

    # Unified legend
    handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, title="Activity", loc="lower right", ncol=5, bbox_to_anchor=(.99, 0.085))
    title = "Hourly PM2.5 Exposure by Day and Activity"
    fig.suptitle(title, fontsize=16, weight='bold')

    save_path = PLOT_DIR / "Agent_Exposure_Plots" / f"{agent_type}_{agent_id}_exp_plot.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_activity_schedule(agent_df, agent_id, agent_type):
    day_order = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

    agent_df = agent_df.copy()
    agent_df["day"] = pd.Categorical(agent_df["day"], categories=day_order, ordered=True)
    agent_df = agent_df.sort_values(["day", "hour_num"])

    fig, ax = plt.subplots(figsize=(15, 5))
    for _, row in agent_df.iterrows():
        day = row["day"]
        hour = row["hour_num"]
        activity = row["activity_nice"]
        color = PALETTE_DICT.get(activity, (0.8, 0.8, 0.8))
        rect = patches.Rectangle((hour, day_order.index(day)), 1, 1,
                                 facecolor=color, edgecolor="gray", linewidth=0.5)
        ax.add_patch(rect)

    # xaxis
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0.5, 24.5, 1))   # centers at 0.5, 1.5, … 23.5
    ax.set_xticklabels([str(h) for h in range(24)], rotation=0)
    ax.set_xlabel("Hour of Day", fontsize=16)
    ax.tick_params(axis='x', labelsize=16)

    # yaxis
    ax.set_ylim(0, len(day_order))
    ax.set_yticks(np.arange(len(day_order)) + 0.5)  # centers at 0.5, 1.5, … 
    ax.set_yticklabels([d.capitalize() for d in day_order], fontsize=16)
    ax.set_ylabel("Day", fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.invert_yaxis()

    handles = [Line2D([0], [0], color=PALETTE_DICT[act], lw=4) for act in NICE_ACTIVITIES]
    fig.legend(handles, NICE_ACTIVITIES, title="Activity", loc="lower center", ncol=11, fontsize='medium', bbox_to_anchor=(0.5, -0.15))
    fig.suptitle(f"Weekly Activity Schedule", fontsize=16, weight='bold')

    save_path = PLOT_DIR / "Agent_Activity_Schedules" / f"{agent_type}_{agent_id}_schedule.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# ======================
# Main
# ======================
def main():
    for agent_id in SELECTED_AGENT_IDS:
        agent = load_agent(agent_id)
        if agent.empty:
            print(f"No agent with ID {agent_id} found; skipping.")
            continue

        agent_type = agent.iloc[0]["agent_type"]
        print(f"Plotting: agent {agent_id} ({agent_type}): {len(agent):,} rows")

        plot_hourly_exposure(agent, agent_id, agent_type)
        plot_activity_schedule(agent, agent_id, agent_type)


if __name__ == "__main__":
    main()
