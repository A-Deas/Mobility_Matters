import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb
from pathlib import Path
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import pickle
import random
import numpy as np

# ======================
# Config
# ======================
RUN_TAG = "run1"
HEXES_BY_PLACE_PATH = "Hexes by Place/hexes_by_place.pkl"

""" Work only """
# SELECTED_PLACE = "unknown"  
# SELECTED_AGENT_TYPE = "wrk_only"  
# SELECTED_AGENT_ID = "2016000563846-105602" 

""" Retired """
# SELECTED_PLACE = "north_salt_lake"  
# SELECTED_AGENT_TYPE = "retired"  
# SELECTED_AGENT_ID = "2015000410724-3549601"

""" Work student """
# SELECTED_PLACE = "vineyard"  
# SELECTED_AGENT_TYPE = "wrk__student"  
# SELECTED_AGENT_ID = "2019HU0645171-5402301"

""" Youth """
SELECTED_PLACE = "unknown"  
SELECTED_AGENT_TYPE = "youth__age_under05"  
SELECTED_AGENT_ID = "2015001242138-4323305"



# # 2015000931212-3227502,student__age10_15,student__age10_15,316.60276113364074
# # 2016000531101-3234401,wrk_only,unknown,881.525294534066

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
def load_agents_from_place(safe_name):
    file = f'Output/{safe_name}/{safe_name}_final_{RUN_TAG}_SWAPPED.parquet'
    place_df = pd.read_parquet(file)
    place_df = place_df.sort_values(["p_id", "tick"], kind="mergesort")
    place_df["hour_num"] = place_df["hour"].str.slice(0, 2).astype(int)
    place_df["activity"] = place_df["activity"].map(ACTIVITY_NAME_MAP) # replace raw activity with nice name
    return place_df

def reverse_hex_lookup():
    with open(HEXES_BY_PLACE_PATH, "rb") as f:
        hexes_by_place_dict = pickle.load(f)
    reverse_lookup = {}
    for place, hexes in hexes_by_place_dict.items():
        for h in hexes:
            reverse_lookup[h] = place
    return reverse_lookup

# ======================
# Plotting
# ======================
def plot_hourly_exposure(agent_df, agent_id, agent_type, agent_home_place, agent_work_place):
    fig, axes = plt.subplots(3, 3, figsize=(15, 8), sharey=True)
    axes = axes.flatten()

    valid_days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    agent_df = agent_df[agent_df["day"].isin(valid_days)].copy()
    agent_df["day"] = pd.Categorical(agent_df["day"], ordered=True, categories=valid_days)

    for i, day in enumerate(agent_df["day"].cat.categories):
        ax = axes[i]
        day_data = agent_df[agent_df["day"] == day]

        sns.barplot(
            data=day_data,
            x="hour_num",
            y="pm2.5",
            hue="activity",
            hue_order=NICE_ACTIVITIES,
            dodge=False,
            ax=ax,
            palette=PALETTE_DICT
        )

        ax.set_title(day.capitalize(), fontsize=16, weight='bold')
        ax.set_xlabel("Hour of Day", fontsize=16)
        ax.set_ylabel("PM2.5", fontsize=16)
        ax.tick_params(axis="both", labelsize=16)
        ax.set_ylim(0, 40)

        ticks = list(range(0, 24, 3)) + [23]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{h:2d}" for h in ticks])

        ax.legend_.remove()

    # Drop unused subplots
    for j in range(len(agent_df["day"].cat.categories), len(axes)):
        fig.delaxes(axes[j])

    # Unified legend
    handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, title="Activity", loc="lower right", ncol=5, bbox_to_anchor=(.99, 0.085))
    fig.suptitle(f"Hourly PM2.5 Exposure by Day and Activity", fontsize=16, weight='bold')

    save_path = Path(f"Plots/Agent Exposure Plots/{agent_type}_{agent_id}.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_activity_schedule(agent_df, agent_id, agent_type, agent_home_place):
    day_order = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

    agent_df = agent_df.copy()
    agent_df["day"] = pd.Categorical(agent_df["day"], categories=day_order, ordered=True)
    agent_df = agent_df.sort_values(["day", "hour_num"])

    fig, ax = plt.subplots(figsize=(15, 5))
    for _, row in agent_df.iterrows():
        day = row["day"]
        hour = row["hour_num"]
        activity = row["activity"]
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

    save_path = Path(f"Plots/Agent Activity Schedules/{agent_type}_{agent_id}.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# ======================
# Main
# ======================
def main():
    reverse_lookup = reverse_hex_lookup()

    safe_name = SELECTED_PLACE
    place_df = load_agents_from_place(safe_name)

    # Split into agents (168 rows per agent)
    place_agents = [place_df.iloc[i:i+168] for i in range(0, len(place_df), 168)]

    # Filter by agent type
    agents_of_type = [a for a in place_agents if a.iloc[0]['agent_type'] == SELECTED_AGENT_TYPE]

    # Get a random agent to plot if no specific one was selected above
    if SELECTED_AGENT_ID:
        agent = next((a for a in agents_of_type if a.iloc[0]['p_id'] == SELECTED_AGENT_ID), None)
        if agent is None:
            print(f"No agent with ID {SELECTED_AGENT_ID} found in {SELECTED_PLACE}")
            return
    else:
        agent = random.choice(agents_of_type)
        
    agent_id = agent.iloc[0]['p_id']
    agent_home_place = SELECTED_PLACE

    # Work place lookup
    work_hexes = agent[agent['activity'] == ACTIVITY_NAME_MAP["work_emp"]]['h3_act_loc'].unique()
    work_hex = work_hexes[0] if len(work_hexes) > 0 else None
    agent_work_place = reverse_lookup.get(work_hex, "Unknown") if work_hex else "Unknown"

    # Plotting
    plot_hourly_exposure(agent, agent_id, SELECTED_AGENT_TYPE, agent_home_place, agent_work_place)
    plot_activity_schedule(agent, agent_id, SELECTED_AGENT_TYPE, agent_home_place)

if __name__ == "__main__":
    main()