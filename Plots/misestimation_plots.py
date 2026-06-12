import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import numpy as np

# ======================
# Config
# ======================
RUN_TAG = "run1"
HEXES_BY_PLACE_PATH = "Hexes by Place/hexes_by_place.pkl"

""" Underestimated """
# SELECTED_PLACE = "unknown"  
# SELECTED_AGENT_TYPE = "wrk_only"  
# SELECTED_AGENT_ID = "2016000563846-105602" 

""" Overestimated """
SELECTED_PLACE = "vineyard"  
SELECTED_AGENT_TYPE = "wrk__student"  
SELECTED_AGENT_ID = "2019HU0645171-5402301"

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

def load_agents_from_place(safe_name):
    file = f'Output/{safe_name}/{safe_name}_final_{RUN_TAG}_SWAPPED.parquet'
    place_df = pd.read_parquet(file)
    place_df = place_df.sort_values(["p_id", "tick"], kind="mergesort")
    place_df["hour_num"] = place_df["hour"].str.slice(0, 2).astype(int)
    # replace raw activity with nice name
    place_df["activity"] = place_df["activity"].map(ACTIVITY_NAME_MAP)
    return place_df

def plot_misestimations(agent_df):
    fig, axes = plt.subplots(3, 3, figsize=(12, 7), sharey=True)
    axes = axes.flatten()

    agent_df = agent_df.copy()
    valid_days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    agent_df = agent_df[agent_df["day"].isin(valid_days)]
    agent_df["day"] = pd.Categorical(agent_df["day"], ordered=True, categories=valid_days)

    # Get static exposure (home value at midnight for each day)
    day_to_index = {'mon': 0, 'tue': 24, 'wed': 48, 'thu': 72, 'fri': 96, 'sat': 120, 'sun': 144}
    home_index_lookup = {day: agent_df.iloc[day_idx]["pm2.5"] for day, day_idx in day_to_index.items()}

    for i, day in enumerate(agent_df["day"].cat.categories):
        ax = axes[i]
        home_exp = home_index_lookup[day]

        df_day = agent_df[agent_df["day"] == day].copy()
        hours = df_day["hour_num"].values
        heights = df_day["pm2.5"].values
        activities = df_day["activity"].tolist()
        at_home = df_day["activity"] == "Home"

        # Static baseline
        ax.axhline(y=home_exp, color="mediumblue", linestyle="-", linewidth=1.5, label="Static Exposure")

        # Bars for dynamic – static (only when not at home)
        for j, hour in enumerate(hours):
            if not at_home.iloc[j]:
                diff = heights[j] - home_exp
                ax.bar(hour, diff, bottom=home_exp, color="crimson", edgecolor="black", linewidth=0.1)

                ax.hlines(y=heights[j], xmin=hour-0.45, xmax=hour+0.45,color="forestgreen", linewidth=1.5)

        # Formatting
        ax.set_title(day.capitalize(), fontsize=16, weight='bold')
        ax.set_xlim(-0.5, 23.5)

        ticks = list(np.arange(0, 24, 4)) + [23]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, fontsize=14)
        ax.set_xlabel("Hour of Day", fontsize=16)

        # ax.set_xticks(np.arange(24))
        # ax.set_xticklabels(activities, rotation=45, ha="right", fontsize=6)
        # ax.set_xlabel("Activity", fontsize=12)

        # Color xtick labels
        # for tick_label, activity in zip(ax.get_xticklabels(), activities):
        #     if activity == "Home":
        #         tick_label.set_color("mediumblue")
        #     else:
        #         tick_label.set_color("crimson")

        if i % 3 == 0:   # only left column
            ax.set_ylabel("PM2.5", fontsize=16)
        else:
            ax.set_ylabel("")

        ymax = max(heights.max(), home_exp) 
        ax.set_ylim(0, ymax + 3)
        ax.set_yticks([0, 15, 30])
        ax.tick_params(axis='y', labelsize=14)

    # Drop unused subplots
    for j in range(len(valid_days), len(axes)):
        fig.delaxes(axes[j])

    # Title and legend once
    # fig.suptitle("Difference Between Static and Dynamic Exposure", fontsize=16, weight='bold')
    mediumblue_line = Line2D([0], [0], color='mediumblue', linestyle='-', linewidth=2, label='Static exposure')
    green_line = Line2D([0], [0], color='forestgreen', linestyle='-', linewidth=2, label='Dynamic exposure')
    red_patch = Patch(facecolor='crimson', label='Difference in exposure')
    fig.legend(handles=[mediumblue_line, green_line, red_patch], loc='lower right', bbox_to_anchor=(.75, 0.10), fontsize=16)

    plt.tight_layout()
    outdir = Path("Plots/Misestimation Plots")
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{SELECTED_AGENT_TYPE}_{SELECTED_AGENT_ID}.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def main():
    safe_name = SELECTED_PLACE
    place_df = load_agents_from_place(safe_name)

    # Split into agents (168 rows per agent)
    place_agents = [place_df.iloc[i:i+168] for i in range(0, len(place_df), 168)]

    # Filter by agent type
    agents_of_type = [a for a in place_agents if a.iloc[0]['agent_type'] == SELECTED_AGENT_TYPE]

    agent = next((a for a in agents_of_type if a.iloc[0]['p_id'] == SELECTED_AGENT_ID), None)

    # Plotting
    plot_misestimations(agent)

if __name__ == "__main__":
    main()