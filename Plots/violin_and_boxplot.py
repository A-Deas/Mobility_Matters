import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

AGENT_TYPE_MAP = {
    "wrk_only": "Worker",
    "wrk__student": "Working student",
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

def load_agent_metrics():
    file = 'Difference Metrics/agent_difference_metrics.parquet'
    agent_df = pd.read_parquet(file)

    # Map code names to nice names
    agent_df["agent_type"] = agent_df["agent_type"].map(AGENT_TYPE_MAP)

    # Create high_error label
    mape_cutoff = agent_df["agent_mape"].quantile(0.98)
    agent_df["high_error"] = (agent_df["agent_mape"] >= mape_cutoff).astype(int)

    # Filter to top 2% highest MAPE agents
    top_mape_df = agent_df[agent_df["agent_mape"] >= mape_cutoff]
    return agent_df, top_mape_df

def boxplot_agent_type_vs_high_error(agent_df):
    order = (agent_df.groupby("agent_type")["non_home_hours"].mean().sort_values(ascending=False).index)
    plt.figure(figsize=(14, 6))

    sns.boxplot(
        x="agent_type", 
        y="non_home_hours", 
        hue="high_error", 
        data=agent_df, 
        order=order, 
        hue_order=[True, False],
        showfliers=False,
        palette={True: "darkorange", False: "deepskyblue"},  # Explicit color mapping for the different group (classified with booleans! So True means highly misestimated.)
        linewidth=1.0, 
        boxprops=dict(edgecolor="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="black")
    )

    plt.xticks(rotation=45, ha='right')
    # plt.title("Mobility differences between highly misestimated versus not agents", fontsize=16, weight='bold')
    plt.ylabel("Mobility (non-home hours)")
    plt.xlabel("Agent Type")

    # Proper filled legend boxes
    blue_patch = Patch(facecolor='deepskyblue', edgecolor='black', label='All other agents')
    orange_patch = Patch(facecolor='darkorange', edgecolor='black', label='Highly misestimated agents')

    plt.legend(handles=[orange_patch, blue_patch], title="Group", loc='upper right')
    plt.tight_layout()
    # ax = plt.gca()
    # ax.set_ylim(bottom=20, top=100)
    plt.savefig(f"Plots/Violin and Boxplots/mobility_boxplot.png", dpi=300, bbox_inches="tight")
    plt.show()

def violin_mobility_plot(agent_df):
    order = (agent_df.groupby("agent_type")["non_home_hours"].mean().sort_values(ascending=False).index)
    plt.figure(figsize=(14, 6))

    ax = sns.violinplot(
        x="agent_type", 
        y="non_home_hours", 
        hue="agent_type",
        data=agent_df, 
        order=order, 
        palette="husl",
        linewidth=1.0,
        inner="box",
        legend=False 
    )

    # Make violin outlines black
    for violin in ax.collections:
        violin.set_edgecolor("black")

    # Make all internal lines (quartiles, whiskers, median) black
    for line in ax.lines:
        line.set_color("black")

    plt.xticks(rotation=45, ha='right')
    # plt.title("Mobility Distributions For All Agent Types", fontsize=16, weight='bold')
    plt.ylabel("Mobility (non-home hours)")
    plt.xlabel("Agent Type")
    plt.tight_layout()
    # ax.set_ylim(bottom=0, top=120)
    plt.savefig(f"Plots/Violin and Boxplots/violin_mobility_plot.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    agent_df, top_mape_df = load_agent_metrics()
    boxplot_agent_type_vs_high_error(agent_df)
    violin_mobility_plot(agent_df)

if __name__ == "__main__":
    main()



