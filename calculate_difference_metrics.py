import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

OUTPUT_DIR = Path("Output")
RUN_TAG = "run1"

def load_agents_from_place(slug_name):
    file = f'Output/{slug_name}/{slug_name}_final_{RUN_TAG}_SWAPPED.parquet'
    place_df = pd.read_parquet(file)
    place_df = place_df.sort_values(["p_id", "tick"], kind="mergesort")
    # print(place_df.head(169))
    return place_df

def compute_diff_err_metrics(place_df):
    agents = [place_df.iloc[i:i+168].copy() for i in range(0, len(place_df), 168)]
    place_records = []

    for agent in agents:
        agent_raw_diffs = []
        agent_abs_diffs = []
        agent_norm_diffs = []
        agent_norm_abs_diffs = []
        agent_static_exposures = []
        agent_dynamic_exposures = []
        non_home_hours = 0
        
        pm_values = agent["pm2.5"].values
        activities = agent["activity"].values

        for day_index in range(0, 168, 24):
            day_home_exp = pm_values[day_index] # pm2.5 home exposure for each day of the week is found at the first hour of each day

            day_slice = slice(day_index, day_index + 24) # Isolate this day of the week from agent's full df (left inclusive, right exclusive)
            daily_exposures = pm_values[day_slice]

            not_home_mask = activities[day_slice] != "home"
            non_home_hours += not_home_mask.sum()

            static_arr = np.array([day_home_exp] * len(daily_exposures))
            dynamic_arr = np.array(daily_exposures)

            # Array computations are performed component wise
            raw_diff_arr = dynamic_arr - static_arr # raw differences
            abs_diff_arr = np.abs(dynamic_arr - static_arr) # absolute differences
            norm_diff_arr = (dynamic_arr - static_arr) / static_arr # vector of normalized differences
            norm_abs_diff_arr = np.abs( (dynamic_arr - static_arr) / static_arr ) # vector of absolute normalized differences

            # Extend dataframes
            agent_raw_diffs.extend(raw_diff_arr)
            agent_abs_diffs.extend(abs_diff_arr)
            agent_norm_diffs.extend(norm_diff_arr)
            agent_norm_abs_diffs.extend(norm_abs_diff_arr)
            agent_static_exposures.extend(static_arr)
            agent_dynamic_exposures.extend(dynamic_arr)

        # Convert everything to arrays to be safe and a little more efficient
        agent_raw_diffs = np.array(agent_raw_diffs)
        agent_abs_diffs = np.array(agent_abs_diffs)
        agent_norm_diffs = np.array(agent_norm_diffs)
        agent_norm_abs_diffs = np.array(agent_norm_abs_diffs)
        static_all_ticks = np.array(agent_static_exposures)
        dynamic_all_ticks = np.array(agent_dynamic_exposures)

        # Compute summary metrics across all ticks in the simulation
        agent_total_raw_diff = np.sum(agent_raw_diffs)
        agent_total_abs_diff = np.sum(agent_abs_diffs) 
        agent_mpe = np.mean(agent_norm_diffs) * 100 
        agent_mape = np.mean(agent_norm_abs_diffs) * 100

        # Grab the agent metadata, these can truly all be grabbed from the first row of the agent's df
        p_id = agent.iloc[0]["p_id"]
        place_name = agent.iloc[0]["place"]
        home_hex = agent.iloc[0]["home_hex"]
        agent_type = agent.iloc[0]["agent_type"]

        place_records.append({
            "p_id": p_id,
            "place": place_name,
            "home_hex": home_hex,
            "agent_type": agent_type,
            "agent_total_raw_diff": agent_total_raw_diff,
            "agent_total_abs_diff": agent_total_abs_diff,
            "agent_mpe": agent_mpe,
            "agent_mape": agent_mape,
            "static_exposures": static_all_ticks,
            "dynamic_exposures": dynamic_all_ticks,
            "non_home_hours": non_home_hours})
    return place_records # looping through all places loops through all agents, so by the end of the place loop, every single agent is accounted for

def main():
    parquet_files = list(OUTPUT_DIR.glob(f"*/**/*_final_{RUN_TAG}_SWAPPED.parquet"))
    agent_diff_records = []

    for parquet_file in parquet_files:
        slug_name = parquet_file.parent.name
        nice_name = slug_name.replace("_", " ").title()

        print(f"Processing {nice_name}...")
        place_df = load_agents_from_place(slug_name)
        place_records = compute_diff_err_metrics(place_df)
        agent_diff_records.extend(place_records)

    agent_diff_df = pd.DataFrame(agent_diff_records)

    # Save the results
    agent_diff_df.to_parquet("Difference Metrics/agent_difference_metrics.parquet", index=False)
    print(f"\nSaved agent-level difference metrics.")

if __name__ == "__main__":
    main()