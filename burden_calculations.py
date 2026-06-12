from pathlib import Path
import numpy as np
import pandas as pd

# RR10 values and their categories (aligned order)
RR10_LIST = [1.115, 1.068, 1.050, 1.037, 1.0065]
RR10_CATEGORIES = [
    "Asthma emergency room visits",
    "Asthma hospital admission",
    "Cardiovascular hospital admission",
    "Respiratory hospital admission",
    "Mortality"
]

def load_agents():
    file = 'Difference Metrics/agent_difference_metrics.parquet'
    agent_df = pd.read_parquet(file)

    # Filter to top 2% highest MAPE agents
    mape_cutoff = agent_df["agent_mape"].quantile(0.98)
    top_mape_df = agent_df[agent_df["agent_mape"] >= mape_cutoff]

    # top_mape_df = agent_df

    print(f"Number of highly misestimated agents (MAPE ≥ 98th percentile): {len(top_mape_df)}")
    return top_mape_df

def compute_agent_means(static_arr, dynamic_arr):
    """ Compute daily averages (7 days) for one agent """
    static_means, dynamic_means = [], []
    for day in range(7):
        sl = slice(day * 24, (day + 1) * 24)  # indices for one day
        static_means.append(np.mean(static_arr[sl]))
        dynamic_means.append(np.mean(dynamic_arr[sl]))
    return np.array(static_means), np.array(dynamic_means)

def compute_global_means(agent_df):
    """ Compute global daily averages across all agents, restricting to underestimated ones """
    all_static, all_dynamic = [], []
    underestimated_count = 0

    for _, row in agent_df.iterrows():
        static_arr = np.asarray(row["static_exposures"])
        dynamic_arr = np.asarray(row["dynamic_exposures"])
        static_means, dynamic_means = compute_agent_means(static_arr, dynamic_arr)

        # Only keep underestimated agents
        # if np.mean(dynamic_means) > np.mean(static_means):
        #     underestimated_count += 1
        #     all_static.extend(static_means)
        #     all_dynamic.extend(dynamic_means)

        # Include all agents
        all_static.extend(static_means)
        all_dynamic.extend(dynamic_means)

    global_static = float(np.mean(all_static))
    global_dynamic = float(np.mean(all_dynamic))

    # print(f"Number of underestimated agents: {underestimated_count}")
    print(f"Global daily static average:  {global_static:.3f}")
    print(f"Global daily dynamic average: {global_dynamic:.3f}")
    return global_static, global_dynamic

def compute_metrics(global_static, global_dynamic, RR10, category):
    """ Compute RR, AF, ratio, and underestimation % for a given RR10 and category """
    rr_static = RR10 ** (global_static / 10.0)
    rr_dynamic = RR10 ** (global_dynamic / 10.0)

    af_static = (rr_static - 1) / rr_static
    af_dynamic = (rr_dynamic - 1) / rr_dynamic

    ratio = af_dynamic / af_static
    under_perc = (ratio - 1) * 100

    return {
        "Category": category,
        "RR10": RR10,
        "AF_dynamic": af_dynamic,
        "AF_static": af_static,
        "Ratio": ratio,
        "% underestimation": under_perc,
    }

def main():
    agent_df = load_agents()
    global_static, global_dynamic = compute_global_means(agent_df)

    results = []
    for RR10, category in zip(RR10_LIST, RR10_CATEGORIES):
        results.append(compute_metrics(global_static, global_dynamic, RR10, category))

    # Pretty print as a table
    results_df = pd.DataFrame(results)
    print("\nResults Table:")
    print(results_df.to_string(index=False, float_format="%.6f"))

if __name__ == "__main__":
    main()