import pandas as pd
import numpy as np

file = 'Difference Metrics/agent_difference_metrics.parquet'
agent_df = pd.read_parquet(file)

# Filter to top 2% highest MAPE agents
mape_cutoff = agent_df["agent_mape"].quantile(0.98)
top_mape_df = agent_df[agent_df["agent_mape"] >= mape_cutoff]

# Compute each agent’s average static vs. dynamic exposures
static_means = top_mape_df["static_exposures"].apply(np.mean)
dynamic_means = top_mape_df["dynamic_exposures"].apply(np.mean)

# Split underestimated (dynamic > static) and overestimated (dynamic < static)
underestimated_top_mape_df = top_mape_df[dynamic_means > static_means]
overestimated_top_mape_df = top_mape_df[dynamic_means < static_means]

# Sort by agent_mape descending
underestimated_sorted = underestimated_top_mape_df.sort_values("agent_mape", ascending=False)
overestimated_sorted = overestimated_top_mape_df.sort_values("agent_mape", ascending=False)

# Columns to save
cols_to_keep = ["p_id", "agent_type", "place", "agent_mape"]

# Save
underestimated_sorted[cols_to_keep].to_csv("Difference Metrics/underestimated_agents_map.csv", index=False)
overestimated_sorted[cols_to_keep].to_csv("Difference Metrics/overestimated_agents_map.csv", index=False)

