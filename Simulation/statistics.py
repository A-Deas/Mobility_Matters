import os
import pandas as pd
import pyarrow.dataset as ds

# -------------------------------
# 1. Count distinct agent IDs across all output files
# -------------------------------

def count_unique_agents(output_dir="Output"):
    unique_agent_ids = set()

    # Traverse through subdirectories in Output/
    for place_name in os.listdir(output_dir):
        place_path = os.path.join(output_dir, place_name)

        # Look for parquet files in each place folder
        for file_name in os.listdir(place_path):
            if file_name.endswith(".parquet"):
                file_path = os.path.join(place_path, file_name)
                print(f"Reading agent file: {file_path}")
                df = pd.read_parquet(file_path, columns=["p_id"])
                unique_agent_ids.update(df["p_id"].unique())

    print(f"\nTotal number of unique agents: {len(unique_agent_ids)}")
    return unique_agent_ids


# -------------------------------
# 2. Count distinct hexes in PM2.5 dataset
# -------------------------------

def count_unique_hexes_pm25():
    dataset = ds.dataset("/Users/p5d/Documents/Python/Utah ABM/PM2.5/conus_pm25_2019_filtered.parquet", format="parquet")

    # We scan only the h3_polyfill field for the given year to keep it efficient
    scanner = dataset.scanner(columns=["h3_polyfill"])
    unique_hexes = set()

    for batch in scanner.to_batches():
        df = batch.to_pandas()
        unique_hexes.update(df["h3_polyfill"].astype(str).unique())

    print(f"\nTotal number of unique hexes in PM2.5 dataset: {len(unique_hexes)}")
    return unique_hexes


# -------------------------------
# Run the counting
# -------------------------------

if __name__ == "__main__":
    # Count agents
    # agent_ids = count_unique_agents("Output")

    # Count hexes
    unique_hexes = count_unique_hexes_pm25()
