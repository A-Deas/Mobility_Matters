from repast4py import context as ctx, core, space, schedule, logging, random
from repast4py.space import ContinuousPoint as cpt, DiscretePoint as dpt, BorderType, OccupancyType, BoundingBox
from repast4py.parameters import create_args_parser, init_params
from mpi4py import MPI
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from collections import defaultdict
from typing import Tuple
from scipy.spatial import cKDTree
from pathlib import Path
import time
import random as pyrandom # so we don't confuse with repast4py's random
import h3
import pickle
import csv

RUN_TAG = "run1"

def load_pm25_lookup(pollution_file):
    df = pd.read_parquet(pollution_file)
    df = df.astype({'h3_polyfill': str, 'day': int, 'value': float})
    
    # Create a nested dictionary: {day: {hex: pm_value, hex: pm_value, ...} }
    lookup = {
        day: day_df.set_index('h3_polyfill')['value'].to_dict()
        for day, day_df in df.groupby('day') }
    return lookup

def load_act_amn_dict():
    with open("Amenities/activity_amenities_dictionary.pkl", "rb") as f:
        act_amn_dict = pickle.load(f)
    return act_amn_dict

def load_hex_amn_dict():
    with open("Amenities/hex_amenities_dictionary.pkl", "rb") as f:
        hex_amn_dict = pickle.load(f)
    return hex_amn_dict

def load_hexes_by_place_dict():
    with open("hexes_by_place.pkl", "rb") as f:
        hexes_by_place_dict = pickle.load(f)
    return hexes_by_place_dict

class Person(core.Agent):
    def __init__(self, id: int, rank: int, schedule_df: pd.DataFrame, hex_to_place: dict):
        super().__init__(id, rank)

        # Constants defining the agents
        self.urban_pop_id = schedule_df.iloc[0].p_id
        self.age = schedule_df.iloc[0].age
        self.sex = schedule_df.iloc[0].sex
        self.nhts_role = schedule_df.iloc[0].nhts_role
        self.home_hex = schedule_df.iloc[0].h3_home
        self.place_name = hex_to_place.get(self.home_hex, "unknown")
        self.prev_act = None
        self.prev_act_hex = None

        # Define custom order
        day_order = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        hour_order = [f"{str(h).zfill(2)}-{str((h+1)).zfill(2)}" for h in range(24)]

        # Convert columns to ordered categorical
        schedule_df["day"] = pd.Categorical(schedule_df["day"], categories=day_order, ordered=True)
        schedule_df["hour"] = pd.Categorical(schedule_df["hour"], categories=hour_order, ordered=True)

        # Sort by day then hour
        self.schedule = schedule_df.sort_values(by=["day", "hour"]).reset_index(drop=True)

    def step(self, tick, lookup_table, act_amn_dict, hex_amn_dict):
        day_order = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        hour_order = [f"{str(h).zfill(2)}-{str((h+1)).zfill(2)}" for h in range(24)]

        day_index = (tick // 24) % 7
        hour_index = tick % 24

        lookup_day = day_index + 4 # I am looking at November 4th (Monday) through November 10th (Sunday), so I need to set lookup_day to 4, 5, 6, ..., 10.

        day_str = day_order[day_index]
        hour_str = hour_order[hour_index]

        match = self.schedule[
            (self.schedule["day"] == day_str) &
            (self.schedule["hour"] == hour_str)
        ]

        match_idx = match.index[0]
        act_row = self.schedule.loc[match_idx]
        act_type = act_row.act
        agent_type = act_row.nhts_role

        # Check if this is the same activity as last tick
        if act_type == self.prev_act and self.prev_act_hex is not None:
            h3_act_hex = self.prev_act_hex
        else:
            # Need to assign a new location (either it's the first tick, or the activity changed)
            h3_act_hex = act_row.h3_act

            if h3_act_hex is None:
                # print(f"[DEBUG] Tick {tick}, act={act_type}, agent={self.urban_pop_id} has no hex.")
                prev_tick = max(tick - 1, 0)
                prev_day_index = (prev_tick // 24) % 7
                prev_hour_index = prev_tick % 24
                prev_day_str = day_order[prev_day_index]
                prev_hour_str = hour_order[prev_hour_index]

                prev_match = self.schedule[
                    (self.schedule["day"] == prev_day_str) &
                    (self.schedule["hour"] == prev_hour_str) ]
                prev_hex = prev_match.iloc[0].h3_act
                # print(prev_hex)

                # Find nearby hexes within ~18km (adjust ring level as needed)
                disk_size = 3  # ~18 km radius (H3 resolution 8); adjust based on the scale you want
                neighbors = h3.grid_disk(prev_hex, disk_size)   # returns all hexagons within k = disc_size
                                                                # rings from the central hexagon. The max distance
                                                                # from the center to any hexagon in the disk is roughly
                                                                # k times the edge length

                # Only keep hexes in Utah (those with data)
                valid_neighbors = [hex for hex in neighbors if hex in lookup_table.get(lookup_day)]

                # Only keep hexes which contain possible amenities for the current activity
                activity_amenities = act_amn_dict.get(act_type, []) # list of possible amenities for this activity

                filtered_neighbors = []
                for hex in valid_neighbors:
                    amenities = hex_amn_dict.get(hex, []) # list of amenities for this hex
                    if any(a in amenities for a in activity_amenities):
                        filtered_neighbors.append(hex)

                # If any hexes matched, randomly choose one of them. Otherwise, fall back to full valid_neighbors set.
                if filtered_neighbors: # is not empty
                    h3_act_hex = pyrandom.choice(filtered_neighbors)
                else:
                    h3_act_hex = pyrandom.choice(valid_neighbors) 

        # Save back to schedule for future steps
        self.schedule.at[match_idx, "h3_act"] = h3_act_hex

        # Save for next tick
        self.prev_act = act_type
        self.prev_act_hex = h3_act_hex

        # Get corresponding pollution value at that hex
        pm_val = lookup_table.get(lookup_day).get(h3_act_hex)

        # Log the entry
        entry = {
            "p_id": self.urban_pop_id,
            "sex": self.sex,
            "age": self.age,
            "place": self.place_name,
            "home_hex": self.home_hex,
            "agent_type": agent_type,
            "tick": tick,
            "day": day_str,
            "hour": hour_str,
            "activity": act_type,
            "h3_act_loc": h3_act_hex,
            "pm2.5": pm_val
        }
        return entry

def load_agents_by_file_partition(folder_path: str, context, rank: int, comm_size: int, hex_to_place: dict):
    folder = Path(folder_path)
    activity_files = sorted(folder.glob("*.parquet")) # full list of files sorted in ascending order  
    # activity_files = [folder / "acts_cohort_49__2_fmt.parquet"] # use one specific file for testing debugging

    # Each rank gets a round-robin slice of the files
    assigned_files = activity_files[rank::comm_size] # Python slice: start at index=rank and take every comm_size file from the list

    total_loaded = 0
    agent_id = 0

    for file in assigned_files:
        df = pd.read_parquet(file)

        # Group the DataFrame by p_id
        grouped = df.groupby("p_id", observed=False) # creates a groupby object that assigns the entire 168 row dataframe to each agent via AND INCLUDING their p_id
                                                     # This really just slices the list in each parquet file into 168 row chunks 
                                                     # The columns of this dataframe are exactly the same as the original parquet file

        for pid, schedule_df in grouped: # schedule_df is the name we are giving to the dataframe assigned to each agent
            agent = Person(agent_id, rank, schedule_df, hex_to_place)
            context.add(agent)
            agent_id += 1
            total_loaded += 1

    print(f"[Rank {rank}] Loaded {total_loaded} agents from {len(assigned_files)} files", flush=True)

class ActivitiesModel:
    def __init__(self, comm, params):
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.context = ctx.SharedContext(comm)
        self.params = params
        self.place_buffers = defaultdict(list) #  Python dictionary, but with a twist: if you try to access a key that doesn’t exist yet, it automatically creates it with an empty list

        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(at=0, interval=1, evt=self.step)
        # self.runner.schedule_repeating_event(1, 1, self.log_agents)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        # Geographic bounds
        self.min_lon = params['min_lon']
        self.max_lon = params['max_lon']
        self.min_lat = params['min_lat']
        self.max_lat = params['max_lat']

        box = BoundingBox(self.min_lon, self.max_lon - self.min_lon, self.min_lat, self.max_lat - self.min_lat, 0, 0)
        self.space = space.SharedCSpace('space', bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple, buffer_size=2, comm=comm, tree_threshold=100)
        self.context.add_projection(self.space)

        self.pm_lookup = load_pm25_lookup(params['pollution_file'])
        self.act_amn_lookup = load_act_amn_dict()
        self.hex_amn_lookup = load_hex_amn_dict()
        self.hexes_by_place = load_hexes_by_place_dict()
        self.hex_to_place = {}
        for place, hexes in self.hexes_by_place.items():
            clean_place = place.replace(".", "").replace(" ", "_").lower()
            for h in hexes:
                self.hex_to_place[h] = clean_place

        # Load agents into the simulation
        load_agents_by_file_partition(
            folder_path=self.params['activity_schedules_folder'],
            context=self.context,
            rank=self.rank,
            comm_size=self.comm.Get_size(),
            hex_to_place=self.hex_to_place)
        
    def flush_place_outputs(self, tick):
        for place_name, buffer in self.place_buffers.items():
            if not buffer:
                continue

            # Clean and use place-specific subfolder
            safe_name = place_name.replace(".", "").replace(" ", "_").lower()
            place_dir = Path("Output") / safe_name
            place_dir.mkdir(parents=True, exist_ok=True)

            # Write new file for this 6-tick window with rank info
            tick_tag = str(tick + 1).zfill(3)
            output_file = place_dir / f"{safe_name}_tick{tick_tag}_rank{self.rank}.parquet"

            # Convert to pandas and make columns safe
            df = pd.DataFrame(buffer)
            for col in ["p_id", "sex", "place", "agent_type", "day", "hour", "activity", "h3_act_loc"]:
                if col in df:
                    df[col] = df[col].astype(str)

            df.to_parquet(
                output_file,
                engine="pyarrow",
                compression="snappy",
                use_dictionary=False,
                index=False
            )

            self.place_buffers[place_name] = []


    def step(self):
        current_tick = self.runner.schedule.tick
        for agent in self.context.agents():
            entry = agent.step(current_tick, self.pm_lookup, self.act_amn_lookup, self.hex_amn_lookup)
            self.place_buffers[agent.place_name].append(entry) # add the agent’s log entry to the list of rows for their home place

        # Every 6 ticks: flush to disk
        if (current_tick + 1) % 6 == 0:
            self.flush_place_outputs(current_tick)

        # Print statements to track the model's progress
        if self.rank == 0:
            if current_tick == 0:
                self._start_time = time.time()
            elif (current_tick + 1) % 6 == 0:
                elapsed = time.time() - self._start_time
                formatted = time.strftime('%H:%M:%S', time.gmtime(elapsed))
                print(f"[Tick {current_tick}] Elapsed time: {formatted}", flush=True)

    def run(self):
        self.runner.execute()

    def at_end(self):
        if self.rank != 0:
            return

        for place_name in self.place_buffers.keys():
            safe_name = place_name.replace(".", "").replace(" ", "_").lower()
            place_dir = Path("Output") / safe_name

            if not place_dir.exists():
                continue

            # Collect all rank-based tick files
            tick_files = sorted(place_dir.glob(f"{safe_name}_tick*_rank*.parquet"))
            if not tick_files:
                continue

            dfs = []
            for file in tick_files:
                try:
                    df = pd.read_parquet(file)
                    dfs.append(df)
                except Exception as e:
                    print(f"⚠️ Failed to read {file}: {e}")

            if not dfs:
                continue

            final_df = pd.concat(dfs, ignore_index=True)
            final_df.sort_values(by=["p_id", "tick"], inplace=True)

            # Save final merged file
            final_file = place_dir / f"{safe_name}_final_{RUN_TAG}.parquet"
            final_df.to_parquet(
                final_file,
                engine="pyarrow",
                compression="snappy",
                use_dictionary=False,
                index=False
            )

            # Clean up
            for f in tick_files:
                f.unlink()

def run(params):
    model = ActivitiesModel(MPI.COMM_WORLD, params)
    model.run()

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    main_start_time = time.time()

    parser = create_args_parser()
    args = parser.parse_args()
    params = init_params(args.parameters_file, args.parameters)
    run(params)

    main_end_time = time.time()
    main_elapsed = main_end_time - main_start_time
    main_formatted = time.strftime('%H:%M:%S', time.gmtime(main_elapsed))

    if rank == 0:
        print(f"Total simulation runtime: {main_formatted}")