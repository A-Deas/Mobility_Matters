import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from shapely.geometry import Polygon
from pathlib import Path
import h3
import duckdb
import polars as pl


# ============================================================
# Paths / configuration
# ============================================================
DIFFERENCE_DB_PATH = "Data/DuckDB_Files/DifferenceMetrics.db"
DIFFERENCE_TABLE = "DifferenceMetrics"

SIMULATION_DB_PATH = "Data/DuckDB_Files/SimulationResults.db"
SIMULATION_TABLE = "SimulationResults"

PM25_DB_PATH = "Data/DuckDB_Files/PM25_Utah.db"
PM25_TABLE = "PM25_Utah"

UTAH_HEX_PATH = "Data/Hex_Sets/utah_hexes_res8.parquet"

PLACES_SHP_PATH = "Shapefiles/cb_2019_49_place_500k/cb_2019_49_place_500k.shp"
UTAH_SHP_PATH = "Shapefiles/tl_2019_49_tract/tl_2019_49_tract.shp"
LAKES_SHP_PATH = "Shapefiles/UtahLakesNHD/LakesNHDHighRes.shp"

PLOT_DIR = Path("Plots/Outlier_Maps/Maps")
DATA_DIR = Path("Plots/Outlier_Maps/Data")

PLOT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2016-07-25"
END_DATE = "2016-07-31"

TOP_K = 1

# ============================================================
# Geometry helpers
# ============================================================
def h3_to_polygon(hex_id):
    coords = h3.cell_to_boundary(hex_id)
    swapped = [(lon, lat) for lat, lon in coords]

    if swapped[0] != swapped[-1]:
        swapped.append(swapped[0])

    return Polygon(swapped)


def build_h3_gdf(hex_ids, crs="EPSG:4326"):
    hex_ids = list(pd.unique(pd.Series(hex_ids).dropna()))

    return gpd.GeoDataFrame(
        {"hex_id": hex_ids},
        geometry=[h3_to_polygon(h) for h in hex_ids],
        crs=crs
    )


def load_boundaries():
    utah = gpd.read_file(UTAH_SHP_PATH).to_crs("EPSG:4326")
    utah_boundary = utah.dissolve()

    places = gpd.read_file(PLACES_SHP_PATH).to_crs("EPSG:4326")

    lakes = gpd.read_file(LAKES_SHP_PATH)
    lakes = lakes[(lakes["InUtah"] == 1) & (lakes["IsMajor"] == 1)]
    lakes = lakes.sort_values("AreaSqKm", ascending=False).head(15)
    lakes = lakes.to_crs("EPSG:4326")

    return utah_boundary, places, lakes


# ============================================================
# Load top MAPD agents
# ============================================================
def load_top_mapd_agents(top_k):
    con = duckdb.connect(DIFFERENCE_DB_PATH, read_only=True)

    top_agents = con.execute(f"""
        SELECT
            *
        FROM {DIFFERENCE_TABLE}
        ORDER BY mapd DESC
        LIMIT {top_k}
    """).df()

    con.close()

    top_agents.insert(0, "mapd_rank", np.arange(1, len(top_agents) + 1))

    print(f"\n--- Top {top_k} MAPD Agents ---")
    print(top_agents[["mapd_rank", "p_id", "mapd", "mean_abs_diff", "max_abs_diff"]].to_string(index=False))

    return top_agents


# ============================================================
# Load one agent's complete weekly schedule
# ============================================================
def load_agent(agent_id):
    con = duckdb.connect(SIMULATION_DB_PATH, read_only=True)

    agent = con.execute(f"""
        SELECT *
        FROM {SIMULATION_TABLE}
        WHERE p_id = ?
        ORDER BY tick
    """, [agent_id]).df()

    con.close()

    print(f"\nLoaded agent p_id={agent_id}, rows={len(agent):,}")

    return agent


# ============================================================
# Grab home and primary work H3 cells
# ============================================================
def get_home_and_work_hexes(agent):
    home_hex = agent["h3_home"].iloc[0]

    work_rows = agent[
        (agent["activity"] == "work_emp") &
        (agent["h3_act"].notna())
    ]

    primary_work_hex = None if len(work_rows) == 0 else work_rows["h3_act"].value_counts().idxmax()

    print("\n--- Agent Locations ---")
    print(f"Home H3: {home_hex}")
    print(f"Primary work H3: {primary_work_hex}")

    return home_hex, primary_work_hex


# ============================================================
# Load weekly PM2.5 data
# ============================================================
def load_pm25_data_week():
    con = duckdb.connect(PM25_DB_PATH, read_only=True)

    pm_df = con.execute(f"""
        SELECT
            h3_polyfill AS hex_id,
            AVG(value) AS weekly_avg_pm25
        FROM {PM25_TABLE}
        WHERE date BETWEEN DATE '{START_DATE}' AND DATE '{END_DATE}'
        GROUP BY h3_polyfill
    """).df()

    con.close()

    return pm_df


# ============================================================
# Plot weekly PM2.5 map with home/work H3 cells
# ============================================================
def plot_utah_weekly_heatmap_with_home_work_halos(agent, mapd_value, mapd_rank, clip_hexes_to_utah=True, clip_halos_to_utah=False, save_path=None):
    utah_boundary, places, lakes = load_boundaries()

    utah_hexes_df = pl.read_parquet(UTAH_HEX_PATH)
    utah_hexes = utah_hexes_df["hex_id"].to_list()

    pm_df = load_pm25_data_week()

    hex_gdf = build_h3_gdf(utah_hexes).merge(pm_df, on="hex_id", how="left")

    if clip_hexes_to_utah:
        hex_gdf = gpd.overlay(hex_gdf, utah_boundary, how="intersection")

    global_min = hex_gdf["weekly_avg_pm25"].min()
    global_max = hex_gdf["weekly_avg_pm25"].max()

    home_hex, work_hex = get_home_and_work_hexes(agent)

    halo_hexes = [home_hex] + ([work_hex] if work_hex is not None else [])
    halo_gdf = build_h3_gdf(halo_hexes)

    if clip_halos_to_utah:
        halo_gdf = gpd.overlay(halo_gdf, utah_boundary, how="intersection")

    cmap = plt.colormaps.get_cmap("RdYlBu_r")

    fig, ax = plt.subplots(figsize=(8, 8))

    utah_boundary.plot(ax=ax, color="lightgrey", edgecolor="black", linewidth=0.6)

    hex_gdf.plot(
        ax=ax,
        column="weekly_avg_pm25",
        cmap=cmap,
        linewidth=0.05,
        edgecolor="white",
        alpha=0.95,
        legend=False,
        vmin=global_min,
        vmax=global_max,
        missing_kwds={"color": "lightgrey", "label": "No Data"}
    )

    places.boundary.plot(ax=ax, color="black", linewidth=0.45, alpha=0.25)
    # lakes.plot(ax=ax, facecolor="dimgrey", edgecolor="dimgrey", linewidth=0.5, alpha=1)
    lakes.boundary.plot(ax=ax, facecolor="dimgrey", color="dimgrey", linewidth=.5, alpha=1)

    # Home
    halo_gdf[halo_gdf["hex_id"] == home_hex].boundary.plot(ax=ax, color="lime", linewidth=7.0, alpha=0.45)
    halo_gdf[halo_gdf["hex_id"] == home_hex].boundary.plot(ax=ax, color="lime", linewidth=2.4, alpha=1.0, label="Home")

    # Work
    if work_hex is not None:
        halo_gdf[halo_gdf["hex_id"] == work_hex].boundary.plot(ax=ax, color="magenta", linewidth=7.0, alpha=0.45)
        halo_gdf[halo_gdf["hex_id"] == work_hex].boundary.plot(ax=ax, color="magenta", linewidth=2.4, alpha=1.0, label="Work")

    sm = ScalarMappable(norm=mcolors.Normalize(vmin=global_min, vmax=global_max), cmap=cmap)

    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04)
    cbar.set_label("Weekly average PM2.5 (µg/m³)", fontsize=10, weight="bold")

    tick_values = np.linspace(global_min, global_max, 5)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{v:.2f}" for v in tick_values])
    cbar.ax.tick_params(labelsize=8)

    # leg = ax.legend(title=f"Agent with max \nMPD/MAPD score: \n{mapd_value:.2f}%", loc="upper right", fontsize=12)
    # leg.get_title().set_fontweight("bold")
    # leg.get_title().set_multialignment("center")
    # leg._legend_box.align = "left"

    ax.set_title(f"Utah — Weekly Average PM2.5 with MAPD Rank {mapd_rank} Home/Work", fontsize=14, weight="bold")
    ax.axis("off")

    plt.tight_layout()

    # if save_path is not None:
    #     plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# ============================================================
# Investigate hourly APDs for top MAPD agents
# ============================================================
def investigate_hourly_apd(top_agents):
    con = duckdb.connect(SIMULATION_DB_PATH, read_only=True)

    con.register("top_agents_df", top_agents[["mapd_rank", "p_id", "mapd"]])

    hourly_apd = con.execute(f"""
        SELECT
            t.mapd_rank,
            s.p_id,
            t.mapd,
            s.tick,
            s.day,
            s.date,
            s.day_hour,
            s.activity,
            s.h3_act,
            s.h3_home,
            s.static_exposure,
            s.dynamic_exposure,

            ABS(s.dynamic_exposure - s.static_exposure) AS absolute_difference,

            100.0 * ABS(s.dynamic_exposure - s.static_exposure) / s.static_exposure AS hourly_apd

        FROM {SIMULATION_TABLE} AS s

        JOIN top_agents_df AS t
            ON s.p_id = t.p_id

        WHERE s.static_exposure > 0

        ORDER BY t.mapd_rank, s.tick
    """).df()

    con.unregister("top_agents_df")
    con.close()

    hourly_apd.to_csv(DATA_DIR / f"top_{len(top_agents)}_mapd_hourly_apd.csv", index=False)

    print(f"\nSaved hourly APD records for top {len(top_agents)} MAPD agents.")


    # ============================================================
    # Agent-level APD / denominator summary
    # ============================================================
    summary_records = []

    for rank, agent_df in hourly_apd.groupby("mapd_rank"):
        agent_df = agent_df.copy()

        p_id = agent_df["p_id"].iloc[0]
        mapd = agent_df["mapd"].iloc[0]

        max_row = agent_df.loc[agent_df["hourly_apd"].idxmax()]

        static_below_1 = (agent_df["static_exposure"] < 1).sum()
        static_below_2 = (agent_df["static_exposure"] < 2).sum()
        static_below_5 = (agent_df["static_exposure"] < 5).sum()

        max_apd_when_static_below_1 = agent_df.loc[agent_df["static_exposure"] < 1, "hourly_apd"].max()

        summary_records.append({
            "mapd_rank": rank,
            "p_id": p_id,
            "mapd": mapd,
            "max_hourly_apd": max_row["hourly_apd"],
            "static_exposure_at_max_apd": max_row["static_exposure"],
            "dynamic_exposure_at_max_apd": max_row["dynamic_exposure"],
            "absolute_difference_at_max_apd": max_row["absolute_difference"],
            "tick_at_max_apd": max_row["tick"],
            "activity_at_max_apd": max_row["activity"],
            "hours_static_below_1": static_below_1,
            "hours_static_below_2": static_below_2,
            "hours_static_below_5": static_below_5,
            "max_apd_when_static_below_1": max_apd_when_static_below_1
        })

    summary_df = pd.DataFrame(summary_records)

    summary_df.to_csv(DATA_DIR / f"top_{len(top_agents)}_mapd_apd_summary.csv", index=False)


    # ============================================================
    # Print summaries
    # ============================================================
    print("\n--- Hourly APD Investigation ---")

    for _, row in summary_df.iterrows():
        print(f"\nMAPD rank {int(row['mapd_rank'])}: {row['p_id']}")
        print(f"MAPD: {row['mapd']:.3f}%")

        print(f"\nMaximum hourly APD: {row['max_hourly_apd']:.3f}%")
        print(f"Static exposure at maximum APD: {row['static_exposure_at_max_apd']:.6f} µg/m³")
        print(f"Dynamic exposure at maximum APD: {row['dynamic_exposure_at_max_apd']:.6f} µg/m³")
        print(f"Absolute difference at maximum APD: {row['absolute_difference_at_max_apd']:.6f} µg/m³")
        print(f"Tick at maximum APD: {int(row['tick_at_max_apd'])}")
        print(f"Activity at maximum APD: {row['activity_at_max_apd']}")

        print(f"\nHours with static exposure < 1 µg/m³: {int(row['hours_static_below_1'])}")
        print(f"Hours with static exposure < 2 µg/m³: {int(row['hours_static_below_2'])}")
        print(f"Hours with static exposure < 5 µg/m³: {int(row['hours_static_below_5'])}")

        if pd.notna(row["max_apd_when_static_below_1"]):
            print(f"Maximum hourly APD when static exposure < 1 µg/m³: {row['max_apd_when_static_below_1']:.3f}%")
        else:
            print("Maximum hourly APD when static exposure < 1 µg/m³: No hours below 1 µg/m³")

    return hourly_apd, summary_df


# ============================================================
# Main
# ============================================================
# ============================================================
# Main
# ============================================================
def main():
    top_agents = load_top_mapd_agents(TOP_K)

    # Investigate hourly APDs for the top K MAPD agents
    hourly_apd, summary_df = investigate_hourly_apd(top_agents)

    # Produce a home/work map for each of the top K MAPD agents
    for _, selected in top_agents.iterrows():
        mapd_rank = int(selected["mapd_rank"])
        agent_id = selected["p_id"]
        mapd_value = selected["mapd"]

        print(f"\nCreating map for MAPD rank {mapd_rank}: {agent_id}")

        selected_agent = load_agent(agent_id)

        plot_utah_weekly_heatmap_with_home_work_halos(
            selected_agent,
            mapd_value=mapd_value,
            mapd_rank=mapd_rank,
            save_path=PLOT_DIR / f"mapd_rank_{mapd_rank}_home_work_map.png"
        )


if __name__ == "__main__":
    main()