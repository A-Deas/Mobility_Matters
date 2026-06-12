import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from shapely.geometry import Polygon
import h3
import duckdb
import polars as pl

PLACES_SHP_PATH = "Shapefiles/cb_2019_49_place_500k/cb_2019_49_place_500k.shp"
UTAH_SHP_PATH   = "Shapefiles/tl_2019_49_tract/tl_2019_49_tract.shp"
LAKES_SHP_PATH = "Shapefiles/UtahLakesNHD/LakesNHDHighRes.shp"

RUN_TAG = "run1"
DIFF_METRICS_PATH = "Difference Metrics/agent_difference_metrics.parquet"

PID = "2018HU0834970-199902"
PLACE = "west_jordan"

# SELECTED_PLACE = "unknown"  
# SELECTED_AGENT_TYPE = "youth__age_under05"  
# SELECTED_AGENT_ID = "2015001242138-4323305"

# SELECTED_PLACE = "unknown"  
# SELECTED_AGENT_TYPE = "wrk_only"  
# SELECTED_AGENT_ID = "2016000563846-105602" 


# ---------------------------
# Geometry helpers
# ---------------------------
def h3_to_polygon(hex_id: str) -> Polygon:
    coords = h3.cell_to_boundary(hex_id)  # (lat, lon)
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
    utah   = gpd.read_file(UTAH_SHP_PATH).to_crs("EPSG:4326")
    utah_boundary = utah.dissolve()

    places = gpd.read_file(PLACES_SHP_PATH).to_crs("EPSG:4326")

    lakes = gpd.read_file(LAKES_SHP_PATH)
    lakes = lakes[(lakes["InUtah"] == 1) & (lakes["IsMajor"] == 1)]
    lakes = lakes.sort_values("AreaSqKm", ascending=False).head(15)
    lakes = lakes.to_crs("EPSG:4326")
    return utah_boundary, places, lakes

# ---------------------------
# Agent selection
# ---------------------------
def load_outlier_agent():
    bad_pid   = PID
    bad_place = PLACE

    file = f"Output/{bad_place}/{bad_place}_final_{RUN_TAG}_SWAPPED.parquet"
    df = pd.read_parquet(file)

    agent = df[df["p_id"] == bad_pid].copy()
    agent = agent.sort_values(["p_id", "tick"], kind="mergesort").reset_index(drop=True)

    print(f"\nLoaded agent p_id={bad_pid}, place={bad_place}, rows={len(agent)}")
    return agent

def get_home_and_work_hexes(agent: pd.DataFrame):
    home_hex = agent["home_hex"].iloc[0]
    work_rows = agent[agent["activity"] == "work_emp"]
    primary_work_hex = None if len(work_rows) == 0 else work_rows["h3_act_loc"].value_counts().idxmax()
    return home_hex, primary_work_hex

# ---------------------------
# PM loading
# ---------------------------
def load_pm25_data_week(month=7, day_min=24, day_max=30):
    pm_df = duckdb.query(f"""
        SELECT
            CAST(pm.h3_polyfill AS VARCHAR) AS hex_id,
            pm.month,
            pm.day,
            CAST(pm.value AS DOUBLE) AS PMValue
        FROM read_parquet('Data/PM2.5/*.parquet') AS pm
        JOIN read_parquet('Data/utah_hexes_res8.parquet') AS ut
          ON pm.h3_polyfill = ut.hex_id
        WHERE pm.month = {month}
          AND pm.day BETWEEN {day_min} AND {day_max}
    """).to_df()
    return pm_df

# ---------------------------
# Plot
# ---------------------------
def plot_utah_weekly_heatmap_with_home_work_halos(
    agent: pd.DataFrame,
    month=7, day_min=24, day_max=30,
    clip_hexes_to_utah=True,
    clip_halos_to_utah=False,   # <- key: keep halos even if just outside UT
):
    utah_boundary, places, lakes = load_boundaries()

    utah_hexes_df = pl.read_parquet("Data/utah_hexes_res8.parquet")
    utah_hexes = set(utah_hexes_df["hex_id"].to_list())

    pm_df = load_pm25_data_week(month=month, day_min=day_min, day_max=day_max)

    weekly_avg_df = (
        pm_df.groupby("hex_id", as_index=False)["PMValue"]
        .mean()
        .rename(columns={"PMValue": "weekly_avg_pm25"})
    )

    hex_gdf = build_h3_gdf(list(utah_hexes)).merge(weekly_avg_df, on="hex_id", how="left")
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
        missing_kwds={"color": "lightgrey", "label": "No Data"},
    )

    # places (thin / subtle)
    places.boundary.plot(ax=ax, color="black", linewidth=0.45, alpha=0.25)

    # lakes
    lakes.boundary.plot(ax=ax, facecolor="dimgrey", color="dimgrey", linewidth=.5, alpha=1)

    # Glow + crisp outlines
    if len(halo_gdf) > 0:
        # home
        halo_gdf[halo_gdf["hex_id"] == home_hex].boundary.plot(ax=ax, color="lime", linewidth=7.0, alpha=.45)
        halo_gdf[halo_gdf["hex_id"] == home_hex].boundary.plot(ax=ax, color="lime", linewidth=2.4, alpha=1.0, label="Home")

        # work
        if work_hex is not None:
            halo_gdf[halo_gdf["hex_id"] == work_hex].boundary.plot(ax=ax, color="magenta", linewidth=7.0, alpha=.45)
            halo_gdf[halo_gdf["hex_id"] == work_hex].boundary.plot(ax=ax, color="magenta", linewidth=2.4, alpha=1.0, label="Work")

    sm = ScalarMappable(norm=mcolors.Normalize(vmin=global_min, vmax=global_max), cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04)
    cbar.set_label("Weekly avg PM2.5 (µg/m³)", fontsize=10, weight="bold")
    tick_values = np.linspace(global_min, global_max, 5)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{v:.2f}" for v in tick_values])
    cbar.ax.tick_params(labelsize=8)

    leg = ax.legend(title="  Agent with Max\nMPD/MAPD Score", loc="upper right", fontsize=12)
    leg.get_title().set_fontweight("bold")
    leg._legend_box.align = "left"

    ax.set_title("Utah — Weekly Avg PM2.5 with Outlier Agent Home/Work", fontsize=14, weight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    agent = load_outlier_agent()
    plot_utah_weekly_heatmap_with_home_work_halos(
        agent,
        month=7, day_min=24, day_max=30
    )

if __name__ == "__main__":
    main()
