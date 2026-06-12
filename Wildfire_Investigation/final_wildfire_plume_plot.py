import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from shapely.geometry import Polygon
from pathlib import Path
import pickle
import h3
import duckdb
import polars as pl

# Constants
PLACES_SHP_PATH = "Shapefiles/cb_2019_49_place_500k/cb_2019_49_place_500k.shp"
COUNTIES_SHP_PATH = "Shapefiles/tl_2016_49_cousub/tl_2016_49_cousub.shp"
LAKES_SHP_PATH = "Shapefiles/UtahLakesNHD/LakesNHDHighRes.shp"
UTAH_SHP_PATH = "Shapefiles/tl_2019_49_tract/tl_2019_49_tract.shp"
PM_DATA_PATH = "PM2.5/conus_pm25_2019_filtered.parquet"
MONTH = 7 # july
DAY_MIN = 25 # mon
DAY_MAX = 31 # sun

def h3_to_polygon(hex_id):
    coords = h3.cell_to_boundary(hex_id)
    swapped = [(lon, lat) for lat, lon in coords]
    if swapped[0] != swapped[-1]:
        swapped.append(swapped[0])
    return Polygon(swapped)

def build_h3_gdf(hexes):
    return gpd.GeoDataFrame(
        {'h3_index': list(hexes)},
        geometry=[h3_to_polygon(h) for h in hexes],
        crs="EPSG:4326")

def load_pm25_data(hexes):
    pm_df = duckdb.query(f"""
        SELECT
            pm.h3_polyfill as hex_id,
            pm.month,
            pm.day,
            pm.value as PMValue
        FROM read_parquet('Data/PM2.5/*.parquet') as pm
        JOIN read_parquet('Data/utah_hexes_res8.parquet') as ut on pm.h3_polyfill = ut.hex_id
        WHERE month = {MONTH}
        AND day >= {DAY_MIN}
        AND day <= {DAY_MAX}
    """).to_df()
    return pm_df

def construct_utah_hex_map(hex_gdf, pm_df, utah_boundary, places, lakes):
    cmap = plt.colormaps.get_cmap('RdYlBu_r') # RdYlBu_r, YlOrRd, RdYlGn_r

    # === 1. Compute weekly average per hex ===
    weekly_avg_df = (
        pm_df.groupby("hex_id", as_index=False)["PMValue"]
        .mean()
        .rename(columns={"PMValue": "weekly_avg_pm25"})
    )

    # === 2. Merge averages into hex grid ===
    hex_gdf = hex_gdf.merge(weekly_avg_df, left_on="h3_index", right_on="hex_id", how="left")

    global_min = 0.10
    global_max = 30.49

    print(f"\nGlobal min = {global_min}")
    print(f"Global max = {global_max}")

    # === 3. Plot ===
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot boundaries first so hexes lay on top    
    utah_boundary.plot(ax=ax, color="black", edgecolor="black", linewidth=1) # Utah state boundary

    # Hexes colored by PM2.5
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

    # places (thin / subtle)
    places.boundary.plot(ax=ax, color="black", linewidth=0.45, alpha=0.25)

    # lakes
    lakes.boundary.plot(ax=ax, facecolor="dimgrey", color="dimgrey", linewidth=.5, alpha=1)

    # Add colorbar
    sm = ScalarMappable(norm=mcolors.Normalize(vmin=global_min, vmax=global_max), cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.set_label('Weekly avg PM2.5 (µg/m³)', fontsize=10, weight='bold')

    # Set the colorbar ticks
    n_ticks = 5
    tick_values = np.linspace(global_min, global_max, n_ticks)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{val:.2f}" for val in tick_values])
    cbar.ax.tick_params(labelsize=8)

    ax.set_title("Utah — Average Weekly PM2.5 Heatmap", fontsize=14, weight='bold')
    ax.axis('off')
    plt.tight_layout()

    # Save if desired
    # plt.savefig("Plots/Heat Maps/full_utah_heat_map.png", dpi=300, bbox_inches="tight")
    plt.show()

def main():
    places = gpd.read_file(PLACES_SHP_PATH).to_crs("EPSG:4326")
    utah = gpd.read_file(UTAH_SHP_PATH).to_crs("EPSG:4326")

    lakes = gpd.read_file(LAKES_SHP_PATH)
    lakes = lakes[(lakes["InUtah"] == 1) & (lakes["IsMajor"] == 1)]
    lakes = lakes.sort_values("AreaSqKm", ascending=False).head(15)
    lakes = lakes.to_crs("EPSG:4326")

    utah_boundary = utah.dissolve()

    utah_hexes_df = pl.read_parquet("Data/utah_hexes_res8.parquet")
    utah_hexes = set(utah_hexes_df["hex_id"].to_list())

    pm_df = load_pm25_data(utah_hexes)

    hex_gdf = build_h3_gdf(utah_hexes)
    hex_gdf = gpd.overlay(hex_gdf, utah_boundary, how="intersection")

    # 🔥 NEW: load largest cluster
    df_main = pd.read_csv("Wildfire_Investigation/final_wildfire_hexes3.csv")
    main_hexes = set(df_main["hex_id"])

    # 🔥 NEW: filter hexes to only main plume
    hex_gdf = hex_gdf[hex_gdf["h3_index"].isin(main_hexes)].copy()

    construct_utah_hex_map(hex_gdf, pm_df, utah_boundary, places, lakes)

if __name__ == "__main__":
    main()