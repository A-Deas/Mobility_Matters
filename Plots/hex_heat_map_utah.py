import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from shapely.geometry import Polygon
from matplotlib.transforms import blended_transform_factory
from pathlib import Path
import h3
import duckdb
import polars as pl

# Constants
PLACES_SHP_PATH = "Shapefiles/cb_2019_49_place_500k/cb_2019_49_place_500k.shp"
COUNTIES_SHP_PATH = "Shapefiles/tl_2016_49_cousub/tl_2016_49_cousub.shp"
LAKES_SHP_PATH = "Shapefiles/UtahLakesNHD/LakesNHDHighRes.shp"
UTAH_SHP_PATH = "Shapefiles/tl_2019_49_tract/tl_2019_49_tract.shp"

UTAH_HEX_PATH = "Data/Hex_Sets/utah_hexes_res8.parquet"

PM_DATA_PATH = "Data/DuckDB_Files/PM25_Utah.db"
PM_TABLE = "PM25_Utah"

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

def load_pm25_data():
    con = duckdb.connect(PM_DATA_PATH, read_only=True)

    pm_df = con.execute(f"""
        SELECT
            h3_polyfill AS hex_id,
            month,
            day,
            value AS PMValue
        FROM {PM_TABLE}
        WHERE month = {MONTH}
        AND day >= {DAY_MIN}
        AND day <= {DAY_MAX}
    """).df()

    con.close()
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

    global_min = hex_gdf["weekly_avg_pm25"].min()
    global_max = hex_gdf["weekly_avg_pm25"].max()

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

    # Add the scale bar
    add_scale_bar(ax)

    ax.set_title("Utah — Average Weekly PM2.5 Heatmap", fontsize=14, weight='bold')
    ax.axis('off')
    plt.tight_layout()

    # Save and show
    plt.savefig("Plots/Heat Maps/full_utah_heat_map.png", dpi=300, bbox_inches="tight")
    plt.show()

def add_scale_bar(
    ax,
    length_km=200,
    x_position=0.03,
    y_position=-0
):
    # --------------------------------------------------------------
    # Add a simple 200 km scale bar below the map.
    # 
    # To do this, we need the map's x coordinates measured in meters.
    # --------------------------------------------------------------
    x_min, x_max = ax.get_xlim()
    map_width = x_max - x_min

    x_start = x_min + x_position * map_width
    x_end = x_start + length_km * 1000 # we want the bar measured in KILOmeters

    transform = blended_transform_factory(ax.transData, ax.transAxes)

    # Main scale-bar line
    ax.plot(
        [x_start, x_end],
        [y_position, y_position],
        transform=transform,
        color="black",
        linewidth=1.5,
        solid_capstyle="butt",
        clip_on=False,
        zorder=20
    )

    # Ticks at 0, 50, 100, 150, and 200 km
    tick_values = [0, 50, 100, 150, 200]
    labeled_ticks = {0, 50, 100, 200}

    for value in tick_values:
        x_tick = x_start + value * 1000

        # Make the two endpoints slightly taller
        tick_height = 0.018 if value in {0, 200} else 0.012

        ax.plot(
            [x_tick, x_tick],
            [y_position - tick_height / 2, y_position + tick_height / 2],
            transform=transform,
            color="black",
            linewidth=1.25,
            clip_on=False,
            zorder=20
        )

        if value in labeled_ticks:
            ax.text(
                x_tick,
                y_position - 0.012,
                f"{value}",
                transform=transform,
                ha="center",
                va="top",
                fontsize=8,
                color="black",
                clip_on=False
            )

    # Unit label to the right of the scale bar
    label_offset = 0.025 * map_width

    ax.text(
        x_end + label_offset,
        y_position,
        "Kilometers",
        transform=transform,
        ha="left",
        va="center",
        fontsize=8,
        color="black",
        clip_on=False
    )

def main():
    places = gpd.read_file(PLACES_SHP_PATH).to_crs("EPSG:4326")
    utah = gpd.read_file(UTAH_SHP_PATH).to_crs("EPSG:4326")

    lakes = gpd.read_file(LAKES_SHP_PATH)
    lakes = lakes[(lakes["InUtah"] == 1) & (lakes["IsMajor"] == 1)]
    lakes = lakes.sort_values("AreaSqKm", ascending=False).head(15)
    lakes = lakes.to_crs("EPSG:4326")

    utah_boundary = utah.dissolve()

    utah_hexes_df = pl.read_parquet(UTAH_HEX_PATH)
    utah_hexes = set(utah_hexes_df["hex_id"].to_list())

    pm_df = load_pm25_data()

    hex_gdf = build_h3_gdf(utah_hexes)

    # Perform the spatial overlay while both layers use EPSG:4326
    hex_gdf = gpd.overlay(hex_gdf, utah_boundary, how="intersection")

    # Project every map layer to UTM Zone 12N. Coordinates will now be measured in meters.
    map_crs = "EPSG:26912"

    hex_gdf = hex_gdf.to_crs(map_crs)
    utah_boundary = utah_boundary.to_crs(map_crs)
    places = places.to_crs(map_crs)
    lakes = lakes.to_crs(map_crs)

    construct_utah_hex_map(hex_gdf, pm_df, utah_boundary, places, lakes)

if __name__ == "__main__":
    main()