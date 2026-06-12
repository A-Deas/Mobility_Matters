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
import math

# Constants
PLACES_SHP_PATH = "Shapefiles/cb_2019_49_place_500k/cb_2019_49_place_500k.shp"
COUNTIES_SHP_PATH = "Shapefiles/tl_2016_49_cousub/tl_2016_49_cousub.shp"
LAKES_SHP_PATH = "Shapefiles/UtahLakesNHD/LakesNHDHighRes.shp"
UTAH_SHP_PATH = "Shapefiles/tl_2019_49_tract/tl_2019_49_tract.shp"

MONTH = 7   # july
DAY_MIN = 25
DAY_MAX = 31

# Antelope Island reference point
ANTELOPE_LAT = 41.06
ANTELOPE_LON = -112.24

# Selection knobs
RADIUS_KM = 75
PM_CUTOFF = 13.0   # keep only hexes with weekly_avg_pm25 >= this


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
        crs="EPSG:4326"
    )


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def load_pm25_data():
    pm_df = duckdb.query(f"""
        SELECT
            pm.h3_polyfill AS hex_id,
            pm.month,
            pm.day,
            pm.value AS PMValue
        FROM read_parquet('Data/PM2.5/*.parquet') AS pm
        JOIN read_parquet('Data/utah_hexes_res8.parquet') AS ut
          ON pm.h3_polyfill = ut.hex_id
        WHERE pm.month = {MONTH}
          AND pm.day >= {DAY_MIN}
          AND pm.day <= {DAY_MAX}
    """).to_df()
    return pm_df


def construct_utah_hex_map(hex_gdf, pm_df, utah_boundary, places, lakes):
    cmap = plt.colormaps.get_cmap('RdYlBu_r')

    # === 1. Compute weekly average per hex across ALL Utah hexes ===
    weekly_avg_df = (
        pm_df.groupby("hex_id", as_index=False)["PMValue"]
        .mean()
        .rename(columns={"PMValue": "weekly_avg_pm25"})
    )

    # === 2. Merge averages into full hex grid ===
    hex_gdf = hex_gdf.merge(
        weekly_avg_df,
        left_on="h3_index",
        right_on="hex_id",
        how="left"
    )

    # === 3. Compute statewide color scale ===
    global_min = hex_gdf["weekly_avg_pm25"].min()
    global_max = hex_gdf["weekly_avg_pm25"].max()

    print(f"\nStatewide min = {global_min}")
    print(f"Statewide max = {global_max}")

    # === 4. Add centroid + distance to Antelope Island ===
    centroids = hex_gdf["h3_index"].apply(h3.cell_to_latlng)
    hex_gdf["centroid_lat"] = centroids.apply(lambda x: x[0])
    hex_gdf["centroid_lon"] = centroids.apply(lambda x: x[1])

    hex_gdf["dist_to_antelope_km"] = hex_gdf.apply(
        lambda row: haversine_km(
            ANTELOPE_LAT,
            ANTELOPE_LON,
            row["centroid_lat"],
            row["centroid_lon"]
        ),
        axis=1
    )

    # === 5. Keep only nearby + high-PM hexes for plotting ===
    plot_gdf = hex_gdf[
        (hex_gdf["dist_to_antelope_km"] <= RADIUS_KM) &
        (hex_gdf["weekly_avg_pm25"] >= PM_CUTOFF)
    ].copy()

    plot_gdf[["h3_index", "weekly_avg_pm25", "dist_to_antelope_km"]] \
    .rename(columns={"h3_index": "hex_id"}) \
    .to_csv("Wildfire_Investigation/candidate_hexes.csv", index=False)

    print("Saved candidate hexes to CSV")

    print(f"Radius cutoff (km) = {RADIUS_KM}")
    print(f"PM cutoff = {PM_CUTOFF}")
    print(f"Hexes plotted = {len(plot_gdf)}")

    # === 6. Plot ===
    fig, ax = plt.subplots(figsize=(8, 8))

    # Utah boundary
    utah_boundary.plot(ax=ax, color="black", edgecolor="black", linewidth=1)

    # Candidate hexes only, but colored using statewide min/max
    plot_gdf.plot(
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

    # places
    places.boundary.plot(ax=ax, color="black", linewidth=0.45, alpha=0.25)

    # lakes
    lakes.boundary.plot(ax=ax, facecolor="dimgrey", color="dimgrey", linewidth=.5, alpha=1)

    # Antelope Island reference point
    ax.scatter(
        [ANTELOPE_LON],
        [ANTELOPE_LAT],
        s=40,
        color="cyan",
        edgecolor="black",
        zorder=5
    )

    # Colorbar using statewide scale
    sm = ScalarMappable(
        norm=mcolors.Normalize(vmin=global_min, vmax=global_max),
        cmap=cmap
    )
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.set_label('Weekly avg PM2.5 (µg/m³)', fontsize=10, weight='bold')

    n_ticks = 5
    tick_values = np.linspace(global_min, global_max, n_ticks)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{val:.2f}" for val in tick_values])
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        f"Antelope Island Region — Weekly Avg PM2.5\n"
        f"Radius ≤ {RADIUS_KM} km, PM2.5 ≥ {PM_CUTOFF}",
        fontsize=14,
        weight='bold'
    )
    ax.axis('off')
    plt.tight_layout()

    # plt.savefig("Plots/Heat Maps/antelope_radius_cutoff_heat_map.png", dpi=300, bbox_inches="tight")
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

    pm_df = load_pm25_data()

    hex_gdf = build_h3_gdf(utah_hexes)
    hex_gdf = gpd.overlay(hex_gdf, utah_boundary, how="intersection")

    construct_utah_hex_map(hex_gdf, pm_df, utah_boundary, places, lakes)


if __name__ == "__main__":
    main()