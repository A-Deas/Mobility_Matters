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


# Constants
HEXES_BY_PLACE_PATH = "Hexes by Place/hexes_by_place.pkl"
PLACES_SHP_PATH = "Shapefiles/cb_2019_49_place_500k/cb_2019_49_place_500k.shp"
PM_DATA_PATH = "PM2.5/conus_pm25_2019_filtered.parquet"
PLACES_OF_INTEREST = ['koosharem', 'millcreek', 'park_city', 'riverton', 'salt_lake_city', 'samak', 'sandy', 
                      'south_jordan', 'taylorsville', 'unknown', 'wendover', 'woodland',
                      'beaver', 'circleville', 'clearfield', 'cottonwood_heights', 'hideout', 'holladay', 'kanab', 
                      'layton', 'marysvale', 'midvale', 'milford', 'murray', 'park_city', 'provo', 'salt_lake_city', 
                      'sandy', 'snyderville', 'south_jordan', 'south_salt_lake', 'taylorsville', 'vineyard', 'wendover', 
                      'west_haven', 'west_jordan', 'west_valley_city', 'woodland']

def h3_to_polygon(hex_id):
    coords = h3.cell_to_boundary(hex_id)
    swapped = [(lon, lat) for lat, lon in coords]
    if swapped[0] != swapped[-1]:
        swapped.append(swapped[0])
    return Polygon(swapped)

def generate_utah_level8_hexes(bbox, resolution=8):
    minx, miny, maxx, maxy = bbox
    lat_lng_pairs = [
        (lat, lon)
        for lat in np.arange(miny, maxy, 0.005)
        for lon in np.arange(minx, maxx, 0.005)]
    hexes = set(h3.latlng_to_cell(lat, lon, resolution) for lat, lon in lat_lng_pairs)
    print(f"Generated {len(hexes)} hexes for Utah.")
    return hexes

def build_h3_gdf(hexes):
    return gpd.GeoDataFrame(
        {'h3_index': list(hexes)},
        geometry=[h3_to_polygon(h) for h in hexes],
        crs="EPSG:4326")

def load_hexes_by_place():
    with open(HEXES_BY_PLACE_PATH, "rb") as f:
        hexes_by_place_dict = pickle.load(f)
        slugified = {key.replace(".", "").replace(" ", "_").lower(): value for key, value in hexes_by_place_dict.items()}
        return slugified

def load_pm25_data():
    df = pd.read_parquet(PM_DATA_PATH)
    df = df.rename(columns={"h3_polyfill": "hex_id", "value": "pm2.5 value"})
    return df[["hex_id", "day", "pm2.5 value"]]

def construct_weekly_hex_map(hexes_by_place, places, hex_gdf, pm_df):
    global_min = 3.2
    global_max = 17.2
    cmap = plt.colormaps.get_cmap('RdYlBu_r')

    # === 1. Compute weekly average per hex ===
    weekly_avg_df = (
        pm_df.groupby("hex_id", as_index=False)["pm2.5 value"]
        .mean()
        .rename(columns={"pm2.5 value": "weekly_avg_pm25"})
    )

    for place_name, hex_cells in sorted(hexes_by_place.items()):
        if place_name not in PLACES_OF_INTEREST:
            continue
        
        print(f"Constructing plot for {place_name}...")
        # Get place boundary
        place_geom = places[places['NAME'] == place_name]

        # Get the bounds of the place to compute aspect ratio
        xmin, ymin, xmax, ymax = place_geom.total_bounds
        width = xmax - xmin
        height = ymax - ymin
        aspect_ratio = width / height if height > 0 else 1

        # Base height in inches; scale width to preserve aspect ratio
        fig_height = 8
        fig_width = fig_height * aspect_ratio
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Get hexes for this place
        hex_subset = hex_gdf[hex_gdf['h3_index'].isin(hex_cells)].copy()

        # Merge in weekly averages
        hex_subset = hex_subset.merge(weekly_avg_df, left_on="h3_index", right_on="hex_id", how="left")

        # Plot base
        place_geom.plot(ax=ax, color="lightgrey", edgecolor="black")

        # Plot hexes with color by weekly average
        hex_subset.plot(
            ax=ax,
            column="weekly_avg_pm25",
            cmap=cmap,
            linewidth=0.1,
            edgecolor="white",
            alpha=0.7,
            legend=False,
            vmin=global_min,
            vmax=global_max,
            missing_kwds={"color": "lightgrey", "label": "No Data"}
        )

        # Add colorbar
        sm = ScalarMappable(norm=mcolors.Normalize(vmin=global_min, vmax=global_max), cmap=cmap)
        mean_val = weekly_avg_df["weekly_avg_pm25"].mean()

        # Add custom colorbar
        sm = ScalarMappable(norm=mcolors.Normalize(vmin=global_min, vmax=global_max), cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Weekly Avg PM2.5 (µg/m³)', fontsize=10, weight='bold')

        # Set custom ticks
        cbar.set_ticks([global_min, (global_min + global_max)/2, global_max])
        cbar.set_ticklabels([f"{global_min:.1f}", f"{(global_min + global_max)/2:.1f}", f"{global_max:.1f}"])
        cbar.ax.tick_params(labelsize=8)


        ax.set_title(f"{place_name} — Average Weekly PM2.5 Heatmap", fontsize=14, weight='bold')
        ax.axis('off')
        ax.set_aspect('auto')
        pad_x = width * 0.5
        pad_y = height * 0.5

        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)


        # Save figure
        output_path = Path("Plots/Heat Maps") / f"{place_name.replace(' ', '_')}_Weekly.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()



def main():
    places = gpd.read_file(PLACES_SHP_PATH).to_crs("EPSG:4326")
    places["NAME"] = places["NAME"].str.replace(".", "", regex=False).str.replace(" ", "_").str.lower()

    pm_df = load_pm25_data()
    hexes_by_place = load_hexes_by_place()

    utah_bbox = places.total_bounds
    hexes = generate_utah_level8_hexes(utah_bbox)
    hex_gdf = build_h3_gdf(hexes)

    construct_weekly_hex_map(hexes_by_place, places, hex_gdf, pm_df)


if __name__ == "__main__":
    main()
