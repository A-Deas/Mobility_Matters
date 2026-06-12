import duckdb
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from shapely.geometry import Polygon
import h3

# -----------------------------
# Paths / constants
# -----------------------------
PLACES_SHP_PATH = "Shapefiles/cb_2019_49_place_500k/cb_2019_49_place_500k.shp"
LAKES_SHP_PATH = "Shapefiles/UtahLakesNHD/LakesNHDHighRes.shp"
UTAH_SHP_PATH = "Shapefiles/tl_2019_49_tract/tl_2019_49_tract.shp"

PM_GLOB = "Data/PM2.5_FullYear/*/*.parquet"
UT_HEXES = "Data/utah_hexes_res8.parquet"
FINAL_HEXES_CSV = "Wildfire_Investigation/final_wildfire_hexes3.csv"

WEEK_MONTH = 7
WEEK_DAY_MIN = 25
WEEK_DAY_MAX = 31

OUT_PATH = "Plots/Wildfire_Effect/base_plot.png"

# -----------------------------
# Helpers
# -----------------------------
def h3_to_polygon(hex_id):
    coords = h3.cell_to_boundary(hex_id)
    swapped = [(lon, lat) for lat, lon in coords]
    if swapped[0] != swapped[-1]:
        swapped.append(swapped[0])
    return Polygon(swapped)

def build_h3_gdf(hexes):
    return gpd.GeoDataFrame(
        {"hex_id": list(hexes)},
        geometry=[h3_to_polygon(h) for h in hexes],
        crs="EPSG:4326"
    )

def load_base_layers():
    places = gpd.read_file(PLACES_SHP_PATH).to_crs("EPSG:4326")

    utah = gpd.read_file(UTAH_SHP_PATH).to_crs("EPSG:4326")
    utah_boundary = utah.dissolve()

    lakes = gpd.read_file(LAKES_SHP_PATH)
    lakes = lakes[(lakes["InUtah"] == 1) & (lakes["IsMajor"] == 1)]
    lakes = lakes.sort_values("AreaSqKm", ascending=False).head(15)
    lakes = lakes.to_crs("EPSG:4326")

    return places, utah_boundary, lakes

# -----------------------------
# Query averages
# -----------------------------
con = duckdb.connect()

year_df = con.execute(f"""
WITH plume_hexes AS (
    SELECT hex_id
    FROM read_csv_auto('{FINAL_HEXES_CSV}')
)

SELECT
    pm.h3_polyfill AS hex_id,
    avg(pm.value) AS avg_pm25
FROM read_parquet('{PM_GLOB}') AS pm
JOIN read_parquet('{UT_HEXES}') AS ut
  ON pm.h3_polyfill = ut.hex_id
JOIN plume_hexes AS ph
  ON pm.h3_polyfill = ph.hex_id
GROUP BY pm.h3_polyfill
ORDER BY pm.h3_polyfill
""").fetchdf()

week_df = con.execute(f"""
WITH plume_hexes AS (
    SELECT hex_id
    FROM read_csv_auto('{FINAL_HEXES_CSV}')
)

SELECT
    pm.h3_polyfill AS hex_id,
    avg(pm.value) AS avg_pm25
FROM read_parquet('{PM_GLOB}') AS pm
JOIN read_parquet('{UT_HEXES}') AS ut
  ON pm.h3_polyfill = ut.hex_id
JOIN plume_hexes AS ph
  ON pm.h3_polyfill = ph.hex_id
WHERE pm.month = {WEEK_MONTH}
  AND pm.day BETWEEN {WEEK_DAY_MIN} AND {WEEK_DAY_MAX}
GROUP BY pm.h3_polyfill
ORDER BY pm.h3_polyfill
""").fetchdf()

print("\nFull-year averages:")
print(year_df.head())

print("\nStudy-week averages:")
print(week_df.head())

# Means for annotation
year_mean = year_df["avg_pm25"].mean()
week_mean = week_df["avg_pm25"].mean()

print(f"\nFull-year plume mean PM2.5: {year_mean:.3f}")
print(f"Study-week plume mean PM2.5: {week_mean:.3f}")

# -----------------------------
# Geometry
# -----------------------------
hexes = year_df["hex_id"].tolist()
hex_gdf = build_h3_gdf(hexes)

year_gdf = hex_gdf.merge(year_df, on="hex_id", how="left")
week_gdf = hex_gdf.merge(week_df, on="hex_id", how="left")

# Shared color scale based on full-year plume values
combined = pd.concat([year_df["avg_pm25"], week_df["avg_pm25"]])
global_min = combined.min()
global_max = combined.max()

print(f"\nYear min/max: {year_df['avg_pm25'].min()} / {year_df['avg_pm25'].max()}")
print(f"Week min/max: {week_df['avg_pm25'].min()} / {week_df['avg_pm25'].max()}")
print(f"Global min/max: {global_min} / {global_max}")

print(f"\nShared color scale min = {global_min}")
print(f"Shared color scale max = {global_max}")

# -----------------------------
# Base layers
# -----------------------------
places, utah_boundary, lakes = load_base_layers()

# Optional clipping
year_gdf = gpd.overlay(year_gdf, utah_boundary, how="intersection")
week_gdf = gpd.overlay(week_gdf, utah_boundary, how="intersection")

# -----------------------------
# Plot
# -----------------------------
cmap = plt.colormaps.get_cmap("RdYlBu_r")

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

panel_info = [
    (axes[0], year_gdf, "Full-Year Average PM2.5", year_mean),
    (axes[1], week_gdf, "Study Week Average PM2.5\n(July 25–31)", week_mean),
]

# -----------------------------
# Compute shared zoom extent
# -----------------------------
minx, miny, maxx, maxy = year_gdf.total_bounds

# Add padding (adjust as needed)
pad_x = (maxx - minx) * 0.5
pad_y = (maxy - miny) * 0.5

SHIFT_X = -0.50   # negative = left
SHIFT_Y =  0.05   # positive = up

width = maxx - minx
height = maxy - miny

x_shift = width * SHIFT_X
y_shift = height * SHIFT_Y

xlim = (minx - pad_x + x_shift, maxx + pad_x + x_shift)
ylim = (miny - pad_y + y_shift, maxy + pad_y + y_shift)

for ax, gdf, title, mean_val in panel_info:
    utah_boundary.boundary.plot(ax=ax, color="black", linewidth=1)

    gdf.plot(
        ax=ax,
        column="avg_pm25",
        cmap=cmap,
        linewidth=0.1,
        edgecolor="white",
        alpha=0.95,
        legend=False,
        vmin=global_min,
        vmax=global_max,
        missing_kwds={"color": "lightgrey"}
    )

    places.boundary.plot(ax=ax, color="black", linewidth=0.45, alpha=0.20)
    lakes.plot(ax=ax, facecolor="dimgrey", edgecolor="dimgrey", linewidth=0.5, alpha=1)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # ax.set_title(title, fontsize=13, weight="bold")
    ax.axis("off")

    # Average value annotation
    # label_text = (
    #     f"Yearly mean = {mean_val:.2f} µg/m³"
    #     if "Full-Year" in title
    #     else f"Study week mean = {mean_val:.2f} µg/m³"
    # )

    # ax.text(
    #     0.97, 0.97,
    #     label_text,
    #     transform=ax.transAxes,
    #     ha="right",
    #     va="bottom",
    #     fontsize=11,
    #     weight="bold",
    #     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="none")
    # )

# Shared colorbar
sm = ScalarMappable(norm=mcolors.Normalize(vmin=global_min, vmax=global_max), cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.06)
cbar.set_label("Average PM2.5 (µg/m³)", fontsize=10, weight="bold")

tick_values = np.linspace(global_min, global_max, 5)
cbar.set_ticks(tick_values)
cbar.set_ticklabels([f"{val:.2f}" for val in tick_values])
cbar.ax.tick_params(labelsize=8)

plt.tight_layout()
# plt.savefig(OUT_PATH, dpi=300)
plt.show()

# print(f"Saved: {OUT_PATH}")