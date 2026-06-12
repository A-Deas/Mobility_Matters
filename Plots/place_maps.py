from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt

PLACES_SHP_PATH = Path("Shapefiles/cb_2019_49_place_500k/cb_2019_49_place_500k.shp")

def highlight_place(place_name: str, name_col: str = "NAME"):
    # Load places
    gdf = gpd.read_file(PLACES_SHP_PATH)

    # Quick sanity check: what columns exist?
    print("Columns:", list(gdf.columns))

    if name_col not in gdf.columns:
        raise KeyError(f"Column '{name_col}' not found. Pick one from the printed columns.")

    # Filter to the place you want
    target = gdf[gdf[name_col].str.lower() == place_name.lower()]

    if target.empty:
        # Show a few examples to help you match the exact spelling
        examples = gdf[name_col].dropna().astype(str).sort_values().unique()[:30]
        raise ValueError(
            f"Place '{place_name}' not found in column '{name_col}'. "
            f"First 30 available names: {examples}"
        )

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    gdf.plot(ax=ax, linewidth=0.5, edgecolor="white", alpha=0.6)     # all places
    target.plot(ax=ax, edgecolor="black", linewidth=1.5)             # highlighted place

    ax.set_title(f"Utah place highlighted: {place_name}")
    ax.set_axis_off()
    plt.show()

if __name__ == "__main__":
    highlight_place("Trenton", name_col="NAME")
