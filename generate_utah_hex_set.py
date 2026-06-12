import polars as pl
import numpy as np
import geopandas as gpd
import h3

UTAH_SHP_PATH = "Shapefiles/tl_2019_49_tract/tl_2019_49_tract.shp"

def generate_utah_level8_hexes(bbox, resolution=8):
    minx, miny, maxx, maxy = bbox
    lat_lng_pairs = [
        (lat, lon)
        for lat in np.arange(miny, maxy, 0.005)
        for lon in np.arange(minx, maxx, 0.005)]
    hexes = set(h3.latlng_to_cell(lat, lon, resolution) for lat, lon in lat_lng_pairs)
    print(f"Generated {len(hexes)} hexes for Utah.")
    return hexes

def main():
    utah = gpd.read_file(UTAH_SHP_PATH).to_crs("EPSG:4326")
    utah_boundary = utah.dissolve()  
    utah_bbox = utah_boundary.total_bounds

    hexes = generate_utah_level8_hexes(utah_bbox)

    df = pl.DataFrame({"hex_id": sorted(hexes)})
    df.write_parquet("New Data/utah_hexes_res8.parquet")
    
if __name__ == "__main__":
    main()
