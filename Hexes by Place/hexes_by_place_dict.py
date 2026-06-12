import geopandas as gpd
import h3
from shapely.geometry import Polygon
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

def h3_to_polygon(hex): # Convert each H3 cell to a Shapely Polygon
    coords = h3.cell_to_boundary(hex)
    swapped = [(lon, lat) for lat, lon in coords] # need to swap coordinates from (lat, lon) to (lon, lat) to match the places geometry

    if swapped[0] != swapped[-1]: # just double check that the ring is closed
        swapped.append(swapped[0])
    return Polygon(swapped)

def generate_utah_level8_hexes(bbox, resolution=8): # generate all H3 hexes within the bounding box of Utah
    minx, miny, maxx, maxy = bbox

    lat_lng_pairs = [
        (lat, lon)
        for lat in np.arange(miny, maxy, 0.005) # need fine granularity to be sure we don't miss any hexes
        for lon in np.arange(minx, maxx, 0.005) # need fine granularity to be sure we don't miss any hexes
    ]

    hexes = set() # initialize the set of hexes
    for lat, lon in lat_lng_pairs:
        hex_cell = h3.latlng_to_cell(lat, lon, resolution)
        hexes.add(hex_cell)

    print(f"\nThe number of hexes covering the entire state of Utah is: {len(hexes)}")
    return hexes

def build_h3_gdf(hexes): # build a GeoDataFrame of hex geometries
    return gpd.GeoDataFrame(
        {'h3_index': list(hexes)},
        geometry=[h3_to_polygon(hex_cell) for hex_cell in hexes], # this our own function that we constructed above
        crs="EPSG:4326" # need this to match the places crs
    )

def get_hexes_by_place(places_gdf, hex_gdf): # return a dictionary mapping place name to set of H3 hexes
    joined = gpd.sjoin(hex_gdf, places_gdf[['NAME', 'geometry']], how='inner', predicate='intersects') # this is the big function which finds the hexes that intersect the place! (via their geometries)
    hex_place_dict = joined.groupby('NAME')['h3_index'].apply(set).to_dict() # group the gpd by NAME of the places, turn the h3_index column into a list, then convert the result to a dictionary

    print(f"The number of places in Utah is: {len(places_gdf)}")
    print(f"Matched {len(joined)} hex cells to place polygons (should be 15,308!)") # Sanity check, this should be 15,308
    return hex_place_dict

def print_results(hexes_by_place): # print out the list of hexes for every place in Utah
    print("\n---------- H3 Hexes by Utah Place ----------")
    for place, hexes in sorted(hexes_by_place.items()):
        print(f"\n{place} ({len(hexes)} hexes):")
        print(sorted(hexes))

PLACES = ['Saratoga Springs', 'Salt Lake City', 'Park City', 'St. George', 'Vernal', 'Logan']
def construct_plots(hexes_by_place, places, hex_gdf):
    for place_name, hex_cells in sorted(hexes_by_place.items()):
        if place_name not in PLACES: # for now, only plot the places of interest
            continue

        fig, ax = plt.subplots(figsize=(8, 8))
        
        place_geom = places[places['NAME'] == place_name] # grab this place's geometry from the full list
        hex_subset = hex_gdf[hex_gdf['h3_index'].isin(hex_cells)] # grab the corresponding hexes from the full list

        place_geom.plot(ax=ax, color='lightgrey', edgecolor='black')
        hex_subset.plot(ax=ax, color='blue', alpha=0.4, edgecolor='white')

        ax.set_title(f"{place_name} with Overlaid H3 Hexes")
        plt.tight_layout()
        plt.show()

def main():
    places = gpd.read_file("Shapefiles/cb_2019_49_place_500k/cb_2019_49_place_500k.shp") # load the 2019 place shapefile
    places = places.to_crs("EPSG:4326")  # convert CRS to match

    utah_bbox = places.total_bounds # Utah bounding box
    hexes = generate_utah_level8_hexes(utah_bbox, resolution=8)
    hex_gdf = build_h3_gdf(hexes)
    hexes_by_place = get_hexes_by_place(places, hex_gdf)

    with open("Hexes by Place/hexes_by_place.pkl", "wb") as f: # save dictionary to a pickle file
        pickle.dump(hexes_by_place, f)
    print("Saved hexes_by_place dictionary.")

    # print_results(hexes_by_place) # comment out if you don't want to print
    # construct_plots(hexes_by_place, places, hex_gdf) # comment out if you don't want to plot

if __name__ == "__main__":
    main()