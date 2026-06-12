# import pandas as pd
# import h3

# df = pd.read_csv("Wildfire_Investigation/candidate_hexes.csv")

# hexes = set(df["hex_id"])

# visited = set()
# clusters = []

# for h in hexes:
#     if h in visited:
#         continue

#     stack = [h]
#     cluster = set()

#     while stack:
#         current = stack.pop()
#         if current in visited:
#             continue

#         visited.add(current)
#         cluster.add(current)

#         neighbors = h3.grid_disk(current, 1)
#         for n in neighbors:
#             if n in hexes and n not in visited:
#                 stack.append(n)

#     clusters.append(cluster)

# # Find largest cluster
# largest_cluster = max(clusters, key=len)

# print(f"Total clusters: {len(clusters)}")
# print(f"Largest cluster size: {len(largest_cluster)}")

# # Filter dataframe
# df_main = df[df["hex_id"].isin(largest_cluster)].copy()

# df_main.to_csv("Wildfire_Investigation/final_wildfire_hexes.csv", index=False)

# print("Saved main plume hexes")

import pandas as pd
import h3

df = pd.read_csv("Wildfire_Investigation/final_wildfire_hexes2.csv")
hexes = set(df["hex_id"])

visited = set()
clusters = []

# --- find connected components ---
for h in hexes:
    if h in visited:
        continue

    stack = [h]
    cluster = set()

    while stack:
        current = stack.pop()
        if current in visited:
            continue

        visited.add(current)
        cluster.add(current)

        neighbors = set(h3.grid_disk(current, 1))
        neighbors.discard(current)

        for n in neighbors:
            if n in hexes and n not in visited:
                stack.append(n)

    clusters.append(cluster)

largest_cluster = max(clusters, key=len)

print(f"Total clusters: {len(clusters)}")
print(f"Largest cluster size before fill: {len(largest_cluster)}")

# --- fill likely interior holes ---
expanded_cluster = set(largest_cluster)

candidate_missing = set()
for h in largest_cluster:
    nbrs = set(h3.grid_disk(h, 1))
    nbrs.discard(h)
    for n in nbrs:
        if n not in largest_cluster:
            candidate_missing.add(n)

filled_hexes = set()

for n in candidate_missing:
    nbrs_of_n = set(h3.grid_disk(n, 1))
    nbrs_of_n.discard(n)

    plume_neighbor_count = sum(1 for x in nbrs_of_n if x in largest_cluster)

    # Conservative hole-fill rule:
    # add only if surrounded by many plume hexes
    if plume_neighbor_count >= 4:
        filled_hexes.add(n)

expanded_cluster.update(filled_hexes)

print(f"Filled hexes added: {len(filled_hexes)}")
print(f"Largest cluster size after fill: {len(expanded_cluster)}")

# Save final set
df_main = pd.DataFrame({"hex_id": sorted(expanded_cluster)})
df_main.to_csv("Wildfire_Investigation/final_wildfire_hexes3.csv", index=False)

print("Saved filled main plume hexes")