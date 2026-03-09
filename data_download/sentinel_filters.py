#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 09:01:27 2025

@author: vpremier
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import box, shape



def get_filtered_baseline(products):
    """
    Filter Sentinel-2 products to keep only the latest processing baseline 
    per scene.

    Each Sentinel-2 product name encodes its processing baseline 
    (e.g. "N0400", "N0500").
    For scenes that have been reprocessed multiple times, we keep only the newest
    (highest baseline number) version.

    Parameters
    ----------
    products : pandas.DataFrame
        DataFrame containing at least a 'Name' column with standard Sentinel-2
        product names (e.g. "S2A_MSIL1C_20150704T101006_N0500_R022_T32TPS_20231011T134419.SAFE").

    Returns
    -------
    products_fltd : pandas.DataFrame
        DataFrame where only the latest baseline per scene (commonName) is retained.
        All scenes that appear only once are kept unchanged.
    """

    # --- 1️⃣ Build 'commonName' (scene identifier ignoring baseline) ---

    products["commonName"] = [
        '_'.join([f.split('_')[i] for i in [0, 1, 2, 4, 5]])
        for f in products["Name"]
    ]

    # --- 2️⃣ Extract processing baseline string (e.g. "N0500") ---
    products["baseline"] = [f.split('_')[3] for f in products["Name"]]

    # --- 3️⃣ Sort by scene and baseline for deterministic order ---
    # (So that 'N0400' < 'N0500', etc.)
    products = products.sort_values(by=["commonName", "baseline"], ascending=[True, True])

    # --- 4️⃣ Identify scenes that have more than one baseline ---
    baseline_counts = (
        products.groupby("commonName")["baseline"]
        .nunique()
        .reset_index(name="n_baselines")
    )

    # Select only those with multiple distinct baselines
    multi_baseline = baseline_counts.loc[baseline_counts["n_baselines"] > 1, "commonName"]

    if not multi_baseline.empty:
        print("⚠️ Warning: found duplicated baselines for some scenes.")

    # --- 5️⃣ For duplicated scenes, keep only the most recent baseline ---
    # Since we've sorted ascending by baseline, 'keep="last"' keeps the highest one (e.g. N0500)
    products_to_filter = products[products["commonName"].isin(multi_baseline)]
    products_unique = (
        products_to_filter
        .drop_duplicates(subset="commonName", keep="last")
    )

    # --- 6️⃣ Merge filtered and non-duplicated scenes back together ---
    products_fltd = pd.concat([
        products[~products["commonName"].isin(multi_baseline)],
        products_unique
    ]).sort_values(by=["commonName"])
    
    
    before = len(products)
    after = len(products_fltd)
    removed = before - after

    print(f"Baseline filter: removed {removed} scenes "
          f"({after}/{before} remaining)")
    
    products_fltd = products_fltd.reset_index(drop=True)
            

    return products_fltd




def get_filtered_date(products):
    """
    Filter Sentinel-2 products by removing geometrically redundant duplicates.

    Some Sentinel-2 scenes are reprocessed multiple times (same commonName),
    resulting in footprints that are almost identical. This function keeps only
    one footprint per scene when two (or more) overlap nearly perfectly.

    The overlap is measured as the ratio of the intersection area to the
    smaller polygon area. If that ratio exceeds a given tolerance (default 0.999),
    the geometries are considered identical and one is discarded.

    Parameters
    ----------
    products : pandas.DataFrame
        DataFrame containing at least:
        - 'Name' : full Sentinel-2 product name
        - 'GeoFootprint' : dict-like GeoJSON footprint (Polygon geometry)

    Returns
    -------
    products_fltd : pandas.DataFrame
        The original products dataframe filtered so that nearly identical
        footprints (within tolerance) are removed.
    """

    # --- 1️⃣ Build a unique 'commonName' to group versions of the same scene ---
    products["commonName"] = [
        '_'.join([f.split('_')[i] for i in [0, 1, 2, 4, 5]]) for f in products["Name"]
    ]

    # Sort for deterministic grouping
    products = products.sort_values(by="Name")

    # --- 2️⃣ Select only products that appear more than once (duplicates) ---
    duplicates = products[products["commonName"].duplicated(keep=False)]
    duplicates = duplicates.sort_values(by=["commonName", "Name"])

    # --- 3️⃣ Convert GeoJSON footprints to Shapely geometries ---
    duplicates["geometry"] = duplicates["GeoFootprint"].apply(shape)
    gdf = gpd.GeoDataFrame(duplicates, geometry="geometry", crs="EPSG:4326")

    # --- 4️⃣ Loop through each group of duplicates to remove near-identical geometries ---
    keep_rows = []
    tol = 0.95  # Overlap tolerance (99.9% overlap → considered identical)

    for cname, group in gdf.groupby("commonName"):
        if len(group) == 1:
            # Only one product for this scene → always keep
            keep_rows.append(group.index[0])
            continue

        geometries = list(group.geometry)
        keep_flags = [True] * len(geometries)

        # Compare all unique pairs of geometries within the group
        for i in range(len(geometries)):
            for j in range(i + 1, len(geometries)):
                g1, g2 = geometries[i], geometries[j]

                # Compute intersection area ratio
                inter = g1.intersection(g2)
                overlap = inter.area / min(g1.area, g2.area)

                # If they overlap almost perfectly, mark one as redundant
                if overlap >= tol:
                    # If overlap is nearly perfect → keep the larger footprint
                    if g1.area >= g2.area:
                        # g1 is larger → remove g2
                        keep_flags[j] = False
                        print(f"{cname}: overlap={overlap:.5f} → removing smaller geometry (j)")
                    else:
                        # g2 is larger → remove g1
                        keep_flags[i] = False
                        print(f"{cname}: overlap={overlap:.5f} → removing smaller geometry (i)")

        # Retain only non-redundant geometries for this scene
        to_keep = group.loc[keep_flags]
        keep_rows.extend(to_keep.index.tolist())
        


    # --- 5️⃣ Build filtered GeoDataFrame and map back to original products ---
    gdf_fltd = gdf.loc[keep_rows]

    # Filter the *original* products DataFrame
    products_fltd = products.loc[products.index.isin(gdf_fltd.index) | 
                                 ~products["commonName"].isin(gdf["commonName"])]
    
    
    # ---- Plot each remaining pair ----
    # for cname, group in gdf.groupby('commonName'):
    #     print(cname)
    #     if len(group) < 2:
    #         continue
       
    #     fig, ax = plt.subplots(figsize=(6, 6))
    #     colors = ['red', 'blue', 'green', 'orange'][:len(group)]
    #     group.plot(ax=ax, facecolor='none', edgecolor=colors, linewidth=2)
       
    #     for idx, row in group.iterrows():
    #         baseline = [s for s in row['Name'].split('_') if s.startswith('N')][0]
    #         ax.annotate(
    #             text=baseline,
    #             xy=row.geometry.centroid.coords[0],
    #             ha='center', fontsize=8, color='black'
    #         )
       
    #     ax.set_title(f"Footprint comparison for {cname}")
    #     ax.set_xlabel("Longitude")
    #     ax.set_ylabel("Latitude")
    #     plt.tight_layout()
    #     plt.show()


    before = len(products)
    after = len(products_fltd)
    removed = before - after
       
    print(f"Date/footprint filter: removed {removed} scenes "
          f"({after}/{before} remaining)")
    
    products_fltd = products_fltd.reset_index(drop=True)
 
    return products_fltd




def filter_RON(products, RON_list):
    """
    Filter Sentinel-2 products by Relative Orbit Number (RON).

    Parameters
    ----------
    products : pd.DataFrame
        DataFrame containing at least a 'Name' column (e.g., 'S2A_MSIL1C_20240704T101031_N0500_R022_T32TPS_20231011T134419.SAFE').
    RON_list : list, optional
        List of RON strings (e.g., ['R022', 'R051']) to keep. 
        If None or empty, the function returns all products unchanged.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only rows with RON in RON_list.
    """

    # ---- Extract the RON code from the product name ----
    products["RON"] = products["Name"].str.split("_").str[4]


    before = len(products)
    products_fltd = products[products["RON"].isin(RON_list)]
    after = len(products_fltd)
    removed = before - after
    
    print(f"RON filter: removed {removed} scenes "
          f"({after}/{before} remaining)")

    products_fltd = products_fltd.reset_index(drop=True)

    return products_fltd

      


