import rasterio
import rasterio.features

import numpy as np
import warnings 
import os
from pathlib import Path
import geopandas as gpd
import pandas as pd

output_path = Path("outputs/proportional_grazing_stocking_gap.csv")

strassburg_pasture_gap_data = Path("data/rasters/pixel_proportion_restorable_pasture_75perc_gap_closure.tif") # yield / max yield, so 0-1

warnings.filterwarnings("ignore", category=RuntimeWarning)

countries_shapefile = Path("data/country_data/geoBoundariesCGAZ_ADM0.shp")
if not countries_shapefile.is_file():
    import _get_country_boundaries
    _get_country_boundaries.get_country_data()

countries_data = gpd.read_file(countries_shapefile)

isoa3_str = "shapeGroup"
country_isos = countries_data[isoa3_str].unique()

def process_country(country_geom, weights, value_dataset, out_shape, transform):
        """weights are assumed to raw km2 values, extra_weights can be anything"""
        
        mask = rasterio.features.geometry_mask(
            [country_geom],
            out_shape=out_shape,
            transform=transform,
            invert=True
        )
    
        raw_weights = np.where(mask, weights, np.nan)
        raw_vals = np.where(mask, value_dataset, np.nan)
        
        valid_indices = (~np.isnan(raw_weights)) & (~np.isnan(raw_vals)) 

        if not np.any(valid_indices):
            return np.nan, np.nan, np.nan, np.nan

        physical_area = np.nansum(raw_weights[valid_indices])

        weights_used = raw_weights[valid_indices]
        vals_used = raw_vals[valid_indices]

        weights_normalised = weights_used / np.nansum(weights_used)

        mean_value = np.nansum(vals_used * weights_normalised)

        pixel_count = vals_used.size

        variance = np.var(vals_used)
        mean_sem = np.sqrt(variance * np.sum(weights_normalised ** 2))

        return mean_value, mean_sem, int(pixel_count), physical_area


output_rows = []

if not output_path.parent.is_dir():
    os.makedirs(output_path.parent, exist_ok=True)

with rasterio.open(strassburg_pasture_gap_data) as gap_src:
    
    crs = gap_src.crs
    transform = gap_src.transform

    gap_data = gap_src.read(1)
    out_shape = gap_data.shape
    
    gap_data = np.where(np.isnan(gap_data), -9999, gap_data)
    gap_data = np.where(gap_data == -9999, 0, gap_data) # this is fine for this purpose

    area_data = np.full_like(gap_data, gap_src.res[0] * gap_src.res[1] / 1e6) # km2 per pixel (works because data is EA)
    
    countries_reprojected = countries_data.to_crs(crs)

    for i, iso in enumerate(country_isos):

        print("Processing country", iso, f"({i+1}/{len(country_isos)})")

        country_geom = countries_reprojected[countries_reprojected[isoa3_str] == iso].geometry.union_all()

        mean_gap, sem_gap, pixel_count, physical_area = process_country(
            country_geom, area_data, gap_data, out_shape, transform
        )

        if np.isnan(mean_gap):
             print(f"Warning: No valid data for country {iso}")

        output_rows.append({
            "iso_a3": iso,
            "mean_gap": mean_gap,
            "sem_gap": sem_gap,
            "pixel_count": pixel_count,
            "physical_area_km2": physical_area
        })
    
output_data = pd.DataFrame(output_rows)
output_data.to_csv(output_path, index=False)