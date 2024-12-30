#!/usr/bin/env python
# coding: utf-8
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import shapely
from pyproj import Proj, transform

def create_proj_grid(AOI_bounds, cell_size, out_epsg, grid_file, samp_pts=False):
    
    """
    creates <grid_file>, (a tiled/mesh grid for parallel processing), covering <AOI_bounds> (the area of interest (AOI) bounding box, 
    in form of shapefile, or geopackage) Each unique, UNQ, grid cell has the <cell_size> (width and height -- in the unit of <out_epsg>)
    <out_epsg> is the coordinate reference system EPSG code (as an integer). If <samp_pts> == True, each grid cell's centroid is saved 
    as a point shapefile,'<grid_file>_sampPts.shp'. 
    Returns proc grid as a geodataframe and outputs a .gpkg file with name <grid_file>.gpkg' (and optional pt file)
    """
    
    gdf_bounds = gpd.read_file(AOI_bounds)
    gdf_proj = gdf_bounds.to_crs(out_epsg)
    xmin, ymin, xmax, ymax = gdf_proj.total_bounds
    n_cells=(xmax-xmin) / cell_size
    grid_cells = []
    for x in np.arange(xmin, xmax+cell_size, cell_size):
        for y in np.arange(ymin, ymax+cell_size, cell_size):
            grid_cells.append(shapely.geometry.box(x, y, x-cell_size, y+cell_size))         
    df_grids = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=out_epsg)
    #df_grids = df_grids.loc[[geom.intersects(gdf_proj.geometry.values[0]) for geom in df_grids.geometry.values]]
    df_grids.loc[:, 'UNQ'] = range(1, df_grids.shape[0]+1)
    
    #os.makedirs(grid_file, exist_ok=True)
    df_grids.to_file(grid_file, driver='GPKG')
    if samp_pts == True:
        pts = gpd.GeoDataFrame(df_grids.drop(columns=["geometry"]), geometry=df_grids.centroid, crs=out_epsg) 
        pts.to_file(grid_file.replace(".", "_sampPts."))
        
    return df_grids