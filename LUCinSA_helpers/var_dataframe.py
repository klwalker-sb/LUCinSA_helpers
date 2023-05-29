#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
import geowombat as gw
import datetime
import rasterio as rio
from rasterio import plot
import matplotlib.pyplot as plt
import shutil
import tempfile
import json
import random
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Proj, transform
from pyproj import CRS
import xarray as xr

def get_pts_in_grid (grid_file, grid_cell, ptfile):
    '''
    loads point file (from .csv with 'XCoord' and 'YCoord' columns) and returns points that overlap a gridcell
    as a geopandas GeoDataFrame. Use this if trying to match/append data to existing sample points
    rather than making a new random sample each time (e.g. if matching Planet and Sentinel points)
    Note that crs of point file is known ahead of time and hardcoded here to match specific grid file.
    '''
    out_path = Path(gridFile).parent

    with tempfile.TemporaryDirectory(dir=out_path) as temp_dir:
        temp_file = Path(temp_dir) / Path(gridFile).name
        shutil.copy(gridFile, temp_file)
        df = gpd.read_file(temp_file)
        crs_grid = df.crs
    print('grid is in: ', crs_grid)  #ESRI:102033

    ptsdf = pd.read_csv(ptFile, index_col=0)
    pts = gpd.GeoDataFrame(ptsdf,geometry=gpd.points_from_xy(ptsdf.XCoord,ptsdf.YCoord),crs=crs_grid)

    bb = df.query(f'UNQ == {grid_cell}').geometry.total_bounds

    grid_bbox = box(bb[0],bb[1],bb[2],bb[3])
    grid_bounds = gpd.GeoDataFrame(gpd.GeoSeries(grid_bbox), columns=['geometry'], crs=crs_grid)
    print(grid_bounds)

    pts_in_grid = gpd.sjoin(pts, grid_bounds, op='within')
    pts_in_grid = pts_in_grid.loc[:,['geometry']]

    print("Of the {} ppts, {} are in gridCell {}. I am actually in here". format (pts.shape[0], pts_in_grid.shape[0],grid_cell))

    #Write to geojson file
    if pts_in_grid.shape[0] > 0:
        pt_clip = Path(os.path.join(out_path,'ptsGrid_'+str(grid_cell)+'.json'))
        pts_in_grid.to_file(pt_clip, driver="GeoJSON")

        return pts_in_grid
        print(pts_in_grid.head(n=5))

def get_variables_at_pts(out_dir, in_dir, polys, spec_indices, numPts, seed, load_samp=False, ptgdb=None):
    '''
    Gets values for all sampled points {'numpts'} in all polygons {'polys'} for all images in {'in_dir'}
    OR gets values for points in a previously generated dataframe {ptgdb} using loadSamp=True.
    output is a dataframe with a pt (named polygonID_pt#)
    on each row and an image index value(named YYYYDDD) in each column
    '''
    
    band_names = [Max,Min,Amp,Avg,CV,Std,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec]
    
    if load_samp == False:
        if polys:
            ptsgdb = get_ran_pts_in_polys (polys, numPts, seed)
        else:
            print('There are no polygons or points to process in this cell')
            return None
    elif load_samp == True:
        ptsgdb = ptgdb

    xy = [ptsgdb['geometry'].x, ptsgdb['geometry'].y]
    coords = list(map(list, zip(*xy)))
    
    for vi in spec_indices:
        for img in os.listdir(in_dir):
            if img.endswith(rf_vars.tif) and vi in img:
                with rio.open(rio.open(os.path.join(in_dir,img), 'r') as comp:
                    #Open each band and get values
                    for b, var in enumerate(band_names):
                        comp.np = comp.read(b+1)
                        varn = ('var_{}_{}'.format(vi,var))
                        ptsgdb[varn] = [sample[b] for sample in comp.sample(coords)]     

    #pd.DataFrame.to_csv(ptsgdb,os.path.join(out_dir,'ptsgdb.csv'), sep=',', index=True)
    return ptsgdb

def make_var_dataframe(out_dir, spec_indices, StartYr, in_dir, imageType, grid_file, cell_list,
                            ground_polys, oldest, newest, npts, seed, load_samp, ptfile):
                                        
    all_pts = pd.DataFrame()

        for cell in cell_list:
            print ('working on cell {}'.format(cell))
            if load_samp == True:
                points = get_pts_in_grid (grid_file, cell, ptfile)
                polys = None
            else:
                polys = Get_polygons_in_grid (grid_file, cell, ground_polys, oldest, newest)
                points = None
                                        
            if load_samp == True:
                polys=None
                pts = get_variables_at_pts(out_dir, in_dir, polys, spec_indices, npts, seed=88, load_samp=True, ptgdb=points)
            else:
                pts = get_variables_at_pts(out_dir, in_dir, polys, spec_indices, npts, seed=88, load_samp=False, ptgdb=None)

            pts.drop(columns=['geometry'], inplace=True)
            all_pts = pd.concat([all_pts, pts])
            pd.DataFrame.to_csv(all_pts,os.path.join(out_dir,'ptsgdb.csv'), sep=',', index=True)


def get_variables_at_pts_external(out_dir, ras_in,ptfile):

    ptsdf = pd.read_csv(ptfile, index_col=0)
    ptsgdb = gpd.GeoDataFrame(ptsdf,geometry=gpd.points_from_xy(ptsdf.XCoord,ptsdf.YCoord),crs='epsg:8858')
    #pts4326 = ptsgdb.to_crs({'init': 'epsg:4326'})
    xy = [ptsgdb['geometry'].x, ptsgdb['geometry'].y]
    coords = list(map(list, zip(*xy)))
    
    with rio.open(ras_in, 'r') as comp:
        comp.np = comp.read(3)
        ptsgdb['B3'] = [sample[2] for sample in comp.sample(coords)]     

    pd.DataFrame.to_csv(ptsgdb,os.path.join(out_dir,'seg_join6.csv'), sep=',', index=True)
    return ptsgdb


#ras_in = 'D:/NasaProject/Paraguay/CropMapsComparison/GCEP30/LGRIP30_2015_S30W60_001_2023014175240_8858.tif'
#ras_in = 'D:/NasaProject/JordanConeLC_AOI_Py/mosaic2018_8858.tif'
#ras_in = 'D:/NasaProject/Paraguay/Potapov_soy/SouthAmerica_Soybean_2021_8858.tif'
#ras_in = 'D:/NasaProject/SegmentationResults_Edges3/SegmentLayers.tif'
#samp_pts = 'D:/NasaProject/Paraguay/sampling/samplePts_FINALdfs/AllPts_Mar2023.csv'
#samp_pts = 'D:/NasaProject/Paraguay/ClassificationModels/RF/seg_join5.csv'
#out_dir = 'D:/NasaProject/Paraguay/ClassificationModels/RF'
#get_variables_at_pts_external(out_dir, ras_in, samp_pts)

# %matplotlib inline
#from shapely.geometry import Point
#from geopandas import datasets, GeoDataFrame, read_file

#polydf = read_file('D:/NasaProject/SegmentationResults_Edges2/SegPred_ftp8020_NASA.gpkg')
#ptsdf = pd.read_csv('D:/NasaProject/Paraguay/sampling/samplePts_FINALdfs/AllPts_Mar2023.csv', index_col=0)
#ptsgdb = gpd.GeoDataFrame(ptsdf,geometry=gpd.points_from_xy(ptsdf.XCoord,ptsdf.YCoord),crs='epsg:8858')
#pts4326 = ptsgdb.to_crs({'init': 'epsg:4326'})
#join_left_df = pts4326.sjoin(polydf, how="left")        
#polySamp = polydf.sjoin(ptsgdb, how="left")  


#polySamp2 = polySamp.dropna(subset=['index_right'])
#print(polySamp2)

#polySamp2.to_file('D:/NasaProject/SegmentationResults_Edges2/polysJoined.gpkg',driver='GPKG')


