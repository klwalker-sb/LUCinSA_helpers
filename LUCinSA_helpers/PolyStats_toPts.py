#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
import geowombat as gw
import datetime
import rasterio
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
from shapely.geometry import box
from shapely.geometry import shape
from shapely.geometry import MultiPoint
from shapely.geometry import Point
from shapely.geometry import Polygon
from geopandas.tools import sjoin
from pyproj import Proj, transform
from pyproj import CRS
import xarray as xr

def GetPtsInGrid (gridFile, gridCell, ptFile):
    '''
    loads point file (from .csv with 'XCoord' and 'YCoord' columns) and returns points that overlap a gridcell
    as a geopandas GeoDataFrame. Use this if trying to match/append data to existing sample points
    rather than making a new random sample each time (e.g. if matching Planet and Sentinel points)
    Note that crs of point file is known ahead of time and hardcoded here to match specific grid file.
    '''
    out_path = Path(gridFile).parent

    '''
    with tempfile.TemporaryDirectory(dir=out_path) as temp_dir:
        temp_file = Path(temp_dir) / Path(gridFile).name
        shutil.copy(gridFile, temp_file)
        df = gpd.read_file(temp_file)
        crs_grid = df.crs
    '''
    df = gpd.read_file(gridFile)
    crs_grid = df.crs
    print('grid is in: ', crs_grid)  #ESRI:102033

    ptsdf = pd.read_csv(ptFile, index_col=0)
    pts = gpd.GeoDataFrame(ptsdf,geometry=gpd.points_from_xy(ptsdf.XCoord,ptsdf.YCoord),crs=crs_grid)

    bb = df.query(f'UNQ == {gridCell}').geometry.total_bounds

    gridBbox = box(bb[0],bb[1],bb[2],bb[3])
    gridBounds = gpd.GeoDataFrame(gpd.GeoSeries(gridBbox), columns=['geometry'], crs=crs_grid)
    print(gridBounds)

    ptsInGrid = gpd.sjoin(pts, gridBounds, op='within')
    ptsInGrid = ptsInGrid.loc[:,['geometry']]

    print("Of the {} ppts, {} are in gridCell {}". format (pts.shape[0], ptsInGrid.shape[0],gridCell))

    #Write to geojson file
    if ptsInGrid is not None:
        ptClip = Path(os.path.join(out_path,'ptsGrid_'+str(gridCell)+'.json'))
        ptsInGrid.to_file(ptClip, driver="GeoJSON")

        return ptsInGrid
        print(ptsInGrid.head(n=5))

def GetPolyStatsAtPts(out_dir, in_dir, cell, polys, numPts, seed, loadSamp=False, ptgdb=None):
    '''
    Gets values for all sampled points {'numpts'} in all polygons {'polys'} for all images in {'in_dir'}
    OR gets values for points in a previously generated dataframe {ptgdb} using loadSamp=True.
    output is a dataframe with a pt (named polygonID_pt#)
    on each row and an image index value(named YYYYDDD) in each column
    '''
    
    if loadSamp == False:
        if polys:
            ptsgdb = getRanPtsInPolys (polys, numPts, seed)
        else:
            print('There are no polygons or points to process in this cell')
            return None
    elif loadSamp == True:
        ptsgdb = ptgdb

    if ptsgdb.shape[0] > 0:
        poly_path = [f for f in os.listdir(in_dir) if f'_{cell}' in f][0]
        polys = gpd.GeoDataFrame.from_file(os.path.join(in_dir, poly_path))
        pts4326 = ptsgdb.to_crs({'init': 'epsg:4326'})
        point_polydata = sjoin(pts4326, polys, how='left')
        #print(point_polydata)
    else:
        point_polydata = None
                          
    #pd.DataFrame.to_csv(point_polydata,os.path.join(out_dir,'points_polydata.csv'), sep=',', index=True)
    return point_polydata



def GetVariablesAtPts(out_dir, in_dir, polys, numPts, seed, loadSamp=False, ptgdb=None):
    '''
    Gets values for all sampled points {'numpts'} in all polygons {'polys'} for all images in {'in_dir'}
    OR gets values for points in a previously generated dataframe {ptgdb} using loadSamp=True.
    output is a dataframe with a pt (named polygonID_pt#)
    on each row and an image index value(named YYYYDDD) in each column
    '''
    
    band_names = ['Band_1','Band_2','Band_3','Band_4']
    
    if loadSamp == False:
        if polys:
            ptsgdb = getRanPtsInPolys (polys, numPts, seed)
        else:
            print('There are no polygons or points to process in this cell')
            return None
    elif loadSamp == True:
        ptsgdb = ptgdb
    
    xy = [ptsgdb['geometry'].x, ptsgdb['geometry'].y]
    coords = list(map(list, zip(*xy)))
    
    poly_path = [f for f in os.listdir(in_dir) if f'_{cell}' in f][0]
    with rasterio.open(os.path.join(in_dir,poly_path), 'r') as comp:
        #Open each band and get values
        for b, var in band_names:
            comp.np = comp.read(b+1)
            ptsgdb[var] = [sample[b] for sample in comp.sample(coords)]     

    pd.DataFrame.to_csv(ptsgdb,os.path.join(out_dir,'ptsgdb.csv'), sep=',', index=True)
    return ptsgdb

def MakeVarDataframe(out_dir, in_dir, gridFile, cellList, groundPolys, oldest, newest, numPts, seed, loadSamp, ptFile):
                                        
    all_pts = pd.DataFrame()
    for cell in cellList:
        print ('working on cell {}'.format(cell))
        if loadSamp == True:
            points = GetPtsInGrid (gridFile, cell, ptFile)
            polys = None
            pts = GetPolyStatsAtPts(out_dir, in_dir, cell, polys, numPts, seed, loadSamp=True, ptgdb=points)
            #pts = GetVariablesAtPts(out_dir, in_dir, polys=None, numPts=3, seed=33, loadSamp=True, ptgdb=samp_pts)
        else:
            polys = GetPolygonsInGrid (gridFile, cell, groundPolys, oldest, newest)
            points = 0
            
        if pts is not None:                            
            pts.drop(columns=['geometry'], inplace=True)
            all_pts = pd.concat([all_pts, pts])
        
    print(all_pts)
    pd.DataFrame.to_csv(all_pts,os.path.join(out_dir,'pts_polyData_add.csv'), sep=',', index=True)


def MakeVarDataframe2(out_dir, in_dir, gridFile, cellList, groundPolys, oldest, newest, numPts, seed, loadSamp, ptFile):
    
    df = gpd.read_file(gridFile)
    crs_grid = df.crs
    print('grid is in: ', crs_grid)  #ESRI:102033

    ptsdf = pd.read_csv(ptFile, index_col=0)
    pts = gpd.GeoDataFrame(ptsdf,geometry=gpd.points_from_xy(ptsdf.XCoord,ptsdf.YCoord),crs=crs_grid)
    pts2 = GetPolyStatsAtPts(out_dir, in_dir, cell, polys, numPts, seed, loadSamp=True, ptgdb=points)                       
    pts2.drop(columns=['geometry'], inplace=True)
        
    print(pts2)
    pd.DataFrame.to_csv(all_pts,os.path.join(out_dir,'pts_segData.csv'), sep=',', index=True)


in_dir = 'D:/NasaProject/Paraguay/Stac_outputs/SegmentStats'
#in_dir = '/home/downspout-cel/paraguay_lc/Segmentation8858_2023/composites_probas/TS_stats/'
out_dir = 'D:/NasaProject/Paraguay/ClassificationModels/RF'
gridFile = 'C:/Users/klobw/Desktop/NasaProject/LUCinLA_grid_8858.gpkg'
samp_pts = 'D:/NasaProject/Paraguay/sampling/samplePts_FINALdfs/AllPts_Mar2023.csv'
cellList = [3881,3882,3847,3702]
#cellList = [3648,3649,3654,3655,3656,3695,3696,3697,3701,3703,3730,3731,3732,3737,3738,3766,3767,3768,3769,3770,3771,3775,3776,3777,3803,3805,3806,3809,3810,3811,3812,3841,3842,3843, 3844,3845,3873,3874,3875,3876,3877,3878,3879,3908,3909,3910,3911,3912,3913,3914,3944,3945,3946,3947,3949,3976,3977,3978,3979,3980,4010,4011,4012,4013,4038,4039,4040,4041,4059,4060,4073,4079]
MakeVarDataframe(out_dir, in_dir, gridFile, cellList, groundPolys=None, oldest=2020, newest=2021, numPts=2, seed=33, loadSamp=True, ptFile=samp_pts)


