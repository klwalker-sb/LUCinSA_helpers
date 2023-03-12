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

    with tempfile.TemporaryDirectory(dir=out_path) as temp_dir:
        temp_file = Path(temp_dir) / Path(gridFile).name
        shutil.copy(gridFile, temp_file)
        df = gpd.read_file(temp_file)
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

    print("Of the {} ppts, {} are in gridCell {}. I am actually in here". format (pts.shape[0], ptsInGrid.shape[0],gridCell))

    #Write to geojson file
    if ptsInGrid.shape[0] > 0:
        ptClip = Path(os.path.join(out_path,'ptsGrid_'+str(gridCell)+'.json'))
        ptsInGrid.to_file(ptClip, driver="GeoJSON")

        return ptsInGrid
        print(ptsInGrid.head(n=5))

def GetVariablesAtPts(out_dir, in_dir, polys, spec_indices, numPts, seed, loadSamp=False, ptgdb=None):
    '''
    Gets values for all sampled points {'numpts'} in all polygons {'polys'} for all images in {'in_dir'}
    OR gets values for points in a previously generated dataframe {ptgdb} using loadSamp=True.
    output is a dataframe with a pt (named polygonID_pt#)
    on each row and an image index value(named YYYYDDD) in each column
    '''
    
    band_names = [Max,Min,Amp,Avg,CV,Std,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec]
    
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
    
    for vi in spec_indices:
        for img in os.listdir(in_dir):
            if img.endswith(RFVars.tif) and vi in img:
                comp = rasterio.open(rasterio.open(os.path.join(in_dir,img), 'r')
                #Open each band and get values
                for b, var in band_names:
                    ras.np = ras.read(b+1)
                    ptsgdb[var] = [sample[b] for sample in src.sample(coords)]     

    #pd.DataFrame.to_csv(ptsgdb,os.path.join(out_dir,'ptsgdb.csv'), sep=',', index=True)
    return ptsgdb

def MakeVarDataframe(out_dir, spec_indices, StartYr, in_dir, imageType, gridFile, cellList,
                            groundPolys, oldest, newest, npts, seed, loadSamp, ptFile):
                                        
    Allpts = pd.DataFrame()

        for cell in cellList:
            print ('working on cell {}'.format(cell))
            if loadSamp == True:
                points = GetPtsInGrid (gridFile, cell, ptFile)
                polys = None
            else:
                polys = GetPolygonsInGrid (gridFile, cell, groundPolys, oldest, newest)
                points = None
                                        
            if loadSamp == True:
                polys=None
                pts = GetVariablesAtPts(out_dir, in_dir, polys, spec_indices, npts, seed=88, loadSamp=True, ptgdb=points)
            else:
                pts = GetVariablesAtPts(out_dir, in_dir, polys, spec_indices, npts, seed=88, loadSamp=False, ptgdb=None)

            pts.drop(columns=['geometry'], inplace=True)
            Allpts = pd.concat([Allpts, pts])
            pd.DataFrame.to_csv(Allpts,os.path.join(out_dir,'ptsgdb.csv'), sep=',', index=True)
