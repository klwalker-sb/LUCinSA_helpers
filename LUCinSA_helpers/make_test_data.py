#!/usr/bin/env python
# coding: utf-8

import os
import sys
from pathlib import Path
import csv
import rasterio as rio
import geopandas as gpd
#from shapely.geometry import box
import rioxarray

def make_test_patch(x,y,img_in,out_name,out_dir,nbsize,res):
    
    bigimg = rioxarray.open_rasterio(img_in)
    
    buf = (res + res/2)   
    grid_bbox = box(x-buf,y-buf,x+buf,y+buf)
    
    patch = bigimg.rio.clip_box(minx=x-buf,miny=y-buf, maxx=x+buf,maxy=y+buf)
    patch.rio.to_raster(os.path.join(out_dir,out_name)

def make_patch_series(in_dir,out_main,ptfile)

nbsize = 9
if isinstance(ptfile, pd.DataFrame):
        ptsdf = ptfile
    else:
        ptsdf = pd.read_csv(ptfile, index_col=0)
    ptsgdb = gpd.GeoDataFrame(ptsdf,geometry=gpd.points_from_xy(ptsdf.XCoord,ptsdf.YCoord),crs=crs_grid)
    for index, row in ptsgdb.iterrows():
        out_dir = os.path.join(out_main,'{}_{}pix'.format(index,nbsize))
        for img in in_dir:
            out_img = os.path.join(out_dir,img) 
            img_in = os.path.join(in_dir,img)
            make test_patch(ptsgdb['geometry'].x,ptsgdb['geometry'].y,img_in,out_name,out_dir,nbsize,10)                    


    