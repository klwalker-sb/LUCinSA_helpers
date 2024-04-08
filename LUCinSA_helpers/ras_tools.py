#!/usr/bin/env python
# coding: utf-8

import os
import sys
from pathlib import Path
import csv
from rasterio.enums import Resampling
import rasterio as rio
import geopandas as gpd
import pandas as pd
#import rioxarray

def make_test_patch(xpt,ypt,img_in,out_name,out_dir,nbsize,res):
    '''
    clips raster to patch of size {nbsize, nbsize} around point with coords {xpt,ypt}
    '''
    
    bigimg = rioxarray.open_rasterio(img_in)
    buf = (res + res/2)   
    #grid_bbox = box(xpt-buf,ypt-buf,xpt+buf,ypt+buf)
    
    patch = bigimg.rio.clip_box(minx=(xpt-buf),miny=(ypt-buf),maxx=(xpt+buf),maxy=(ypt+buf))
    patch.rio.to_raster(os.path.join(out_dir,out_name))

def make_test_set(ptfile, xpt,ypt,img_in,out_name,out_dir,nbsize,res):
    '''
    makes test patches of size {nbsize, nbsize} for all points in {ptfile}
    ptfile must have fields XCoord and YCoord in same input as images (here hardcoded as epsg:8858)
    '''
    if isinstance(ptfile, pd.DataFrame):
        ptsdf = ptfile
    else:
        ptsdf = pd.read_csv(ptfile, index_col=0)
    ptsgdb = gpd.GeoDataFrame(ptsdf,geometry=gpd.points_from_xy(ptsdf.XCoord,ptsdf.YCoord),crs='epsg:8858')
    for index, row in ptsgdb.iterrows():
        out_dir1 = '{}_{}pix'.format(index,nbsize)
        out_dir = os.path.join(out_main,out_dir1)
        if not os.path.exists(out_dir):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        for img in os.listdir(in_dir):
            if img.endswith('tif') and img.startswith('2020'):
                out_name = os.path.join(out_dir,'{}_{}'.format(ind,img)) 
                img_in = os.path.join(in_dir,img)
                make_test_patch(ptsgdb['geometry'].x[index],ptsgdb['geometry'].y[index],img_in,out_name,out_dir,nbsize,10)   
    
def downsample_images(cell_list, in_dir_main, local_dir, common_str, out_dir_main, new_res):

    cells = []
    if isinstance(cell_list, list):
        cells = cell_list
    elif isinstance(cell_list, str) and cell_list.endswith('.csv'): 
        with open(cell_list, newline='') as cell_file:
            for row in csv.reader(cell_file):
                cells.append (row[0])
    elif isinstance(cell_list, int) or isinstance(cell_list, str): # if runing individual cells as array via bash script
        cells.append(cell_list) 
        
    for cell in cells:
        cell_path = os.path.join(in_dir_main,'{:06d}'.format(int(cell)), local_dir)
        if not os.path.exists(cell_path):
            print('there is no {} folder for cell {}.'.format(local_dir, cell))
        else:
            out_dir = os.path.join(out_dir_main,'{:06d}'.format(int(cell)), local_dir)
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            
            in_files = [f for f in os.listdir(cell_path) if f.startswith(common_str)]
            with rio.open(os.path.join(cell_path,in_files[0]), 'r') as ex_src:
                scale_factor_x = ex_src.res[0] / int(new_res)
                scale_factor_y = ex_src.res[1] / int(new_res) 
                
            for in_file in in_files:
                with rio.open(os.path.join(cell_path,in_file), 'r') as src_in:
                    profile = src_in.profile.copy()
                    data = src_in.read(
                        out_shape=(
                            src_in.count,
                            int(src_in.height * scale_factor_y),
                            int(src_in.width * scale_factor_x)
                        ),
                        resampling=Resampling.bilinear
                    )

                    transform = src_in.transform * src_in.transform.scale(
                        (1 / scale_factor_x),
                        (1 / scale_factor_y)
                    )
                    
                    profile.update({"height": data.shape[-2],
                        "width": data.shape[-1],
                        "transform": transform})

                with rio.open(os.path.join(out_dir,in_file), "w", **profile) as outdata:
                    outdata.write(data)                            
    