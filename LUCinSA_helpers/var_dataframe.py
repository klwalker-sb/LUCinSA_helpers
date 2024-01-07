#!/usr/bin/env python
# coding: utf-8

import os
import sys
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
import csv
from LUCinSA_helpers.ts_profile import get_pts_in_grid, get_polygons_in_grid

def get_variables_at_pts(out_dir, in_dir, polys, spec_indices, si_vars, numpts, seed, load_samp=False, ptgdb=None):
    '''
    Gets values for all sampled points {'numpts'} in all polygons {'polys'} for all images in {'in_dir'}
    OR gets values for points in a previously generated dataframe {ptgdb} using loadSamp=True.
    output is a dataframe with a pt (named polygonID_pt#)
    on each row and an image index value(named YYYYDDD) in each column
    '''
    if load_samp == False:
        if polys:
            ptsgdb = get_ran_pts_in_polys (polys, numpts, seed)
        else:
            print('There are no polygons or points to process in this cell')
            return None
    elif load_samp == True:
        ptsgdb = ptgdb

    xy = [ptsgdb['geometry'].x, ptsgdb['geometry'].y]
    coords = list(map(list, zip(*xy)))
    
    print(spec_indices)
    for vi in spec_indices:
        sys.stdout.write("working on {} \n".format(vi))
        print(in_dir)
        for img in os.listdir(in_dir):
            if img.endswith('RFVars.tif') and vi in img:
                sys.stdout.write('Extracting variables from: {} \n'.format(img))
                comp = rio.open(os.path.join(in_dir,img),'r')
                #Open each band and get values
                for b, var in enumerate(si_vars):
                    sys.stdout.write('{}:{}, '.format(b,var))
                    comp.np = comp.read(b+1)
                    varn = ('var_{}_{}'.format(vi,var))
                    ptsgdb[varn] = [sample[b] for sample in comp.sample(coords)]
    #pd.DataFrame.to_csv(ptsgdb,os.path.join(out_dir,'ptsgdb.csv'), sep=',', index=True)
    return ptsgdb

def make_var_dataframe(out_dir, spec_indices, si_vars, in_dir, grid_file, cell_list,
                            ground_polys, oldest, newest, npts, seed, load_samp, ptfile):
    
    all_pts = pd.DataFrame()
    if isinstance(cell_list, list):
        cells = cell_list
    elif cell_list.endswith('.csv'): 
        cells = []
        with open(cell_list, newline='') as cell_file:
            for row in csv.reader(cell_file):
                cells.append (row[0])
    else:
        print('cell_list needs to be a list or path to .csv file with list')
    for cell in cells:
        print ('working on cell {}'.format(cell))
        if load_samp == True:
            sys.stdout.write('loading sample from points for cell {} \n'.format(cell))
            points = get_pts_in_grid (grid_file, cell, ptfile)
            polys = None
        else:
            sys.stdout.write('loading sample from polygons for cell {} \n'.format(cell))
            polys = get_polygons_in_grid (grid_file, cell, ground_polys, oldest, newest)
            points = None
          
        if isinstance(points, gpd.GeoDataFrame) or polys is not None:
            var_dir = os.path.join(in_dir,'{:06d}'.format(int(cell)),'comp')
           
            if load_samp == True:
                polys=None
                pts = get_variables_at_pts(out_dir, var_dir, polys, spec_indices, si_vars, npts, seed=88, load_samp=True, ptgdb=points)
            else:
                pts = get_variables_at_pts(out_dir, var_dir, polys, spec_indices, si_vars, npts, seed=88, load_samp=False, ptgdb=None)

            pts.drop(columns=['geometry'], inplace=True)
            all_pts = pd.concat([all_pts, pts])
          
        else:
            sys.stdout.write('skipping this cell \n')
            pass
    
    pts_out = pd.DataFrame.to_csv(all_pts,os.path.join(out_dir,'ptsgdb.csv'), sep=',', index=True)

    pts_in = pd.read_csv(ptfile, index_col=0)
    rfdf = all_pts.merge(pts_in, left_index=True, right_index=True)
    pd.DataFrame.to_csv(rfdf,os.path.join(out_dir,'RFdf.csv'), sep=',', index=True)
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


