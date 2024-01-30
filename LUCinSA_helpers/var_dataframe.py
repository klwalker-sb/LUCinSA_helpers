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
from LUCinSA_helpers.ts_profile import get_pts_in_grid, get_polygons_in_grid, get_ran_pts_in_polys
from LUCinSA_helpers.rf import getset_feature_model
from LUCinSA_helpers.ts_composite import make_ts_composite

def clip_singleton_feat_to_cell(singleton_feat,singleton_feat_dict,out_dir,clip_to_ras):
    with open(singleton_feat_dict, 'r+') as singleton_feat_dict:
        dic = json.load(singleton_feat_dict)
    if not singleton_feat in dic:
        sys.stderr.write('ERROR: do not know path for {}. Add to singleton_var_dict and rerun'.format(singleton_feat))
    else:
        sf_path = dic[singleton_feat]['path']
        sf_col = dic[singleton_feat]['col']
        sys.stdout.write('getting {} from {}'.format(singleton_feat,sf_path))    
        singleton_clipped = os.path.join(out_dir,'{}.tif'.format(singleton_feat))
        if not os.path.isfile(singleton_clipped):
            ## clip large singleton raster to extent of other rasters in stack for grid cell
            src_small = gdal.Open(clip_to_ras)
            ulx, xres, xskew, uly, yskew, yres  = src_small.GetGeoTransform()
            lrx = ulx + (src_small.RasterXSize * xres)
            lry = uly + (src_small.RasterYSize * yres)
            geometry = [[ulx,lry], [ulx,uly], [lrx,uly], [lrx,lry]]
            roi = [Polygon(geometry)]
            with rio.open(clip_to_ras) as src0:
                out_meta = src0.meta.copy()
                out_meta.update({"count":1})
            with rio.open(sf_path) as src:
                out_image, transformed = rio.mask.mask(src, roi, crop = True)
                with rio.open(singleton_clipped, 'w', **out_meta) as dst:
                    dst.write(out_image)
                    
        return singleton_clipped                 
                        
def get_variables_at_pts(in_dir, out_dir, feature_model, feature_mod_dict, start_yr, polys, numpts, seed, load_samp=False, ptgdb=None):
    '''
    Gets values for all sampled points {'numpts'} in all polygons {'polys'} for all images in {'in_dir'}
    OR gets values for points in a previously generated dataframe {ptgdb} using loadSamp=True.
    output is a dataframe with a pt (named polygonID_pt#)
    on each row and an image index value(named YYYYDDD) in each column
    '''
    stack_path = os.path.join(in_dir,'{}_{}_stack.tif'.format(feature_model, start_yr))
    #stack_path = os.path.join(in_dir,'stack.tif')
    if not os.path.isfile(stack_path):
        print('need to create variable stack for {}_{} first.'.format(feature_model, start_yr))
        ptsgdb = None
    
    else:
        band_names = getset_feature_model(feature_mod_dict, feature_model)[4]
    
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
    
        sys.stdout.write('Extracting variables from stack')
        with rio.open(os.path.join(in_dir,'{}_{}_stack.tif'.format(feature_model, start_yr)),'r') as comp:
        #with rio.open(os.path.join(in_dir,'stack.tif'),'r') as comp:
            #Open each band and get values
            for b, band in enumerate(band_names):
                sys.stdout.write('{}:{}'.format(b,band))
                comp.np = comp.read(b+1)
                varn = ('var_{}'.format(band))
                ptsgdb[varn] = [sample[b] for sample in comp.sample(coords)]
            #pd.DataFrame.to_csv(ptsgdb,os.path.join(out_dir,'ptsgdb.csv'), sep=',', index=True)
    
    return ptsgdb

def make_var_dataframe(in_dir, out_dir, grid_file, cell_list, feature_model, feature_mod_dict, start_yr,
                            polyfile, oldest, newest, npts, seed, load_samp, ptfile):
    
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
        var_dir = os.path.join(in_dir,'{:06d}'.format(int(cell)),'comp')
        print ('working on cell {}'.format(cell))
        if load_samp == True:
            sys.stdout.write('loading sample from points for cell {} \n'.format(cell))
            points = get_pts_in_grid (grid_file, cell, ptfile)
            polys = None
        else:
            sys.stdout.write('loading sample from polygons for cell {} \n'.format(cell))
            polys = get_polygons_in_grid (grid_file, cell, polyfile, oldest, newest)
            points = None
        
        sys.stdout.write('looking for {}_{}_stack.tif in {} to extract variables'.format(feature_model,start_yr,var_dir))
        if isinstance(points, gpd.GeoDataFrame) or polys is not None:
            if load_samp == True:
                polys=None
                pts = get_variables_at_pts(var_dir, out_dir, 
                                           feature_model, feature_mod_dict, start_yr, 
                                           polys, npts, seed=88, load_samp=True, ptgdb=points)
            else:
                pts = get_variables_at_pts(var_dir, out_dir, 
                                           feature_model, feature_mod_dict, start_yr,
                                           polys, npts, seed=88, load_samp=False, ptgdb=None)
            if pts is not None:
                pts.drop(columns=['geometry'], inplace=True)
                all_pts = pd.concat([all_pts, pts])
          
        else:
            sys.stdout.write('skipping this cell \n')
            pass
    
    pts_out = pd.DataFrame.to_csv(all_pts,os.path.join(out_dir,'ptsgdb.csv'), sep=',', index=True)

    pts_in = pd.read_csv(ptfile, index_col=0)
    rfdf = all_pts.merge(pts_in, left_index=True, right_index=True)
    pd.DataFrame.to_csv(rfdf,os.path.join(out_dir,'RFdf_{}_{}.csv'.format(feature_model,start_yr)), sep=',', index=True)
    pd.DataFrame.to_csv(all_pts,os.path.join(out_dir,'ptsgdb_{}-{}.csv'.format(feature_model,start_yr)), sep=',', index=True)
    
def append_feature_dataframe(in_dir, ptfile, feat_df, cell_list, grid_file, out_dir, start_yr, start_mo, spec_indices, si_vars,
                             spec_indices_pheno=None, pheno_vars=None, singleton_vars=None, singleton_var_dict=None, poly_vars=None, 
                             poly_var_path=None, scratch_dir=None):
    all_pts = pd.DataFrame()

    cells = []
    if isinstance(cell_list, list):
        cells = cell_list
    elif cell_list.endswith('.csv'): 
        with open(cell_list, newline='') as cell_file:
            for row in csv.reader(cell_file):
                cells.append (row[0])   
    elif isinstance(cell_list, int) or isinstance(cell_list, str): # if runing individual cells as array via bash script
        cells.append(cell_list) 
        
    for cell in cells:
        sys.stderr.write('working on cell {}... \n'.format(cell))
        cell_dir = os.path.join(in_dir,'{:06d}'.format(int(cell)))
        ptsgdb = get_pts_in_grid (grid_file, int(cell), ptfile)
        if scratch_dir:
            out_dir_int = os.path.join(scratch_dir,'{}'.format(cell))
        else:
            out_dir_int = os.path.join(cell_dir,'comp')       

        xy = [ptsgdb['geometry'].x, ptsgdb['geometry'].y]
        coords = list(map(list, zip(*xy)))
        
        for sip in spec_indices_phen:
            if sip is not None and sip != 'None':
                comp_dir = os.path.join(cell_dir,'comp',sip)
                for temp in ['wet','dry']:
                    sys.stderr.write('extracting {} pheno vars for {}... \n'.format(temp,sip))
                    pvars = [si for si in sip if si.split("_")[1] == temp]
                    phen_bands = [f'maxv_{temp}',f'maxd_{temp}',f'sosv_{temp}',f'sosd_{temp}',
                                   f'rog{temp}',f'eosv{temp}',f'eosd{temp}',f'ros{temp}',f'los{temp}']
                    if len(wet_vars) > 0:
                        phen_comp = os.path.join(comp_dir, '{:06d}_{}_{}_Phen{}.tif'.format(int(cell),start_yr,si,temp.upper()))
                        if os.path.exists(phen_comp) == True:
                            sys.stderr.write('getting variables from existing stack')
                        else:
                            sys.stderr.write('no existing stack. calculating new varaibles...'
                                make_ts_composite(cell,cell_dir,comp_dir,start_yr,start_mo,sip,phen_bands)
                        phen_vars = rio.open(phen_comp,'r')
                        for b, band in enumerate(phen_bands):
                            sys.stdout.write('{}:{}'.format(b,band))
                            phen_vars.np = phen_vars.read(b+1)
                            varn = ('var_{}_{}'.format(si,band))
                            ptsgdb[varn] = [sample[b] for sample in phen_vars.sample(coords)]

        for si in spec_indices:
            if si is not None and si != ' ' and si !='None':
                sys.stderr.write('extracting {}... \n'.format(si))
                comp_dir = os.path.join(cell_dir,'comp',si)
                img_dir = os.path.join(cell_dir,'brdf_ts','ms',si)
                    if os.path.isdir(img_dir):
                        new_vars = make_ts_composite(cell, img_dir, out_dir_int, start_yr, start_mo, si, si_vars)
                        comp = rio.open(new_vars,'r')
                        for b, band in enumerate(si_vars):
                            sys.stdout.write('{}:{}'.format(b,band))
                            comp.np = comp.read(b+1)
                            varn = ('var_{}_{}'.format(si,band))
                            ptsgdb[varn] = [sample[b] for sample in comp.sample(coords)]                
                    else:
                        sys.stderr.write ('no index {} created for {} \n'.format(si,cell))
                                             
        if poly_vars is not None and poly_vars != 'None':
            for pv in poly_vars:
                ppath = os.path.join(poly_var_path,'{}_{}.tif'.format(pv,cell))
                if os.path.isfile(ppath):
                    with rio.open(ppath, 'r') as src:
                        vals = src.read(1)
                        varn = 'var_poly_{}'.format(pv)
                        ptsgdb[varn] = [sample[0] for sample in src.sample(coords)] 
                else: 
                    sys.stderr.write ('no var {} created for {} \n'.format(pv,cell))   
        
        if singleton_vars is not None and singleton_vars != 'None':
            for sing in singleton_vars:
                sys.stderr.write('extracting {}... \n'.format(sing))
                with open(singleton_var_dict, 'r+') as singleton_feat_dict:
                    dic = json.load(singleton_feat_dict)
                if not sing in dic:
                    sys.stderr.write ('error:')
                else: 
                    sf_path = dic[sing]['path']
                    sf_col = dic[sing]['col']
                    with rio.open(sf_path, 'r') as src2:
                        vals = src2.read(1)
                        varn = 'var_sing_{}'.format(sing)
                        ptsgdb[varn] = [sample[0] for sample in src2.sample(coords)] 
    
        ptsgdb.drop(columns=['geometry'], inplace=True)
        all_pts = pd.concat([all_pts, ptsgdb])

    pts_out = pd.DataFrame.to_csv(all_pts,os.path.join(out_dir,'appended_vars.csv'), sep=',', index=True)

    return pts_out

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


