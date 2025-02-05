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
from LUCinSA_helpers.pheno import make_pheno_vars

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
    if not os.path.exists(stack_path):
        if 'Poly' in feature_model and 'NoPoly' not in feature_model:
            nopoly_model = feature_model.replace('Poly','NoPoly')
            stack_path = os.path.join(in_dir,'{}_{}_stack.tif'.format(feature_model, start_yr))
    if not os.path.exists(stack_path):       
        sys.stderr.write('path {} does not exist. \n'.format(stack_path))
        sys.stderr.write('need to create variable stack for {}_{} first. \n'.format(feature_model, start_yr))
        ptsgdb = None
    
    else:
        #band_names = getset_feature_model(feature_mod_dict, feature_model)[7]
    
        if load_samp == False:
            if polys:
                ptsgdb = get_ran_pts_in_polys (polys, numpts, seed)
            else:
                sys.stderr.write('There are no polygons or points to process in this cell \n')
                return None
        elif load_samp == True:
            ptsgdb = ptgdb

        xy = [ptsgdb['geometry'].x, ptsgdb['geometry'].y]
        coords = list(map(list, zip(*xy)))
    
        sys.stdout.write('Extracting variables from stack \n')
        
        with gw.open(stack_path) as src0:
            #sys.stdout.write('{}'.format(src0.attrs))
            band_names = src0.attrs['descriptions']
        
        with rio.open(stack_path ,'r') as comp:
            #Open each band and get values
            for b, band in enumerate(band_names):
                sys.stdout.write('{}:{},'.format(b,band))
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
        sys.stderr.write('cell_list needs to be a list or path to .csv file with list \n')
    for cell in cells:
        var_dir = os.path.join(in_dir,'{:06d}'.format(int(cell)),'comp')
        sys.stderr.write('working on cell {} \n'.format(cell))
        if load_samp == True:
            sys.stdout.write('loading sample from points for cell {} \n'.format(cell))
            points = get_pts_in_grid (grid_file, cell, ptfile)
            polys = None
        else:
            sys.stdout.write('loading sample from polygons for cell {} \n'.format(cell))
            polys = get_polygons_in_grid (grid_file, cell, polyfile, oldest, newest)
            points = None
        
        sys.stdout.write('looking for {}_{}_stack.tif in {} to extract variables... \n'.format(feature_model,start_yr,var_dir))
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
    
    pts_in = pd.read_csv(ptfile, index_col=0)
    rfdf = all_pts.merge(pts_in, left_index=True, right_index=True)
    pd.DataFrame.to_csv(rfdf,os.path.join(out_dir,'forchecking_ptsfeats_{}_{}.csv'.format(feature_model,start_yr)), sep=',', index=True)
    pd.DataFrame.to_csv(all_pts,os.path.join(out_dir,'ptsfeats_{}_{}.csv'.format(feature_model,start_yr)), sep=',', index=True)
    
def append_feature_dataframe(in_dir, ptfile, feat_df, cell_list, grid_file, out_dir, start_yr, start_mo, spec_indices, si_vars,
                             spec_indices_pheno=None, pheno_vars=None, singleton_vars=None, singleton_var_dict=None, poly_vars=None, 
                             poly_var_path=None, combo_bands=None, scratch_dir=None):
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
        if scratch_dir is not None and scratch_dir != 'None':
            out_dir_int = os.path.join(scratch_dir,'{}'.format(cell))
        else:
            out_dir_int = os.path.join(cell_dir,'comp')       

        xy = [ptsgdb['geometry'].x, ptsgdb['geometry'].y]
        coords = list(map(list, zip(*xy)))
        
        if spec_indices_pheno is not None and spec_indices_pheno != 'None':
            for sip in spec_indices_pheno:
                if sip is not None and sip != 'None':
                    comp_dir = os.path.join(cell_dir,'comp', sip)
                    img_dir = os.path.join(cell_dir,'brdf_ts','ms', sip)
                    if os.path.isdir(img_dir):
                        for temp in ['wet','dry']:
                            pvars = [s for s in pheno_vars if s.split("_")[1] == temp]
                            if len(pvars) > 0:
                                sys.stderr.write('extracting {} pheno vars for {}... \n'.format(temp,sip))
                                if len(pvars) == 1:
                                    phen_comp = os.path.join(comp_dir, '{:06d}_{}.tif'.format(int(cell),pvars[0]))
                                    phen_bands = pvars
                                else:
                                    phen_comp = os.path.join(comp_dir, '{:06d}_{}_{}_Phen_{}.tif'.format(int(cell),start_yr,sip,temp))
                                    #sys.stderr.write('{} \n'.format(pvars))
                                    phen_bands = [f'maxv_{temp}', f'minv_{temp}', f'med_{temp}', f'slp_{temp}', f'numrot_{temp}',
                                              f'posd_{temp}', f'posv_{temp}', f'numlow_{temp}', f'tosd_{temp}', f'p1amp_{temp}',
                                              f'sosd_{temp}', f'sosv_{temp}', f'eosd_{temp}', f'eosv_{temp}', f'rog_{temp}', 
                                              f'ros_{temp}',f'los_{temp}']  
                                sys.stderr.write('looking for {} \n'.format(phen_comp)) 
                                if os.path.exists(phen_comp) == True:
                                    sys.stderr.write('getting variables from existing stack \n')
                                else:
                                    continue
                                    #sys.stderr.write('no existing stack. calculating new varaibles...')
                                    #phen_comp = make_pheno_vars(cell,img_dir,comp_dir,start_yr,start_mo,sip,phen_bands,500,[30,0])
                                phen_vars = rio.open(phen_comp,'r')
                                for b, band in enumerate(phen_bands):
                                    sys.stdout.write('{}:{}'.format(b,band))
                                    phen_vars.np = phen_vars.read(b+1)
                                    varn = ('var_{}_{}'.format(sip,band))
                                    ptsgdb[varn] = [sample[b] for sample in phen_vars.sample(coords)]
                    else:
                        sys.stderr.write ('no index {} created for {} \n'.format(sip,cell))
                        
        if spec_indices is not None and spec_indices != ' ' and spec_indices !='None':
            for si in spec_indices:
                if si is not None and si != ' ' and si !='None':
                    sys.stderr.write('looking at {}... \n'.format(si))
                    if scratch_dir is not None and scratch_dir != 'None':
                        out_dir_int = os.path.join(scratch_dir,'{}'.format(cell))
                    else:
                        out_dir_int = os.path.join(cell_dir,'comp',si)       
                    img_dir = os.path.join(cell_dir,'brdf_ts','ms',si)
                    if os.path.isdir(img_dir):
                        for siv in si_vars:
                            sys.stderr.write('getting {}... \n'.format(siv))
                            if os.path.exists(os.path.join(out_dir_int,'{}.tif'.format(siv))):
                                comp = os.path.join(out_dir_int,'{}.tif'.format(siv))
                            else:
                                continue
                                #comp = make_ts_composite(cell, img_dir, out_dir_int, start_yr, start_mo, si, siv)
                            with rio.open(comp, 'r') as src:
                                vals = src.read(1)
                                varn = ('var_{}_{}'.format(si,siv))
                                sys.stdout.write(' adding {}:{} \n'.format(si,siv))
                                ptsgdb[varn] = [sample[0] for sample in src.sample(coords)]                
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
                    sys.stderr.write ('error: need to enter {} into singular feature dictionary with path info.'.format(sing))
                else: 
                    sf_path = dic[sing]['path']
                    sf_col = dic[sing]['col']
                    with rio.open(sf_path, 'r') as src2:
                        vals = src2.read(1)
                        varn = 'var_sing_{}'.format(sing)
                        ptsgdb[varn] = [sample[0] for sample in src2.sample(coords)] 
    
        ptsgdb.drop(columns=['geometry'], inplace=True)
        all_pts = pd.concat([all_pts, ptsgdb])

    sys.stdout.write('done extracting variables. output to appended_vars.csv in {}'.format(out_dir))
    pts_out = pd.DataFrame.to_csv(all_pts,os.path.join(out_dir,'appended_vars.csv'), sep=',', index=True)

    return pts_out
                                             
def reduce_variable_dataframe(ptdf, drop_indices, drop_vars, drop_combo, out_dir, new_feature_mod, feature_mod_dict):
    ptsdf = pd.read_csv(ptdf, index_col=0)
    cols = list(ptsdf.columns)
    drop_cols = set()
    if drop_indices is not None and drop_indices != 'None' and len(drop_indices) > 0: 
            for i in drop_indices:
                drop = [c for c in cols if c.split('_')[1] == i]
                drop_cols.update(drop)
    if drop_vars is not None and drop_vars != 'None' and len(drop_vars) > 0: 
            for v in drop_vars:
                drop = [c for c in cols if c.split('_')[2] == v.split('_')[0] and c.split('_')[3] == v.split('_')[1]]
                drop_cols.update(drop)
    if drop_combo is not None and drop_combo != 'None' and len(drop_combo) > 0: 
            for cb in drop_combo:
                drop = [c for c in cols if c==cb]
                drop_cols.update(drop)
    ptsdf.drop(list(drop_cols), axis=1, inplace=True)
    print('dropping {} from model'.format(drop_cols))
    final_cols = [c for c in cols if c not in list(drop_cols)]
    print('new model has bands: {}'.format(final_cols))
    new_model = getset_feature_model(feature_mod_dict,new_feature_mod,spec_indices=None,si_vars=None,
                                     spec_indices_pheno=None,pheno_vars=None, singleton_vars=None,poly_vars=None, combo_bands=final_cols)
    df_out = pd.DataFrame.to_csv(ptsdf,os.path.join(out_dir,'ptsgdb_{}.csv'.format(new_feature_mod)), sep=',', index=True)
    print('created new model {} in {}'.format(new_feature_mod,out_dir))
    
    return df_out
                                             
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


