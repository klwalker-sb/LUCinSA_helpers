#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
import datetime
import rasterio as rio
import numpy as np
import geowombat as gw
import xarray as xr
import pandas as pd

def add_var_to_stack(arr, si_var, attrs, out_dir, comp_band_names, ras_list, **gw_args):
    arr.attrs = attrs
    ras = os.path.join(out_dir,f'{si_var}.tif')
    print(f'making {si_var} raster') 
    arr.gw.to_raster(ras,**gw_args)
    ras_list.append(ras)
    comp_band_names.append(si_var)
    
def prep_ts_variable_bands(si_vars,ts_stack,ds_stack, out_dir,temp,start_doy,comp_band_names,ras_list,**gw_args):
   
    with gw.open(ts_stack, time_names = ds_stack) as src:
        attrs = src.attrs.copy()
     
    if any(v in si_vars for v in [f'maxv_{temp}',f'amp_{temp}', f'maxd_{temp}', f'maxdc_{temp}']):
        mmax = src.max(dim='time')
        if f'maxv_{temp}' in si_vars:
            add_var_to_stack(mmax,f'maxv_{temp}',attrs,out_dir,comp_band_names,ras_list,**gw_args)
    if any(v in si_vars for v in [f'minv_{temp}',f'amp_{temp}', f'mind_{temp}',f'mindc_{temp}']):
        mmin = src.min(dim='time')
        if f'minv_{temp}' in si_vars:
            add_var_to_stack(mmin,f'minv_{temp}',attrs,out_dir,comp_band_names,ras_list,**gw_args)
    if f'amp_{temp}' in si_vars:
        aamp = mmax - mmin
        add_var_to_stack(aamp,f'amp_{temp}',attrs,out_dir,comp_band_names,ras_list,**gw_args)
    if f'avg_{temp}' in si_vars or f'cv_{temp}' in si_vars:
        aavg = src.mean(dim='time').astype('int16')
        if f'avg_{temp}' in si_vars:
            add_var_to_stack(aavg,f'avg_{temp}',attrs,out_dir,comp_band_names,ras_list,**gw_args)
    if f'med_{temp}' in si_vars:
        mmed = src.median(dim='time').astype('int16')
        add_var_to_stack(mmed,f'med_{temp}',attrs,out_dir,comp_band_names,ras_list,**gw_args)
    if f'sd_{temp}' in si_vars or f'cv_{temp}' in si_vars:
        sstd = src.std(dim='time')
        if f'sd_{temp}' in si_vars:
            add_var_to_stack(sstd,f'sd_{temp}',attrs,out_dir,comp_band_names,ras_list,**gw_args)
    if f'cv_{temp}' in si_vars:
        ccv = ((sstd * 1000) / aavg).astype('int16')
        add_var_to_stack(ccv,f'cv_{temp}',attrs,out_dir,comp_band_names,ras_list,**gw_args)
    if any(v in si_vars for v in [f'maxd_{temp}', f'maxdc_{temp}']):
        maxd = src.idxmax(dim='time',skipna=True)
        maxd1 = maxd.dt.dayofyear.astype('int16')
        ## add 365 to doy if it passed into the next year to avoid jump in values from Dec31 to Jan 1
        ## (keep everything >= start_doy as is. If passes into next year, doy will be < start_doy, so add 365)
        maxd2 = maxd1.where(maxd1 >= start_doy, maxd1 + 365)
        #max_date2 = max_date2.astype('int16')
        if f'maxd_{temp}' in si_vars:
            add_var_to_stack(maxd2,f'maxd_{temp}',attrs,out_dir,comp_band_names,ras_list,**gw_args)
    if f'maxdc_{temp}' in si_vars:
        maxd_360 = 2 * np.pi * maxd1/365
        maxd_cos = 100 * (np.cos(max_360) + 1)
        maxd_cos = maxd_cos.astype('int16')
        add_var_to_stack(maxd_cos,f'maxdc_{temp}',attrs,out_dir,comp_band_names,ras_list,**gw_args)
    if f'mind_{temp}' in si_vars or f'mindc_{temp}' in si_vars:
        mind = src.idxmin(dim='time',skipna=True)
        mind1 = mind.dt.dayofyear.astype('int16')
        ## add 365 to doy if it passed into the next year to avoid jump in values from Dec31 to Jan 1
        ## (keep everything >= start_doy as is. If passes into next year, doy will be < start_doy, so add 365)
        mind2 = mind1.where(mind1 >= start_doy, mind1 + 365)
        if f'mind_{temp}' in si_vars:
            add_var_to_stack(mind2,f'mind_{temp}',attrs,out_dir,comp_band_names,ras_list,**gw_args)
    if f'mindc_{temp}' in si_vars:
        mind_360 = 2 * np.pi * mind1 / 365
        mind_cos = 100 * (np.cos(mind_360) + 1)
        mind_cos = mind_cos.astype('int16')
        add_var_to_stack(mind_cos,f'mindc_{temp}',attrs,out_dir,comp_band_names,ras_list,**gw_args)

def get_monthly_ts(si_vars, img_dir, start_yr, start_mo, comp_band_names, ras_list):
    for img in sorted(os.listdir(img_dir)):
        if (start_mo == 1 and img.startswith(str(start_yr))) or (start_mo > 1 and img.startswith(str(int(start_yr)+1))):
            if img.endswith('020.tif') and 'Jan_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                comp_band_names.append('Jan_20')
                print('added Jan')
        if (start_mo <= 2 and img.startswith(str(start_yr))) or (start_mo > 2 and img.startswith(str(int(start_yr)+1))):
            if img.endswith('051.tif') and 'Feb_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                comp_band_names.append('Feb_20')
                print('added Feb')
        if (start_mo <= 3 and img.startswith(str(start_yr))) or (start_mo > 3 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('079.tif') | img.endswith('080.tif')) and 'Mar_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                comp_band_names.append('Mar_20')
                print('added Mar')
        if (start_mo <= 4 and img.startswith(str(start_yr))) or (start_mo > 4 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('110.tif') | img.endswith('111.tif')) and 'Apr_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                comp_band_names.append('Apr_20')
                print('added Apr')
        if (start_mo <= 5 and img.startswith(str(start_yr))) or (start_mo > 5 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('140.tif') | img.endswith('141.tif')) and 'May_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                comp_band_names.append('May_20')
                print('added May')
        if (start_mo <= 6 and img.startswith(str(start_yr))) or (start_mo > 6 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('171.tif') | img.endswith('172.tif')) and 'Jun_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                comp_band_names.append('Jun_20')
                print('added Jun')
        if (start_mo <= 7 and img.startswith(str(start_yr))) or (start_mo > 7 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('201.tif') | img.endswith('202.tif')) and 'Jul_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                comp_band_names.append('Jul_20')
                print('added July')
        if (start_mo <= 8 and img.startswith(str(start_yr))) or (start_mo > 8 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('232.tif') | img.endswith('233.tif')) and 'Aug_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                comp_band_names.append('Aug_20')
                print('added Aug')
        if (start_mo <= 9 and img.startswith(str(start_yr))) or (start_mo > 9 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('263.tif') | img.endswith('264.tif')) and 'Sep_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                comp_band_names.append('Sep_20')
                print('added Sep')
        if (start_mo <= 10 and img.startswith(str(start_yr))) or (start_mo > 10 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('293.tif') | img.endswith('294.tif')) and 'Oct_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                comp_band_names.append('Oct_20')
                print('added Oct')
        if (start_mo <= 11 and img.startswith(str(start_yr))) or (start_mo > 11 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('324.tif') | img.endswith('325.tif')) and 'Nov_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                comp_band_names.append('Nov_20')
                print('added Nov')
        if (start_mo <=12 and img.startswith(str(start_yr))):
            if (img.endswith('354.tif') | img.endswith('355.tif')) and 'Dec_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                comp_band_names.append('Dec_20')
                print('added Dec')

def make_ts_composite(grid_cell,img_dir,out_dir,start_yr,start_mo,spec_index,si_vars):

    ##si_vars shoud be list. If fed via bash script, will be string; need to reparse as list:
    # (this is now in main)
    #if isinstance(si_vars, list):
    #    si_vars == si_vars
    #elif si_vars.startswith('['):
    #    si_vars = si_vars[1:-1].split(',')

    if not os.path.exists(out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        print('made new directory: {}'.format(out_dir))

    ## get stack from images in time-series directory that match year and month
    ts_stack = []
    ts_stack_wet = []
    ts_stack_dry = []
    ds_stack = []
    ds_stack_wet = []
    ds_stack_dry = []
    for img in sorted(os.listdir(img_dir)):
        if img.endswith('.tif'):
            ## ts images are named YYYYDDD with YYYY=year and DDD=doy
            img_date = pd.to_datetime(img[:7],format='%Y%j')
            ## use year to filter to mapping year and doy to parse seasons regardless of year
            img_yr = int(img[:4])
            img_doy = int(img[4:7])
            start_doy = int (30.5 * start_mo) - 30
            if (img_yr == int(start_yr) and img_doy >= start_doy) or (img_yr == (int(start_yr)+1) and img_doy < start_doy):
                ts_stack.append(os.path.join(img_dir,img))
                ds_stack.append(img_date)
                ## if in wet season (hardcoded here as 1/Nov-1/Mar), add to wet season subset
                if img_doy > 306 or img_doy < 61:
                    ts_stack_wet.append(os.path.join(img_dir,img))
                    ds_stack_wet.append(img_date)
            ## if in dry season of first year (hardcoded here as 1/July-15/Sep), add to dry season subset
            if (img_yr == int(start_yr) and (img_doy < 259 and img_doy > 183)):
                ts_stack_dry.append(os.path.join(img_dir,img))
                ds_stack_dry.append(img_date)
    
    ras_list = []
    comp_band_names = []
    
    gw_args = {'verbose':1,'n_workers':4,'n_threads':1,'n_chunks':200, 'gdal_cache':64,'overwrite':True}
                  
    ## Convert image stack to XArray using geowombat:
    yr_bands = [b for b in si_vars if "_" not in b or b.split("_")[1] == 'yr']
    if len(yr_bands) > 0:
        prep_ts_variable_bands(yr_bands, ts_stack, ds_stack, out_dir,'yr',start_doy, comp_band_names, ras_list, **gw_args)

    wet_bands = [b for b in si_vars if "_" in b and b.split("_")[1] == 'wet']
    if len(wet_bands) > 0:
        prep_ts_variable_bands(wet_bands, ts_stack_wet, ds_stack_wet, out_dir,'wet', start_doy, comp_band_names, ras_list, **gw_args)
            
    dry_bands = [b for b in si_vars if "_" in b and b.split("_")[1] == 'dry']
    if len(dry_bands) > 0:
        prep_ts_variable_bands(dry_bands, ts_stack_dry, ds_stack_dry, out_dir,'dry', start_doy, comp_band_names, ras_list, **gw_args)
    
    mo_bands = [b for b in si_vars if b.split("_")[1] == '20']
    if len(mo_bands) > 0:
        get_monthly_ts(mo_bands, img_dir, start_yr, start_mo, comp_band_names, ras_list)
        
    #print('ras_list:{}'.format(ras_list))
    #print('comp_band_names:{}'.format(comp_band_names))
    print('writing stack for si_vars:{}'.format(si_vars))
    if len(ras_list)<len(si_vars):
        print('oops--got an unknown band')
        out_ras = None

    else:
        ##Start writing output composite
        with rio.open(ras_list[0]) as src0:
            meta = src0.meta
            meta.update(count = len(ras_list))
            
        # Read each layer and write it to stack
        
        if len(ras_list)>12:
            out_ras = os.path.join(out_dir,'{:06d}_{}_{}_RFVars.tif'.format(int(grid_cell),start_yr,spec_index))
        elif len(ras_list)==12:
            out_ras = os.path.join(out_dir,'{:06d}_{}_{}_monthly.tif'.format(int(grid_cell),start_yr,spec_index))
        elif len(ras_list)==4:
            out_ras = os.path.join(out_dir,'{:06d}_{}_{}_{}{}{}{}.tif'.
                                   format(int(grid_cell),start_yr,spec_index,comp_band_names[0],comp_band_names[1],comp_band_names[2],comp_band_names[3]))
        elif len(ras_list)==2:
            out_ras = os.path.join(out_dir,'{:06d}_{}_{}_{}{}.tif'.
                                   format(int(grid_cell),start_yr,spec_index,comp_band_names[0],comp_band_names[1]))
        elif len(ras_list)==1:
            out_ras = os.path.join(out_dir,'{:06d}_{}_{}_{}.tif'.
                                   format(int(grid_cell),start_yr,spec_index,comp_band_names[0]))
        else:
            out_ras = os.path.join(out_dir,'{:06d}_{}_{}_{}{}{}.tif'.
                                   format(int(grid_cell),start_yr,spec_index,comp_band_names[0],comp_band_names[1],comp_band_names[2]))
            
        with rio.open(out_ras, 'w', **meta) as dst:
            for id, layer in enumerate(ras_list, start=1):
                with rio.open(layer) as src1:
                    dst.write(src1.read(1),id)
            dst.descriptions = tuple(comp_band_names)

    return out_ras, comp_band_names