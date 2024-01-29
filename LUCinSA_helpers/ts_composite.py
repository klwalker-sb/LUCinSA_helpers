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

def add_var_to_stack(arr, si_var, attrs, out_dir, band_names, ras_list, **gw_args):
    arr.attrs = attrs
    ras = os.path.join(out_dir,f'{si_var}.tif')
    print(f'making {si_var} raster') 
    arr.gw.to_raster(ras,**gw_args)
    ras_list.append(ras)
    band_names.append(si_var)
    
def prep_ts_variable_bands(si_vars,ts_stack,ds_stack, out_dir,temp,start_doy,band_names,ras_list,**gw_args):
   
    with gw.open(ts_stack, time_names = ds_stack) as src:
        attrs = src.attrs.copy()
     
    #if any(v in si_vars for v in [f'maxv_{temp}',f'amp_{temp}', f'maxd_{temp}', f'maxdc_{temp}', f'rog_{temp}', f'ros_{temp}']):
    if any(v in si_vars for v in [f'maxv_{temp}',f'amp_{temp}', f'maxd_{temp}', f'maxdc_{temp}']):
        mmax = src.max(dim='time')
        if f'maxv_{temp}' in si_vars:
            add_var_to_stack(mmax,f'maxv_{temp}',attrs,out_dir,band_names,ras_list,**gw_args)
    if any(v in si_vars for v in [f'minv_{temp}',f'amp_{temp}', f'mind_{temp}',f'mindc_{temp}']):
        mmin = src.min(dim='time')
        if f'minv_{temp}' in si_vars:
            add_var_to_stack(mmin,f'minv_{temp}',attrs,out_dir,band_names,ras_list,**gw_args)
    if f'amp_{temp}' in si_vars:
        aamp = mmax - mmin
        add_var_to_stack(aamp,f'amp_{temp}',attrs,out_dir,band_names,ras_list,**gw_args)
    if f'avg_{temp}' in si_vars or f'cv_{temp}' in si_vars:
        aavg = src.mean(dim='time').astype('int16')
        if f'avg_{temp}' in si_vars:
            add_var_to_stack(aavg,f'avg_{temp}',attrs,out_dir,band_names,ras_list,**gw_args)
    if f'sd_{temp}' in si_vars or f'cv_{temp}' in si_vars:
        sstd = src.std(dim='time')
        if f'sd_{temp}' in si_vars:
            add_var_to_stack(sstd,f'sd_{temp}',attrs,out_dir,band_names,ras_list,**gw_args)
    if f'cv_{temp}' in si_vars:
        ccv = (sstd * 1000) / aavg
        add_var_to_stack(ccv,f'cv_{temp}',attrs,out_dir,band_names,ras_list,**gw_args)
    #if any(v in si_vars for v in [f'maxd_{temp}', f'maxdc_{temp}', f'sosv_{temp}',f'sosd_{temp}',
    #                              f'eosv_{temp}',f'eosd_{temp}',f'rog_{temp}', f'ros_{temp}']):
    if any(v in si_vars for v in [f'maxd_{temp}', f'maxdc_{temp}', f'sosv_{temp}',f'sosd_{temp}',f'eosv_{temp}',f'eosd_{temp}']):
        maxd = src.idxmax(dim='time',skipna=True)
        maxd1 = maxd.dt.dayofyear.astype('int16')
        ## add 365 to doy if it passed into the next year to avoid jump in values from Dec31 to Jan 1
        ## (keep everything >= start_doy as is. If passes into next year, doy will be < start_doy, so add 365)
        maxd2 = maxd1.where(maxd1 >= start_doy, maxd1 + 365)
        #max_date2 = max_date2.astype('int16')
        if f'maxd_{temp}' in si_vars:
            add_var_to_stack(maxd2,f'maxd_{temp}',attrs,out_dir,band_names,ras_list,**gw_args)
    if f'maxdc_{temp}' in si_vars:
        maxd_360 = 2 * np.pi * maxd1/365
        maxd_cos = 100 * (np.cos(max_360) + 1)
        maxd_cos = maxd_cos.astype('int16')
        add_var_to_stack(maxd_cos,f'maxdc_{temp}',attrs,out_dir,band_names,ras_list,**gw_args)
    if f'mind_{temp}' in si_vars or f'mindc_{temp}' in si_vars:
        mind = src.idxmin(dim='time',skipna=True)
        mind1 = mind.dt.dayofyear.astype('int16')
        ## add 365 to doy if it passed into the next year to avoid jump in values from Dec31 to Jan 1
        ## (keep everything >= start_doy as is. If passes into next year, doy will be < start_doy, so add 365)
        mind2 = mind1.where(mind1 >= start_doy, mind1 + 365)
        if f'mind_{temp}' in si_vars:
            add_var_to_stack(mind2,f'mind_{temp}',attrs,out_dir,band_names,ras_list,**gw_args)
    if f'mindc_{temp}' in si_vars:
        mind_360 = 2 * np.pi * mind1 / 365
        mind_cos = 100 * (np.cos(mind_360) + 1)
        mind_cos = mind_cos.astype('int16')
        add_var_to_stack(mind_cos,f'mindc_{temp}',attrs,out_dir,band_names,ras_list,**gw_args)
    #if any(v in si_vars for v in [f'sosd_{temp}',f'sosv_{temp}',f'los_{temp}',f'rog_{temp}']):
    if any(v in si_vars for v in [f'sosd_{temp}',f'sosv_{temp}']):
        ## Greenup calcs adapted from Digital Earth Africa, temporal.py
        sosv_path = os.path.join(out_dir,f'sosv_{temp}.tif')
        sosd_path = os.path.join(out_dir,f'sosd_{temp}.tif')
        if os.path.exists(sosd_path) == False or os.path.exists(sosv_path) == False: 
            src_c = src.chunk({"time": -1})
            greenup = src_c.where(src_c['time'] < maxd)
            green_deriv = src_c.differentiate("time")
            pos_green_deriv = green_deriv.where(green_deriv > 0)
            pos_greenup = greenup.where(~np.isnan(pos_green_deriv))
            med_g = pos_greenup.median("time")
            dist = np.abs(pos_greenup - med_g)
            mask = dist.isnull().all("time")
            distfill = dist.fillna(dist.max() + 1)
            ## get time index for start of greenup
            sos = distfill.idxmin(dim="time",skipna=True).where(~mask)
            ## date output should be as day of year for comparison with other years
            sosd = sos.dt.dayofyear.fillna(start_doy).astype('int16')
            ## add 365 to doy if it passes into the next year to avoid jump in values from Dec31 to Jan 1
            ## (keep everything >= start_doy as is. If passes into next year, doy will be < start_doy, so add 365)
            sosd1 = sosd.where(sosd >= start_doy, sosd + 365)
            if f'sosd_{temp}' in si_vars:
                if os.path.exists(sosd_path) == False:
                    add_var_to_stack(sosd1,f'sosd_{temp}',attrs,out_dir,band_names,ras_list,**gw_args)
                else:
                    ras_list.append(sosd_path)
                    band_names.append('sosd_{temp}')
        #if f'sosv_{temp}' in si_vars or f'rog_{temp}' in si_vars:
        if f'sosv_{temp}' in si_vars:
            ## get index value at date of start of season
            ## Note: if sosv is calculated before adding other bands to stack, those bands will fail to write due to lock problem
            if os.path.exists(sosv_path) == False:
                sosv = src.sel(time=sos, method='nearest').astype('int16')
                add_var_to_stack(sosv,f'sosv_{temp}',attrs,out_dir,band_names,ras_list,**gw_args)
            else:
                ras_list.append(sosv_path)
                band_names.append('sosv_{temp}')
    #if any(v in si_vars for v in [f'eosd_{temp}',f'eosv_{temp}',f'los_{temp}',f'ros_{temp}']):
    if any(v in si_vars for v in [f'eosd_{temp}',f'eosv_{temp}']):
        ## Senescence calcs adapted from Digital Earth Africa, temporal.py
        eosv_path = os.path.join(out_dir,f'eosv_{temp}.tif')
        eosd_path = os.path.join(out_dir,f'eosd_{temp}.tif')
        if os.path.exists(eosd_path) == False or os.path.exists(eosv_path) == False: 
            src_c = src.chunk({"time": -1})
            brownup = src_c.where(src_c['time'] > maxd)
            brown_deriv = src_c.differentiate("time")
            neg_brown_deriv = brown_deriv.where(brown_deriv < 0)
            neg_brownup = brownup.where(~np.isnan(neg_brown_deriv))
            med_b = neg_brownup.median("time")
            dist = np.abs(neg_brownup - med_b)
            mask = dist.isnull().all("time")
            distfill = dist.fillna(dist.max() + 1)
            ## get time index for start of senescence
            eos = distfill.idxmin(dim="time",skipna=True).where(~mask)
            ## date output should be as day of year for comparison with other years
            eosd = eos.dt.dayofyear.fillna(start_doy).astype('int16')
            ## add 365 to doy if it passes into the next year to avoid jump in values from Dec31 to Jan 1
            ## (keep everything >= start_doy as is. If passes into next year, doy will be < start_doy, so add 365)
            eosd1 = eosd.where(eosd >= start_doy, eosd + 365)
            if f'eosd_{temp}' in si_vars:
                if os.path.exists(eosd_path) == False:
                    add_var_to_stack(eosd1,f'eosd_{temp}',attrs,out_dir,band_names,ras_list,**gw_args)
                else:
                    ras_list.append(eosd_path)
                    band_names.append('eosd_{temp}')
        #if f'eosv_{temp}' in si_vars or f'ros_{temp}' in si_vars:
        if f'eosv_{temp}' in si_vars:
            if os.path.exists(eosv_path) == False:
                ## get index value at date of start of season
                ## Note: if sosv is calculated before adding sosd1 to stack, sosd1 will fail to write due to lock problem
                eosv = src.sel(time=eos, method='nearest').astype('int16')
                add_var_to_stack(eosv,f'eosv_{temp}',attrs,out_dir,band_names,ras_list,**gw_args)
            else:
                ras_list.append(eosv_path)
                band_names.append('eosv_{temp}')
    # This is causing file lock errors if sosv or eosv are calculated prior. Hacky solution below.
    #if f'rog_{temp}' in si_vars:
    #    with rio.open(os.path.join(out_dir,f'sosv_{temp}.tif')) as src:
    #        ssov = src.read()
    #    rog = (mmax - sosv) / (maxd2 - sosd1)
    #    rog = rog.astype('int16')
    #    add_var_to_stack(rog,f'rog_{temp}',attrs,out_dir,band_names,ras_list,**gw_args)
    #if f'ros_{temp}' in si_vars:
    #    ros = (mmax - eosv) / (maxd2 - eosd1) 
    #    ros = ros.astype('int16')
    #    add_var_to_stack(ros,f'ros_{temp}',attrs,out_dir,band_names,ras_list,**gw_args)
    #if f'los_{temp}' in si_vars:
    #    los = eosd1 - sosd1
    #    add_var_to_stack(los,f'los_{temp}',attrs,out_dir,band_names,ras_list,**gw_args)

def get_monthly_ts(si_vars, img_dir, start_yr, start_mo, band_names, ras_list):
    for img in sorted(os.listdir(img_dir)):
        if (start_mo == 1 and img.startswith(str(start_yr))) or (start_mo > 1 and img.startswith(str(int(start_yr)+1))):
            if img.endswith('020.tif') and 'Jan_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                band_names.append('Jan_20')
        if (start_mo <= 2 and img.startswith(str(start_yr))) or (start_mo > 2 and img.startswith(str(int(start_yr)+1))):
            if img.endswith('051.tif') and 'Feb_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                band_names.append('Feb_20')
        if (start_mo <= 3 and img.startswith(str(start_yr))) or (start_mo > 3 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('079.tif') | img.endswith('080.tif')) and 'Mar_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                band_names.append('Mar_20')
        if (start_mo <= 4 and img.startswith(str(start_yr))) or (start_mo > 4 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('110.tif') | img.endswith('111.tif')) and 'Apr_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                band_names.append('Apr_20')
        if (start_mo <= 5 and img.startswith(str(start_yr))) or (start_mo > 5 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('140.tif') | img.endswith('141.tif')) and 'May_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                band_names.append('May_20')
        if (start_mo <= 6 and img.startswith(str(start_yr))) or (start_mo > 6 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('171.tif') | img.endswith('172.tif')) and 'Jun_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                band_names.append('Jun_20')
        if (start_mo <= 7 and img.startswith(str(start_yr))) or (start_mo > 7 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('201.tif') | img.endswith('202.tif')) and 'Jul_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                band_names.append('Jul_20')
        if (start_mo <= 8 and img.startswith(str(start_yr))) or (start_mo > 8 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('232.tif') | img.endswith('233.tif')) and 'Aug_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                band_names.append('Aug_20')
        if (start_mo <= 9 and img.startswith(str(start_yr))) or (start_mo > 9 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('263.tif') | img.endswith('264.tif')) and 'Sep_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                band_names.append('Sep_20')
        if (start_mo <= 10 and img.startswith(str(start_yr))) or (start_mo > 10 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('293.tif') | img.endswith('294.tif')) and 'Oct_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                band_names.append('Oct_20')
        if (start_mo <= 11 and img.startswith(str(start_yr))) or (start_mo > 11 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('324.tif') | img.endswith('325.tif')) and 'Nov_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                band_names.append('Nov_20')
        if (start_mo <=12 and img.startswith(str(start_yr))):
            if (img.endswith('354.tif') | img.endswith('355.tif')) and 'Dec_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
                band_names.append('Dec_20')

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
                ## if in wet season (hardcoded here as 15/Oct-15/Apr), add to wet season subset
                if img_doy > 288 or img_doy < 105:
                    ts_stack_wet.append(os.path.join(img_dir,img))
                    ds_stack_wet.append(img_date)
            ## if in dry season of first year (hardcoded here as 15/Apr-15/Oct), add to dry season subset
            if (img_yr == int(start_yr) and (img_doy < 288 or img_doy > 105)):
                ts_stack_dry.append(os.path.join(img_dir,img))
                ds_stack_dry.append(img_date)
    
    ras_list = []
    band_names = []
    
    gw_args = {'verbose':1,'n_workers':4,'n_threads':1,'n_chunks':200, 'gdal_cache':64,'overwrite':True}
    
    ## Note: this is a hacky workaround of gdal cache problem to compute rog after computing ssov
    ##   (ssov compute creates some kind of lock that I cant release)
    rate_bands = [b for b in si_vars if b.split("_")[0] in ['rog','ros']]
    for f in rate_bands:
        temp = f.split("_")[1]
        ts_stack = ts_stack_wet if temp == 'wet' else ts_stack_dry if temp == 'dry' else ts_stack
        ds_stack = ds_stack_wet if temp == 'wet' else ds_stack_dry if temp == 'dry' else ds_stack
        maxv_path = os.path.join(out_dir,f'maxv_{temp}.tif')
        maxd_path = os.path.join(out_dir,f'maxd_{temp}.tif')
        if os.path.exists(maxv_path) == False:
            prep_ts_variable_bands([f'maxv_{temp}'],ts_stack, ds_stack, out_dir,temp, start_doy, band_names, ras_list, **gw_args)
        elif f'maxv_{temp}' in si_vars:
                ras_list.append(maxv_path)
                band_names.append(f'maxv_{temp}')
        if os.path.exists(maxd_path) == False:
            prep_ts_variable_bands([f'maxd_{temp}'],ts_stack, ds_stack, out_dir,temp, start_doy, band_names, ras_list, **gw_args)
        elif f'maxd_{temp}' in si_vars:
                ras_list.append(maxd_path)
                band_names.append(f'maxd_{temp}')
        si_vars = [i for i in si_vars if i not in [f'maxv_{temp}',f'maxd_{temp}']]
        if f == f'rog_{temp}':
            sosv_path = os.path.join(out_dir,f'sosv_{temp}.tif')
            sosd_path = os.path.join(out_dir,f'sosd_{temp}.tif')
            if os.path.exists(sosv_path) == False or os.path.exists(sosd_path) == False:
                prep_ts_variable_bands([f'sosv_{temp}'],ts_stack, ds_stack, out_dir,temp, start_doy, band_names, ras_list, **gw_args)
            elif f'sosv_{temp}' in si_vars:
                ras_list.append(sosv_path)
                band_names.append(f'sosv_{temp}')
            if os.path.exists(sosd_path) == False:
                prep_ts_variable_bands([f'sosd_{temp}'],ts_stack, ds_stack, out_dir,temp, start_doy, band_names, ras_list, **gw_args)
            elif f'sosd_{temp}' in si_vars:
                ras_list.append(sosd_path)
                band_names.append(f'sosd_{temp}')
            with gw.open(sosv_path) as sosv:
                attrs = sosv.attrs.copy()
                with gw.open(maxv_path) as maxv:
                    with gw.open(maxd_path) as maxd:
                        with gw.open(sosd_path) as sosd:
                            rog = (maxv - sosv) / (maxd - sosd)
                            rog = rog.where(rog > 0, 0).astype('int16')    
            add_var_to_stack(rog,f'rog_{temp}',attrs,out_dir,band_names,ras_list,**gw_args)
            si_vars = [i for i in si_vars if i not in [f'sosv_{temp}',f'sosd_{temp}',f'rog_{temp}']]
        if f == f'ros_{temp}':
            eosv_path = os.path.join(out_dir,f'eosv_{temp}.tif')
            eosd_path = os.path.join(out_dir,f'eosd_{temp}.tif')
            if os.path.exists(eosv_path) == False or os.path.exists(eosd_path) == False:   
                prep_ts_variable_bands([f'eosv_{temp}'],ts_stack, ds_stack, out_dir,temp, start_doy, band_names, ras_list, **gw_args)
            elif f'eosv_{temp}' in si_vars:
                ras_list.append(eosv_path)
                band_names.append(f'eosv_{temp}')
            if os.path.exists(eosd_path) == False:
                prep_ts_variable_bands([f'eosd_{temp}'],ts_stack, ds_stack, out_dir,temp, start_doy, band_names, ras_list, **gw_args)
            elif f'eosd_{temp}' in si_vars:
                ras_list.append(eosd_path)
                band_names.append(f'eosd_{temp}')
            with gw.open(eosv_path) as eosv:
                attrs = eosv.attrs.copy()
                with gw.open(maxv_path) as maxv:
                    with gw.open(maxd_path) as maxd:
                        with gw.open(eosd_path) as eosd:
                            ros = (maxv - eosv) / (eosd - maxd) 
                            ros = ros.where(ros > 0, 0).astype('int16')    
            add_var_to_stack(ros,f'ros_{temp}',attrs,out_dir,band_names,ras_list,**gw_args)
            si_vars = [i for i in si_vars if i not in [f'eosv_{temp}',f'eosd_{temp}',f'ros_{temp}']]
    length_bands = [b for b in si_vars if b.split("_")[0] == 'los']
    for f in length_bands:
        ts_stack = ts_stack_wet if temp == 'wet' else ts_stack_dry if temp == 'dry' else ts_stack
        ds_stack = ds_stack_wet if temp == 'wet' else ds_stack_dry if temp == 'dry' else ds_stack
        temp = f.split("_")[1]
        sosd_path = os.path.join(out_dir,f'sosd_{temp}.tif')   
        if os.path.exists(sosd_path) == False:
            prep_ts_variable_bands([f'sosd_{temp}'],ts_stack, ds_stack, out_dir,temp, start_doy, band_names, ras_list, **gw_args)
        elif f'sosd_{temp}' in si_vars:
            ras_list.append(sosd_path)
            band_names.append(f'sosd_{temp}')
        eosd_path = os.path.join(out_dir,f'eosd_{temp}.tif')   
        if os.path.exists(eosd_path) == False:
            prep_ts_variable_bands([f'eosd_{temp}'],ts_stack, ds_stack, out_dir,temp, start_doy, band_names, ras_list, **gw_args)   
        elif f'eosd_{temp}' in si_vars:
            ras_list.append(eosd_path)
            band_names.append(f'eosd_{temp}')
        with gw.open(sosd_path) as sosd:
            attrs = eosv.attrs.copy()
            with gw.open(eosd_path) as eosd:
                los = eosd - sosd
                los = los.where(los > 0, 0).astype('int16')
        add_var_to_stack(los,f'los_{temp}',attrs,out_dir,band_names,ras_list,**gw_args)
        si_vars = [i for i in si_vars if i not in [f'sosd_{temp}',f'eosd_{temp}',f'los_{temp}']]
        print('si vars remaining to create: {}'.format(si_vars))
              
    ## Convert image stack to XArray using geowombat:
    yr_bands = [b for b in si_vars if "_" not in b or b.split("_")[1] == 'yr']
    if len(yr_bands) > 0:
        prep_ts_variable_bands(si_vars, ts_stack, ds_stack, out_dir,'yr',start_doy, band_names, ras_list, **gw_args)

    wet_bands = [b for b in si_vars if "_" in b and b.split("_")[1] == 'wet']
    if len(wet_bands) > 0:
        prep_ts_variable_bands(si_vars, ts_stack_wet, ds_stack_wet, out_dir,'wet', start_doy, band_names, ras_list, **gw_args)
            
    dry_bands = [b for b in si_vars if "_" in b and b.split("_")[1] == 'dry']
    if len(dry_bands) > 0:
        prep_ts_variable_bands(si_vars, ts_stack_dry, ds_stack_dry, out_dir,'dry', start_doy, band_names, ras_list, **gw_args)
    
    mo_bands = [b for b in si_vars if b.split("_")[1] == '20']
    if len(mo_bands) > 0:
        get_monthly_ts(si_vars, img_dir, start_yr, start_mo, band_names, ras_list)
        
    print('ras_list:{}'.format(ras_list))
    print('band_names:{}'.format(band_names))
    print('writing stack for si_vars:{}'.format(band_names))
    if len(ras_list)<len(si_vars):
        print('oops--got an unknown band')

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
        elif len(ras_list)==9:
            if band_names[0].split('_')[1] == 'wet':
                out_ras = os.path.join(out_dir,'{:06d}_{}_{}_PhenWet.tif'.format(int(grid_cell),start_yr,spec_index))
            elif band_names[0].split('_')[1] == 'dry':
                out_ras = os.path.join(out_dir,'{:06d}_{}_{}_PhenDry.tif'.format(int(grid_cell),start_yr,spec_index))
        elif len(ras_list)==4:
            out_ras = os.path.join(out_dir,'{:06d}_{}_{}_{}{}{}{}.tif'.
                                   format(int(grid_cell),start_yr,spec_index,band_names[0],band_names[1],band_names[2],band_names[3]))
        elif len(ras_list)==2:
            out_ras = os.path.join(out_dir,'{:06d}_{}_{}_{}{}.tif'.
                                   format(int(grid_cell),start_yr,spec_index,band_names[0],band_names[1]))
        elif len(ras_list)==1:
            out_ras = os.path.join(out_dir,'{:06d}_{}_{}_{}.tif'.
                                   format(int(grid_cell),start_yr,spec_index,band_names[0]))
        else:
            out_ras = os.path.join(out_dir,'{:06d}_{}_{}_{}{}{}.tif'.
                                   format(int(grid_cell),start_yr,spec_index,band_names[0],band_names[1],band_names[2]))
            
        with rio.open(out_ras, 'w', **meta) as dst:
            for id, layer in enumerate(ras_list, start=1):
                with rio.open(layer) as src1:
                    dst.write(src1.read(1),id)
            dst.descriptions = tuple(band_names)

    return out_ras