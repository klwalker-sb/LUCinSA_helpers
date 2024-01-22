#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
#import geowombat as gw
import datetime
import rasterio as rio
import numpy as np

def make_ts_composite(grid_cell,img_dir,out_dir,start_yr,start_mo,spec_index,si_vars):

    import geowombat as gw

    ##si_vars shoud be list. If fed via bash script, will be string; need to reparse as list:
    # (this is now in main)
    #if isinstance(si_vars, list):
    #    si_vars == si_vars
    #elif si_vars.startswith('['):
    #    si_vars = si_vars[1:-1].split(',')

    ras_list = []

    if not os.path.exists(out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        print('made new directory: {}'.format(out_dir))

    ##Convert image stack to XArray using geowombat:
    ts_stack = []
    ts_stack_wet = []
    ts_stack_dry = []
    ds_stack = []
    ds_stack_wet = []
    ds_stack_dry = []
    for img in sorted(os.listdir(img_dir)):
        if img.endswith('.tif'):
            img_yr = int(img[:4])
            img_doy = int(img[4:7])
            start_doy = int (30.5 * start_mo) - 30
            if (img_yr == int(start_yr) and img_doy >= start_doy) or (img_yr == (int(start_yr)+1) and img_doy < start_doy):
                ts_stack.append(os.path.join(img_dir,img))
                ds_stack.append(img_doy)
                ## if in wet season (hardcoded here as 1/Nov-1/Apr), add to wet season subset
                if img_doy > 305 or img_doy < 90:
                    ts_stack_wet.append(os.path.join(img_dir,img))
                    ds_stack_wet.append(img_doy)
            ## if in dry season of first year (hardcoded here as 1/May-1/Oct), add to dry season subset
            if (img_yr == int(start_yr) and img_doy < 274 and img_doy > 121):
                ts_stack_dry.append(os.path.join(img_dir,img))
                ds_stack_dry.append(img_doy)
                
    with gw.open(ts_stack, time_names= ds_stack) as src:
        attrs = src.attrs.copy()

    if 'Max_yr' in si_vars or 'Amp_yr' in si_vars or 'Maxd_yr' in si_vars or 'Maxdc_yr' in si_vars:
        mmax = src.max(dim='time')
        if 'Max_yr' in si_vars:
            mmax.attrs = attrs
            ras = os.path.join(out_dir,'Max_yr.tif')
            print('making max raster')
            mmax.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
            ras_list.append(ras)
    if 'Min_yr' in si_vars or 'Amp_yr' in si_vars or 'Mind_yr' in si_vars or 'Mindc_yr' in si_vars:
        mmin = src.min(dim='time')
        if 'Min_yr' in si_vars:
            mmin.attrs = attrs
            print('making min raster')
            ras = os.path.join(out_dir,'Min_yr.tif')
            ras_list.append(ras)
            mmin.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'Amp_yr' in si_vars:
        aamp = mmax - mmin
        aamp.attrs = attrs
        print('making amp raster')
        ras = os.path.join(out_dir,'Amp_yr.tif')
        ras_list.append(ras)
        aamp.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'Avg_yr' in si_vars or 'CV_yr' in si_vars:
        aavg = src.mean(dim='time')
        if 'Avg_yr' in si_vars:
            aavg.attrs = attrs
            ras = os.path.join(out_dir,'Avg_yr.tif')
            ras_list.append(ras)
            print('making avg raster')
            aavg.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'Std_yr' in si_vars or 'CV_yr' in si_vars:
        sstd = src.std(dim='time')
        if 'Std_yr' in si_vars:
            sstd.attrs = attrs
            ras = os.path.join(out_dir,'Std_yr.tif')
            ras_list.append(ras)
            print('making std raster')
            sstd.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'CV_yr' in si_vars:
        sstdv_b = sstd*1000
        cv = sstdv_b/aavg
        cv.attrs = attrs
        ras = os.path.join(out_dir,'CV_yr.tif')
        ras_list.append(ras)
        print('making coef of var raster')
        cv.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'Maxd_yr' in si_vars or 'Maxdc_yr' in si_vars:
        max_date = src.idxmax(dim='time',skipna=True)
        if 'Maxd_yr' in si_vars:
            max_date.attrs = attrs
            ras = os.path.join(out_dir,'Maxd_yr.tif')
            ras_list.append(ras)
            print('making maxDate raster')
            max_date.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'Maxdc_yr' in si_vars:
        max_date_360 = 2 * np.pi * max_date/365
        max_date_cos = 100 * (np.cos(max_date_360) + 1)
        max_date_cos.attrs = attrs
        ras = os.path.join(out_dir,'Maxdc_yr.tif')
        ras_list.append(ras)
        max_date_cos.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'Mind_yr' in si_vars or 'Mindc_yr' in si_vars:
        min_date = src.idxmin(dim='time',skipna=True)
        if 'Mind_yr' in si_vars:
            min_date.attrs = attrs
            ras = os.path.join(out_dir,'Mind_yr.tif')
            ras_list.append(ras)
            print('making minDate raster')
            min_date.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'Mindc_yr' in si_vars:
        min_date_360 = 2 * np.pi * min_date/365
        min_date_cos = 100 * (np.cos(min_date_360) + 1)
        min_date_cos.attrs = attrs
        ras = os.path.join(out_dir,'Mindc_yr.tif')
        ras_list.append(ras)
        min_date_cos.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    
    wet_bands = [b for b in si_vars if b.split("_")[1] == 'wet']
    if len(wet_bands) > 0:
        with gw.open(ts_stack_wet, time_names= ds_stack_wet) as src_wet:
            attrs_wet = src_wet.attrs.copy()
            
        if 'Max_wet' in si_vars or 'Amp_wet' in si_vars or 'Maxd_wet' in si_vars or 'Maxdc_wet' in si_vars:
            mmax_wet = src_wet.max(dim='time')
            if 'Max_wet' in si_vars:
                mmax_wet.attrs = attrs_wet
                ras = os.path.join(out_dir,'Max_wet.tif')
                print('making max wet raster')
                mmax_wet.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
                ras_list.append(ras)
        if 'Min_wet' in si_vars or 'Amp_wet' in si_vars or 'Mind_wet' in si_vars or 'Mindc_wet' in si_vars:
            mmin_wet = src_wet.min(dim='time')
            if 'Min_wet' in si_vars:
                mmin_wet.attrs = attrs_wet
                print('making min_wet raster')
                ras = os.path.join(out_dir,'Min_wet.tif')
                ras_list.append(ras)
                mmin_wet.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
        if 'Amp_wet' in si_vars:
            aamp_wet = mmax_wet - mmin_wet
            aamp_wet.attrs = attrs_wet
            print('making amp wet raster')
            ras = os.path.join(out_dir,'Amp_wet.tif')
            ras_list.append(ras)
            aamp_wet.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
        if 'Maxd_wet' in si_vars or 'Maxdc_wet' in si_vars or 'SOS_wet' in si_vars:
            max_date_wet = src_wet.idxmax(dim='time',skipna=True)
            if 'Maxd_wet' in si_vars:
                max_date_wet.attrs = attrs_wet
                ras = os.path.join(out_dir,'Maxd_wet.tif')
                ras_list.append(ras)
                print('making maxDate wet raster')
                max_date_wet.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
        if 'Maxdc_wet' in si_vars:
            max_date_360_wet = 2 * np.pi * max_date_wet/365
            max_date_cos_wet = 100 * (np.cos(max_date_360_wet) + 1)
            max_date_cos_wet.attrs = attrs_wet
            ras = os.path.join(out_dir,'Maxdc_wet.tif')
            ras_list.append(ras)
            max_date_cos_wet.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
        if 'Mind_wet' in si_vars or 'Mindc_wet' in si_vars:
            min_date_wet = src_wet.idxmin(dim='time',skipna=True)
            if 'Mind_wet' in si_vars:
                min_date_wet.attrs = attrs_wet
                ras = os.path.join(out_dir,'Mind_wet.tif')
                ras_list.append(ras)
                print('making minDate raster')
                min_date_wet.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
        if 'Mindc_wet' in si_vars:
            min_date_360_wet = 2 * np.pi * min_date_wet/365
            min_date_cos_wet = 100 * (np.cos(min_date_360_wet) + 1)
            min_date_cos_wet.attrs = attrs_wet
            ras = os.path.join(out_dir,'Mindc_wet.tif')
            ras_list.append(ras)
            min_date_cos_wet.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
        '''
        if 'SOS_wet' in si_vars:
            #greenup = src_wet.where(src_wet['time'] < max_date_wet)
            green_deriv = src_wet.differentiate("time")
            pos_green_deriv = green_deriv.where(green_deriv > 0)
            pos_greenup = greenup.where(~np.isnan(pos_green_deriv))
            #median = pos_greenup.median("time")
            distance = abs(pos_greenup - median)
            SOS_wet = distance.min("time")
            SOS_wet.attrs = attrs_wet
            ras = os.path.join(out_dir,'SOS_wet.tif')
            ras_list.append(ras)
            SOS_wet.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
        '''          
    dry_bands = [b for b in si_vars if b.split("_")[1] == 'dry']
    if len(dry_bands) > 0:
        with gw.open(ts_stack_dry, time_names= ds_stack_dry) as src_dry:
            attrs_dry = src_dry.attrs.copy()
            
        if 'Max_dry' in si_vars or 'Amp_dry' in si_vars or 'Maxd_dry' in si_vars or 'Maxdc_dry' in si_vars:
            mmax_dry = src_dry.max(dim='time')
            if 'Max_dry' in si_vars:
                mmax_dry.attrs = attrs_dry
                ras = os.path.join(out_dir,'Max_dry.tif')
                print('making max dry raster')
                mmax_dry.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
                ras_list.append(ras)
        if 'Min_dry' in si_vars or 'Amp_dry' in si_vars or 'Mind_dry' in si_vars or 'Mindc_dry' in si_vars:
            mmin_dry = src_dry.min(dim='time')
            if 'Min_dry' in si_vars:
                mmin_dry.attrs = attrs_dry
                print('making min_dry raster')
                ras = os.path.join(out_dir,'Min_dry.tif')
                ras_list.append(ras)
                mmin_dry.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
        if 'Amp_dry' in si_vars:
            aamp_dry = mmax_dry - mmin_dry
            aamp_dry.attrs = attrs_dry
            print('making amp dry raster')
            ras = os.path.join(out_dir,'Amp_dry.tif')
            ras_list.append(ras)
            aamp_dry.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
        if 'Maxd_dry' in si_vars or 'Maxdc_dry' in si_vars:
            max_date_dry = src_dry.idxmax(dim='time',skipna=True)
            if 'Maxd_dry' in si_vars:
                max_date_dry.attrs = attrs_dry
                ras = os.path.join(out_dir,'Maxd_dry.tif')
                ras_list.append(ras)
                print('making maxDate dry raster')
                max_date_dry.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
        if 'Maxdc_dry' in si_vars:
            max_date_360_dry = 2 * np.pi * max_date_dry/365
            max_date_cos_dry = 100 * (np.cos(max_date_360_dry) + 1)
            max_date_cos_dry.attrs = attrs_dry
            ras = os.path.join(out_dir,'Maxdc_dry.tif')
            ras_list.append(ras)
            max_date_cos_dry.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
        if 'Mind_dry' in si_vars or 'Mindc_dry' in si_vars:
            min_date_dry = src_dry.idxmin(dim='time',skipna=True)
            if 'Mind_dry' in si_vars:
                min_date_dry.attrs = attrs_dry
                ras = os.path.join(out_dir,'Mind_dry.tif')
                ras_list.append(ras)
                print('making min date_dry raster')
                min_date_dry.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
        if 'Mindc_dry' in si_vars:
            min_date_360_dry = 2 * np.pi * min_date_dry/365
            min_date_cos_dry = 100* (np.cos(min_date_360_dry) + 1)
            min_date_cos_dry.attrs = attrs_dry
            ras = os.path.join(out_dir,'Mindc_dry.tif')
            ras_list.append(ras)
            min_date_cos_dry.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
                
    for img in sorted(os.listdir(img_dir)):
        if (start_mo == 1 and img.startswith(str(start_yr))) or (start_mo > 1 and img.startswith(str(int(start_yr)+1))):
            if img.endswith('020.tif') and 'Jan_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
        if (start_mo <= 2 and img.startswith(str(start_yr))) or (start_mo > 2 and img.startswith(str(int(start_yr)+1))):
            if img.endswith('051.tif') and 'Feb_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
        if (start_mo <= 3 and img.startswith(str(start_yr))) or (start_mo > 3 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('079.tif') | img.endswith('080.tif')) and 'Mar_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
        if (start_mo <= 4 and img.startswith(str(start_yr))) or (start_mo > 4 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('110.tif') | img.endswith('111.tif')) and 'Apr_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
        if (start_mo <= 5 and img.startswith(str(start_yr))) or (start_mo > 5 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('140.tif') | img.endswith('141.tif')) and 'May_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
        if (start_mo <= 6 and img.startswith(str(start_yr))) or (start_mo > 6 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('171.tif') | img.endswith('172.tif')) and 'Jun_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
        if (start_mo <= 7 and img.startswith(str(start_yr))) or (start_mo > 7 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('201.tif') | img.endswith('202.tif')) and 'Jul_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
        if (start_mo <= 8 and img.startswith(str(start_yr))) or (start_mo > 8 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('232.tif') | img.endswith('233.tif')) and 'Aug_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
        if (start_mo <= 9 and img.startswith(str(start_yr))) or (start_mo > 9 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('263.tif') | img.endswith('264.tif')) and 'Sep_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
        if (start_mo <= 10 and img.startswith(str(start_yr))) or (start_mo > 10 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('293.tif') | img.endswith('294.tif')) and 'Oct_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
        if (start_mo <= 11 and img.startswith(str(start_yr))) or (start_mo > 11 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('324.tif') | img.endswith('325.tif')) and 'Nov_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
        if (start_mo <=12 and img.startswith(str(start_yr))):
            if (img.endswith('354.tif') | img.endswith('355.tif')) and 'Dec_20' in si_vars:
                ras_list.append(os.path.join(img_dir,img))
   
    
    with gw.open(ts_stack, time_names= ds_stack) as src:
        attrs = src.attrs.copy()
        
    print(ras_list)
    if len(ras_list)<len(si_vars):
        print('oops--got an unknown band; Current Band options are Max,Min,Amp,Avg,CV,Std,MaxDate,MaxDateCos,MinDate,MinDateCos,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec')

    else:
        ##Start writing output composite
        with rio.open(ras_list[1]) as src0:
            meta = src0.meta
            meta.update(count = len(ras_list))
            
        # Read each layer and write it to stack
        
        if len(ras_list)>12:
            out_ras = os.path.join(out_dir,'{:06d}'.format(int(grid_cell))+'_'+str(start_yr)+spec_index+'_RFVars.tif')
        elif len(ras_list)==12:
            out_ras = os.path.join(out_dir,'{:06d}'.format(int(grid_cell))+'_'+str(start_yr)+spec_index+'_monthly.tif')
        elif len(ras_list)==4:
            out_ras = os.path.join(out_dir,'{:06d}'.format(int(grid_cell))+'_'+str(start_yr)+spec_index+'_'+ si_vars[0]+si_vars[1]+si_vars[2]+si_vars[3]+'.tif')
        elif len(ras_list)==2:
            out_ras = os.path.join(out_dir,'{:06d}'.format(int(grid_cell))+'_'+str(start_yr)+spec_index+'_'+si_vars[0]+si_vars[1]+'.tif')
        elif len(ras_list)==1:
            out_ras = os.path.join(out_dir,'{:06d}'.format(int(grid_cell))+'_'+str(start_yr)+spec_index+'_'+si_vars[0]+'.tif')
        else:
            out_ras = os.path.join(out_dir,'{:06d}'.format(int(grid_cell))+'_'+str(start_yr)+spec_index+'_'+ si_vars[0]+si_vars[1]+si_vars[2]+'.tif') 
        with rio.open(out_ras, 'w', **meta) as dst:
            for id, layer in enumerate(ras_list, start=1):
                with rio.open(layer) as src1:
                    dst.write(src1.read(1),id)

    return out_ras