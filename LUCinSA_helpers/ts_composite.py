#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
#import geowombat as gw
import datetime
import rasterio as rio
import numpy as np

def make_ts_composite(grid_cell,img_dir,out_dir,start_yr,start_mo,spec_index,bands_out):

    import geowombat as gw

    ##bands_out shoud be list. If fed via bash script, will be string; need to reparse as list:
    # (this is now in main)
    #if isinstance(bands_out, list):
    #    bands_out == bands_out
    #elif bands_out.startswith('['):
    #    bands_out = bands_out[1:-1].split(',')

    ras_list = []

    if not os.path.exists(out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        print('made new directory: {}'.format(out_dir))

    ##Convert Image stack to XArray using geowombat:
    ts_stack = []
    ds_stack = []
    for img in sorted(os.listdir(img_dir)):
        if img.endswith('.tif'):
            img_yr = int(img[:4])
            img_doy = int(img[4:7])
            start_doy = int (30.5 * start_mo) - 30
            if (img_yr == int(start_yr) and img_doy >= start_doy) or (img_yr == (int(start_yr)+1) and img_doy < start_doy):
                ts_stack.append(os.path.join(img_dir,img))
                ds_stack.append(img_doy)
    with gw.open(ts_stack, time_names= ds_stack) as src:
        attrs = src.attrs.copy()

    print(bands_out)

    if 'Max' in bands_out or 'Amp' in bands_out or 'MaxDate' in bands_out or 'MaxDateCos' in bands_out:
        mmax = src.max(dim='time')
        if 'Max' in bands_out:
            mmax.attrs = attrs
            ras = os.path.join(out_dir,'Max.tif')
            print('making max raster')
            mmax.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
            ras_list.append(ras)
    if 'Min' in bands_out or 'Amp' in bands_out or 'MinDate' in bands_out or 'MinDateCos' in bands_out:
        mmin = src.min(dim='time')
        if 'Min' in bands_out:
            mmin.attrs = attrs
            print('making min raster')
            ras = os.path.join(out_dir,'Min.tif')
            ras_list.append(ras)
            mmin.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'Amp' in bands_out:
        aamp = mmax - mmin
        aamp.attrs = attrs
        print('making amp raster')
        ras = os.path.join(out_dir,'Amp.tif')
        ras_list.append(ras)
        aamp.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'Avg' in bands_out or 'CV' in bands_out:
        aavg = src.mean(dim='time')
        if 'Avg' in bands_out:
            aavg.attrs = attrs
            ras = os.path.join(out_dir,'Avg.tif')
            ras_list.append(ras)
            print('making avg raster')
            aavg.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'Std' in bands_out or 'CV' in bands_out:
        sstd = src.std(dim='time')
        if 'Std' in bands_out:
            sstd.attrs = attrs
            ras = os.path.join(out_dir,'Std.tif')
            ras_list.append(ras)
            print('making std raster')
            sstd.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'CV' in bands_out:
        sstdv_b = sstd*1000
        cv = sstdv_b/aavg
        cv.attrs = attrs
        ras = os.path.join(out_dir,'CV.tif')
        ras_list.append(ras)
        print('making coef of var raster')
        cv.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'MaxDate' in bands_out or 'MaxDateCos' in bands_out:
        max_date = src.idxmax(dim='time',skipna=True)
        if 'MaxDate' in bands_out:
            max_date.attrs = attrs
            ras = os.path.join(out_dir,'MaxDate.tif')
            ras_list.append(ras)
            print('making maxDate raster')
            max_date.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'MaxDateCos' in bands_out:
        max_date_360 = max_date * max_date/360
        max_date_cos = np.cos(max_date_360)
        max_date_cos.attrs = attrs
        ras = os.path.join(out_dir,'max_date_cos.tif')
        ras_list.append(ras)
        max_date_cos.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'MinDate' in bands_out or 'MinDateCos' in bands_out:
        min_date = src.idxmin(dim='time',skipna=True)
        if 'MinDate' in bands_out:
            min_date.attrs = attrs
            ras = os.path.join(out_dir,'MinDate.tif')
            ras_list.append(ras)
            print('making minDate raster')
            min_date.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'MinDateCos' in bands_out:
        min_date_360 = min_date * min_date/360
        min_date_cos = np.cos(min_date_360)
        min_date_cos.attrs = attrs
        ras = os.path.join(out_dir,'MinDateCos.tif')
        ras_list.append(ras)
        min_date_cos.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)

    for img in sorted(os.listdir(img_dir)):
        if (start_mo == 1 and img.startswith(str(start_yr))) or (start_mo > 1 and img.startswith(str(int(start_yr)+1))):
            if img.endswith('020.tif') and 'Jan' in bands_out:
                ras_list.append(os.path.join(img_dir,img))
        if (start_mo <= 2 and img.startswith(str(start_yr))) or (start_mo > 2 and img.startswith(str(int(start_yr)+1))):
            if img.endswith('051.tif') and 'Feb' in bands_out:
                ras_list.append(os.path.join(img_dir,img))
        if (start_mo <= 3 and img.startswith(str(start_yr))) or (start_mo > 3 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('079.tif') | img.endswith('080.tif')) and 'Mar' in bands_out:
                ras_list.append(os.path.join(img_dir,img))
        if (start_mo <= 4 and img.startswith(str(start_yr))) or (start_mo > 4 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('110.tif') | img.endswith('111.tif')) and 'Apr' in bands_out:
                ras_list.append(os.path.join(img_dir,img))
        if (start_mo <= 5 and img.startswith(str(start_yr))) or (start_mo > 5 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('140.tif') | img.endswith('141.tif')) and 'May' in bands_out:
                ras_list.append(os.path.join(img_dir,img))
        if (start_mo <= 6 and img.startswith(str(start_yr))) or (start_mo > 6 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('171.tif') | img.endswith('172.tif')) and 'Jun' in bands_out:
                ras_list.append(os.path.join(img_dir,img))
        if (start_mo <= 7 and img.startswith(str(start_yr))) or (start_mo > 7 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('201.tif') | img.endswith('202.tif')) and 'Jul' in bands_out:
                ras_list.append(os.path.join(img_dir,img))
        if (start_mo <= 8 and img.startswith(str(start_yr))) or (start_mo > 8 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('232.tif') | img.endswith('233.tif')) and 'Aug' in bands_out:
                ras_list.append(os.path.join(img_dir,img))
        if (start_mo <= 9 and img.startswith(str(start_yr))) or (start_mo > 9 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('263.tif') | img.endswith('264.tif')) and 'Sep' in bands_out:
                ras_list.append(os.path.join(img_dir,img))
        if (start_mo <= 10 and img.startswith(str(start_yr))) or (start_mo > 10 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('293.tif') | img.endswith('294.tif')) and 'Oct' in bands_out:
                ras_list.append(os.path.join(img_dir,img))
        if (start_mo <= 11 and img.startswith(str(start_yr))) or (start_mo > 11 and img.startswith(str(int(start_yr)+1))):
            if (img.endswith('324.tif') | img.endswith('325.tif')) and 'Nov' in bands_out:
                ras_list.append(os.path.join(img_dir,img))
        if (start_mo <=12 and img.startswith(str(start_yr))):
            if (img.endswith('354.tif') | img.endswith('355.tif')) and 'Dec' in bands_out:
                ras_list.append(os.path.join(img_dir,img))
    
    print(ras_list)
    if len(ras_list)<len(bands_out):
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
            out_ras = os.path.join(out_dir,'{:06d}'.format(int(grid_cell))+'_'+str(start_yr)+spec_index+'_'+ bands_out[0]+bands_out[1]+bands_out[2]+bands_out[3]+'.tif')
        elif len(ras_list)==2:
            out_ras = os.path.join(out_dir,'{:06d}'.format(int(grid_cell))+'_'+str(start_yr)+spec_index+'_'+bands_out[0]+bands_out[1]+'.tif')
        elif len(ras_list)==1:
            out_ras = os.path.join(out_dir,'{:06d}'.format(int(grid_cell))+'_'+str(start_yr)+spec_index+'_'+bands_out[0]+'.tif')
        else:
            out_ras = os.path.join(out_dir,'{:06d}'.format(int(grid_cell))+'_'+str(start_yr)+spec_index+'_'+ bands_out[0]+bands_out[1]+bands_out[2]+'.tif') 
        with rio.open(out_ras, 'w', **meta) as dst:
            for id, layer in enumerate(ras_list, start=1):
                with rio.open(layer) as src1:
                    dst.write(src1.read(1),id)

    return out_ras