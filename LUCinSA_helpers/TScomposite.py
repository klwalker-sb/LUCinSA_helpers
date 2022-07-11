#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
import geowombat as gw
import datetime
import rasterio

def MakeTSComposite(gridCell,img_dir,out_dir,StartYr,spec_index,BandsOut):
    ##BandsOut is list, fed to args as string. need to reparse as list:
    if BandsOut.startswith('['):
        BandsOut = BandsOut[1:-1].split(',')
        #BandsOut = list(map(str, BandsOut))

    rasList = []

    if not os.path.exists(out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        print('made new directory: {}'.format(out_dir))

    ##Convert Image stack to XArray using geowombat:
    TStack = []
    DStack = []
    for img in sorted(os.listdir(img_dir)):
        if img.endswith('.tif'):
            imgYr = int(img[:4])
            imgDoy = int(img[4:7])
            if imgYr == StartYr:
                TStack.append(os.path.join(img_dir,img))
                DStack.append(imgDoy)
    with gw.open(TStack, time_names= DStack) as src:
        attrs = src.attrs.copy()

    print(BandsOut)

    if 'Max' in BandsOut or 'Amp' in BandsOut or 'MaxDate' in BandsOut or 'MaxDateCos' in BandsOut:
        Max = src.max(dim='time')
        if 'Max' in BandsOut:
            Max.attrs = attrs
            ras = os.path.join(out_dir,'Max.tif')
            print('making max raster')
            Max.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
            rasList.append(ras)
    if 'Min' in BandsOut or 'Amp' in BandsOut or 'MinDate' in BandsOut or 'MinDateCos' in BandsOut:
        Min = src.min(dim='time')
        if 'Min' in BandsOut:
            Min.attrs = attrs
            print('making min raster')
            ras = os.path.join(out_dir,'Min.tif')
            rasList.append(ras)
            Min.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'Amp' in BandsOut:
        Amp = Max - Min
        Amp.attrs = attrs
        print('making amp raster')
        ras = os.path.join(out_dir,'Amp.tif')
        rasList.append(ras)
        Amp.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'Avg' in BandsOut or 'CV' in BandsOut:
        Avg = src.mean(dim='time')
        if 'Avg' in BandsOut:
            Avg.attrs = attrs
            ras = os.path.join(out_dir,'Avg.tif')
            rasList.append(ras)
            print('making avg raster')
            Avg.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'Std' in BandsOut or 'CV' in BandsOut:
        Std = src.std(dim='time')
        if 'Std' in BandsOut:
            Std.attrs = attrs
            ras = os.path.join(out_dir,'Std.tif')
            rasList.append(ras)
            print('making std raster')
            Std.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'CV' in BandsOut:
        StdvB = Std.dot(1000)
        CV = StdvB/Avg
        CV.attrs = attrs
        ras = os.path.join(out_dir,'CV.tif')
        rasList.append(ras)
        print('making coef of var raster')
        CV.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'MaxDate' in BandsOut or 'MaxDateCos' in BandsOut:
        MaxDate = src.idxmax(dim='time',skipna=True)
        if 'MaxDate' in BandsOut:
            MaxDate.attrs = attrs
            ras = os.path.join(out_dir,'MaxDate.tif')
            rasList.append(ras)
            print('making maxDate raster')
            MaxDate.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'MaxDateCos' in BandsOut:
        MaxDate360 = MaxDate * MaxDate/360
        MaxDateCos = xr.ufuncs.cos(MaxDate360)
        MaxDateCos.attrs = attrs
        ras = os.path.join(out_dir,'MaxDateCos.tif')
        rasList.append(ras)
        MaxDateCos.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'MinDate' in BandsOut or 'MinDateCos' in BandsOut:
        MinDate = src.idxmin(dim='time',skipna=True)
        if 'MinDate' in BandsOut:
            MinDate.attrs = attrs
            ras = os.path.join(out_dir,'MinDate.tif')
            rasList.append(ras)
            print('making minDate raster')
            MinDate.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)
    if 'MinDateCos' in BandsOut:
        MinDate360 = MinDate * MinDate/360
        MinDateCos = xr.ufuncs.cos(MinDate360)
        MinDateCos.attrs = attrs
        ras = os.path.join(out_dir,'MinDateCos.tif')
        rasList.append(ras)
        MinDateCos.gw.to_raster(ras,verbose=1,n_workers=4,n_threads=2,n_chunks=200, overwrite=True)

    for img in sorted(os.listdir(img_dir)):
        if img.startswith(str(StartYr)):
            if img.endswith('020.tif') and 'Jan' in BandsOut:
                rasList.append(os.path.join(img_dir,img))
            if img.endswith('051.tif') and 'Feb' in BandsOut:
                rasList.append(os.path.join(img_dir,img))
            if (img.endswith('079.tif') | img.endswith('080.tif')) and 'Mar' in BandsOut:
                rasList.append(os.path.join(img_dir,img))
            if (img.endswith('110.tif') | img.endswith('111.tif')) and 'Apr' in BandsOut:
                rasList.append(os.path.join(img_dir,img))
            if (img.endswith('140.tif') | img.endswith('141.tif')) and 'May' in BandsOut:
                rasList.append(os.path.join(img_dir,img))
            if (img.endswith('171.tif') | img.endswith('172.tif')) and 'Jun' in BandsOut:
                rasList.append(os.path.join(img_dir,img))
            if (img.endswith('201.tif') | img.endswith('202.tif')) and 'Jul' in BandsOut:
                rasList.append(os.path.join(img_dir,img))
            if (img.endswith('232.tif') | img.endswith('233.tif')) and 'Aug' in BandsOut:
                rasList.append(os.path.join(img_dir,img))
            if (img.endswith('263.tif') | img.endswith('264.tif')) and 'Sep' in BandsOut:
                rasList.append(os.path.join(img_dir,img))
            if (img.endswith('293.tif') | img.endswith('294.tif')) and 'Oct' in BandsOut:
                rasList.append(os.path.join(img_dir,img))
            if (img.endswith('324.tif') | img.endswith('325.tif')) and 'Nov' in BandsOut:
                rasList.append(os.path.join(img_dir,img))
            if (img.endswith('354.tif') | img.endswith('355.tif')) and 'Dec' in BandsOut:
                rasList.append(os.path.join(img_dir,img))
    
    print(rasList)
    if len(rasList)<len(BandsOut):
        print('oops--got an unknown band; Current Band options are Max,Min,Amp,Avg,CV,Std,MaxDate,MaxDateCos,MinDate,MinDateCos,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec')

    else:
        ##Start writing output composite
        with rasterio.open(rasList[1]) as src0:
            meta = src0.meta
            meta.update(count = len(rasList))
            
        # Read each layer and write it to stack
        
        if len(BandsOut)==12:
            outRas = os.path.join(out_dir,'{:06d}'.format(int(gridCell))+'_'+str(StartYr)+spec_index+'_monthly.tif')
        elif len(BandsOut)==4:
            outRas = os.path.join(out_dir,'{:06d}'.format(int(gridCell))+'_'+str(StartYr)+spec_index+'_'+ BandsOut[0]+BandsOut[1]+BandsOut[2]+BandsOut[3]+'.tif')
        elif len(BandsOut)==2:
            outRas = os.path.join(out_dir,'{:06d}'.format(int(gridCell))+'_'+str(StartYr)+spec_index+'_'+BandsOut[0]+BandsOut[1]+'.tif')
        elif len(BandsOut)==1:
            outRas = os.path.join(out_dir,'{:06d}'.format(int(gridCell))+'_'+str(StartYr)+spec_index+'_'+BandsOut[0]+'.tif')
        else:
            outRas = os.path.join(out_dir,'{:06d}'.format(int(gridCell))+'_'+str(StartYr)+spec_index+'_'+ BandsOut[0]+BandsOut[1]+BandsOut[2]+'.tif')
        
        with rasterio.open(outRas, 'w', **meta) as dst:
            for id, layer in enumerate(rasList, start=1):
                with rasterio.open(layer) as src1:
                    dst.write(src1.read(1),id)