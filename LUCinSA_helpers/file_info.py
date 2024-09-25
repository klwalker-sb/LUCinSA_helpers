#!/usr/bin/env python
# coding: utf-8

import os
import sys
import datetime
import xarray as xr
import rasterio as rio
import numpy as np

def get_img_date(img, image_type, data_source=None):
    '''
    old method. TODO: Make sure current is working on old GEE images and delete this:  
    elif data_source == 'GEE':   
        if img_base.startswith('L1C'):
            YYYY = int(img_base[19:23])
            MM = int(img_base[23:25])
            DD = int(img_base[25:27])
        elif img_base.startswith('LC') or img_base.startswith('LT') or img_base.startswith('LE'):
            YYYY = int(img_base[17:21])
            MM = int(img_base[21:23])
            DD = int(img_base[23:25]
     '''
    if '.' in img:
        img_base = os.path.basename(img)
    else:
        img_base = img
    
    if 'Smooth' in image_type:  #Expects images to be named YYYYDDD
        YYYY = int(img_base[:4])
        doy = int(img_base[4:7])
    elif image_type not in ['Sentinel2','Landsat','Landsat5','Landsat7','Landsat8','Landsat9','AllRaw']:
        print ('Currently valid image types are Smooth,Smooth_old,Sentinel,Landsat(5,7,8,9) and AllRaw. You put {}'.format(image_type))
    else:
        if image_type == 'Sentinel' and 'brdf' not in str(img_base):
            YYYYMMDD = img_base.split('_')[2][:8]
        else:
            YYYYMMDD = img_base.split('_')[3][:8]
        YYYY = int(YYYYMMDD[:4])
        MM = int(YYYYMMDD[4:6])
        DD = int(YYYYMMDD[6:8])
   
        ymd = datetime.datetime(YYYY, MM, DD)
        doy = int(ymd.strftime('%j'))
    
    return YYYY, doy

def get_valid_pix_per(img_path):
    if img_path.endswith('.tif'):
        with rio.open(img_path) as src:
            no_data = src.nodata
            img = src.read(4)
        allpix = img.shape[0]*img.shape[1]
        nanpix = np.count_nonzero(np.isnan(img))
        validpix = allpix-nanpix
    elif img_path.endswith('.nc'):
        with xr.open_dataset(img_path) as xrimg:
            xr_idx = xrimg['nir']
        xr_idx_valid = xr_idx.where(xr_idx < 10000)
        allpix = xr_idx.count() 
        validpix = xr_idx_valid.count()
    
    validper = (validpix/allpix)    
    return validper

def get_closest_image(img_dir, image_type, data_source, target_yr, target_day):
    '''
    returns path of image closest to {target_day (DDD)} of {target_year (YYYY)} in directory of images, {img_dir}
    Currently looks for images with name YYYYDDD* (DDD is day-of-year(1-365)) if image_type = Smooth
    or images with name YYYYMMDD as first 8 characters after third '_' if image_type = Landsat or Sentinel
    '''
    
    if data_source == 'stac' and 'brdf' not in str(img_dir):
        if image_type.startswith('Landsat'):
            imdir = os.path.join(img_dir,'landsat')
        elif image_type.startswith('Sentinel'):
            imdir = os.path.join(img_dir,'sentinel2')
    else:
        imdir = img_dir
            
    if image_type == 'Smooth':
        img_list = [get_img_date(i, 'Smooth', data_source)[1] for i in os.listdir(imdir) if (i.endswith('.tif') and get_img_date(i, 'Smooth', data_source)[0] == target_yr)]
        
    else:
        img_files = []
        img_list = []
        for img in os.listdir(imdir):
            if img.endswith(tuple(['.nc','tif'])):
                if data_source == 'GEE' or ('brdf' not in str(imdir)):
                    img_typ = os.path.basename(img)[:4]
                elif data_source == 'stac' and ('brdf' in str(imdir)):
                    img_typ = os.path.basename(img).split('_')[1][:4]
                
                if (image_type in ['Sentinel2','AllRaw'] and img_typ[:2] in ['S2','L1']) or (image_type in ['Landsat','AllRaw'] and img_typ[:2] in['LC','LT','LE']) or (image_type == 'Landsat7' and img_typ == 'LE07') or (image_type == 'Landsat8' and img_typ == 'LC08') or (image_type == 'Landsat9' and img_typ == 'LC09') or (image_type == 'Landsat5' and img_typ == 'LT05'):
                    if all(x not in img for x in ['angles','cloudless']):
                        img_files.append(img)
                        img_date = get_img_date(img, image_type, data_source)
                        if img_date[0] == target_yr:
                            img_list.append(img_date[1])
                
    if len(img_list) == 0: 
        print('     there are no {} images for target year {}'.format(image_type, target_yr))
        closestimg = None
        
    else:       
        closest_day = min(img_list, key=lambda x:abs(x-target_day))
        #print('closest day for {} is: {}'.format(imageType, str(TargetYr)+str(closestDay)))
        closestMM = (datetime.datetime(target_yr, 1, 1) + datetime.timedelta(closest_day - 1)).month
        closestDD = (datetime.datetime(target_yr, 1, 1) + datetime.timedelta(closest_day - 1)).day
    
        valid_imgs = [] #There can be more than one image if more than one path overlaps cell; pick the largest file
        
        if image_type == 'Smooth':
            for fname in os.listdir(imdir):
                if str(target_yr)+str(closest_day) in fname:
                    closestimg = os.path.join(imdir,fname)
                    print(closestimg)
                    #validImgs.append(os.path.join(imdir,fname))
        else:
            for fname in img_files:
                if str(target_yr)+'{:02d}'.format(closestMM)+'{:02d}'.format(closestDD) in fname and fname.endswith(tuple(['.nc','tif'])) and all(x not in fname for x in ['angles','cloudless']):
                    #print('image match with {} of file size {}'.format(fname, os.stat(imgPath).st_size))
                    valid_imgs.append(os.path.join(imdir,fname))
                    
            closestimg = max(valid_imgs, key =  lambda x: os.stat(x).st_size)
        print('{} with file size {}'.format(closestimg, os.stat(closestimg).st_size))
            
    return closestimg

def get_img_from_planetary_hub(iid):
    from pystac_client import Client
    import planetary_computer

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    collect = catalog.get_collection("landsat-c2-l2")
    img = collect.get_item(id=iid)
    
    return img
