#!/usr/bin/env python
# coding: utf-8

import os
import sys
import datetime
import xarray as xr
import rasterio as rio
import numpy as np

def GetImgDate(img, imageType, data_source=None):
    '''
    old method. TODO: Make sure current is working on old GEE images and delete this:  
    elif data_source == 'GEE':   
        if imgBase.startswith('L1C'):
            YYYY = int(imgBase[19:23])
            MM = int(imgBase[23:25])
            DD = int(imgBase[25:27])
        elif imgBase.startswith('LC') or imgBase.startswith('LT') or imgBase.startswith('LE'):
            YYYY = int(imgBase[17:21])
            MM = int(imgBase[21:23])
            DD = int(imgBase[23:25]
     '''
    if '.' in img:
        imgBase = os.path.basename(img)
    else:
        imgBase = img
    
    if imageType == 'Smooth':  #Expects images to be named YYYYDDD
        YYYY = int(imgBase[:4])
        doy = int(imgBase[4:7])
    elif imageType not in ['Sentinel','Landsat','Landsat5','Landsat7','Landsat8','Landsat9','AllRaw']:
        print ('Currently valid image types are Smooth,Sentinel,Landsat(5,7,8,9) and AllRaw. You put {}'.format(imageType))
    else:
        if imageType == 'Sentinel' and 'brdf' not in str(imgBase):
            YYYYMMDD = imgBase.split('_')[2][:8]
        else:
            YYYYMMDD = imgBase.split('_')[3][:8]
        YYYY = int(YYYYMMDD[:4])
        MM = int(YYYYMMDD[4:6])
        DD = int(YYYYMMDD[6:8])
   
        ymd = datetime.datetime(YYYY, MM, DD)
        doy = int(ymd.strftime('%j'))
    
    return YYYY, doy

def GetValidPixPer(imgPath):
    if imgPath.endswith('.tif'):
        with rio.open(imgPath) as src:
            no_data = src.nodata
            img = src.read(4)
        allpix = img.shape[0]*img.shape[1]
        NANpix = np.count_nonzero(np.isnan(img))
        validpix = allpix-NANpix
    elif imgPath.endswith('.nc'):
        with xr.open_dataset(imgPath) as xrimg:
            xr_idx = xrimg['nir']
        xr_idx_valid = xr_idx.where(xr_idx < 10000)
        allpix = xr_idx.count() 
        validpix = xr_idx_valid.count()
    
    validper = (validpix/allpix)    
    return validper

def GetClosestImage(img_dir, imageType, data_source, TargetYr, TargetDay):
    '''
    returns path of image closest to {TargetDay (DDD)} of {TargetYear (YYYY)} in directory of images, {img_dir}
    Currently looks for images with name YYYYDDD* (DDD is day-of-year(1-365)) if imageType = Smooth
    or images with name YYYYMMDD as first 8 characters after third '_' if imageType = Landsat or Sentinel
    '''
    imgFiles = []
    imgList = []
    
    if data_source == 'stac' and 'brdf' not in str(img_dir):
        if imageType.startswith('Landsat'):
            imdir = os.path.join(img_dir,'landsat')
        elif imageType.startswith('Sentinel'):
            imdir = os.path.join(img_dir,'sentinel2')
    else:
        imdir = img_dir
            
    if imageType == 'Smooth':
        for img in os.listdir(imdir):
            if img.endswith('.tif'):
                imgDate = GetImgDate(img, 'Smooth', data_source)
                if imgDate[0] == TargetYr:
                    imgList.append(imgDate[1])
    
    else:
        for img in os.listdir(imdir):
            if img.endswith(tuple(['.nc','tif'])):
                if data_source == 'GEE' or ('brdf' not in str(imdir)):
                    imgTyp = os.path.basename(img)[:4]
                elif data_source == 'stac' and ('brdf' in str(imdir)):
                    imgTyp = os.path.basename(img).split('_')[1][:4]
                
                if (imageType in ['Sentinel','AllRaw'] and imgTyp[:2] in ['S2','L1']) or (imageType in ['Landsat','AllRaw'] and imgTyp[:2] in['LC','LT','LE']) or (imageType == 'Landsat7' and imgTyp == 'LE07') or (imageType == 'Landsat8' and imgTyp == 'LC08') or (imageType == 'Landsat9' and imgTyp == 'LC09') or (imageType == 'Landsat5' and imgTyp == 'LT05'):
                    if all(x not in img for x in ['angles','cloudless']):
                        imgFiles.append(img)
                        imgDate = GetImgDate(img, imageType, data_source)
                        if imgDate[0] == TargetYr:
                            imgList.append(imgDate[1])
                
    if len(imgList) == 0: 
        print('     there are no {} images for target year {}'.format(imageType, TargetYr))
        closestimg = None
        
    else:       
        closestDay = min(imgList, key=lambda x:abs(x-TargetDay))
        #print('closest day for {} is: {}'.format(imageType, str(TargetYr)+str(closestDay)))
        closestMM = (datetime.datetime(TargetYr, 1, 1) + datetime.timedelta(closestDay - 1)).month
        closestDD = (datetime.datetime(TargetYr, 1, 1) + datetime.timedelta(closestDay - 1)).day
    
        validImgs = [] #There can be more than one image if more than one path overlaps cell; pick the largest file
        
        if imageType == 'Smooth':
            for fname in os.listdir(imdir):
                if str(TargetYr)+str(closestDay) in fname:
                    closestimg = os.path.join(imdir,fname)
                    print(closestimg)
                    #validImgs.append(os.path.join(imdir,fname))
        else:
            for fname in imgFiles:
                if str(TargetYr)+'{:02d}'.format(closestMM)+'{:02d}'.format(closestDD) in fname and fname.endswith(tuple(['.nc','tif'])) and all(x not in fname for x in ['angles','cloudless']):
                    #print('image match with {} of file size {}'.format(fname, os.stat(imgPath).st_size))
                    validImgs.append(os.path.join(imdir,fname))
                    
            closestimg = max(validImgs, key =  lambda x: os.stat(x).st_size)
        print('{} with file size {}'.format(closestimg, os.stat(closestimg).st_size))
            
    return closestimg

def GetImgFromPlanetaryHub(iid):
    from pystac_client import Client
    import planetary_computer

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    collect = catalog.get_collection("landsat-c2-l2")
    img = collect.get_item(id=iid)
    
    return img
