#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from rasterio import plot
from rasterio.plot import show
from rasterio.plot import show_hist
import rasterio as rio
import xarray as xr
import numpy as np

### For visualizing index image (Process check 1)
def exploreBand(img, band):

    print('plotting {} band of image {}'.format(band, img))
    fig, axarr = plt.subplots(1, 3, figsize=(15,5))
    
    if img.endswith('.tif'):
        print ('this is in .tif format')
        with rio.open(img) as src:
            TSamp = src.read(4)
            print('The no data value is:', src.nodata)
            print('min and max nir are {} - {}'.format(TSamp.min(),TSamp.max()))
            plot.show_hist(source=src, bins=50, histtype='stepfilled', alpha=0.5,  ax=axarr[0])
            axarr[0].set_title("original data distribution")
            plot.show(TSamp)
            axarr[1].set_title('{} band'.format(band))
            axarr[1].axis('off')
        
    elif img.endswith('.nc'):
        print ('this is in .nc format')
        with xr.open_dataset(img) as xrimg:
            xrcrs = xrimg.crs
            print(xrcrs)  ##epsg:32632
            #print(xrimg.variables)
            bands=[i for i in xrimg.data_vars]
            print(bands)
            #print(xrimg.coords)
            
        ### Check that x and y values correspond to UTM coords
        print("Coord range is: y: {}-{}. x: {}-{}".format(
          xrimg[band]["y"].values.min(), 
          xrimg[band]["y"].values.max(),
          xrimg[band]["x"].values.min(), 
          xrimg[band]["x"].values.max()))
    
        ### Get single band (Opens as Dataset)
        xr_idx = xrimg[band]
    
        ##Note outliers (while nodata = None)
        origHist = xr_idx.plot.hist(color="purple")
        
        xr_idx_clean = xr_idx.where(xr_idx < 10000)
        maskedHist = xr_idx_clean.plot.hist(color="purple")
        print('The no data value is:', xr_idx.rio.nodata)
    
        TSamp = xr_idx_clean

        xr_idx.plot.hist(color="purple",  ax=axarr[0])
        axarr[0].set_title("original data")
        xr_idx_clean.plot.hist(color="purple", ax=axarr[1])
        axarr[1].set_title("masked data")
        TSamp.plot(x="x",y="y", ax=axarr[2])
        axarr[2].set_title('{} band'.format(band))
        axarr[2].axis('off')
        
    return axarr

def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

def gammacorr(band, gamma):
    return np.power(band, 1/gamma)

def GetrbgImg(image, gamma):
    
    if image.endswith('.tif'):
        with rio.open(image) as src:
            red = src.read(3, masked=True)
            green = src.read(2, masked=True)
            blue = src.read(1, masked=True)
        
        red_n = normalize(red)
        green_n = normalize(green)
        blue_n = normalize(blue)
        
    elif image.endswith('.nc'):
        with xr.open_dataset(image) as xrimg:
            red = xrimg['red'].where(xrimg['red'] != 65535)
            green = xrimg['green'].where(xrimg['green'] != 65535)
            blue = xrimg['blue'].where(xrimg['blue'] != 65535)
            #print (red.min(), red.max())
   
        red_g=gammacorr(red, gamma)
        blue_g=gammacorr(blue, gamma)
        green_g=gammacorr(green, gamma)

        red_n = normalize(red_g)
        green_n = normalize(green_g)
        blue_n = normalize(blue_g)

    rgb = np.dstack([red_n, green_n, blue_n])
    
    return rgb
