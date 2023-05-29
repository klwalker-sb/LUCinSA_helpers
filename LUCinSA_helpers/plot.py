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
def explore_band(img, band):

    print('plotting {} band of image {}'.format(band, img))
    fig, axarr = plt.subplots(1, 3, figsize=(15,5))
    
    if img.endswith('.tif'):
        print ('this is in .tif format')
        with rio.open(img) as src:
            ts_samp = src.read(4)
            ts_samp_clean = src.read(4, masked=True)
            print('The no data value is:', src.nodata)
            print('min and max nir are {} - {}'.format(ts_samp.min(),ts_samp.max()))
        plot.show_hist(source=ts_samp, bins=50, histtype='stepfilled', alpha=0.5,  ax=axarr[0])
        axarr[0].set_title("original data distribution")
        plot.show_hist(source=ts_samp_clean, bins=50, histtype='stepfilled', alpha=0.5,  ax=axarr[1])
        axarr[1].set_title('masked_data')
        plot.show(ts_samp_clean)
        axarr[2].set_title('{} band'.format(band))
        axarr[2].axis('off')
        
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
        orig_hist = xr_idx.plot.hist(color="purple")
        
        xr_idx_clean = xr_idx.where((xr_idx > 0) & (xr_idx < 10000))
        masked_hist = xr_idx_clean.plot.hist(color="purple")
        print('The no data value is:', xr_idx.rio.nodata)
    
        ts_samp = xr_idx_clean

        xr_idx.plot.hist(color="purple",  ax=axarr[0])
        axarr[0].set_title("original data")
        xr_idx_clean.plot.hist(color="purple", ax=axarr[1])
        axarr[1].set_title("masked data")
        ts_samp.plot(x="x",y="y", ax=axarr[2])
        axarr[2].set_title('{} band'.format(band))
        axarr[2].axis('off')
        
    return axarr

def normalize(array):
    array_min, array_max = array.min(), array.max()
    if array_min < 0:
        array_min = 0
    return (array - array_min) / (array_max - array_min)

def gammacorr(band, gamma):
    return np.power(band, 1/gamma)

def get_rbg_img(image, gamma):
    
    if image.endswith('.tif'):
        with rio.open(image) as src:
            red0 = src.read(3, masked=True)
            green0 = src.read(2, masked=True)
            blue0 = src.read(1, masked=True)
        red = np.ma.array(red0, mask=np.isnan(red0))
        red[red < 0] = 0
        green = np.ma.array(green0, mask=np.isnan(green0))
        green[green < 0] = 0
        blue = np.ma.array(blue0, mask=np.isnan(blue0))
        blue[blue < 0] = 0
        
    elif image.endswith('.nc'):
        with xr.open_dataset(image) as xrimg:
            red = xrimg['red'].where((xrimg['red'] > 0) & (xrimg['red'] < 10000))
            green = xrimg['green'].where((xrimg['green'] > 0) & (xrimg['green'] < 10000))
            blue = xrimg['blue'].where((xrimg['blue'] > 0) & (xrimg['blue'] < 10000))
            #print (red.min(), red.max())
   
    red_g=gammacorr(red, gamma)
    blue_g=gammacorr(blue, gamma)
    green_g=gammacorr(green, gamma)

    red_n = normalize(red_g)
    green_n = normalize(green_g)
    blue_n = normalize(blue_g)

    rgb = np.dstack([red_n, green_n, blue_n])
    
    return rgb
