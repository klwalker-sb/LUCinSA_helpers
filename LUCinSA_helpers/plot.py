#!/usr/bin/env python
# coding: utf-8
import os
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio import plot
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio as rio
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import pyproj
from pyproj import Proj, transform
import geowombat as gw
import xarray
from IPython.display import Image
from PIL import Image as PILImage
#import rioxarray
#from shapely.geometry import box
#from shapely.geometry import shape
#from shapely.geometry import MultiPoint
#from shapely.geometry import Point
#from shapely.geometry import Polygon
from ipywidgets import Label
from ipyleaflet  import Map, GeoData, basemaps, LayersControl, ImageOverlay, Marker
from localtileserver import get_leaflet_tile_layer, TileClient
#from rio_cogeo.cogeo import cog_translate
#from rio_cogeo.profiles import cog_profiles

#pyver = float((sys.version)[:3])
#if pyver >= 3.8:
#    import geemap   ##Note geemap doesn't work with Python 3.6 which is native on cluster, but works in a conda envt


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

def get_coords(**kwargs):
    '''
    Note: if selected_coords in initialized on same cell, this will add results, but cannot pass list
    '''
    if kwargs.get('type') == 'click':
        label = Label()
        label.value = str(kwargs.get('coordinates'))
        coords =eval(label.value) 
        selected_coords.append(coords)
        print(selected_coords)
        return selected_coords
    
def convert_and_print_coord_list(coord_list,img_crs, out_dir):
    coord_list_lat = [c[1] for c in coord_list]
    coord_list_lon = [c[0] for c in coord_list]
    ### Convert list of coordinates back to original CRS and print to file:
    coord_listX = []
    coord_listY = []
    transformer = pyproj.Transformer.from_crs("epsg:4326", img_crs)
    for pt in transformer.itransform(coord_list):
        print('{:.3f} {:.3f}'.format(pt[0],pt[1]))
        coord_listX.append(pt[0])
        coord_listY.append(pt[1])
        coords = {'XCoord':coord_listX,'YCoord':coord_listY, 'lat':coord_list_lat,'lon':coord_list_lon}
    coorddb = pd.DataFrame(coords)
    coorddb = coorddb.astype({'XCoord':'float','YCoord':'float', 'lat':'float', 'lon':'float'})
    coord_path = os.path.join(out_dir,'SelectedCoords.csv')
    coorddb.to_csv(coord_path, sep=',', na_rep='NaN', index=True)
    return coord_path

def get_values_at_coords(coord_list, coord_crs, img, bands):
    
    ptsval = {}
    if isinstance(coord_list, pd.DataFrame):
        ptsdf = coord_list
    else:
        ptsdf = pd.read_csv(coord_list)

    pts = gpd.GeoDataFrame(ptsdf,geometry=gpd.points_from_xy(ptsdf.XCoord,ptsdf.YCoord),crs=coord_crs)
    xy = [pts['geometry'].x, pts['geometry'].y]
    coords = list(map(list, zip(*xy)))
    
    #if 'Smooth' in image_type:
    if img.endswith('.tif'):
        #img_name = os.path.basename(img)[:7]
        with rio.open(img, 'r') as src:
            for b in bands:
                ptsval[b] = [sample[b-1] for sample in src.sample(coords)]

    #elif 'Sentinel' in image_type or 'Landsat' in image_type or image_type == 'AllRaw':
    elif img.endswith('.nc'): 
        #YYYY, doy = get_img_date(img, image_type, data_source=None)
        #img_name = str(YYYY)+str(doy)
        xrimg = xr.open_dataset(img)
        for b in bands:
            xr_val = xrimg[b.where(xrimg[b] < 10000)]

            vals=[]
            for index, row in pts.iterrows():
                thispt_val = xr_val.sel(x=pts['geometry'].x[index],y=pts['geometry'].y[index], method='nearest', tolerance=30)
                this_val = thispt_val.values
                vals.append(this_val)
                ptsval[b] = vals
    
    return ptsval

def add_shpfile_overlay(shp, ptfile, inputCRS, polyfile=None):
    ### TO Add A shapefile to map (optional):
    if shp != None:
        if shp == 'point':
            if ptfile.endswith('.txt'):
                ptsdf = pd.read_table(ptfile, index_col=0, sep=",")
                shps = gpd.GeoDataFrame(ptsdf,geometry=gpd.points_from_xy(ptsdf.XCoord,ptsdf.YCoord),crs=inputCRS)
            elif ptfile.endswith('.csv'):
                ptsdf = pd.read_csv(ptfile, index_col=0)
                shps = gpd.GeoDataFrame(ptsdf,geometry=gpd.points_from_xy(ptsdf.XCoord,ptsdf.YCoord),crs=inputCRS)
            else:
                shps = gpd.read_file(ptfile)
        elif shp == 'poly':
            shps = gpd.read_file(polyfile)
        shps_ll = shps.to_crs("EPSG:4326")
        viewshp = GeoData(geo_dataframe = shps_ll)
    return viewshp

def img_to_cog(img_path, cog_path):
    from rio_cogeo.cogeo import cog_translate
    from rio_cogeo.profiles import cog_profiles
    
    img_cog = (os.path.join(cog_path,'COG.tif'))
    cog_translate(img_path, cog_path, cog_profiles.get("deflate"))
    
def reproject_to_ll(img, out_dir):
    '''
    tanslates .tif images from native resolution to lat-lon
    if image is .nc, first convert to .tif with nc_to_tif, then run this on the .tif
    '''
    target_crs = 'epsg:4326' # Global lat-lon coordinate system (ipyleaflet only uses lat lon)
    img_ll = os.path.join(out_dir,'temp_img_ll.tif')
    with rio.open(img) as src:
        if src.crs == target_crs:
            print('image is already in latlon')
            img_ll = img
        else:
            print('original image is in {}, translating to lat_lon...'.format(src.crs))
            kwargs=src.meta.copy()
            transform, width, height = calculate_default_transform(src.crs, target_crs, src.width, src.height, *src.bounds)
            kwargs.update({'crs': target_crs,'transform': transform,'width': width,'height': height})
            print(kwargs)
            with rio.open(img_ll, 'w+', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rio.band(src, i),
                        destination=rio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest)
            print(dst.bounds)
               
    return img_ll

def nc_to_tif(img, out_dir):
    band_names = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    with gw.open(img, band_names=band_names, chunks={'band': -1, 'y': 512, 'x': 512}, engine='h5netcdf') as src:
        tif_out = ((src.chunk({'y': 512,'x': 512})).assign_attrs(**src.attrs))
        temp_file = os.path.join(out_dir,'temp_img_tif.tif')
        if Path(temp_file).is_file():
            Path(temp_file).unlink()
        tif_out.gw.save(temp_file, overwrite=True)
    
    return temp_file
    
def get_img_center(img,out_dir=None):
    '''
    returns lat-lon coordinates of image center point for .tif or .nc images
    '''
    target_crs = 'epsg:4326'
    if img.endswith('.tif'):
        with rio.open(img) as src:
            if src.crs != target_crs:
                print('image is in {}...projecting to latlon'.format(src.crs))
                img = translate_image(img, out_dir)
            else:
                print('image is now in {}'.format(src.crs))
            img_centerT = src.xy(src.height // 2, src.width // 2)
            img_center = [img_centerT[1],img_centerT[0]]
            sw = [src.bounds[1],src.bounds[0]]
            ne = [src.bounds[3],src.bounds[2]]
            img_bounds = [sw, ne]
            print('Image center is at: {}'.format(img_center))
            print('SW and NW corners are at: {}'.format(img_bounds))
    elif img.endswith('.nc'):
        with xr.open_dataset(img) as xrimg:
            crs_to_latlon = pyproj.Transformer.from_crs(xrimg.crs, target_crs, always_xy=True)
            lonB, latB = crs_to_latlon.transform([xrimg.x.min(), xrimg.x.max()],[xrimg.y.min(), xrimg.y.max()])
            img_bounds = [(latB[0],lonB[0]),(latB[1],lonB[1])]
            img_center = (latB[0]+((latB[1]-latB[0])/2), lonB[0]+((lonB[1]-lonB[0])/2))
    else:
        print('only set up to read .tif and .nc files at the moment')
    
    return img_center
            
def show_interactive_img(img, open_port, out_dir=None):
    '''
    Note: these steps work, but does not work when passed through method -- need to enter lines directly in cell
    '''
    if img.endswith('.tif'):
        tile_client = TileClient(img,port=open_port)
        cent = tile_client.center()
    elif img.endswith('nc'):
        cent = get_img_center(img, out_dir)
        img_disp = nc_to_tif(img, out_dir)
        #img_ll = reproject_to_ll(img_disp,out_dir) #not needed anymore
        tile_client = TileClient(img_disp,port=open_port)
    else:
        print('need to add this filetype to the show_interactive_img method')
    
    #if numbands == 3: #TODO: get this from image, not parameter
    #img_ll = translate_image(img_disp)
    #img_center = get_img_center(img_ll)
    ## note: ipleaflet will now find center directly and reproject on the fly for .tifs
    m = Map(center=cent, zoom=12, basemap=basemaps.Esri.WorldImagery)
    t = get_leaflet_tile_layer(tile_client, band=[1,2,3])
    #m.add_layer(t)

    return m,t

def make_thumbnails(img_dir,thumbnail_dir,gamma,reduct_factor=10):
    if not os.path.exists(thumbnail_dir):
        os.makedirs(thumbnail_dir)
    imgs = [f for f in os.listdir(img_dir) if f.endswith(tuple(['.nc','.tif']))]
    for i, img in enumerate(imgs):
        imgi = get_rbg_img(os.path.join(img_dir,img),gamma)
        imgis = imgi[::reduct_factor, ::reduct_factor]
        fig = plt.figure(figsize=(20,20),dpi=80)
        plt.imsave(os.path.join(thumbnail_dir,"{}.png".format(img)), imgis)
        plt.close(fig)
        
def view_thumbnails(img_dir,thumbnail_dir,out_file,gamma,exclude,include,yrs,reduct_factor=10):
    if not os.path.exists(thumbnail_dir):
        make_thumbnails(img_dir, thumbnail_dir, gamma, reduct_factor=10)
    if len(os.listdir(thumbnail_dir)) == 0:
        make_thumbnails(img_dir, thumbnail_dir, gamma, reduct_factor=10)
    if exclude:
        to_view = [f for f in os.listdir(thumbnail_dir) if f.endswith('.png') and exclude not in f]   
    if include:
        to_view = [f for f in os.listdir(thumbnail_dir) if f.endswith('.png') and include in f]
    else:
        to_view = [f for f in os.listdir(thumbnail_dir) if f.endswith('.png')]
    if yrs:
    #if len(yrs) > 0:
        images = [f for f in to_view if f.split('_')[3].startswith(yrs)]
    else:
        images = to_view
        
    columns = 10
    space=1
    rows = len(images) // columns
    if len(images) % columns:
        rows += 1
    width_max = max([PILImage.open(os.path.join(thumbnail_dir,image)).width for image in images])
    height_max = max([PILImage.open(os.path.join(thumbnail_dir,image)).height for image in images])
    background_width = width_max*columns + (space*columns)-space
    background_height = height_max*rows + (space*rows)-space
    background = PILImage.new('RGBA', (background_width, background_height), (255, 255, 255, 255))
    x = 0
    y = 0
    for i, image in enumerate(images):
        img = PILImage.open(os.path.join(thumbnail_dir,image))
        x_offset = int((width_max-img.width)/2)
        y_offset = int((height_max-img.height)/2)
        background.paste(img, (x+x_offset, y+y_offset))
        x += width_max + space
        if (i+1) % columns == 0:
            y += height_max + space
            x = 0
    background.save(out_file)