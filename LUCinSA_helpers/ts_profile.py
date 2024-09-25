
import os
import sys
from pathlib import Path
import rasterio as rio
from rasterio import plot
import matplotlib.pyplot as plt
import shutil
import tempfile
import json
import random
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Proj, transform
from pyproj import CRS
from shapely.geometry import box
from shapely.geometry import shape
from shapely.geometry import MultiPoint
from shapely.geometry import Point
from shapely.geometry import Polygon
from rasterio.plot import show
import xarray as xr
from LUCinSA_helpers.file_info import get_img_date


def get_coord_at_row_col (img, spec_index, row, col):
    df = explore_band(img, spec_index)
    test_samp = df[row,col]
    print('Test Samp at coords x={}, y={} is {}'.format(test_samp['x'].values,test_samp['y'].values, test_samp.values))
    return test_samp['x'].values, test_samp['y'].values

def get_val_at_XY(img, spec_index, xcoord,ycoord):
    df = explore_band(img, spec_index)
    one_point = df[spec_index].sel(x=xcoord,y=ycoord, method='nearest', tolerance=15)
    print('value at {},{}={}'.format(xcoord,ycoord,one_point.values))
    return one_point.values

def get_polygons_in_aoi(out_dir, aoi_file, poly_path, oldest=2018, newest=2020):
    '''
    Filters polygon layer to contain only those overlapping selected AOI
    Allows filtering by 'FirstYrObs','ObsYr','Year'or'Acquired' to remove polygons obseserved during years outside the period of interest
    Outputs new polygon set to a .json file
    '''
    all_polys = gpd.read_file(poly_path)
    aoi = gpd.read_file(aoi_file)
    # Warn if both layers not in same coordinate system:
    if (all_polys.crs != aoi.crs):
        print("AOI and ground truth files are NOT are in the same coordinate system!",
          all_polys.crs, aoi.crs)

    # Clip polygons to AOI
    polys_in_aoi = gpd.clip(all_polys, aoi)

    print("Of the {} polygons, {} are in AOI". format (all_polys.shape[0], polys_in_aoi.shape[0]))
    if 'FirstYrObs' in polys_in_aoi.columns:
        yr_filter = polys_in_aoi['FirstYrObs']
    elif 'ObsYr' in polys_in_aoi.columns:
        yr_filter = polys_in_aoi['ObsYr']
    elif 'Year' in polys_in_aoi.columns:
        yr_filter = polys_in_aoi['Year']
    elif 'Acquired' in polys_in_aoi.columns:
        yr_filter = polys_in_aoi['Acquired'][:4]
    
    ##Filter out polygons that were observed before year set as 'oldest' or after year set as 'newest'
    if oldest > 0:
        polys_in_aoi = polys_in_aoi[int(yr_filter) >= oldest]
    if newest > 0:
        polys_in_aoi = polys_in_aoi[int(yr_filter) <= newest]
        print("{} polygons observed between {} and {} in AOI".format(len(polys_in_aoi),oldest,newest))

    poly_clip = Path(os.path.join(out_dir,'polysInAOI.json'))
    polys_in_aoi.to_file(poly_clip, driver="GeoJSON")

    return poly_clip


def get_pts_in_grid (grid_file, grid_cell, ptfile):
    '''
    loads point file (from .csv with 'XCoord' and 'YCoord' columns) and returns points that overlap a gridcell
    as a geopandas GeoDataFrame. Use this if trying to match/append data to existing sample points
    rather than making a new random sample each time (e.g. if matching Planet and Sentinel points)
    Note that crs of point file is known ahead of time and hardcoded here to match specific grid file.
    '''
    out_path = Path(grid_file).parent

    with tempfile.TemporaryDirectory(dir=out_path) as temp_dir:
        temp_file = Path(temp_dir) / Path(grid_file).name
        shutil.copy(grid_file, temp_file)
        df = gpd.read_file(temp_file)
        crs_grid = df.crs
        print(crs_grid)
    #sys.stderr.write('grid is in: {}'.format(crs_grid))  #ESRI:102033
    #sys.stderr.write('{}'.format(df))

    if isinstance(ptfile, pd.DataFrame):
        ptsdf = ptfile
    else:
        ptsdf = pd.read_csv(ptfile, index_col=0)
    pts = gpd.GeoDataFrame(ptsdf,geometry=gpd.points_from_xy(ptsdf.XCoord,ptsdf.YCoord),crs=crs_grid)
    
    if df.shape[0] > 1:
        bb = df.query(f'UNQ == {grid_cell}').geometry.total_bounds
    else:
        bb = df.geometry.total_bounds
    sys.stderr.write('bb = {} \n'.format(bb))
    
    grid_bbox = box(bb[0],bb[1],bb[2],bb[3])
    grid_bounds = gpd.GeoDataFrame(gpd.GeoSeries(grid_bbox), columns=['geometry'], crs=crs_grid)
    print(grid_bounds)

    pts_in_grid = gpd.sjoin(pts, grid_bounds, predicate='within')
    pts_in_grid = pts_in_grid.loc[:,['geometry']]

    print("Of the {} ppts, {} are in gridCell {}". format (pts.shape[0], pts_in_grid.shape[0],grid_cell))

    #Write to geojson file
    if pts_in_grid.shape[0] > 0:
        pt_clip = Path(os.path.join(out_path,'ptsGrid_'+str(grid_cell)+'.json'))
        pts_in_grid.to_file(pt_clip, driver="GeoJSON")

        return pts_in_grid
        print(pts_in_grid.head(n=5))


def get_polygons_in_grid (grid_file, grid_cell, poly_path, oldest=2018, newest=2020):
    '''
    Filters polygon layer to contain only those overlapping selected grid cell (allows for iteration through grid)
    Allows filtering by 'FirstYrObs','ObsYr','Year'or'Acquired' to remove polygons obseserved during years outside the period of interest
    Outputs new polygon set to a .json file stored in the gridcell directory
    '''
    polys = gpd.read_file(poly_path)
    out_path = Path(grid_file).parent

    with tempfile.TemporaryDirectory(dir=out_path) as temp_dir:
        temp_file = Path(temp_dir) / Path(grid_file).name
        shutil.copy(grid_file, temp_file)
        df = gpd.read_file(temp_file)
        crs_grid = df.crs

    bb = df.query(f'UNQ == {grid_cell}').geometry.total_bounds

    grid_bbox = box(bb[0],bb[1],bb[2],bb[3])
    grid_bounds = gpd.GeoDataFrame(gpd.GeoSeries(grid_bbox), columns=['geometry'], crs=crs_grid)
    polys_in_grid = gpd.overlay(grid_bounds, polys, how='intersection')

    print("Of the {} polygons, {} are in grid_cell {}". format (polys.shape[0], polys_in_grid.shape[0],grid_cell))

    ##Filter out polygons that were observed before year set as 'oldest' or after year set as 'newest'
    if 'FirstYrObs' in polys_in_grid.columns:
        yr_filter = polys_in_grid['FirstYrObs']
    elif 'ObsYr' in polys_in_grid.columns:
        yr_filter = polys_in_grid['ObsYr']
    elif 'Year' in polys_in_grid.columns:
        yr_filter = polys_in_grid['Year']
    elif 'Acquired' in polys_in_grid.columns:
        yr_filter = polys_in_grid['Acquired'][:4]
        
    if oldest > 0:
        polys_in_grid = polys_in_grid[int(yr_filter) >= oldest]
    if newest > 0:
        polys_in_grid = polys_in_grid[int(yr_filter) <= newest]
        print("{} polygons first observed between {} and {} in AOI".format(len(polys_in_grid),oldest,newest))

    #Write to geojson file
    if polys_in_grid.shape[0] > 0:
        poly_clip = Path(os.path.join(out_path,'polysGrid_'+str(grid_cell)+'.json'))
        polys_in_grid.to_file(poly_clip, driver="GeoJSON")

        return poly_clip


def plot_poly_on_index(zoom_poly, img, poly_file):
    '''
    Plots a polygon file {'polyFile'} on top of a .tiff image {'Img'}
    and zooms to a selected poly {'zoomPoly'}
    '''
    #fig, ax = plt.subplots(figsize=(12,12),subplot_kw={'projection': ccrs.epsg(32632)})
    fig, ax = plt.subplots(figsize=(12,12))

    if img.endswith('.tif'):
        with rio.open(img) as src:
            img1 = src.read()
        plot.show(img1, ax=ax)
    elif img.endswith('.nc'):
        with xr.open_dataset(img) as xrimg:
            xr_band = xrimg.nir
            img1 = xr_band.where(xr_band < 10000)
        img1.plot(x="x",y="y")
        plot.show(img1, ax=ax)

    polys = gpd.read_file(poly_file)
    polys.plot(ax=ax, facecolor='none', edgecolor='orangered')

    #Get bounds to zoom into polygon specified in arguments
    polybds = polys.query(f'FID == {zoom_poly}').bounds
    minx=int((polybds['minx'][1])-100)
    maxx=int((polybds['maxx'][1])+100)
    miny=int((polybds['miny'][1])-100)
    maxy=int((polybds['maxy'][1])+100)

    plt.axis([minx,maxx,miny,maxy])
    polys.head(n=5)


def get_ran_pt_in_poly(polyg, seed):
    '''
    Returns a shapely Point object for a random point within a polygon {'polyg'}
    '''
    minx, miny, maxx, maxy = polyg.bounds
    while True:
        np.random.seed(seed)
        pp = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polyg.contains(pp):
            #print(pp)
            return pp


def get_ran_pts_in_polys(polys, npts, seed=88):
    '''
    Returns a geodataframe with geometry column containing shapely Point objects
    With {'npts'} random samples from each polygon in {'polys'}
    ({'polys'} is path to json file with polygon geometries)
    '''
    with open(polys, 'r') as polysrc:
        polys2 = json.load(polysrc)
        #print(polys['type']) #FeatureCollection
        poly_list = []
        pt_dict = {}
        for poly in polys2['features']:
            poly_list.append(poly['geometry'])
            polyobj = shape(poly['geometry'])
            for i in range(0,npts):
                pt_name = str(poly['properties']['id'])+'_'+str(i+1)
                pt_in_poly = get_ran_pt_in_poly(polyobj, seed)
                pt_dict[pt_name] = pt_in_poly

    ptsdb = pd.DataFrame.from_dict(pt_dict, orient='index')
    ptsgdb = gpd.GeoDataFrame(ptsdb, geometry=ptsdb[0])
    ptsgdb.drop(columns=[0], inplace=True)

    return ptsgdb


def get_coords_for_poly_samp_in_AOI(out_dir, polygons, aoi, oldest=2018, newest=2018, nPts=2, seed=88):
    '''
    Gets random points in all polygons in AOI. returns a csv file with X and Y coords
    '''
    polys = (out_dir, aoi, polygons, oldest, newest)
    pts = get_ran_pts_in_polys(polys, nPts, seed)
    pts.crs = "epsg:32632"
    print(pts.crs)
    ###If converting to Lat Lon:
    pts['lat_lon'] = pts['geometry'].to_crs("epsg:4326")
    pts['Lon']=pts['lat_lon'].x
    pts['Lat']=pts['lat_lon'].y
    ###If using UTM:
    pts['XCoord']=pts['geometry'].x
    pts['YCoord']=pts['geometry'].y
    pts.drop(columns=['geometry', 'lat_lon'], inplace=True)
    pd.DataFrame.to_csv(pts, os.path.join(out_dir,'SamplePts.csv'), sep=',', na_rep='NaN', index=True)
    pts.head(n=5)


def plot_pts_in_polys(polys, npts, samp_id=1, zoom=200, seed=88):
    '''
    sample plot of polygons {'polys'} with sample points overlaid
    {'sampID'} is sample polygon to zoom in to
    {'zoom'} is buffer around zoom polygon for extent (smaller zooms in closer)
    '''

    polys3 = gpd.read_file(polys)
    ax = polys3.plot(color='gray')
    pti = get_ran_pts_in_polys (polys, npts, seed)
    plt.scatter(pti['geometry'].x, pti['geometry'].y, color='red')
    plt.axis([pti['geometry'][sampID].x-zoom, pti['geometry'][sampID].x+zoom,
              pti['geometry'][sampID].y-zoom, pti['geometry'][sampID].y+zoom])


def calculate_raw_index(nir_val, b2_val, spec_index):
    if spec_index == 'evi2':
        index_val =  10000* 2.5 * ((nir_val/10000 - b2_val/10000) / (nir_val/10000 + 1.0 + 2.4 * b2_val/10000))
    elif spec_index == 'gcvi':
        index_val = 10000* (nir_val - b2_val) / ((nir_val + b2_val) + 1e-9)
    elif spec_index == 'ndvi':
        index_val = 10000* (nir_val - b2_val) / ((nir_val + b2_val) + 1e-9)
    elif spec_index == 'savi':
        lfactor = .5 #(0-1, 0=very green, 1=very arid. .5 most common. Some use negative vals for arid env)
        index_val = 10000* (1 + lfactor) * ((nir_val - b2_val) / (nir_val + b2_val + lfactor))
    elif spec_index == 'msavi':
        index_val =  5000 * (2 * nir_val/10000 + 1) - ((2 * nir_val/10000 + 1)**2 - 8*(nir_val/10000 - b2_val/10000))**1/2
    elif spec_index == 'ndmi':
        index_val = 10000* (nir_val - b2_val) / ((nir_val + b2_val) + 1e-9)
    elif spec_index == 'ndwi':
        index_val = 10000* (b2_val - nir_val) / ((b2_val + nir_val) + 1e-9)
    elif spec_index == 'wi':  #note nir_val is actually swir1 here
        index_v = nir_val + b2_val
        print(index_v)
        if index_v > 5000:
            index_val = 1
        else:
            index_val = 10000 * (1.0 - (float(index_v) / 5000.0))
    elif spec_index == 'nir':
        index_val = nir_val
    elif spec_index in ['swir1','swir2','red','green']:
        index_val = b2_val
   
    return index_val
                     
def get_index_vals_at_pts(out_dir, ts_stack, image_type, polys, spec_index, num_pts, seed, load_samp=False, ptgdb=None):
    '''
    Gets values for all sampled points {'numpts'} in all polygons {'polys'} for all images in {'TStack'}
    OR gets values for points in a previously generated dataframe {ptgdb} using loadSamp=True.
    If imageType == 'TS', indices are assumed to already be calculated and
    {'TStack'} is a list of image paths, with basenames = YYYYDDD of image acquisition (DDD is Julien day, 1-365)
    If imageType == 'L1C' images are still in raw .nc form (6 bands) and indices are calculated here
    {'TStack'} is a list of image paths from which YYYYDDD info can be extracted
    output is a dataframe with a pt (named polygonID_pt#)
    on each row and an image index value(named YYYYDDD) in each column
    '''
    print('getting index values...')
    if load_samp == False:
        if polys:
            ptsgdb = get_ran_pts_in_polys (polys, num_pts, seed)
        else:
            print('There are no polygons or points to process in this cell')
            return None
    elif load_samp == True:
        ptsgdb = ptgdb

    xy = [ptsgdb['geometry'].x, ptsgdb['geometry'].y]
    coords = list(map(list, zip(*xy)))
    
    pt_dict={}
    for img in ts_stack:
        img_date = get_img_date(img, image_type)
        #print('img_date={}'.format(img_date))
        #imgDay = str('{:03d}'.format(img_date[1]))
        img_name = str(img_date[0])+(f"{img_date[1]:03d}")
        if 'Smooth'in image_type:
            with rio.open(img, 'r') as src:
                #ptsgdb[img_name] = [sample[0] for sample in src.sample(coords)]
                pt_dict[img_name] = [sample[0] for sample in src.sample(coords)]
        elif image_type in ['Sentinel2','Landsat','AllRaw']:
            xrimg = xr.open_dataset(img)
            xr_nir = xrimg['nir'].where(xrimg['nir'] < 10000)
            #xr_nir = xrimg['nir'].map({>9999: np.nan, < 10000: xrimg['nir']})
            if spec_index in ['evi2','msavi','ndvi','savi','wi','kndvi','red']:
                xr_red = xrimg['red'].where(xrimg['red'] < 10000)
                #xr_red = xrimg['red'].map({>9999: np.nan, < 10000: xrimg['red']})
            if spec_index in ['ndmi','wi','swir1']:
                xr_swir1 = xrimg['swir1'].where(xrimg['swir1'] < 10000)
                #xr_swir1 = xrimg['swir1'].map({>9999: np.nan, < 10000: xrimg['swir1']})
            elif spec_index in ['ndwi','gcvi','green']:
                xr_green = xrimg['green'].where(xrimg['green'] < 10000)
                #xr_green = xrimg['green'].map({>9999: np.nan, < 10000: xrimg['green']})
            elif spec_index in ['swir2']:
                xr_swir2 = xrimg['swir2'].where(xrimg['swir2'] < 10000)
                #xr_swir2 = xrimg['swir2'].map({>9999: np.nan, < 10000: xrimg['swir2']})
            elif spec_index in ['nir']:
                 pass
            #else: print('{} is not specified or does not have current method'.format(spec_index))

            pt_vals = []
            for index, row in ptsgdb.iterrows():
                if spec_index == 'wi':  #note thisptnir is actually swir1 here
                    thispt_nir = xr_swir1.sel(x=ptsgdb['geometry'].x[index],y=ptsgdb['geometry'].y[index],
                                method='nearest', tolerance=30)
                else:
                    thispt_nir = xr_nir.sel(x=ptsgdb['geometry'].x[index],y=ptsgdb['geometry'].y[index],
                                        method='nearest', tolerance=30)
                nir_val = thispt_nir.values
                
                    
                if spec_index in ['evi2','msavi','ndvi','savi','wi','kndvi','red']:
                    thispt_b2 = xr_red.sel(x=ptsgdb['geometry'].x[index],y=ptsgdb['geometry'].y[index],
                                method='nearest', tolerance=30)
                    b2_val = thispt_b2.values
                elif spec_index in ['ndmi','swir1']:
                    thispt_b2 = xr_swir1.sel(x=ptsgdb['geometry'].x[index],y=ptsgdb['geometry'].y[index],
                                method='nearest', tolerance=30)
                    b2_val = thispt_b2.values
                elif spec_index in ['ndwi','gcvi','green']:
                    thispt_b2 = xr_green.sel(x=ptsgdb['geometry'].x[index],y=ptsgdb['geometry'].y[index],
                                method='nearest', tolerance=30)
                    b2_val = thispt_b2.values
                elif spec_index in ['swir2']:
                    thispt_b2 = xr_swir2.sel(x=ptsgdb['geometry'].x[index],y=ptsgdb['geometry'].y[index],
                                method='nearest', tolerance=30)
                    b2_val = thispt_b2.values
                elif spec_index in ['nir']:
                    b2_val = nir_val

                #print('b2_val = {} for image {}.'.format(b2_val, img_name))

                index_val = calculate_raw_index(nir_val, b2_val, spec_index)
                pt_vals.append(index_val)
            pt_dict[img_name] = pt_vals

        else: print ('Currently valid image types are Smooth,Smooth_old,Sentinel,Landsat and AllRaw. You put {}'.format(image_type))

    ptdf = pd.DataFrame.from_dict(pt_dict, orient='columns')
    ptsgdb = pd.concat([ptsgdb,ptdf], axis=1)                           
    pd.DataFrame.to_csv(ptsgdb,os.path.join(out_dir,'ptsgdb.csv'), sep=',', index=True)
    print(ptsgdb)
    return ptsgdb

def get_timeseries_for_pts_multicell(out_dir, spec_index, start_yr, end_yr, img_dir, image_type, grid_file, cell_list,
                            polyfile=None, oldest=0, newest=0, npts=3, seed=88, load_samp=False, ptfile=None, filter_class=None):
    '''
    Returns datetime dataframe of values for sampled pts (n={'npts}) for each polygon in {'polys'}
    OR for previously generated points with {load_samp}=True and {pt_file}=path to .csv file
     (.csv file needs 'XCoord' and 'YCoord' fields (in this case, polyfile, oldest, newest, npts and seed are not used))
    for all images of {image_type} acquired between {'start_yr'} and {'end_yr'} in {'TS_Directory'}
    imageType can be 'Sentinel', 'Landsat', or 'All'
    Output format is a datetime object with date (YYYY-MM-DD) on each row and sample name (polygonID_pt#) in columns
    '''

    allpts = pd.DataFrame()

    for cell in cell_list:
        ts_stack = []
        print ('working on cell {}'.format(cell))
        if load_samp == True:
            polys = None
            if filter_class is not None:
                point_df = pd.read_csv(ptfile, index_col=0)
                selpts = point_df[point_df['Class']==filter_class]
                points = get_pts_in_grid (grid_file, cell, selpts)
            else:    
                points = get_pts_in_grid (grid_file, cell, ptfile)
        else:
            polys = get_polygons_in_grid (grid_file, cell, polyfile, oldest, newest)
            points = None
        if isinstance(points, gpd.GeoDataFrame) or polys is not None:
            if 'Smooth' in image_type:
                if image_type == 'Smooth_old':
                    cell_dir = os.path.join(img_dir,'{:06d}'.format(cell),'brdf_ts_old','ms',spec_index)
                if image_type == 'Smooth':
                    cell_dir = os.path.join(img_dir,'{:06d}'.format(cell),'brdf_ts','ms',spec_index)
                for img in os.listdir(cell_dir):
                    if img.endswith('.tif'):
                        img_yr, img_doy = get_img_date(img, image_type, data_source=None)
                        if ((img_yr > start_yr) or (img_yr == start_yr and img_doy >= 150)) and ((img_yr < end_yr) or (img_yr == end_yr and img_doy <= 150)):
                            ts_stack.append(os.path.join(img_dir,cell_dir,img))
            else:
                cell_dir = os.path.join(img_dir,'{:06d}'.format(cell),'brdf')
                if image_type == 'Sentinel2':
                    matchstr = ['S2']
                if image_type == 'Landsat5':
                    matchstr = ['LT']
                if image_type == 'Landsat7':
                    matchstr = ['LE']
                if image_type == 'Landsat8':
                    matchstr = ['LC']  #also Landsat9
                elif image_type == 'Landsat':
                    matchstr = ['LC','LT','LE']
                elif image_type == 'AllRaw':
                    matchstr = ['S2','LC','LT','LE']
                
                for img in os.listdir(cell_dir):
                    if img.endswith('.nc') and img.split('_')[1][:2] in matchstr and 'angles' not in img:
                        img_yr, img_doy = get_img_date(img, image_type, data_source=None)
                        if ((img_yr > start_yr) or (img_yr == start_yr and img_doy >= 150)) and ((img_yr < end_yr) or (img_yr == end_yr and img_doy <= 150)):
                            ts_stack.append(os.path.join(cell_dir,img))

            ts_stack.sort()
            if load_samp == True:
                polys=None
                pts = get_index_vals_at_pts(out_dir, ts_stack, image_type, polys, spec_index, npts, seed=88, load_samp=True, ptgdb=points)
            else:
                pts = get_index_vals_at_pts(out_dir, ts_stack, image_type, polys, spec_index, npts, seed=88, load_samp=False, ptgdb=None)

            pts.drop(columns=['geometry'], inplace=True)
            allpts = pd.concat([allpts, pts])

        else:
            print('skipping this cell')
            pass

    ts = allpts.transpose()
    ts['date'] = [pd.to_datetime(e[:4]) + pd.to_timedelta(int(e[4:]) - 1, unit='D') for e in ts.index]
    ##Note columns are all object due to mask. Need to change to numeric or any NA will result in  NA in average.
    #print(ts.dtypes)
    cols = ts.columns[ts.dtypes.eq('object')]
    for c in cols:
        ts[c] = ts[c].astype(float)
    #print(TS.dtypes)
    ##There are a lot of 9s...
    #ts = ts.replace(9, np.nan)
    ts.set_index('date', drop=True, inplace=True)
    ts=ts.sort_index()

    ts['ALL'] = ts.mean(axis=1)
    ts['stdv'] = ts.std(axis=1)

    pd.DataFrame.to_csv(ts, os.path.join(out_dir,'TS_{}_{}-{}.csv'.format(spec_index, start_yr, end_yr)), sep=',', na_rep='NaN', index=True)
    return ts


def load_ts_from_file(ts_file):
    ts = pd.read_csv(ts_file)
    ts.set_index('date', drop=True, inplace=True)
    ts.index = pd.to_datetime(ts.index)
    ts = ts.sort_index()

    return ts


def convert_timeseries_to_monthly_doy(full_ts_dir,mo,new_dir):
    '''
    Splits full time-series dataframe into monthly chunks for easier comparison with Planet(which are already
    chunked in processing). Note output index is now 'imDay' with format YYDDD
    '''
    for f in os.listdir(full_ts_dir):
        if f.endswith('.csv'):
            df = pd.read_csv(os.path.join(full_ts_dir,f))
            df.set_index('date', drop=True, inplace=True)
            df.index = pd.to_datetime(df.index)
            dfmo = df[df.index.month == int(mo)]
            dfmo.index = dfmo.index.strftime('%Y%j')
            dfmo.index.names = ['imDay']
            new_name = mo+f[4:]
            print(new_name)
            pd.DataFrame.to_csv(dfmo, os.path.join(full_ts_dir,new_dir,new_name), sep=',', na_rep='NaN', index=True)


def convert_full_timeseries_to_doy(tsdf, yr, start_day=0, end_day=365):
    new_df = tsdf[tsdf.index.year == yr]
    #change index to just month-day:
    new_df.index = new_df.index.dayofyear
    new_df = new_df[new_df.index >= start_day]
    new_df = new_df[new_df.index <= end_day]

    return new_df


def load_planet_ts(main_path, mid_path, dl_type, aoi, yr, mos, band):
    '''
    Loads raw time series chunks outputted from cluster Planet processing and concatenates them together into
    a single dataframe depending on arguments {AOI}('All' if want to join all), {DLType}('DLsOld' | 'NoDLs')
    {Yr}, {mos}(in format 'MM-MM') and band. Assumes files are together in {midPath} directory and are
    named MMYYYY<whatever>.csv, with [whatever] containing DLType, AOI and band strings.
    '''
    focus_data = []
    for ts in os.listdir(os.path.join(main_path,mid_path)):
        if aoi == 'All':
            if int(ts[0:2])>=int(mos[0:2]) and int(ts[0:2])<=int(mos[3:5]) and yr in ts and dl_type in ts and band in ts:
                ts_path = os.path.join(main_path,mid_path,ts)
                tsdf = pd.read_csv(ts_path)
                focus_data.append(tsdf)
        else:
            if int(ts[0:2])>=int(mos[0:2]) and int(ts[0:2])<=int(mos[3:5]) and yr in ts and dl_type in ts and band in ts and aoi in ts:
                ts_path = os.path.join(main_path,mid_path,ts)
                tsdf = pd.read_csv(ts_path)
                focus_data.append(tsdf)

    framed_ts = pd.concat(focus_data)
    framed_ts.set_index('imDay', drop=True, inplace=True)
    framed_ts.index = pd.to_datetime(framed_ts.index, format='%Y%j')
    framed_ts['ALL'] = framed_ts.mean(axis=1)
    framed_ts['stdv'] = framed_ts.std(axis=1)

    return framed_ts


