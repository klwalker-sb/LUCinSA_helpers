
import os
from pathlib import Path
import rasterio
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


def getCoordAtRowCol (Img, spec_index, row, col):
    df = exploreBand(Img, spec_index)
    testSamp = df[row,col]
    print('Test Samp at coords x={}, y={} is {}'.format(testSamp['x'].values,testSamp['y'].values, testSamp.values))
    return testSamp['x'].values, testSamp['y'].values

def getValatXY(Img, spec_index, xcoord,ycoord):
    df = exploreBand(Img, spec_index)
    one_point = df[spec_index].sel(x=xcoord,y=ycoord, method='nearest', tolerance=15)
    print('value at {},{}={}'.format(xcoord,ycoord,one_point.values))
    return one_point.values

def GetPolygonsInAOI(out_dir, AOI_file, polyPath, oldest=2018, newest=2020):
    '''
    Filters polygon layer to contain only those overlapping selected AOI
    Allows filtering by 'FirstYObs' to remove DLs constructed before or after period of interest
    Outputs new polygon set to a .json file
    '''
    AllPolys = gpd.read_file(polyPath)
    AOI = gpd.read_file(AOI_file)
    # Warn if both layers not in same coordinate system:
    if (AllPolys.crs != AOI.crs):
        print("AOI and ground truth files are NOT are in the same coordinate system!",
          AllPolys.crs, AOI.crs)

    # Clip polygons to AOI
    polysInAOI = gpd.clip(AllPolys, AOI)

    print("Of the {} polygons, {} are in AOI". format (AllPolys.shape[0], polysInAOI.shape[0]))

    ##Filter out polygons that were observed before year set as 'oldest' or after year set as 'newest'
    if oldest > 0:
        polysInAOI = polysInAOI[polysInAOI['FirstYrObs'] >= oldest]
    if newest > 0:
        polysInAOI = polysInAOI[polysInAOI['FirstYrObs'] <= newest]
        print("{} DLs first seen between {} and {} in AOI".format(len(polysInAOI),oldest,newest))

    polyClip = Path(os.path.join(out_dir,'polysInAOI.json'))
    polysInAOI.to_file(polyClip, driver="GeoJSON")

    return polyClip


def GetPtsInGrid (gridFile, gridCell, ptFile):
    '''
    loads point file (from .csv with 'XCoord' and 'YCoord' columns) and returns points that overlap a gridcell
    as a geopandas GeoDataFrame. Use this if trying to match/append data to existing sample points
    rather than making a new random sample each time (e.g. if matching Planet and Sentinel points)
    Note that crs of point file is known ahead of time and hardcoded here to match specific grid file.
    '''
    out_path = Path(gridFile).parent

    with tempfile.TemporaryDirectory(dir=out_path) as temp_dir:
        temp_file = Path(temp_dir) / Path(gridFile).name
        shutil.copy(gridFile, temp_file)
        df = gpd.read_file(temp_file)
        crs_grid = df.crs
    print('grid is in: ', crs_grid)  #ESRI:102033

    ptsdf = pd.read_csv(ptFile, index_col=0)
    pts = gpd.GeoDataFrame(ptsdf,geometry=gpd.points_from_xy(ptsdf.XCoord,ptsdf.YCoord),crs=crs_grid)

    bb = df.query(f'UNQ == {gridCell}').geometry.total_bounds

    gridBbox = box(bb[0],bb[1],bb[2],bb[3])
    gridBounds = gpd.GeoDataFrame(gpd.GeoSeries(gridBbox), columns=['geometry'], crs=crs_grid)
    print(gridBounds)

    ptsInGrid = gpd.sjoin(pts, gridBounds, op='within')
    ptsInGrid = ptsInGrid.loc[:,['geometry']]

    print("Of the {} ppts, {} are in gridCell {}. I am actually in here". format (pts.shape[0], ptsInGrid.shape[0],gridCell))

    #Write to geojson file
    if ptsInGrid.shape[0] > 0:
        ptClip = Path(os.path.join(out_path,'ptsGrid_'+str(gridCell)+'.json'))
        ptsInGrid.to_file(ptClip, driver="GeoJSON")

        return ptsInGrid
        print(ptsInGrid.head(n=5))


def GetPolygonsInGrid (gridFile, gridCell, polyPath, oldest=2018, newest=2020):
    '''
    Filters polygon layer to contain only those overlapping selected grid cell (allows for iteration through grid)
    Allows filtering by 'FirstYObs' to remove DLs constructed before or after period of interest
    Outputs new polygon set to a .json file stored in the gridcell directory
    '''
    polys = gpd.read_file(polyPath)
    out_path = Path(gridFile).parent

    with tempfile.TemporaryDirectory(dir=out_path) as temp_dir:
        temp_file = Path(temp_dir) / Path(gridFile).name
        shutil.copy(gridFile, temp_file)
        df = gpd.read_file(temp_file)
        crs_grid = df.crs

    bb = df.query(f'UNQ == {gridCell}').geometry.total_bounds

    gridBbox = box(bb[0],bb[1],bb[2],bb[3])
    gridBounds = gpd.GeoDataFrame(gpd.GeoSeries(gridBbox), columns=['geometry'], crs=crs_grid)
    polysInGrid = gpd.overlay(gridBounds, polys, how='intersection')

    print("Of the {} polygons, {} are in gridCell {}". format (polys.shape[0], polysInGrid.shape[0],gridCell))

    ##Filter out polygons that were observed before year set as 'oldest' or after year set as 'newest'
    if oldest > 0:
        polysInGrid = polysInGrid[polysInGrid['FirstYrObs'] >= oldest]
    if newest > 0:
        polysInGrid = polysInGrid[polysInGrid['FirstYrObs'] <= newest]
        print("{} DLs first seen between {} and {} in AOI".format(len(polysInGrid),oldest,newest))

    #Write to geojson file
    if polysInGrid.shape[0] > 0:
        polyClip = Path(os.path.join(out_path,'polysGrid_'+str(gridCell)+'.json'))
        polysInGrid.to_file(polyClip, driver="GeoJSON")

        return polyClip


def plotPolyOnIndex(zoomPoly, Img, polyFile):
    '''
    Plots a polygon file {'polyFile'} on top of a .tiff image {'Img'}
    and zooms to a selected poly {'zoomPoly'}
    '''
    #fig, ax = plt.subplots(figsize=(12,12),subplot_kw={'projection': ccrs.epsg(32632)})
    fig, ax = plt.subplots(figsize=(12,12))

    if Img.endswith('.tif'):
        with rasterio.open(Img) as src:
            img = src.read()
        plot.show(img, ax=ax)
    elif Img.endswith('.nc'):
        with xr.open_dataset(Img) as xrimg:
            xr_band = xrimg.nir
            img = xr_band.where(xr_band < 10000)
        img.plot(x="x",y="y")
        plot.show(img, ax=ax)

    polys = gpd.read_file(polyFile)
    polys.plot(ax=ax, facecolor='none', edgecolor='orangered')

    #Get bounds to zoom into polygon specified in arguments
    polybds = polys.query(f'FID == {zoomPoly}').bounds
    minx=int((polybds['minx'][1])-100)
    maxx=int((polybds['maxx'][1])+100)
    miny=int((polybds['miny'][1])-100)
    maxy=int((polybds['maxy'][1])+100)

    plt.axis([minx,maxx,miny,maxy])
    polys.head(n=5)


def getRanPtInPoly(polyg, seed):
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


def getRanPtsInPolys(polys, npts, seed=88):
    '''
    Returns a geodataframe with geometry column containing shapely Point objects
    With {'npts'} random samples from each polygon in {'polys'}
    ({'polys'} is path to json file with polygon geometries)
    '''
    with open(polys, 'r') as polysrc:
        polys2 = json.load(polysrc)
        #print(polys['type']) #FeatureCollection
        polyList = []
        ptDict = {}
        for poly in polys2['features']:
            polyList.append(poly['geometry'])
            polyobj = shape(poly['geometry'])
            for i in range(0,npts):
                ptName = str(poly['properties']['id'])+'_'+str(i+1)
                pt_in_poly = getRanPtInPoly(polyobj, seed)
                ptDict[ptName] = pt_in_poly

    ptsdb = pd.DataFrame.from_dict(ptDict, orient='index')
    ptsgdb = gpd.GeoDataFrame(ptsdb, geometry=ptsdb[0])
    ptsgdb.drop(columns=[0], inplace=True)

    return ptsgdb


def getCoordsForPolySampInAOI(out_dir, polygons, AOI, oldest=2018, newest=2018, nPts=2, seed=88):
    '''
    Gets random points in all polygons in AOI. returns a csv file with X and Y coords
    '''
    polys = GetPolygonsInAOI(out_dir, AOI, polygons, oldest, newest)
    pts = getRanPtsInPolys(polys, nPts, seed)
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


def PlotPtsInPolys(polys, npts, sampID=1, zoom=200, seed=88):
    '''
    sample plot of polygons {'polys'} with sample points overlaid
    {'sampID'} is sample polygon to zoom in to
    {'zoom'} is buffer around zoom polygon for extent (smaller zooms in closer)
    '''

    polys3 = gpd.read_file(polys)
    ax = polys3.plot(color='gray')
    pti = getRanPtsInPolys (polys, npts, seed)
    plt.scatter(pti['geometry'].x, pti['geometry'].y, color='red')
    plt.axis([pti['geometry'][sampID].x-zoom, pti['geometry'][sampID].x+zoom,
              pti['geometry'][sampID].y-zoom, pti['geometry'][sampID].y+zoom])


def CalculateRawIndex(nir_val, b2_val, spec_index):
    if spec_index == 'evi2':
        index_val =  2.5 * ((nir_val - b2_val) / (nir_val + 1.0 + 2.4 * b2_val))
    elif spec_index == 'ndvi':
        index_val = (nir_val - b2_val) / ((nir_val + b2_val) + 1e-9)
    elif spec_index == 'savi':
        lfactor = .5 #(0-1, 0=very green, 1=very arid. .5 most common. Some use negative vals for arid env)
        index_val = (1 + lfactor) * ((nir_val - b2_val) / (nir_val + b2_val + lfactor))
    elif spec_index == 'msavi':
        index_val =  1/2 * (2 * nir_val + 1) - ((2 * nir_val + 1)**2 - 8*(nir_val - b2_val))**1/2
    elif spec_index == 'ndmi':
        index_val = (nir_val - b2_val) / ((nir_val + b2_val) + 1e-9)
    elif spec_index == 'ndwi':
        index_val = (b2_val - nir_val) / ((b2_val + nir_val) + 1e-9)
    elif spec_index == 'nir':
        index_val = nir_val
    elif spec_index in ['swir1','swir2','red','green']:
        index_val = b2_val

    return index_val

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
    elif imageType not in ['Sentinel','Landsat','AllRaw']:
        print ('Currently valid image types are Smooth,Sentinel,Landsat and AllRaw. You put {}'.format(imageType))
    else:
        YYYYMMDD = img.split('_')[3][:8]
        YYYY = int(YYYYMMDD[:4])
        MM = int(YYYYMMDD[4:6])
        DD = int(YYYYMMDD[6:8])
   
    ymd = datetime.datetime(YYYY, MM, DD)
    doy = int(ymd.strftime('%j'))
    
    return YYYY, doy

def GetIndexValsAtPts(out_dir, TSstack, imageType, polys, spec_index, numPts, seed, loadSamp=False, ptgdb=None):
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
    print('now I am here in GetIndexValsAtPts')
    if loadSamp == False:
        if polys:
            ptsgdb = getRanPtsInPolys (polys, numPts, seed)
        else:
            print('There are no polygons or points to process in this cell')
            return None
    elif loadSamp == True:
        ptsgdb = ptgdb

    xy = [ptsgdb['geometry'].x, ptsgdb['geometry'].y]
    coords = list(map(list, zip(*xy)))

    for img in TSstack:
        imgDate = GetImgDate(img, imageType)
        #print('imgDate={}'.format(imgDate))
        #imgDay = str('{:03d}'.format(imgDate[1]))
        img_name = str(imgDate[0])+(f"{imgDate[1]:03d}")
        if imageType == 'Smooth':
            with rasterio.open(img, 'r') as src:
                ptsgdb[img_name] = [sample[0] for sample in src.sample(coords)]
        elif imageType in ['Sentinel','Landsat','AllRaw']:
            xrimg = xr.open_dataset(img)
            xr_nir = xrimg['nir'].where(xrimg['nir'] < 10000)
            #xr_nir = xrimg['nir'].map({>9999: np.nan, < 10000: xrimg['nir']})
            if spec_index in ['evi2','msavi','ndvi','savi','red']:
                xr_red = xrimg['red'].where(xrimg['red'] < 10000)
                #xr_red = xrimg['red'].map({>9999: np.nan, < 10000: xrimg['red']})
            elif spec_index in ['ndmi','swir1']:
                xr_swir1 = xrimg['swir1'].where(xrimg['swir1'] < 10000)
                #xr_swir1 = xrimg['swir1'].map({>9999: np.nan, < 10000: xrimg['swir1']})
            elif spec_index in ['ndwi','green']:
                xr_green = xrimg['green'].where(xrimg['green'] < 10000)
                #xr_green = xrimg['green'].map({>9999: np.nan, < 10000: xrimg['green']})
            elif spec_index in ['swir2']:
                xr_swir2 = xrimg['swir2'].where(xrimg['swir2'] < 10000)
                #xr_swir2 = xrimg['swir2'].map({>9999: np.nan, < 10000: xrimg['swir2']})
            elif spec_index in ['nir']:
                 pass
            else: print('{} is not specified or does not have current method'.format(spec_index))

            ptVals = []
            for index, row in ptsgdb.iterrows():
                thispt_nir = xr_nir.sel(x=ptsgdb['geometry'].x[index],y=ptsgdb['geometry'].y[index],
                                        method='nearest', tolerance=30)
                nir_val = thispt_nir.values

                if spec_index in ['evi2','msavi','ndvi','savi','red']:
                    thispt_b2 = xr_red.sel(x=ptsgdb['geometry'].x[index],y=ptsgdb['geometry'].y[index],
                                method='nearest', tolerance=30)
                    b2_val = thispt_b2.values
                elif spec_index in ['ndmi','swir1']:
                    thispt_b2 = xr_swir1.sel(x=ptsgdb['geometry'].x[index],y=ptsgdb['geometry'].y[index],
                                method='nearest', tolerance=30)
                    b2_val = thispt_b2.values
                elif spec_index in ['ndwi','green']:
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

                indexVal = CalculateRawIndex(nir_val, b2_val, spec_index)
                ptVals.append(indexVal)
            ptsgdb[img_name] = ptVals

        else: print ('Currently valid image types are Smooth,Sentinel,Landsat and AllRaw. You put {}'.format(imageType))

    pd.DataFrame.to_csv(ptsgdb,os.path.join(out_dir,'ptsgdb.csv'), sep=',', index=True)
    return ptsgdb

def GetTimeSeriesForPts_MultiCell(out_dir, spec_index, StartYr, EndYr, img_dir, imageType, gridFile, cellList,
                            groundPolys, oldest, newest, npts, seed, loadSamp, ptFile):
    '''
    Returns datetime dataframe of values for sampled pts (n={'npts}) for each polygon in {'polys'}
    OR for previously generated points with {loadSamp}=True and {ptFile}=path to .csv file
     (.csv file needs 'XCoord' and 'YCoord' fields (in this case, groundpolys, oldest, newest, npts and seed are not used))
    for all images of {imageType} acquired between {'StartYr'} and {'EndYr'} in {'TS_Directory'}
    imageType can be 'Sentinel', 'Landsat', or 'All'
    Output format is a datetime object with date (YYYY-MM-DD) on each row and sample name (polygonID_pt#) in columns
    '''

    Allpts = pd.DataFrame()

    for cell in cellList:
        TStack = []
        print ('working on cell {}'.format(cell))
        if loadSamp == True:
            points = GetPtsInGrid (gridFile, cell, ptFile)
            polys = None
        else:
            polys = GetPolygonsInGrid (gridFile, cell, groundPolys, oldest, newest)
            points = None
        if isinstance(points, gpd.GeoDataFrame) or polys is not None:
            if imageType == 'Smooth':
                cellDir = os.path.join(img_dir,'{:06d}'.format(cell),'brdf_ts','ms',spec_index)
                for img in os.listdir(cellDir):
                    if img.endswith('.tif'):
                        imgYr = int(img[:4])
                        imgDoy = int(img[4:7])
                        if ((imgYr > StartYr) or (imgYr == StartYr and imgDoy >= 150)) and ((imgYr < EndYr) or (imgYr == EndYr and imgDoy <= 150)):    
                            TStack.append(os.path.join(img_dir,cellDir,img))
            else:
                cellDir = os.path.join(img_dir,'{:06d}'.format(cell),'brdf')

                if imageType == 'Sentinel':
                    for img in os.listdir(cellDir):
                        if img.startswith('L1C') and 'angles' not in img:
                            imgYr = int(img[19:23])
                            if imgYr >= StartYr and imgYr <= EndYr:
                                TStack.append(os.path.join(img_dir,cellDir,img))
                if imageType == 'Landsat':
                    for img in os.listdir(cellDir):
                        if img.startswith('LC') or img.startswith('LT') or img.startswith('LE'):
                            if 'angles' not in img:
                                imgYr = int(img[17:21])
                                if imgYr >= StartYr and imgYr <= EndYr:
                                    TStack.append(os.path.join(img_dir,cellDir,img))
                if imageType == 'AllRaw':
                    for img in os.listdir(cellDir):
                        if img.startswith('L1C') and 'angles' not in img:
                            imgYr = int(img[19:23])
                            if imgYr >= StartYr and imgYr <= EndYr:
                                    TStack.append(os.path.join(img_dir,cellDir,img))
                        elif img.startswith('LC') or img.startswith('LT') or img.startswith('LE'):
                            if 'angles' not in img:
                                imgYr = int(img[17:21])
                                if imgYr >= StartYr and imgYr <= EndYr:
                                    TStack.append(os.path.join(img_dir,cellDir,img))

            TStack.sort()
            if loadSamp == True:
                polys=None
                pts = GetIndexValsAtPts(out_dir, TStack, imageType, polys, spec_index, npts, seed=88, loadSamp=True, ptgdb=points)
            else:
                pts = GetIndexValsAtPts(out_dir, TStack, imageType, polys, spec_index, npts, seed=88, loadSamp=False, ptgdb=None)

            pts.drop(columns=['geometry'], inplace=True)
            Allpts = pd.concat([Allpts, pts])

        else:
            print('skipping this cell')
            pass

    TS = Allpts.transpose()
    TS['date'] = [pd.to_datetime(e[:4]) + pd.to_timedelta(int(e[4:]) - 1, unit='D') for e in TS.index]
    ##Note columns are all object due to mask. Need to change to numeric or any NA will result in  NA in average.
    #print(TS.dtypes)
    cols = TS.columns[TS.dtypes.eq('object')]
    for c in cols:
        TS[c] = TS[c].astype(float)
    #print(TS.dtypes)
    ##There are a lot of 9s...
    #TS = TS.replace(9, np.nan)
    TS.set_index('date', drop=True, inplace=True)
    TS=TS.sort_index()

    TS['ALL'] = TS.mean(axis=1)
    TS['stdv'] = TS.std(axis=1)

    pd.DataFrame.to_csv(TS, os.path.join(out_dir,'TS_{}_{}-{}.csv'.format(spec_index, StartYr, EndYr)), sep=',', na_rep='NaN', index=True)
    return TS


def LoadTSfromFile(TSfile):
    TS = pd.read_csv(TSfile)
    TS.set_index('date', drop=True, inplace=True)
    TS.index = pd.to_datetime(TS.index)
    TS = TS.sort_index()

    return TS


def ConvertTimeSeriesToMonthlyDoy(FullTS_dir,mo,newdir):
    '''
    Splits full time-series dataframe into monthly chunks for easier comparison with Planet(which are already
    chunked in processing). Note output index is now 'imDay' with format YYDDD
    '''
    for f in os.listdir(FullTS_dir):
        if f.endswith('.csv'):
            df = pd.read_csv(os.path.join(FullTS_dir,f))
            df.set_index('date', drop=True, inplace=True)
            df.index = pd.to_datetime(df.index)
            dfmo = df[df.index.month == int(mo)]
            dfmo.index = dfmo.index.strftime('%Y%j')
            dfmo.index.names = ['imDay']
            new_name = mo+f[4:]
            print(new_name)
            pd.DataFrame.to_csv(dfmo, os.path.join(FullTS_dir,newdir,new_name), sep=',', na_rep='NaN', index=True)


def Convert_FullTimeSeries_toDOY(TSdf, Yr, StartDay=0, EndDay=365):
    newDF = TSdf[TSdf.index.year == Yr]
    #change index to just month-day:
    newDF.index = newDF.index.dayofyear
    newDF = newDF[newDF.index >= StartDay]
    newDF = newDF[newDF.index <= EndDay]

    return newDF


def LoadPlanetTS(mainPath, midPath, DLType, AOI, Yr, mos, band):
    '''
    Loads raw time series chunks outputted from cluster Planet processing and concatenates them together into
    a single dataframe depending on arguments {AOI}('All' if want to join all), {DLType}('DLsOld' | 'NoDLs')
    {Yr}, {mos}(in format 'MM-MM') and band. Assumes files are together in {midPath} directory and are
    named MMYYYY<whatever>.csv, with [whatever] containing DLType, AOI and band strings.
    '''
    focusData = []
    for ts in os.listdir(os.path.join(mainPath,midPath)):
        if AOI == 'All':
            if int(ts[0:2])>=int(mos[0:2]) and int(ts[0:2])<=int(mos[3:5]) and Yr in ts and DLType in ts and band in ts:
                tsPath = os.path.join(mainPath,midPath,ts)
                tsdf = pd.read_csv(tsPath)
                focusData.append(tsdf)
        else:
            if int(ts[0:2])>=int(mos[0:2]) and int(ts[0:2])<=int(mos[3:5]) and Yr in ts and DLType in ts and band in ts and AOI in ts:
                tsPath = os.path.join(mainPath,midPath,ts)
                tsdf = pd.read_csv(tsPath)
                focusData.append(tsdf)

    FramedTS = pd.concat(focusData)
    FramedTS.set_index('imDay', drop=True, inplace=True)
    FramedTS.index = pd.to_datetime(FramedTS.index, format='%Y%j')
    FramedTS['ALL'] = FramedTS.mean(axis=1)
    FramedTS['stdv'] = FramedTS.std(axis=1)

    return FramedTS


