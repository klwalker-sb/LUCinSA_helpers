#!/usr/bin/env python
# coding: utf-8

import os
import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import pyproj
import pickle
from shapely.geometry import box

def PrintFilesInDirectory(in_dir,endstring,printList=False,out_dir=None,dataSource='stac'):
    '''
    Can generate a dataframe with a list of all files in folder for quick exploration. 
    Option to print to file for deeper look.
    If the directory of interest is a brdf directory, can extract dates and sensor types from file names
    using '.nc' as {endstring} and {brdfDir=True} and generate quick graphs of images by year and month below.
    '''
    fileList = []
    for f in os.listdir(in_dir):
        if f.endswith(endstring) and 'angles' not in f:
            fileList.append(f)
    filedf = pd.DataFrame(fileList, columns = ['file'])
    
    if 'brdf' in str(in_dir):
        if dataSource == 'GEE':
            filedf['sensor'] = filedf['file'].str[:4]
            filedf['date'] = np.where((filedf['sensor']=='L1C_'),filedf['file'].str[19:27],filedf['file'].str[17:25])
        elif dataSource == 'stac':
            #filedf['sensor'] = str(filedf['file']).split('_')[1][:4]
            filedf['sensor'] = filedf.apply(lambda x: x['file'].split('_')[1][:4], axis=1)
            filedf['date'] = filedf.apply(lambda x: x['file'].split('_')[3][:8], axis=1)
        filedf['yr'] = filedf['date'].str[:4]
        filedf['yrmo'] = filedf['date'].str[:6]
        sorted_files = filedf.sort_values(by='date')
    
    if printList == True:
        pd.DataFrame.to_csv(filedf, os.path.join(out_dir,'FileList.txt'), sep=',', na_rep='.', index=False)   
        
    return filedf


def GetImgListFromDb(sensor, in_dir, gridCell,Yrs, dataSource='stac'):
    '''
    returns list of images in database for year range (Yrs) for selected directory (raw or brdf)
    and sensor ('Landsat', 'Sentinel').
    Yrs is a list in format [YYYY, YYYY]
    '''
    if dataSource == 'stac':
        if 'brdf' not in str(in_dir):
            if sensor == 'Landsat':
                scene_info = (Path('{}/00{}/landsat/scene.info'.format(in_dir,gridCell)))
            elif sensor == 'Sentinel':
                scene_info = (Path('{}/00{}/sentinel/scene.info'.format(in_dir,gridCell)))
    print('Getting scene info: ',scene_info)

    pyver = float((sys.version)[:3])
    if pyver < 3.8:
        print('python version is {}'.format(pyver))
        ### note this may not work depending on what version of Pandas you have. Maybe upgrade Pandas
        import pickle5 as pickle
        with open(scene_info, 'rb') as file1:         
            df = pickle.load(file1)
    else:
        df = pd.read_pickle(scene_info).set_index('date')
        df.index.rename(None, inplace=True)
        df = df.assign(date=df.index)
        df = df.sort_index()
    
    if Yrs:
        df = df[f'{Yrs[0]}-01-01':f'{Yrs[1]}-12-30']

    return df



def GetImgListFromCat(sensor, gridCell, gridFile, Yrs=None):
    '''
    Gets list of images available on Planetary Hub for range of years {Yrs} [YYYY, YYYY]
    for {gridCell} CCCC and sensor {'Landsat' or 'Sentinel'}
    returns dataFrame with column of image ids
    '''

    import pystac_client
    import planetary_computer as pc
    from pystac_client import Client
    from pystac.extensions.eo import EOExtension as eo

    grid = gpd.read_file(gridFile)
    if grid.crs != pyproj.CRS.from_epsg(4326):
        grid = grid.to_crs('epsg:4326')
    bb = grid.query(f'UNQ == {[gridCell]}').geometry.total_bounds

    if Yrs == None:
        datetime="2010-01-01/2022-12-30"
    else:
        TimeSlice=f"{Yrs[0]}-01-01/{Yrs[1]}-12-30"
        
    if sensor == 'Landsat':
        collect=["landsat-c2-l2"]
    elif sensor == 'Sentinel':
        collect=["sentinel-2-l2a"]
        
    api = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1/")
    search = api.search(bbox=bb,
        datetime=TimeSlice,
        collections=collect,
        query={"eo:cloud_cover": {"lt": 90}},
        max_items = 10000)

    items = list(search.get_items())
    CatIDs = pd.DataFrame({'id':[o.id for o in items]})
    print(f"Returned {len(items)} Items")

    return CatIDs


def CheckForMissingFiles(df, in_dir, gridCell, sensor, dataSource='stac'):
    '''
    Checks whether all of the files in the database {df} are in the directory {in_dir}
    returns list of image ids in db but not in directory
    '''
    if dataSource == 'stac':
        if 'brdf' not in str(in_dir):
            if sensor == 'Landsat':
                df['file_path'] = df.apply(lambda x: Path('{}/00{}/landsat/{}.tif'.format(in_dir,gridCell,x['id'])), axis=1)
            elif sensor == 'Sentinel':
                df['file_path'] = df.apply(lambda x: Path('{}/00{}/sentinel/{}.tif'.format(in_dir,gridCell,x['id'])), axis=1)  
        else:
            df['file_path'] = df.apply(lambda x: Path('{}/{}.tif'.format(in_dir,gridCell,x['id'])), axis=1)
            #TODO: modify to reflect change in basename (this does not work as is for brdf directory) 
    df['file_path_exists'] = df.apply(lambda x: x['file_path'].is_file(), axis=1)
    dfe = df.loc[df.file_path_exists]
    dfno = df.loc[df.file_path_exists==False]
    missing = dfno.id.values.tolist()
    print('Of the {} files in the db, {} are missing from the directory'.format(df.shape[0],len(missing)))
    
    return missing


def CompareFilesToDb(sensor, dbSource, in_dir, gridCell, gridFile, Yrs, dataSource='stac'):
    '''
    Compares files in directory {in_dir} to those in database {dbSource}.
    dbSource = 'Local'=scene.info , 'Remote'=PlanetaryComputer, or 'Both'
    returns list of image ids in db but not in directory
    if dbSource == 'Both', also compares remote and local databases to check whether there are remote files
    that were never seen by local database (never attempted to download) for cell/sensor/years 
    '''
    if dbSource in['Local','Both']:
        print (f'Checking files against local database for {Yrs[0]}-{Yrs[1]}...')
        dfLocal = GetImgListFromDb(sensor, in_dir, gridCell, Yrs, dataSource='stac')
        missingLocal = CheckForMissingFiles(dfLocal, in_dir, gridCell, sensor, dataSource='stac')
    if dbSource in['Remote','Both']:
        print (f'Checking files agaist remote database for {Yrs[0]}-{Yrs[1]}...')
        dfRemote = GetImgListFromCat(sensor, gridCell, gridFile, Yrs)
        missingRemote = CheckForMissingFiles(dfRemote, in_dir, gridCell, sensor, dataSource='stac')
    if dbSource == 'Both':
        print (f'Checking local db remote database for {Yrs[0]}-{Yrs[1]}...')
        LocalList = dfLocal['id'].values.tolist()
        RemoteList = dfRemote['id'].values.tolist()
        MissingFromLocalDb = list(set(RemoteList) - set(LocalList))
        print('the following Files are in the remote catelog but not in the local database:',MissingFromLocalDb)
    if dbSource == 'Local':
        return missingLocal
    elif dbSource == 'Remote':
        return missingRemote
    else:
        return missingLocal, missingRemote, MissingFromLocalDb

