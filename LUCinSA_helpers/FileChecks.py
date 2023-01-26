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
            
    if len(fileList)==0:
        print('there are no files ending with {} in directory {}'.format(endstring, in_dir))
        return None
    else:
        filedf = pd.DataFrame(fileList, columns = ['file'])
    
        if 'brdf' in str(in_dir):
            if dataSource == 'GEE':
                filedf['sensor'] = filedf['file'].str[:4]
                filedf['date'] = np.where((filedf['sensor']=='L1C_'),filedf['file'].str[19:27],filedf['file'].str[17:25])
            elif dataSource == 'stac':
                filedf['sensor'] = filedf.apply(lambda x: x['file'].split('_')[1][:4], axis=1)
                filedf['date'] = filedf.apply(lambda x: x['file'].split('_')[3][:8], axis=1)
                filedf['base']=filedf.apply(lambda x: "_".join(x['file'].split("_")[:4]), axis=1)
            filedf['yr'] = filedf['date'].str[:4]
            filedf['yrmo'] = filedf['date'].str[:6]
            sorted_files = filedf.sort_values(by='date')
    
        if printList == True:
            pd.DataFrame.to_csv(filedf, os.path.join(out_dir,'FileList.txt'), sep=',', na_rep='.', index=False)   
        
        return filedf


def PrintFilesInMultipleDirectories(full_dir,sub_dir,endstring,printList=False,out_dir=None):
    '''
    This will return a dataframe with all files ending in {endstring} in each directory named {sub_dir} within the {full_dir}
    If the desired sub_dir is a brdf directory ({brdf=True}), adds correct date and year info to dataframe for plotting
    Drops duplicated filenames to reveal the number of images coming from unique Sentinel/Landsat scenes.
    Will print final dataframe to file in {out_dir} with {printList=True}
    '''
    fileList = []
    multiFileList = []
    for x in full_dir.iterdir():
        if x.is_dir():
            for sx in x.iterdir():
                if os.path.basename(sx) == sub_dir:
                    fileSet = PrintFilesInDirectory(sx,endstring,printList=False,out_dir=None,dataSource='stac')
                    multiFileList.append(fileSet)
    fullFiledf = pd.concat(multiFileList)
    numCells = len(multiFileList)
    lenOrig = len(fullFiledf)
    uniqueImgs = fullFiledf.drop_duplicates(subset=['base'],keep='first')
    print('There are {} processed images from {} unique Sentinel/Landsat images over {} cells.'.format(lenOrig,len(uniqueImgs),numCells))
    
    if printList == True:
        pd.DataFrame.to_csv(uniqueImgs, os.path.join(out_dir,'ALLFileList.txt'), sep=',', na_rep='.', index=False)   
        
    return uniqueImgs

def GetImgListFromDb(sensor, in_dir, gridCell,Yrs, dataSource='stac'):
    '''
    returns list of images in database for year range (Yrs) for selected directory (raw or brdf)
    and sensor ('Landsat', 'Sentinel').
    Yrs is a list in format [YYYY, YYYY]
    '''
    if dataSource == 'stac':
        if 'brdf' not in str(in_dir):
            if sensor == 'Landsat':
                scene_info = (Path('{}/{:06d}/landsat/scene.info'.format(in_dir,gridCell)))
                if os.path.exists(scene_info) == False:
                    print('There is no scene.info file in the Landsat directory for cell {}'.format(gridCell))
                    return None
            elif sensor == 'Sentinel':
                scene_info = (Path('{}/{:06d}/sentinel2/scene.info'.format(in_dir,gridCell)))
                if os.path.exists(scene_info) == False:
                    print('There is no scene.info file in the Sentinel directory for cell {}'.format(gridCell))
                    return None
    
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



def GetImgListFromCat(sensor, gridCell, gridFile, Yrs=None, cat='default'):
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
        TimeSlice="2010-01-01/2022-12-30"
    else:
        TimeSlice=f"{Yrs[0]}-01-01/{Yrs[1]}-12-30"
        
    if sensor == 'Landsat':
        collect=["landsat-c2-l2"]
        api = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1/")

    elif sensor == 'Sentinel':
        if cat == 'default':
            collect=['sentinel-s2-l2a-cogs']  #if using element84
            api =  pystac_client.Client.open("https://earth-search.aws.element84.com/v0")
        elif cat == 'planetary':
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

def ComparePlanetaryHub_w_Element84 (sensor, gridCell, gridFile, Yrs=None):
    '''
    Compares list of images for time slice that are available on Planetary Hub vs Element84
    Currently focused on Sentinel2.
    '''
    PH_df = GetImgListFromCat(sensor, gridCell, gridFile, Yrs, cat='planetary')
    if sensor == 'Sentinel':
        E84_df = GetImgListFromCat(sensor, gridCell, gridFile, Yrs, cat='default')
        E84_df['id2'] = E84_df.apply(lambda x: x['id'].split("_")[0] +'_'+ x['id'].split("_")[1] +'_'+ x['id'].split("_")[2], axis=1)
        E84_ids = E84_df['id2'].values.tolist()
        PH_df['id2'] = PH_df.apply(lambda x: x['id'].split("_")[0] +'_'+ x['id'].split("_")[4][1:6] +'_'+ x['id'].split("_")[2][:8], axis=1)
        PH_ids = PH_df['id2'].values.tolist()
       
    E84_notin_Planetary = list(set(E84_ids) - set(PH_ids))
    Planetary_notin_E84 = list(set(PH_ids) - set(E84_ids))
          
    return E84_notin_Planetary,Planetary_notin_E84

    
def CheckForMissingFiles(df, in_dir, gridCell, sensor, dataSource='stac'):
    '''
    Checks whether all of the files in the database {df} are in the directory {in_dir}
    returns list of image ids in db but not in directory
    '''
    if dataSource == 'stac':
        if 'brdf' not in str(in_dir):
            if sensor == 'Landsat':
                df['file_path'] = df.apply(lambda x: Path('{}/{:06d}/landsat/{}.tif'.format(in_dir,gridCell,x['id'])), axis=1)
            elif sensor == 'Sentinel':
                df['file_path'] = df.apply(lambda x: Path('{}/{:06d}/sentinel2/{}.tif'.format(in_dir,gridCell,x['id'])), axis=1)
        else:
            df['file_path'] = df.apply(lambda x: Path('{}/{}.tif'.format(in_dir,gridCell,x['id'])), axis=1)
            #TODO: modify to reflect change in basename (this does not work as is for brdf directory) 
    df['file_path_exists'] = df.apply(lambda x: x['file_path'].is_file(), axis=1)
    dfe = df.loc[df.file_path_exists]
    dfno = df.loc[df.file_path_exists==False]
    missing = dfno.id.values.tolist()
    print('Of the {} files in the {} db, {} are missing from the directory'.format(df.shape[0],sensor,len(missing)))
    return missing

def WhyMissingFiles(sensor,missing_list,numToAllow):
    print('checking reason for missing files...')
    dl = 0
    if sensor == 'Sentinel':
        print(missing_list)
        StillMissing = missing_list
        if len(missing_list) <= numToAllow:
            dl = 1
    if sensor == 'Landsat':
        missing_df = pd.DataFrame(missing_list)
        missing_df['sensor'] = missing_df.apply(lambda x: x[0].split('_')[0][:4], axis=1)
        missing_df['yr'] = missing_df.apply(lambda x: int(x[0].split('_')[3][:4]), axis=1)
        missing_df['L7except'] = np.where((missing_df['sensor']=='LE07') & (missing_df['yr'] >= 2017),1,0)
        numL7 = sum(missing_df['L7except'])
        if numL7 > 0:
            print (f'{numL7} of the missing images are L7 images in or after 2017; Assuming this is intentional.' +
               '(if not, change the <L7end_yr> parameter in the downloading script)')
        StillMissing = missing_df[missing_df['L7except']==0]
        if len(StillMissing) == 0:
            print('No other missing files!')
            dl = 1
        else:
            print('There are {} files missing for other reasons:'.format(len(StillMissing)))
            SM = StillMissing[0].to_list()
            print(SM)
            
        if len(StillMissing) <= numToAllow:
            dl = 1

    return dl, SM

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
        print (f'Checking local db against remote database for {Yrs[0]}-{Yrs[1]}...')
        LocalList = dfLocal['id'].values.tolist()
        RemoteList = dfRemote['id'].values.tolist()
        MissingFromLocalDb = list(set(RemoteList) - set(LocalList))
        print('the following Files are in the remote catalog but not in the local database:',MissingFromLocalDb)
    if dbSource == 'Local':
        return missingLocal
    elif dbSource == 'Remote':
        return missingRemote
    else:
        return missingLocal, missingRemote, MissingFromLocalDb

def GetCellStatus(in_dir, gridCell,gridFile,Yrs,dataSource='stac'):

    LSdl = 0
    s2dl = 0

    ##Check if files have been downloaded for cell:
    LS = GetImgListFromDb('Landsat', in_dir, gridCell,Yrs, dataSource='stac')
    LS_dir = Path('{}/{:06d}/landsat'.format(in_dir,gridCell))
    if os.path.exists(LS_dir) == False:
        print('There is no Landsat directory for cell {}'.format(gridCell))
    else:
        LS_imgs = PrintFilesInDirectory(LS_dir,'.tif',printList=False,out_dir=None,dataSource='stac')
        if LS_imgs is None:
            print('There are no images in the Landsat directory for cell {}'.format(gridCell))
        LScheck = CompareFilesToDb('Landsat', 'Both', in_dir, gridCell, gridFile, Yrs, dataSource='stac')
        if len(LScheck[0]) == 0 and len(LScheck[1]) == 0:
            print('There are no missing Landsat downloads!')
            LSdl = 1
        else:
            missCheck = WhyMissingFiles('Landsat',LScheck[0],3)
            LSdl = missCheck[0]

    S2 = GetImgListFromDb('Sentinel', in_dir, gridCell,Yrs, dataSource='stac')
    S2_dir = Path('{}/{:06d}/sentinel2'.format(in_dir,gridCell))
    if os.path.exists(S2_dir) == False:
        print('There is no Sentinel directory for cell {}'.format(gridCell))
    else:
        S2_imgs = PrintFilesInDirectory(S2_dir,'.tif',printList=False,out_dir=None,dataSource='stac')
        if S2_imgs is None:
            print('There are no images in the Sentinel directory for cell {}'.format(gridCell))
        S2check = CompareFilesToDb('Sentinel', 'Both', in_dir, gridCell, gridFile, Yrs, dataSource='stac')
        print('checking reason for missing files...')
        if len(S2check[0]) == 0 and len(S2check[1]) == 0:
            print('There are no missing Sentinel downloads!')
            S2dl = 1
        else: 
            missCheck = WhyMissingFiles('Sentinel',S2check[0],3)
            S2dl = missCheck[0]