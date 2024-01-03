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
import xarray as xr
import rasterio as rio
from shapely.geometry import box
import matplotlib.pyplot as plt

def print_files_in_directory(in_dir,endstring,print_list=False,out_dir=None,data_source='stac'):
    '''
    Can generate a dataframe with a list of all files in folder for quick exploration. 
    Option to print to file for deeper look.
    If the directory of interest is a brdf directory, can extract dates and sensor types from file names
    using '.nc' as {endstring} and {brdfDir=True} and generate quick graphs of images by year and month below.
    '''
    file_list = [f for f in os.listdir(in_dir) if f.endswith(endstring) and 'angles' not in f]
    if len(file_list)==0:
        print('there are no files ending with {} in directory {}'.format(endstring, in_dir))
        return None
    else:
        filedf = pd.DataFrame(file_list, columns = ['file'])
    
        if 'brdf' in str(in_dir):
            if data_source == 'GEE':
                filedf['sensor'] = filedf['file'].str[:4]
                filedf['date'] = np.where((filedf['sensor']=='L1C_'),filedf['file'].str[19:27],filedf['file'].str[17:25])
            elif data_source == 'stac':
                filedf['sensor'] = filedf.apply(lambda x: x['file'].split('_')[1][:4], axis=1)
                filedf['date'] = filedf.apply(lambda x: x['file'].split('_')[3][:8], axis=1)
                filedf['base']=filedf.apply(lambda x: "_".join(x['file'].split("_")[:4]), axis=1)
            filedf['yr'] = filedf['date'].str[:4]
            filedf['yrmo'] = filedf['date'].str[:6]
            sorted_files = filedf.sort_values(by='date')
    
        if print_list == True:
            pd.DataFrame.to_csv(filedf, os.path.join(out_dir,'{}_FileList.txt'.format(os.path.basename(in_dir))), sep=',', na_rep='.', index=False)   
        
        return filedf

def print_files_in_multiple_directories(full_dir,sub_dir,endstring,print_list=False,out_dir=None):
    '''
    This will return a dataframe with all files ending in {endstring} in each directory named {sub_dir} within the {full_dir}
    If the desired sub_dir is a brdf directory ({brdf=True}), adds correct date and year info to dataframe for plotting
    Drops duplicated filenames to reveal the number of images coming from unique Sentinel/Landsat scenes.
    Will print final dataframe to file in {out_dir} with {print_list=True}
    '''
    file_list = []
    multi_file_list = []
    for x in full_dir.iterdir():
        if x.is_dir():
            for sx in x.iterdir():
                if os.path.basename(sx) == sub_dir:
                    file_set = print_files_in_directory(sx,endstring,print_list=False,out_dir=None,data_source='stac')
                    multi_file_list.append(file_set)
    fullfile_df = pd.concat(multi_file_list)
    num_cells = len(multi_file_list)
    len_orig = len(fullfile_df)
    unique_imgs = fullfile_df.drop_duplicates(subset=['base'],keep='first')
    print('There are {} processed images from {} unique Sentinel/Landsat images over {} cells.'.format(len_orig,len(unique_imgs),num_cells))
    
    if print_list == True:
        pd.DataFrame.to_csv(unique_imgs, os.path.join(out_dir,'ALLFileList.txt'), sep=',', na_rep='.', index=False)   
        
    return unique_imgs

def reconstruct_db(processing_info_path,landsat_path,sentinel2_path,brdf_path):
    '''
    This checks for an existing processing.info database and creates one if needed from download and brdf folders.
    This is only for cases of corruption or accidental deletion. -- 
       processing.info is normally created as files are downloaded -- 
    Note: It is best to use original database whenever possible, as this will not recreate error notes,
       nor populate the numpix or coreg shift_x and shift_y columns that are in the original db
    '''
    modified = False
    
    if os.path.exists(brdf_path):
        brdf_files = [fi for fi in os.listdir(brdf_path) if fi.endswith('.nc')]
    else:
        brdf_files = []
    if os.path.exists(landsat_path):
        landsat_files = [fi for fi in os.listdir(landsat_path) if fi.endswith('.tif')]
    else:
        landsat_files = []
    if os.path.exists(sentinel2_path):
        sentinel2_files = [fi for fi in os.listdir(sentinel2_path) if fi.endswith('.tif')]
    else:
        sentinel2_files = []
        
    if len(landsat_files) + len(sentinel2_files) + len(brdf_files) == 0:
        print('no images have been downloaded')
    else:
        ## Make new processing db if it does not already exist:
        if not processing_info_path.is_file():
            processing_dict = {}
            # First check for for files in the brdf folder (these are at the most processed stage)
            if len(brdf_files) > 0:
                for b in brdf_files:
                    # get corresponding dl id:
                    if b.split("_")[1].startswith('L'):
                        dlid = '{}_{}_{}_{}_{}_{}'.format(b.split("_")[1],
                                                         'L2SP',
                                                          b.split("_")[2][4:10],
                                                          b.split("_")[3],
                                                          b.split("_")[2][10:12],
                                                          b.split("_")[2][12:14])
                    else:
                        dlid = '{}_{}_{}_{}_{}'.format(b.split("_")[1],
                                                       b.split("_")[2][4:9],
                                                       b.split("_")[3],
                                                       b.split("_")[2][9:10],
                                                       b.split("_")[2][10:13])
                    bp = 'True' if b.split('_')[0] == 'L3B' else ('False' if b.split('_')[0] == 'L3A' else np.nan)
                    processing_dict[dlid] = {'dl':'{}'.format(landsat_path,dlid),
                                           'beforeDB':True,
                                           'brdf_id':'{}'.format(b),
                                           'brdf':'True',
                                           'brdf_error':np.nan,
                                           'bandpass':bp}
            # If no files in brdf folder, reconstruct db from download folders
            else:
                for f in landsat_files:
                    processing_dict[os.path.splitext(f)[0]]={'dl':'{}/{}'.format(landsat_path,f),
                                                             'beforeDB':True}
                for s in sentinel2_files:
                    processing_dict[os.path.splitext(s)[0]]={'dl':'{}/{}'.format(sentinel2_path,s),
                                                             'beforeDB':True}
            new_processing_info = pd.DataFrame.from_dict(processing_dict,orient='index')
            new_processing_info.rename_axis('id', axis=1, inplace=True)
            pd.to_pickle(new_processing_info, processing_info_path)
            print(f'{len(new_processing_info)} images downloaded and added to database.')
            
        # read in existing db (can be the one that was just created or pre-existing):
        processing_db = pd.read_pickle(processing_info_path)
        
        ## to fix issues from older version of db already created for some cells:
        if 'id' not in processing_db:
            processing_db.rename_axis('id', axis=1, inplace=True)
        #if processing_db.index != 'id':
        #    print('removing original index column and setting it to id column')
        #    processing_db.set_index('id', drop=True, inplace=True)
        
        print(f'{len(processing_db)} records in db. {len(landsat_files)} landsat and {len(sentinel2_files)} sentinel images in downloads.')

        if len(processing_db) >= len(landsat_files) + len(sentinel2_files):
            print('all downloaded images have probably been added to db already')
        else:
            print('adding images to db...')
            new_dls = {}
            for f in landsat_files:
                if os.path.splitext(f)[0] in processing_db.values:
                    continue
                else:
                    new_dls[os.path.splitext(f)[0]]={'dl':'{}/{}'.format(landsat_path,f),'beforeDB':True}
            for s in sentinel2_files:
                if os.path.splitext(s)[0] in processing_db.values:
                    continue
                else:
                    new_dls[os.path.splitext(s)[0]]={'dl':'{}/{}'.format(sentinel2_path,s),',beforeDB':True}
        
            if len(new_dls)>0:
                new_dl_db = pd.DataFrame.from_dict(new_dls,orient='index')
                new_dl_db.rename_axis('id', axis=1, inplace=True)
                processing_db.append(new_dl_db)
                modified = True
            
        if os.path.exists(brdf_path):
            if 'brdf' in processing_db:
                print('brdf data already in database')
            
            else: 
                print('adding brdf info to db...')
                processing_db['brdf_id'] = np.nan
                processing_db['brdf_error'] = np.nan
                processing_db['brdf'] = np.nan
                processing_db['bandpass'] = np.nan
                for idx, row in processing_db.iterrows():
                    match=None
                    #print(idx)
                    for fi in os.listdir(brdf_path):
                        if fi.endswith('.nc'):
                            if idx.startswith('S'):  
                                if (idx.split('_')[1] in fi.split('_')[2]) and (idx.split('_')[2] == fi.split('_')[3]):
                                    match = fi
                            elif idx.startswith('L'): 
                                if (idx.split('_')[0] == fi.split('_')[1]) and (idx.split('_')[2] in fi.split('_')[2]) and (idx.split('_')[3] == fi.split('_')[3]):
                                    match = fi
                    #print(f'match:{match}')
                    processing_db.at[idx,'brdf_id']=match
                    if match is not None:
                        if match.split('_')[0] == 'L3B':
                            processing_db.at[idx,'bandpass']=True
                        elif match.split('_')[0] == 'L3A':
                            processing_db.at[idx,'bandpass']=False
                
                modified = True
            
            num_coreged_files = len([fi for fi in os.listdir(brdf_path) if fi.endswith('coreg.nc')])
            print(f'{num_coreged_files} images have been coreged')
            if num_coreged_files == 0:
                print('coregistration has not yet occured. Processing database is up to date')
            else:
                if 'shift_x' in processing_db:
                    print('coreg data has already been added to database')
                else:
                    print('adding coreg info to db...')
                    processing_db['coreg'] = np.nan
                    processing_db['shift_x'] = np.nan
                    processing_db['shift_y'] = np.nan
                    processing_db['coreg_error'] = np.nan
                    for idx, row in processing_db.iterrows():
                        match=None
                        #print(idx)
                        for fi in os.listdir(brdf_path):
                            if fi.endswith('.nc'):
                                if idx.startswith('S'):
                                    if (idx.split('_')[1] in fi.split('_')[2]) and (idx.split('_')[2] == fi.split('_')[3]):
                                        match = fi 
                                elif idx.startswith('L'): 
                                    if (idx.split('_')[0] == fi.split('_')[1]) and (idx.split('_')[2] in fi.split('_')[2]) and (idx.split('_')[3] == fi.split('_')[3]):
                                        match = fi
                        #print(f'match:{match}')
                        if match is not None:
                            if 'coreg' in match:
                                processing_db.at[idx,'coreg']=True
                            elif match.endswith('X.nc'):
                                processing_db.at[idx,'coreg']=False
                                processing_db.at[idx,'coreg_error']='unknown'
                            else:
                                processing_db.at[idx,'coreg']='NaN'                           
                    modified = True                        
        else:
            print('brdfs have not yet been created. Processing database is up to date')

        if modified == True:
            pd.to_pickle(processing_db, processing_info_path)
            print('saving new database')
        
        return processing_db
    
def read_db(db_path,db_version):
    pyver = float((sys.version)[:3])
    if pyver < 3.8:
        print('python version is {}'.format(pyver))
        ### note this may not work depending on version of Pandas. If it doesn't, maybe upgrade Pandas
        import pickle5 as pickle
        with open(db_path, 'rb') as file1:         
            df = pickle.load(file1)
    else:
        if db_version == 'current':
            df = pd.read_pickle(db_path)
            df['sensor']=df.index.map(lambda x:(x[:4].lower()))
            # get date -- in current version, this is in different part of id for Landsat and Sentinel2
            df_lan = df[df['sensor'].str.startswith('l')]
            df_lan['date'] = df_lan.index.map(lambda x: int(x.split('_')[3][:8]))
            df_sen = df[df['sensor'].str.startswith('s')]
            df_sen['date'] = df_sen.index.map(lambda x: int(x.split('_')[2][:8]))
            df = pd.concat([df_lan,df_sen],axis=0)
            df['yr']=df.date.map(lambda x:(int(str(x)[:4])))
        else:
            df = pd.read_pickle(db_path).set_index('date')
            df.index.rename(None, inplace=True)
            df = df.assign(date=df.index)
            df = df.sort_index()
    
    return df


def get_img_list_from_db(in_dir, grid_cell, sensor, yrs, data_source='stac'):
    '''
    returns list of images in database for year range (yrs) for selected directory (raw or brdf)
    and sensor ('Landsat', 'Sentinel').
    yrs is a list in format [YYYY, YYYY]
    '''
    if data_source == 'stac':
        scene_info_combo = (Path('{}/{:06d}/processing.info'.format(in_dir,grid_cell)))
        if os.path.exists(scene_info_combo):
            df = read_db(scene_info_combo,'current')
            if yrs:
                df_out0 = df[(df['date']>=int('{}0101'.format(yrs[0]))) & (df['date']<int('{}0101'.format(yrs[1])))]
        else:
            ## The following is all for backwards compatibility for files processed with previous versions of eostac
            if 'brdf' not in str(in_dir):
            #original downloads for stac data are in separate landsat and sentinel2 folders. Each has its own scene.info file
                if sensor.lower().startswith('l') or sensor=='All':
                    scene_info_l = (Path('{}/{:06d}/landsat/scene.info'.format(in_dir,int(grid_cell))))
                    if os.path.exists(scene_info_l) == False:
                        print('There is no scene.info file in the Landsat directory for cell {}'.format(grid_cell))
                        landsat_df = None
                    else: landsat_df = read_db(scene_info_l,'old')
                    if sensor == 'All':
                        scene_info_s = (Path('{}/{:06d}/sentinel2/scene.info'.format(in_dir,int(grid_cell))))
                        if os.path.exists(scene_info_s) == False:
                            print('There is no scene.info file in the Sentinel directory for cell {}'.format(grid_cell))
                            sentinel_df == None
                        else:
                            sentinel_df = read_db(scene_info_s,'old')
                            if landsat_df is None:
                                df = sentinel_df
                            else:
                                df = pd.concat([landsat_df,sentinel_df],axis=0)
                    else:
                        df = landsat_df        
                elif sensor.lower().startswith('s'):
                    scene_info_s = (Path('{}/{:06d}/sentinel2/scene.info'.format(in_dir,int(grid_cell))))
                    if os.path.exists(scene_info_s) == False:
                        print('There is no scene.info file in the Sentinel directory for cell {}'.format(grid_cell))
                    else: df = read_db(scene_info_s, 'old')
                else:
                    print('sensor options are All, s, l, lt05, le07, lc08, lc09 {}'.format(sensor))           
            else:
                scene_info = (os.path.join(in_dir,'scene.info'))
                df = read_db(scene_info,'old')
            if yrs:
                df_out0 = df['{}-01-01:{}-12-31'.format(yrs[0],yrs[1])]
        
        if sensor !='All':
            print('filtering returned dataset to {}...'.format(sensor))
            if len(sensor) == 1:
                df_out = df_out0.loc[df_out0.sensor.str.startswith(sensor)]
            else:
                df_out = df_out0.loc[df.sensor==sensor]

    return df

def separate_missing_db_files(in_dir,grid_cell,image_type='All', yrs=None,data_source='stac'):
    df = get_img_list_from_db(in_dir, grid_cell, image_type, yrs, data_source)          
    if 'brdf' in in_dir:
        df['file_path'] = df.out_id.apply(lambda x: os.path.join(in_dir,x))
    else:
        df['file_path'] = df.id.apply(lambda x: os.path.join(in_dir,x+'.tif'))

    df['file_path_exists'] = df.file_path.apply(lambda x: Path(x).is_file())
    df_existing = df.loc[df.file_path_exists]
    df_missing = df.loc[~df.file_path_exists]
    
    return df_existing, df_missing

def get_valid_pix_per(img_path):
    if imgPath.endswith('.tif'):
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
    
    if allpix == 0:
        validper = 0
    else:
        validper = (validpix/allpix)    
    return validper

def check_valid_pixels(raw_dir, brdf_dir, grid_cell, image_type='All', yrs=None, data_source='stac'):
    df_brdf = separate_missing_db_files(brdf_dir,grid_cell,imageType,yrs,data_source)[0]          
    landsat_dir = Path('{}/{:06d}/landsat'.format(raw_dir,int(grid_cell)))
    sentinel_dir = Path('{}/{:06d}/sentinel2'.format(raw_dir,int(grid_cell)))
    df_brdf['orig_file_path'] = df_brdf.apply(lambda x: os.path.join(landsat_dir,x['id']+'.tif') if x['sensor'].startswith('l') else os.path.join(sentinel_dir,x['id']+'.tif'), axis=1)
    df_brdf['ValidPix_orig'] = df_brdf.orig_file_path.apply(lambda x: int(get_valid_pix_per(x)*100))
    df_brdf['ValidPix_brdf'] = df_brdf.file_path.apply(lambda x: int(get_valid_pix_per(x)*100))
    out_df = os.path.join(raw_dir,'{:06d}'.format(grid_cell),'processed_imgs_{}.info'.format(image_type))
    df_brdf.to_pickle(out_df)
        
    return df_brdf

def get_img_list_from_cat(sensor, grid_cell, grid_file, yrs=None, cat='default'):
    '''
    Gets list of images available on Planetary Hub for range of years {Yrs} [YYYY, YYYY]
    for {gridCell} CCCC and sensor {'Landsat' or 'Sentinel'}
    returns dataFrame with column of image ids
    '''
    import pystac_client
    import planetary_computer as pc
    from pystac_client import Client
    from pystac.extensions.eo import EOExtension as eo

    grid = gpd.read_file(grid_file)
    if grid.crs != pyproj.CRS.from_epsg(4326):
        grid = grid.to_crs('epsg:4326')
    bb = grid.query(f'UNQ == {[grid_cell]}').geometry.total_bounds

    if yrs == None:
        TimeSlice="2010-01-01/2022-12-30"
    else:
        TimeSlice=f"{yrs[0]}-01-01/{yrs[1]}-12-30"
        
    if sensor.startswith('l'):
        collect=["landsat-c2-l2"]
        api = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1/")

    elif sensor == ('s'):
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
    cat_ids = pd.DataFrame({'id':[o.id for o in items]})
    print(f"Returned {len(items)} Items")

    return cat_ids

def compare_planetaryHub_w_element84 (sensor, grid_cell, grid_file, yrs=None):
    '''
    Compares list of images for time slice that are available on Planetary Hub vs Element84
    Currently focused on Sentinel2.
    '''
    ph_df = get_img_list_from_cat(sensor, grid_cell, grid_file, yrs, cat='planetary')
    if sensor == 'Sentinel':
        e84_df = get_img_list_from_cat(sensor, grid_cell, grid_file, yrs, cat='default')
        e84_df['id2'] = e84_df.apply(lambda x: x['id'].split("_")[0] +'_'+ x['id'].split("_")[1] +'_'+ x['id'].split("_")[2], axis=1)
        e84_ids = e84_df['id2'].values.tolist()
        ph_df['id2'] = ph_df.apply(lambda x: x['id'].split("_")[0] +'_'+ x['id'].split("_")[4][1:6] +'_'+ x['id'].split("_")[2][:8], axis=1)
        ph_ids = ph_df['id2'].values.tolist()
       
    e84_notin_planetary = list(set(e84_ids) - set(ph_ids))
    planetary_notin_e84 = list(set(ph_ids) - set(e84_ids))
          
    return e84_notin_planetary,planetary_notin_e84

    
def check_for_missing_files(df, in_dir, grid_cell, sensor, data_source='stac'):
    '''
    Checks whether all of the files in the database {df} are in the directory {in_dir}
    returns list of image ids in db but not in directory
    '''
    if data_source == 'stac':
        if 'brdf' not in str(in_dir):
            if sensor.lower().startswith('l'):
                df['file_path'] = df.apply(lambda row: Path('{}/{:06d}/landsat/{}.tif'.format(in_dir,grid_cell,row.name)), axis=1)         
            elif sensor.lower().startswith('s'):
                df['file_path'] = df.apply(lambda row: Path('{}/{:06d}/sentinel2/{}.tif'.format(in_dir,grid_cell,row.name)), axis=1)
        else:
            df['file_path'] = df.apply(lambda row: Path('{}/{}.tif'.format(in_dir,grid_cell,row.name)), axis=1)
            #TODO: modify to reflect change in basename (this does not work as is for brdf directory) 
    df['file_path_exists'] = df.apply(lambda x: x['file_path'].is_file(), axis=1)
    dfe = df.loc[df.file_path_exists]
    dfno = df.loc[df.file_path_exists==False]
    missing = dfno.index.values.tolist()
    print('Of the {} files in the {} db, {} are missing from the directory'.format(df.shape[0],sensor,len(missing)))
    return missing

def why_missing_files(sensor,missing_list,num_to_allow):
    '''
    takes list of missing files in db and determines which are missing because they are Landsat7 after stop year (thus probably
    intentionally skipped). This was prior to creation of processing.db -- Now there is a column 'skipped' that gives this info directly.
    NO LONGER BEING USED
    '''
    print('checking reason for missing files...')
    dl = 0
    if sensor.lower().startswith('s'):
        print(missing_list)
        still_missing = missing_list
        if len(missing_list) <= num_to_allow:
            dl = 1
    if sensor.lower().startswith('l'):
        missing_df = pd.DataFrame(missing_list)
        missing_df['sensor'] = missing_df.apply(lambda x: x[0].split('_')[0][:4], axis=1)
        missing_df['yr'] = missing_df.apply(lambda x: int(x[0].split('_')[3][:4]), axis=1)
        missing_df['L7except'] = np.where((missing_df['sensor']=='LE07') & (missing_df['yr'] >= 2017),1,0)
        numL7 = sum(missing_df['L7except'])
        if numL7 > 0:
            print (f'{numL7} of the missing images are L7 images in or after 2017; Assuming this is intentional.' +
               '(if not, change the <L7end_yr> parameter in the downloading script)')
        still_missing = missing_df[missing_df['L7except']==0]
        if len(still_missing) == 0:
            print('No other missing files!')
            dl = 1
        else:
            print('There are {} files missing for other reasons:'.format(len(still_missing)))
            sm = still_missing[0].to_list()
            print(sm)
            
        if len(still_missing) <= num_to_allow:
            dl = 1

    return dl, sm

def compare_files_to_db(sensor, db_source, in_dir, grid_cell, grid_file, yrs, data_source='stac'):
    '''
    Compares files in directory {in_dir} to those in database {dbSource}.
    dbSource = 'Local'=scene.info , 'Remote'=PlanetaryComputer, or 'Both'
    returns list of image ids in db but not in directory
    if dbSource == 'Both', also compares remote and local databases to check whether there are remote files
    that were never seen by local database (never attempted to download) for cell/sensor/years 
    '''
    if db_source in['local','both']:
        print (f'Checking files against local database for {yrs[0]}-{yrs[1]}...')
        df_local = get_img_list_from_db(in_dir, grid_cell, sensor, yrs, data_source='stac')
        missing_local = check_for_missing_files(df_local, in_dir, grid_cell, sensor, data_source='stac')
    if db_source in['remote','both']:
        print (f'Checking files agaist remote database for {yrs[0]}-{yrs[1]}...')
        df_remote = get_img_list_from_cat(sensor, grid_cell, grid_file, yrs)
        missing_remote = check_for_missing_files(df_remote, in_dir, grid_cell, sensor, data_source='stac')
    if db_source == 'both':
        print (f'Checking local db against remote database for {yrs[0]}-{yrs[1]}...')
        local_list = df_local.index.values.tolist()
        remote_list = df_remote['id'].values.tolist()
        missing_from_localdb = list(set(remote_list) - set(local_list))
        print('the following Files are in the remote catalog but not in the local database:',missing_from_localdb)
    if db_source == 'local':
        return missing_local
    elif db_source == 'remote':
        return missing_remote
    else:
        return missing_local, missing_remote, missing_from_localdb

def get_cell_status(dl_dir, processed_dir, grid_cell, yrs, out_dir, data_source='stac'):
    
    cell_dict = {}
 
    if os.path.exists(Path('{}/{:06d}/processing.info'.format(dl_dir,int(grid_cell)))):
        processing_db = get_img_list_from_db(dl_dir, grid_cell, 'All', yrs, data_source='stac')
        yrdf1 =  processing_db.groupby(['yr']).size().to_frame('seen')

        ax = processing_db.groupby(['yr','sensor']).size().unstack().plot(kind='bar', stacked=True, figsize=(20, 5), 
                                     title=('Number images processed per year for cell {}'.format(grid_cell)))
        fig1 = ax.get_figure()
        fig1_path = os.path.join(out_dir,'{}_images_processed_by_sensor.png'.format(grid_cell))
        fig1.savefig(fig1_path, format="png")
        print('saved fig1 to {}'.format(fig1_path))
        
        if 'redownload' in processing_db:
            processing_errors1 = processing_db[processing_db['redownload'] == True]
            cell_dict['dl_errors']= int(processing_errors1.shape[0])
        if 'brdf_error' in processing_db:
            processing_errors2 = processing_db[~processing_db['brdf_error'].isnull()]
            cell_dict['brdf_errors']= int(processing_errors2.shape[0])
        if 'skip' in processing_db:
            skip = processing_db[processing_db['skip'] == True]
            cell_dict['skipped']=int(skip.shape[0])
            processed0 = processing_db[processing_db['skip']!=True]
            yrdf2 = processed0.groupby(['yr']).size().to_frame('processed')
            processed = processed0[processed0['redownload']!=True]
        else:
            processed = processing_db
        if 'brdf' in processed:
            brdf = processed[processed['brdf']==True]
            cell_dict['num_processed']=processed.shape[0]
            cell_dict['num_brdf']=brdf.shape[0]
        else:
            cell_dict['num_brdf']='brdf step not complete'
        if 'shift_x' in processed:
            coreged = processed0[processed0['coreg'] != False] 
            yrdf3 = coreged.groupby(['yr']).size().to_frame('coreged')
            cell_dict['num_coreged']=coreged.shape[0]
            processed_sentinel = processed[processed.index.str.startswith('S')]
            creg_sentinel = processed_sentinel[processed_sentinel['coreg']==True]
            cell_dict['num_processed_s2']=processed_sentinel.shape[0]
            cell_dict['num_coreg_s2']=creg_sentinel.shape[0]
            cell_dict['per_coreg_s2']=cell_dict['num_coreg_s2'] / cell_dict['num_processed_s2']
            creg_sentinel['abs_shift_x']=creg_sentinel['shift_x'].abs()
            creg_sentinel['abs_shift_y']=creg_sentinel['shift_y'].abs()
            cell_dict['avg_x_shift_s2']=creg_sentinel['abs_shift_x'].mean()
            cell_dict['avg_y_shift_s2']=creg_sentinel['abs_shift_y'].mean()
            cell_dict['med_x_shift_s2']=creg_sentinel['abs_shift_x'].median()
            cell_dict['med_y_shift_s2']=creg_sentinel['abs_shift_y'].median()
            processed_L5 = processed[processed.index.str.startswith('LT05')]
            creg_L5 = processed_L5[processed_L5['coreg']==True]
            cell_dict['num_processed_L5']=processed_L5.shape[0]
            cell_dict['num_coreg_L5']=creg_L5.shape[0]
            cell_dict['per_coreg_L5']=cell_dict['num_coreg_L5'] / cell_dict['num_processed_L5']
            creg_L5['abs_shift_x']=creg_L5['shift_x'].abs()
            creg_L5['abs_shift_y']=creg_L5['shift_y'].abs()
            cell_dict['avg_x_shift_L5']=creg_L5['abs_shift_x'].mean()
            cell_dict['avg_y_shift_L5']=creg_L5['abs_shift_y'].mean()
            cell_dict['med_x_shift_L5']=creg_L5['abs_shift_x'].median()
            cell_dict['med_y_shift_L5']=creg_L5['abs_shift_y'].median()
            processed_L7 = processed[processed.index.str.startswith('LE07')]
            creg_L7 = processed_L7[processed_L7['coreg']==True]
            cell_dict['num_processed_L7']=processed_L7.shape[0]
            cell_dict['num_coreg_L7']=creg_L7.shape[0]
            cell_dict['per_coreg_L7']=cell_dict['num_coreg_L7'] / cell_dict['num_processed_L7']
            creg_L7['abs_shift_x']=creg_L7['shift_x'].abs()
            creg_L7['abs_shift_y']=creg_L7['shift_y'].abs()
            cell_dict['avg_x_shift_L7']=creg_L7['abs_shift_x'].mean()
            cell_dict['avg_y_shift_L7']=creg_L7['abs_shift_y'].mean()
            cell_dict['med_x_shift_L7']=creg_L7['abs_shift_x'].median()
            cell_dict['med_y_shift_L7']=creg_L7['abs_shift_y'].median()
        else:
            cell_dict['num_coreged']='coreg step not complete'
            yrdf3 = yrdf2
        yrdf4 = yrdf1.join(yrdf2)
        yrdf = yrdf4.join(yrdf3)
        yrdf['excluded'] = yrdf['seen'] - yrdf['processed']
        yrdf['low quality'] = yrdf['processed'] - yrdf['coreged']
        yrdf.rename(columns={'coreged': 'used'},inplace=True)

        ax2 = yrdf[["used", "uncoreged", "excluded"]].plot(kind="bar", stacked=True, 
                color=['black','grey','white'], edgecolor = "black", figsize=(20, 5), 
                title=('Processing status for cell {}'.format(grid_cell)))
        fig2 = ax2.get_figure()
        fig2_path = os.path.join(out_dir,'{}_images_processed_by_stat.png'.format(grid_cell))
        fig2.savefig(fig2_path, format="png")
        print('saved fig2 to {}'.format(fig2_path))
        for idx in ['evi2','gcvi','wi','kndvi','nbr','ndmi']:
            idx_path = Path('{}/{:06d}/brdf_ts/ms/{}'.format(processed_dir,int(grid_cell),idx))
            if os.path.exists(idx_path):
                ts = sorted([f for f in os.listdir(idx_path) if f.endswith('tif')])
                if len(ts)>0:
                    cell_dict['index_{}'.format(idx)]='{}-{}'.format(ts[0][:4],ts[-1][:4])
    
    else:
        ##Check if files have been downloaded for cell:
        ls_dir = Path('{}/{:06d}/landsat'.format(dl_dir,grid_cell))
        if os.path.exists(ls_dir) == False:
            print('There is no Landsat directory for cell {}'.format(grid_cell))
        else:
            ls_imgs = print_files_in_directory(ls_dir,'.tif',print_list=False,out_dir=None,data_source='stac')
            if ls_imgs is None:
                print('There are no images in the Landsat directory for cell {}'.format(grid_cell))
     
        s2_dir = Path('{}/{:06d}/sentinel2'.format(dl_dir,grid_cell))
        if os.path.exists(s2_dir) == False:
            print('There is no Sentinel directory for cell {}'.format(grid_cell))
        else:
            s2_imgs = print_files_in_directory(s2_dir,'.tif',print_list=False,out_dir=None,data_source='stac')
            if s2_imgs is None:
                print('There are no images in the Sentinel directory for cell {}'.format(grid_cell))
        cell_dict = {}
        
    return cell_dict, fig1_path, fig2_path

def update_cell_status_db(status_db_path, cell_list, dl_dir, processed_dir, yrs, data_source='stac'):
 
    #status_db_path = os.path.join(in_dir2,cell_processing_post.csv)
    if Path(status_db_path).is_file():
        status_dict = pd.read_csv(Path(status_db_path),index_col=[0]).to_dict(orient='index')
    else:
        status_dict = {}
        
    for cell_id in cell_list:
        print('processing cell {}'.format(cell_id))
        new_dict_entry = get_cell_status(dl_dir, processed_dir, cell_id, yrs, data_source='stac')
        if cell_id in status_dict:
            status_dict[cell_id].update(new_dict_entry)
        else:
            status_dict[cell_id]=new_dict_entry
            
    updated_processing_info = pd.DataFrame.from_dict(status_dict,orient='index')
    updated_processing_info.rename_axis('cell_id', axis=1, inplace=True)
    pd.DataFrame.to_csv(updated_processing_info, status_db_path, index='cell_id')

    return updated_processing_info

    status_dict[grid_cell]={}
            
def get_num_valid_pix_for_stac(file_list, date_list=None):
    import geowombat as gw  #This throws an error in jupyter notebooks running on older systems, so better to import only as needed
    
    if file_list[0].endswith('.nc'):
        with xr.open_mfdataset(
            file_list,
            chunks={'band': -1, 'y': 256, 'x': 256},
            concat_dim='time',
            combine='nested',
            engine='h5netcdf'
        ) as src:
            validpix = src.nir.where(src.nir < 10000).count(dim='time').chunk({'y': 256, 'x': 256})
            validpix = validpix.squeeze()
    
    elif file_list[0].endswith('tif'):                                                            
        with gw.open(
            file_list,
            time_names=date_list,
            band_names=['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
            overlap='min'   # We use overlap='min' because the nodata values are 32768 or 10000.
        ) as src:
            validpix = src.sel(band=['nir']).transpose('time','band','y','x').where(lambda x: x != x.nodatavals[0]).count(dim='time')
            validpix = validpix.squeeze()
    else:
        print('can only count valid pix for .nc and .tif files currently')
        validpix = None
    
    return validpix