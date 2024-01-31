#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np
import rasterio as rio
from collections.abc import Iterable
from osgeo import gdal, ogr, gdal_array
from rasterio import mask as mask
import osgeo  # needed only if running from Windows
from shapely.geometry import Polygon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
#import seaborn as sn
from joblib import dump, load
import csv
from LUCinSA_helpers.ts_composite import make_ts_composite

## Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()


def get_class_col(lc_mod,lut):
    if lc_mod == 'All':
        class_col = 'LC25'
    elif lc_mod == "trans_cats":
        class_col = 'LCTrans'
    elif lc_mod == 'crop_nocrop':
        class_col = 'LC2'
    elif lc_mod == 'crop_nocrop_medcrop':
        class_col = 'LC3'
    elif lc_mod == 'crop_nocrop_medcrop_tree':
        class_col = 'LC4'
    elif lc_mod == 'veg':
        class_col = 'LC5'
    elif lc_mod == 'cropType':
        class_col = 'LC_crops'
    elif lc_mod.startswith('single'):
        lc_base =  lc_mod.split('_')[1].lower()
        target_class = lut.index[lut['USE_NAME'].map(lambda s: lc_base in s.lower())].to_list()
        if len(target_class) == 0:
            print('no match found for {} in lut'.format(lc_base))
        elif len(target_class) > 1:
            print ('there are more than one entries with {} in USE_NAME column of lut'.format(lc_base))
        else:
            class_col = 'LC1'
            if class_col not in lut.columns:
                print('making new virtual {} column in lut'.format(lc_base))
                lut['LC1'] = 0
                lut['LC1_name'] = 'no_{}'.format(lc_base)
                lut.at[target_class[0],'LC1'] = 1
                lut.at[target_class[0],'LC1_name'] = lc_base
    else:
        print('current options for lc_mod are All, LCTrans, LC2, LC3, LC4, LC5, LC_crops and single_X with X as any category. You put {}'.format(lc_mod))
    
    '''
    elif lc_mod == 'crop_post':
        #where prediction == 1...
        df_train = df_train[df_train['LC2'] == 1]
        print('there are {} sample points after removing non-crop'.format(df_train.shape[0]))
        y = df_train['LC_UNQ']
        #TODO: need two step prediction methods
    elif lc_mod == 'nocrop_post':
        df_train = df_train[df_train['LC2'] == 0]
        print('there are {} sample points after removing crop'.format(df_train.shape[0]))
        #TODO: need two step prediction methods
           
    elif lc_mod in ['HighVeg', 'MedVeg', 'LowVeg', 'NoVeg']:
        df_train['label'] = 0
        df_train.loc[df_train['Class1'] == classification,'label'] = 1
        y = df_train['label']
    elif 'Crops' in lc_mod or 'Grass' in lc_mod:
        df_train['label'] = 0
        if 'Mate' in lc_mod:
            df_low = df_train[(df_train['Class1'] == 'LowVeg') | (df_train['USE_NAME'] =='Crops-Yerba-Mate')]
        else:
            df_low = df_train[df_train['Class1'] == 'LowVeg']
        print('There are {} LOWVeg records'.format(str(len(df_low))))
        df_low.loc[df_low['USE_NAME'] == Class,'label'] = 1
        X = df_low[vars_RF]
        y = df_low['label']
    elif lc_mod == 'LowVeg_Lev2':
        df_low = df_train[df_train['Class1'] == 'LowVeg']
        X = df_low[rf_vars]
        y = df_low['USE_NAME']
    '''
        
    return class_col,lut
        
def separate_holdout(training_pix_path, out_dir):
    '''
    USE THIS WHEN WE AUGMENT BY POLYGON
    Generates separate pixel databases for training data and 20% field-level holdout
    Use this instead of generate_holdout() to fit a model to an exsisting holdout set
       to avoid having points from the same polygon in both the training and holdout sets (as this would inflate accuracy)
    '''
    #holdout_set = pd.read_csv(holdout_field_path)
    #pixels = pd.read_csv(training_pix_path)
    
    ## if there is no 'field_id' in the pixel dataset, use the following two lines (but now 'field_id' is aready in pixels)
    #holdout_set['unique_id'] = holdout_set['unique_id'].apply(str)
    #pixels['field_id'] = pixels['pixel_id'].str[:10]
    
    pixels_holdouts = pixels[pixels.field_id.isin(holdout_set['unique_id'])]
    pixels_holdouts['set']='HOLDOUT'
    pixels_training = pixels[~pixels.field_id.isin(holdout_set['unique_id'])]
    pixels_training['set']='TRAINING'

    print("original training set had {} rows. Current training set has {} rows and holdout has {} rows."
          .format(len(pixels), len(pixels_training), len(pixels_holdouts)))
    
    training_pix_path2 = os.path.join(out_dir,'V4_Model_training_FieldLevel_toTrainNH.csv')
    pd.DataFrame.to_csv(pixels_training, training_pix_path2, sep=',', na_rep='NaN', index=False)
    holdout_field_pix_path = os.path.join(out_dir,'V4_Model_testing_FieldLevel_Holdout_FullFieldx.csv')
    pd.DataFrame.to_csv(pixels_holdouts, holdout_field_pix_path, sep=',', na_rep='NaN', index=False)
    
    return training_pix_path2, holdout_field_pix_path

def get_confusion_matrix(pred_col, obs_col, lut, lc_mod_map, lc_mod_acc, print_cm=False, out_dir=None, model_name=None):
    '''
    returns confusion matrix with optional regrouping of classes based on LUT 
    classification schema and class columns defined in get_class_col
    '''
    cmdf = pd.DataFrame()
    cmdf['obs'] = obs_col
    cmdf['pred'] = pred_col
    
    if lc_mod_map.startswith('single'):
        cm=pd.crosstab(cmdf['obs'],cmdf['pred'],margins=True)
        print(cm)
    else: 
        map_cat = get_class_col(lc_mod_map,lut)[0]
        acc_cat = get_class_col(lc_mod_acc,lut)[0]
        print('getting confusion matrix based on {}...'.format(acc_cat))
        cmdf2 = cmdf.merge(lut[['LC_UNQ','{}_name'.format(acc_cat)]], left_on='obs', right_on='LC_UNQ',how='left')
        cmdf2.rename(columns={'{}_name'.format(acc_cat):'obs_reclass'}, inplace=True)
        cmdf2.drop(['LC_UNQ'],axis=1,inplace=True)
        cmdf3 = cmdf2.merge(lut[['LC_UNQ', '{}_name'.format(acc_cat)]], left_on='pred', right_on='LC_UNQ',how='left')
        cmdf3.rename(columns={'{}_name'.format(acc_cat):'pred_reclass'}, inplace=True)
        cmdf3.drop(['LC_UNQ'],axis=1,inplace=True)
        cm=pd.crosstab(cmdf3['obs_reclass'],cmdf3['pred_reclass'],margins=True)
        
    cm['correct'] = cm.apply(lambda x: x[x.name] if x.name in cm.columns else 0, axis=1)
    cm['sumcol'] = cm.apply(lambda x: cm.loc['All', x.name] if x.name in cm.columns else 0)
    cm['UA'] = cm['correct']/cm['All']
    cm['PA'] = cm['correct']/cm['sumcol']
 
    '''
    crops = lut.loc[lut['LC2'] == 1]
    no_crops = lut.loc[lut['LC2'] == 0]
    crop_cats = crops[class_col].to_list()
    crop_cols = [i for i in crop_cats if i in cm.columns]
    
    cm['Crop'] = cm[crop_cols].sum(axis=1)
    nocrop_cats = no_crops[class_col].to_list()
    nocrop_cols = [i for i in nocrop_cats if i in cm.columns]
    cm['NoCrop'] = cm[nocrop_cols].sum(axis=1)
    '''
    #print(f'Confusion Matrix: {cm}')
    if print_cm == True:
        pd.DataFrame.to_csv(cm,os.path.join(out_dir,f'CM_{model_name}.csv'), sep=',', index=True)
    
    return cm

def quick_accuracy (X_test, y_test, rf_model, lc_mod, out_dir,model_name,lut):
    
    predicted = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    print(f'Out-of-bag score estimate: {rf_model.oob_score_:.3}')
    print(f'Mean accuracy score: {accuracy:.3}')
    
    cm = get_confusion_matrix(predicted, y_test,lut, lc_mod, lc_mod, out_dir,model_name,lut)                    
    
    return accuracy, cm

def prep_test_train(df_in, out_dir, class_col, mod_name):
    
    if isinstance(df_in, pd.DataFrame):
        df_in = df_in
    else:
        df_in = pd.read_csv(df_in, index_col=0)
    print('there are {} pts in the full data set'.format(df_in.shape[0]))
   
    # remove unknown and other entries where class is not specified a given level (i.e. crop_low if crop type is desired)
    df_in = df_in[df_in[class_col] < 99]
    print('there are {} sample points after removing those without clear class'.format(df_in.shape[0]))
    
    #Separate training and holdout datasets to avoid confusion with numbering
    training_pix_path = os.path.join(out_dir,f'{mod_name}_TRAINING.csv')
    holdout_pix_path = os.path.join(out_dir,f'{mod_name}_HOLDOUT.csv')
    df_train = df_in[df_in['TESTSET20'] == 0]
    pd.DataFrame.to_csv(df_train, training_pix_path, sep=',', na_rep='NaN', index=False)
    df_test = df_in[df_in['TESTSET20'] == 1]
    pd.DataFrame.to_csv(df_test, holdout_pix_path, sep=',', na_rep='NaN', index=False)
    
    return(training_pix_path, holdout_pix_path)

def multiclass_rf(trainfeatures, out_dir, mod_name, lc_mod, importance_method, ran_hold, lut):
    
    df_train = pd.read_csv(trainfeatures)
    print('There are {} training features'.format(df_train.shape[0]))
    
    class_col = get_class_col(lc_mod,lut)[0]
    y = df_train[class_col]
           
    vars_rf = [col for col in df_train if col.startswith('var_')]
    X = df_train[vars_rf]
    
    #print (pd.Series(y).value_counts())
    X = X.values
    y = y.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = .01, random_state=ran_hold)

    rf = RandomForestClassifier(n_estimators=500, oob_score=True)
    rf = rf.fit(X_train,y_train)
    dump(rf, os.path.join(out_dir,'{}_RFmod.joblib'.format(mod_name)))

    cm = quick_accuracy (X_test, y_test, rf, lc_mod, out_dir,mod_name,lut)

    if importance_method != None:
        if importance_method == "Impurity":
            var_importances = pd.Series(rf.feature_importances_, index=vars_rf)
            pd.Series.to_csv(var_importances,os.path.join(out_dir,'VarImportance_{}.csv'.format(mod_name)),sep=',',index=True)
        elif importance_method == "Permutation":
            result = permutation_importance(rf, X_test, y_test, n_repeats=10,random_state=ran_hold, n_jobs=2)
            var_importances = pd.Series(result.importances_mean, index=vars_rf)
            pd.Series.to_csv(var_importances,os.path.join(out_dir,'VarImportance_{}.csv'.format(mod_name)),sep=',', index=True)

    return rf, cm

def get_holdout_scores(holdoutpix, rf_model, class_col, out_dir):
    ## Save info for extra columns and drop (model is expecting only variable input columns)
    
    if isinstance(holdoutpix, pd.DataFrame):
        holdout_pix = holdoutpix
    else:
        holdout_pix = pd.read_csv(holdoutpix)
        
    holdout_labels = holdout_pix[class_col]
    h_IDs = holdout_pix['OID_']
    print('number of holdout pixels = {}'.format(len(holdout_pix)))
    
    ## Get list of variables to include in model:
    vars = [col for col in holdout_pix if col.startswith('var_')]
    
    holdout_fields = holdout_pix[vars]
    #print(holdout_fields.head())

    ## Calculate scores
    #holdout_fields_predicted = rf_model.predict_proba(holdout_fields)
    holdout_fields_predicted = rf_model.predict(holdout_fields)
    
    ## Add extra columns back in
    holdout_fields = pd.concat([holdout_fields,pd.Series(holdout_fields_predicted),holdout_labels,h_IDs],axis=1)
    new_cols = [-3,-2,-1]
    new_names = ["pred","label","OID"]
    old_names = holdout_fields.columns[new_cols]
    holdout_fields.rename(columns=dict(zip(old_names, new_names)), inplace=True)

    ## Print to file
    pd.DataFrame.to_csv(holdout_fields, os.path.join(out_dir,'Holdouts_predictions.csv'), sep=',', na_rep='NaN', index=True)
   
    return holdout_fields

def getset_feature_model(feature_mod_dict,feature_model,spec_indices=None,si_vars=None,spec_indices_pheno=None,pheno_vars=None
                         singleton_vars=None,poly_vars=None, combo_bands=None):
    
    with open(feature_mod_dict, 'r+') as feature_model_dict:
        dic = json.load(feature_model_dict)
        if feature_model in dic:
            spec_indices = dic[feature_model]['spec_indices']
            si_vars = dic[feature_model]['si_vars']
            spec_indices_pheno = dic['spec_indices_pheno']
            pheno_vars = dic['pheno_vars']
            singleton_vars = dic[feature_model]['singleton_vars']
            poly_vars = dic[feature_model]['poly_vars']
            combo_bands = dic[feature_model]['combo_bands']
            band_names = dic[feature_model]['band_names']
            print('using existing model: {} \n spec_indices = {} \n si_vars = {} \n pheno_vars = {} on {} \n singleton_vars={} \n poly_vars = {}'
                  .format(feature_model, spec_indices, si_vars, pheno_vars, spec_indices_pheno, singleton_vars, poly_vars))
        else:
            dic[feature_model] = {}
            dic[feature_model]['spec_indices'] = spec_indices
            dic[feature_model]['si_vars'] = si_vars
            dic[feature_model]['singleton_vars'] = singleton_vars
            dic[feature_model]['poly_vars'] = poly_vars
            dic[feature_model]['spec_indices_pheno'] = spec_indices_pheno
            dic[feature_model]['pheno_vars'] = pheno_vars
            dic[feature_model]['combo_bands'] = combo_bands
            
            band_names = []
            if len(combo_bands) > 0:
                band_names.append(combo_bands)
            if spec_indices_pheno is not None and spec_indices_pheno != 'None':
                for sip in spec_indices_pheno:
                    for pv in pheno_vars:
                        band_names_append('{}_{}'.format(sip,pv))
            if spec_indices is not None and spec_indices != 'None':
                for si in spec_indices:
                    for sv in si_vars:
                        band_names.append('{}_{}'.format(si,sv))
            if singleton_vars is not None and singleton_vars != 'None':
                for sin in singleton_vars:
                    band_names.append('sing_{}'.format(sin))
            if poly_vars is not None and poly_vars != 'None':       
                for pv in poly_vars:
                    band_names.append('poly_{}'.format(pv))
            all_bands = list(set(band_names))
            dic[feature_model]['band_names'] = all_bands
            with open(feature_mod_dict, 'w') as new_feature_model_dict:
                json.dump(dic, new_feature_model_dict)
            print('created new model: {} \n spec_indices = {} \n si_vars = {} \n pheno_vars = {} on {} \n singleton_vars={} \n singleton_vars = {} \n poly_vars = {}'
                  .format(feature_model, spec_indices, si_vars, pheno_vars, spec_indices_pheno, singleton_vars, poly_vars))
        
    return spec_indices,si_vars,spec_indices_pheno,pheno_vars,singleton_vars,poly_vars,combo_bands,band_names
    
def make_variable_stack(in_dir,cell_list,feature_model,start_yr,start_mo,spec_indices,si_vars,spec_indices_pheno,pheno_vars,feature_mod_dict,
                        singleton_vars=None, singleton_var_dict=None, poly_vars=None, poly_var_path=None, combo_vars=None,                                         scratch_dir=None):
    
    # get model paramaters if model already exists in dict. Else create new dict entry for this model
    spec_indices, si_vars, spec_indices_pheno, pheno_vars, singleton_vars, poly_vars, combo_bands, band_names = getset_feature_model(
                                                                   feature_mod_dict, 
                                                                   feature_model, 
                                                                   spec_indices, 
                                                                   si_vars,
                                                                   spec_indices_pheno,
                                                                   pheno_vars
                                                                   singleton_vars, 
                                                                   poly_vars,
                                                                   combo_bands)
    
    cells = []
    if isinstance(cell_list, list):
        cells = cell_list
    elif cell_list.endswith('.csv'): 
        with open(cell_list, newline='') as cell_file:
            for row in csv.reader(cell_file):
                cells.append (row[0])
    elif isinstance(cell_list, int) or isinstance(cell_list, str): # if runing individual cells as array via bash script
        cells.append(cell_list) 
    
    for cell in cells:
        cell_dir = os.path.join(in_dir,'{:06d}'.format(int(cell)))
        
        # set the path for the temporary output files prior to final stacking
        if scratch_dir:
            out_dir = os.path.join(scratch_dir,'{}'.format(cell))
        else:
            out_dir = os.path.join(cell_dir,'comp')        
        
        #stack_path = os.path.join(cell_dir,'comp','{}_{}_stack.tif'.format(feature_model,start_yr))
        stack_path = os.path.join(cell_dir,'comp','stack.tif')
        #if os.path.isfile(stack_path):
        #    sys.stderr.write('stack file already exists for model {}'.format(feature_model))
        if 1==2:
        else:
            stack_paths = []
            num_bands_all = 0
            sys.stdout.write('making variable stack for cell {}'.format(cell))
            for si in spec_indices:
                img_dir = os.path.join(cell_dir,'brdf_ts','ms',si)
                if os.path.isdir(img_dir):
                    new_bands = make_ts_composite(cell, img_dir, out_dir, start_yr, start_mo, si, si_vars)
                    with rio.open(new_bands) as src:
                        num_bands = src.count
                    if num_bands < len(si_vars):
                        sys.stderr.write('ERROR: not all variables could be calculated for {}'.format(si))
                    else:
                        stack_paths.append(new_bands)
                        sys.stdout.write('Added {} with {} bands \n'.format(si,num_bands))
                else:sys.stderr.write('ERROR: missing spec index: {}'.format(si))        
            
            if len(stack_paths) < len(spec_indices):
                sys.stderr.write('ERROR: did not find ts data for all the requested spec_indices')              
            else:
                num_bands_all = num_bands_all + num_bands
                if singleton_vars is not None and singleton_vars != 'None':
                    ## Clips portion of singleton raster corresponding to gridcell 
                    ## and saves with stack files (if doesn't already exist there)
                    for sf in singleton_vars:
                        with open(singleton_var_dict, 'r+') as singleton_feat_dict:
                            dic = json.load(singleton_feat_dict)
                            if sf in dic: 
                                sf_path = dic[sf]['path']
                                sf_col = dic[sf]['col']
                                sys.stdout.write('getting {} from {}'.format(sf,sf_path))    
                            else:
                                sys.stderr.write('ERROR: do not know path for {}. Add to singleton_var_dict and rerun'.format(sf))
                                sys.exit()

                        singleton_clipped = os.path.join(cell_dir,'comp','{}.tif'.format(sf))
                        if os.path.isfile(singleton_clipped):
                            stack_paths.append(singleton_clipped)
                        else:
                            ## clip large singleton raster to extent of other rasters in stack for grid cell
                            small_ras = stack_paths[0]
                            src_small = gdal.Open(small_ras)
                            ulx, xres, xskew, uly, yskew, yres  = src_small.GetGeoTransform()
                            lrx = ulx + (src_small.RasterXSize * xres)
                            lry = uly + (src_small.RasterYSize * yres)
                            geometry = [[ulx,lry], [ulx,uly], [lrx,uly], [lrx,lry]]
                            roi = [Polygon(geometry)]
                            with rio.open(small_ras) as src0:
                                out_meta = src0.meta.copy()
                                out_meta.update({"count":1})
                            with rio.open(sf_path) as src:
                                out_image, transformed = rio.mask.mask(src, roi, crop = True)
                            with rio.open(singleton_clipped, 'w', **out_meta) as dst:
                                dst.write(out_image)
                            stack_paths.append(singleton_clipped)
                        num_bands_all = num_bands_all + 1
                if poly_vars is not None and poly_vars != 'None':
                    sys.stdout.write('getting poly variables... \n')
                    for pv in poly_vars:
                        poly_path = os.path.join(poly_var_path,'{}_{}.tif'.format(pv,cell))
                        if os.path.isfile(poly_path):
                            ## pred_area is in m2 with vals too big for stack datatype. Rescale:
                            if pv == 'pred_area':
                                with rio.open(poly_path) as src:
                                    vals = src.read([1])
                                    profile = src.profile
                                    if(profile['dtype']) == 'float64':
                                        scaled_vals = vals / 100
                                        profile.update(dtype = 'uint16')
                                        new_area_file = os.path.join(out_dir,"pred_area_scaled.tif")
                                        with rio.open(new_area_file, mode="w",**profile) as new_area:
                                            new_area.write(scaled_vals)
                                stack_paths.append(new_area_file)
                            else:
                                stack_paths.append(poly_path)
                            num_bands_all = num_bands_all + 1
                        else:
                            sys.stderr.write('variable {} does not exist for cell {}'.format(pv,cell))
                                      
                sys.stdout.write('Final stack will have {} bands \n'.format(num_bands_all))
                sys.stdout.write('band names = {} \n'.format(band_names))
                sys.stdout.write('making variable stack... \n')

                output_count = 0
                indexes = []
                for path in stack_paths:
                    #sys.stdout.write('reading from {} \n'.format(path))
                    with rio.open(path, 'r') as src:
                        src_indexes = src.indexes
                        indexes.append(src_indexes)
                        output_count += len(src_indexes)

                with rio.open(stack_paths[0],'r') as src0:
                    kwargs = src0.meta
                    kwargs.update(count = output_count)
                
                with rio.open(os.path.join(cell_dir,'comp','{}_{}_stack.tif'.format(feature_model,start_yr)),'w',**kwargs) as dst:
                #with rio.open(os.path.join(cell_dir,'comp','stack.tif'),'w',**kwargs) as dst:
                    for path, index in zip(stack_paths, indexes):
                        with rio.open(path) as src:
                            if isinstance(index, int):
                                data = src.read(index)
                                dst.write(data, dst_idx)
                                dst_idx += 1
                            elif isinstance(index, Iterable):
                                data = src.read(index)
                                dst.write(data, range(dst_idx, dst_idx + len(index)))
                                dst_idx += len(index)
                    dst.descriptions = tuple(band_names)
            print('done writing {}_{}_stack.tif for cell {}'.format(feature_model,start_yr,cell))
            return stack_path        
                
def classify_raster(var_stack,rf_path,class_img_out):

    img_ds = gdal.Open(var_stack, gdal.GA_ReadOnly)

    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
    for b in range(img.shape[2]):
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    img_as_array = img[:, :, :np.int(img.shape[2])].reshape(new_shape)

    sys.stdout.write('Reshaped from {o} to {n} \n'.format(o=img.shape, n=img_as_array.shape))

    img_as_array = np.nan_to_num(img_as_array)
    
    rf = load(rf_path) #this load is from joblib -- careful if there are other packages with 'Load' module

    try:
        class_prediction = rf.predict(img_as_array)
    except MemoryError:
        slices = int(round(len(img_as_array)/2))

        test = True

        while test == True:
            try:
                class_preds = list()

                temp = rf.predict(img_as_array[0:slices+1,:])
                class_preds.append(temp)

                for i in range(slices,len(img_as_array),slices):
                    sys.stdout.write('{} %, derzeit: {} \n'.format((i*100)/(len(img_as_array)), i))
                    temp = rf.predict(img_as_array[i+1:i+(slices+1),:])
                    class_preds.append(temp)

            except MemoryError as error:
                slices = slices/2
                sys.stdout.write('Not enought RAM, new slices = {} \n'.format(slices))

            else:
                test = False
        else:
            sys.stdout.write('Class prediction was successful without slicing! \n')

    # concatenate all slices and re-shape it to the original extent
    try:
        class_prediction = np.concatenate(class_preds,axis = 0)
    except NameError:
        sys.stdout.write('No slicing was necessary! \n')

    print(class_prediction)
    class_prediction = class_prediction.reshape(img[:, :, 0].shape)
    sys.stdout.write('Reshaped back to {} \n'.format(class_prediction.shape))

    cols = img.shape[1]
    rows = img.shape[0]

    class_prediction.astype(np.float16)

    driver = gdal.GetDriverByName("gtiff")
    outdata = driver.Create(class_img_out, cols, rows, 1, gdal.GDT_UInt16)
    outdata.SetGeoTransform(img_ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(img_ds.GetProjection())##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(class_prediction)
    outdata.FlushCache() ##saves to disk!!
    sys.stdout.write('Image saved to: {}'.format(class_img_out))
         
def rf_model(df_in, out_dir, lc_mod, importance_method, ran_hold, model_name, lut):
    if isinstance(df_in, pd.DataFrame):
        df = df_in
    else:
        df = pd.read_csv(df_in)
    class_col,lut = get_class_col(lc_mod,lut)
    print('class_col = {}'.format(class_col))
    if '{}_name'.format(class_col) in df.columns:
        df2 = df
    else:
        df2 = df.merge(lut[['USE_NAME','{}'.format(class_col),'{}_name'.format(class_col)]], left_on='Class',right_on='USE_NAME', how='left')
    train, ho = prep_test_train(df2, out_dir, class_col, model_name)
    rf = multiclass_rf(train, out_dir, model_name, lc_mod, importance_method, ran_hold, lut)
    score = get_holdout_scores(ho, rf[0], class_col, out_dir)
    
    return rf, score

def rf_classification(in_dir, cell_list, df_in, feature_model, start_yr, start_mo, samp_mod_name, feature_mod_dict, singleton_var_dict, rf_mod, img_out, spec_indices=None, si_vars=None, spec_indices_pheno=None, pheno_vars=None, singleton_vars=None, poly_vars=None, poly_var_path=None, combo_bands=None, lc_mod=None, lut=None, importance_method=None, ran_hold=29, out_dir=None, scratch_dir=None):
    
    spec_indices,si_vars,spec_indices_pheno,pheno_vars,singleton_vars,poly_vars,band_names = getset_feature_model(
                                                                  feature_mod_dict,
                                                                  feature_model,
                                                                  spec_indices,
                                                                  si_vars,
                                                                  spec_indices_pheno,
                                                                  pheno_vars,
                                                                  singleton_vars,
                                                                  poly_vars,
                                                                  combo_bands)
    
    model_name = '{}_{}_{}'.format(feature_model, samp_mod_name,start_yr)
    
    cells = []
    if isinstance(cell_list, list):
        cells = cell_list
    elif cell_list.endswith('.csv'): 
        with open(cell_list, newline='') as cell_file:
            for row in csv.reader(cell_file):
                cells.append (row[0])
    elif isinstance(cell_list, int) or isinstance(cell_list, str): # if runing individual cells as array via bash script
        cells.append(cell_list) 
                
    for cell in cells:            
    # make variable stack if it does not exist (for example for cells without sample pts)
    # -- will not be remade if a file named {feature_model}_{start_year}_stack.tif already exists in ts_dir/comp
        var_stack = make_variable_stack(in_dir,cell,feature_model,start_yr,start_mo,spec_indices,si_vars,feature_mod_dict,
                        singleton_vars=None, singleton_var_dict=None, poly_vars=None, poly_var_path=None, scratch_dir=None)
        cell_dir = os.path.join(in_dir,'{:06d}'.format(int(cell)))
        if img_out == None:
            class_img_out = os.path.join(cell_dir,'comp','{}.tif'.format(model_name))
        else:
            class_img_out = img_out
        if rf_mod != None and os.path.isfile(rf_mod):
            sys.stdout.write('using existing model... \n')
            classify_raster(var_stack, rf_mod, class_img_out)
        else:
            sys.stdout.write('creating rf model... \n')
            rf_mod = rf_model(df_in, out_dir, lc_mod, importance_method, ran_hold, model_name, lut)
            classify_raster(var_stack, rf_mod,class_img_out)