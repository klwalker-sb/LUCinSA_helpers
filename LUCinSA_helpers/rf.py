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
import geowombat as gw
import xarray as xr
import dask
from tqdm import tqdm
from contextlib import ExitStack
import osgeo  # needed only if running from Windows
from shapely.geometry import Polygon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
#import seaborn as sn
from joblib import dump, load
import joblib
import csv
from LUCinSA_helpers.ts_composite import make_ts_composite
from LUCinSA_helpers.pheno import make_pheno_vars

## Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

def get_class_col(lc_mod,lut):
    if lc_mod == 'all':
        class_col = 'LC25'
    elif lc_mod == 'max':
        class_col = 'LC32'
    elif lc_mod == "trans_cats":
        class_col = 'LCTrans'
    elif lc_mod == 'cropNoCrop':
        class_col = 'LC2'
    elif lc_mod == 'crop_nocrop_mixcrop':
        class_col = 'LC3sm'
    elif lc_mod == 'crop_nocrop_medcrop':
        class_col = 'LC3'
    elif lc_mod == 'crop_nocrop_medcrop_tree':
        class_col = 'LC4'
    elif lc_mod == 'veg_with_crop':
        class_col = 'LC8'
    elif lc_mod == 'veg_with_cropType':
        class_col = 'LC10'
    elif lc_mod == 'veg':
        class_col = 'LC5'
    elif lc_mod == 'cropType':
        class_col = 'LC_crops'
    elif lc_mod.startswith('single'):
        lc_base =  lc_mod.split('_')[1].lower()
        target_class = lut.index[lut['USE_NAME'].map(lambda s: lc_base in s.lower())].to_list()
        if len(target_class) == 0:
            print(f'no match found for {lc_base} in lut \n')
        elif len(target_class) > 1:
            sys.stderr.write(f'there are more than one entries with {lc_base} in USE_NAME column of lut \n')
        else:
            class_col = 'LC1'
            if class_col not in lut.columns:
                sys.stderr.write(f'making new virtual {lc_base} column in lut \n')
                lut['LC1'] = 0
                lut['LC1_name'] = f'no_{lc_base}'
                lut.at[target_class[0],'LC1_name'] = lc_base
    else:
        sys.stderr.write(f'current options for lc_mod are: all, LCTrans, cropNoCrop, crop_nocrop_mixcrop, crop_nocrop_medcrop, crop_nocrop_medcrop_tree, veg, veg_with_crop, veg_with_cropType, cropType, and single_X with X as any category. You put {lc_mod} \n')
    
    '''
    elif lc_mod == 'crop_post':
        #where prediction == 1...
        df_train = df_train[df_train['LC2'] == 1]
        print(f'there are {df_train.shape[0]} sample points after removing non-crop \n')
        y = df_train['LC_UNQ']
        #TODO: need two step prediction methods
    elif lc_mod == 'nocrop_post':
        df_train = df_train[df_train['LC2'] == 0]
        print(f'there are {df_train.shape[0]} sample points after removing crop \n')
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
        print(f'There are {str(len(df_low))} LOWVeg records \n')
        df_low.loc[df_low['USE_NAME'] == Class,'label'] = 1
        X = df_low[vars_RF]
        y = df_low['label']
    elif lc_mod == 'LowVeg_Lev2':
        df_low = df_train[df_train['Class1'] == 'LowVeg']
        X = df_low[rf_vars]
        y = df_low['USE_NAME']
    '''
        
    return class_col,lut

def balance_training_data(lut, pixdf, out_dir, cutoff, mix_factor):
    '''
    balances class samples based on map proportion, relative to sample size for class with max map proportion
    (this estimated map proportion is a column named "perLC25E" in the LUT )
    allows a minimum threshold to be set {cutoff} so that sample sizes are not reduced below the minimum
    allows a factor to be set for mixed (heterogeneous) classes to sample them more heavily than main classes
        (the maximum value will depend on the available samples for these classes. Current max is ~12)
    '''
    if isinstance(lut, pd.DataFrame):
        lut = lut
    else:
        lut=pd.read_csv(lut)
    if isinstance(pixdf, pd.DataFrame):
        pixdf = pixdf
    else:
        pixdf = pd.read_csv(pixdf)
        
    if 'LC25_name' not in pixdf:
        pixdf = pixdf.merge(lut[['USE_NAME','LC25','LC25_name','perLC25E']], left_on='Class', right_on='USE_NAME', how='left')

    ## get estimated class percents (this is estimated from other maps and in column in LUT)
    classprev = lut.sort_values('perLC25E')[["perLC25E", "LC25_name"]]
    classprev= classprev.dropna()
    classprev = classprev.drop_duplicates(subset = "perLC25E") 
    tot = classprev["perLC25E"].sum()
    # print(tot)     # should be 1
   
    ## get highest class percent
    mmax = classprev["perLC25E"].max()
    ## convert all class percents to ratio of highest
          ## (keeping all samples from mmax class (n), n = mmax * TotalSamp  -> TotalSamp = n/mmax )
    classprev["perLC25E"] = classprev["perLC25E"]/mmax
    ## get number of samples for each class
    counts = pixdf['LC25_name'].value_counts().reset_index()
    counts.columns = ['LC25_name', 'counts']
    print(counts)
    print(f'Total sample size before balancing is: {sum(counts["counts"])}')
    
    ratiodf = classprev.merge(counts, left_on="LC25_name", right_on="LC25_name", how='left')
    maxsamp = ratiodf.at[ratiodf['perLC25E'].idxmax(), 'counts']
    print(f'samp size for class with max proportion is {maxsamp}')
    ##  Get resample ratio based on class proportion. if existing samples are < cutoff, keep all samples. 
    ##    Otherwise reduce according to proportion, but do not allow to go below cutoff
    ratiodf['ratios'] = np.where(ratiodf["counts"] < cutoff, 1, 
                          np.where(ratiodf["perLC25E"] * maxsamp < ratiodf["counts"], 
                             np.maximum((cutoff / ratiodf["counts"]), (ratiodf["perLC25E"] * maxsamp / ratiodf["counts"])),   
                            1))
    
    ## Allow for separate treatment of mixed classes based on mix_factor
    mixed_classes = ["Mixed-VegEdge", "Mixed-path", "Crops-mix"]
    ratiodf['ratios'] = np.where(ratiodf["LC25_name"].isin(mixed_classes), (ratiodf['ratios'] * mix_factor), ratiodf['ratios'])
    
    ## Use random column to select samples (already in df here, for easy reproduction, but could make new ran column from 0-1)
    pixdf_ratios_rebal = pixdf.merge(ratiodf[['LC25_name','ratios']],left_on="LC25_name", right_on="LC25_name", how='left')
    pixdf_ratios_rebal = pixdf_ratios_rebal[pixdf_ratios_rebal['rand'] < pixdf_ratios_rebal['ratios']]
    print(pixdf_ratios_rebal['LC25_name'].value_counts())
    totsamp = sum(pixdf_ratios_rebal['LC25_name'].value_counts())
    print(f'Total sample size after balancing is: {totsamp}')
    
    pixdf_path = os.path.join(out_dir,f'pixdf_bal{cutoff}mix{mix_factor}.csv')
    pd.DataFrame.to_csv(pixdf_ratios_rebal, pixdf_path)
    
    return pixdf_ratios_rebal

    
def separate_holdout(training_pix_path, holdout_field_pix_path, out_dir):
    '''
    USE THIS WHEN WE AUGMENT BY POLYGON
    Generates separate pixel databases for training data and 20% field-level holdout
    Use this instead of generate_holdout() to fit a model to an exsisting holdout set
       to avoid having points from the same polygon in both the training and holdout sets (as this would inflate accuracy)
    '''
    #training_pix_path2 = os.path.join(out_dir,'V4_Model_training_FieldLevel_toTrainNH.csv')
    #pd.DataFrame.to_csv(training_pixels, training_pix_path, sep=',', na_rep='NaN', index=False)
    #holdout_field_pix_path = os.path.join(out_dir,'V4_Model_testing_FieldLevel_Holdout_FullFieldx.csv')
    #pd.DataFrame.to_csv(pixels_holdouts, holdout_field_pix_path, sep=',', na_rep='NaN', index=False)
    
    holdout_set = pd.read_csv(holdout_field_path)
    pixels = pd.read_csv(training_pix_path)
    
    ## if there is no 'field_id' in the pixel dataset, use the following two lines (but now 'field_id' is aready in pixels)
    #holdout_set['unique_id'] = holdout_set['unique_id'].apply(str)
    #pixels['field_id'] = pixels['pixel_id'].str[:10]
    
    pixels_holdouts = pixels[pixels.field_id.isin(holdout_set['unique_id'])]
    pixels_holdouts['set']='HOLDOUT'
    pixels_training = pixels[~pixels.field_id.isin(holdout_set['unique_id'])]
    pixels_training['set']='TRAINING'

    print(f"original training set had {len(pixels)} rows. Current training set has {len(pixels_training)} rows and holdout has {len(pixels_holdouts)} rows. \n")
    
    return training_pix_path, holdout_field_pix_path

def get_confusion_matrix(pred_col, obs_col, class_lut, lc_mod_map, lc_mod_acc, print_cm=False, out_dir=None, model_name=None):
    '''
    returns confusion matrix with optional regrouping of classes based on LUT 
    classification schema and class columns defined in get_class_col
    '''
    if isinstance(class_lut, pd.DataFrame):
        lut = class_lut
    else:
        lut = pd.read_csv(class_lut)
    
    cmdf = pd.DataFrame()
    cmdf['pred'] = pred_col
    cmdf['obs'] = obs_col
    
    if lc_mod_map.startswith('single'):
        cm=pd.crosstab(cmdf['pred'],cmdf['obs'],margins=True)
    else: 
        map_cat = get_class_col(lc_mod_map,lut)[0]
        acc_cat = get_class_col(lc_mod_acc,lut)[0]
        print(f'getting confusion matrix based on {acc_cat}...')
        cmdf2 = cmdf.merge(lut[['LC_UNQ',f'{acc_cat}_name']], left_on='obs', right_on='LC_UNQ',how='left')
        cmdf2.rename(columns={f'{acc_cat}_name':'obs_reclass'}, inplace=True)
        cmdf2.drop(['LC_UNQ'],axis=1,inplace=True)
        cmdf3 = cmdf2.merge(lut[['LC_UNQ', f'{acc_cat}_name']], left_on='pred', right_on='LC_UNQ',how='left')
        cmdf3.rename(columns={f'{acc_cat}_name':'pred_reclass'}, inplace=True)
        cmdf3.drop(['LC_UNQ'],axis=1,inplace=True)
        cm=pd.crosstab(cmdf3['pred_reclass'],cmdf3['obs_reclass'],margins=True)
    cm['correct'] = cm.apply(lambda x: x[x.name] if x.name in cm.columns else 0, axis=1)
    cm['sumcol'] = cm.apply(lambda x: cm.loc['All', x.name] if x.name in cm.columns else 0)
    cm['UA'] = (cm['correct']/cm['All']).round(3)
    cm['PA'] = (cm['correct']/cm['sumcol']).round(3)
    cm['F1'] = ((2 * cm['UA'] * cm['PA'])/(cm['UA'] + cm['PA'])).round(3)
    cm['F_5'] = ((1.5 * cm['UA'] * cm['PA'])/(.5 * cm['UA'] + cm['PA'])).round(3)
    cm['F_25'] = ((1.25 * cm['UA'] * cm['PA'])/(.25 * cm['UA'] + cm['PA'])).round(3)
    total = cm.at['All','correct']
    cm.at['All','UA'] = ((cm['correct'].sum() - total) / total).round(3)
    cm.at['All','PA'] = ((cm['correct'].sum() - total) / total).round(3)
    if acc_cat == 'LC2':
        cm.at['All','F1']=cm.at['crop','F1']
        TP = cm.at['crop', 'crop']
        FP = cm.at['crop', 'nocrop']
        FN = cm.at['nocrop', 'crop']
        TN = cm.at['nocrop','nocrop']
        All = TP + FP + FN + TN
        cm['Kappa'] = (2*(TP*TN - FN*FP)/((TP+FP)*(FP+TN)+(TP+FN)*(FN+TN))).round(3)
    
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
        mod_path = os.path.join(out_dir,f'{model_name}_{lc_mod_acc}.csv')
        pd.DataFrame.to_csv(cm, mod_path, sep=',', index=True)
    
    return cm

def quick_accuracy(X_test, y_test, rf_model, lc_mod, out_dir,model_name,lut):
    
    predicted = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    sys.stderr.write(f'Out-of-bag score estimate: {rf_model["forest"].oob_score_:.3} \n')
    sys.stderr.write(f'Mean accuracy score: {accuracy:.3} \n')
    
    cm = get_confusion_matrix(predicted, y_test,lut, lc_mod, lc_mod, out_dir,model_name,lut)                    
    
    return accuracy, cm

def smallag_acc(rf_scores, lut, cutoff='1ha',print_cm=False,out_dir=None,model_name=None):
    '''
    gets confusion matrix for subset of data with just smallholder fields and noncrop classified as mixed (smallholder) ag 
    '''
    smallag = rf_scores.loc[(rf_scores[f'smalls_{cutoff}'] == 1) | (rf_scores['pred'] == 35)]
    sys.stderr.write('There are {smallag.shape[0]} {cutoff} smallholder samples in the holdout.')
    smcm = get_confusion_matrix(smallag['pred'], smallag['label'],lut,'all','cropNoCrop',False,None,None)

    if print_cm == True:
        mod_path = os.path.join(out_dir,f'{model_name}_smallholderAcc_{cutoff}.csv')
        pd.DataFrame.to_csv(smcm, mod_path, sep=',', index=True)
    
    return smcm

def wave(cm_path, metric, weights = False):

    if weights == False:
        weights = np.ones(30)

    elif weights == "CT":
        # CropType = Corn, NoCrop, Rice, ShrubCrop, Smallholder, Soybeans, Sugar, TreeCrops, All (All is not correct in current cms)
        weights = [1, 1, 1, 1, 2, 1, 1, 1, 0]
        
    cm = pd.read_csv(cm_path)

    cm['weighted'] = cm.apply(lambda row: row[metric] * weights[row.name], axis=1)
    cmpos = cm[cm['weighted']>0]
    score = cmpos['weighted'].sum() / cmpos.shape[0] 

    return score

def overall_wave(cnc_metrics, ct_path, model_name,pixdf):
    # Read the cropNoCrop matrix and extract values
    cnc = pd.read_csv(cnc_metrics, index_col = 0)
    cnc_partial = cnc[cnc["Model"] == model_name]

    # now calculate weighted values from the cropType matrix:
    #ct = pd.read_csv(ct_path, index_col = 0)
    ct_partial = [wave(ct_path, metric='UA', weights = "CT"), 
                  wave(ct_path, metric='PA', weights = "CT"), 
                  wave(ct_path, metric='F1', weights = "CT")]

    overall_metrics = pd.DataFrame({"Model": [f"{model_name}"],
                             "UA": [(2 * cnc_partial["UA"].values[0] + ct_partial[0])/3],
                             "PA": [(2 * cnc_partial["PA"].values[0] + ct_partial[1])/3],
                             "F1": [(2 * cnc_partial["F1"].values[0] + ct_partial[2])/3],
                             "1ha_UA": [cnc_partial["1ha_UA"].values[0]],
                             "1ha_PA": [cnc_partial["1ha_PA"].values[0]],
                             "1ha_F1": [cnc_partial["1ha_F1"].values[0]],
                             "halfha_UA": [cnc_partial["halfha_UA"].values[0]],       
                             "halfha_PA": [cnc_partial["halfha_PA"].values[0]],
                             "halfha_F1": [cnc_partial["halfha_F1"].values[0]],
                             "Num_obs": [pixdf["LC25_name"].shape[0]],
                             "Num_sm_1ha" : [pixdf['smlhld_1ha'].sum()],                    
                             "Num_sm_halfha" : [pixdf['smlhd_halfha'].sum()]})
    return overall_metrics

    
def build_weighted_accuracy_table(out_dir,model_name,rf_out,df_in,lut, binary=False, second_cm=False, ho_path=None):
    '''
    take the cms and find take the averages of the accuracies and F1 scores
    '''
    if isinstance(df_in, pd.DataFrame):
        pixdf = df_in
    else:
        pixdf = pd.read_csv(df_in)
        
    metrics_dir = os.path.join(out_dir,'metrics')
    os.makedirs(metrics_dir, exist_ok=True) 
    
    if binary == True:
        types = ["cropNoCrop"]
    else:
        types = ["cropNoCrop", "cropType", "veg", "all"] 
    
    for idx, i in enumerate(types):
        sub_dir = os.path.join(out_dir,f'{i}_cms')
        os.makedirs(sub_dir, exist_ok=True) 
        cm = get_confusion_matrix(rf_out[1]['pred'],rf_out[1]['label'],lut,'all',i,print_cm=True,out_dir=sub_dir,model_name=model_name) 
        cmsm1ha = smallag_acc(rf_out[1], lut, cutoff='1ha',print_cm=True,out_dir=sub_dir,model_name=model_name)
        cmsmhalfha = smallag_acc(rf_out[1], lut, cutoff='halfha',print_cm=True,out_dir=sub_dir,model_name=model_name)
        
        ## Get metrics from printed confusion matrices:
        if i == 'cropNoCrop':
            if  second_cm == True:
                mc_correct = get_mixed_crop_holdout_score(ho_path, rf_out[0], out_dir, lut)
            else:
                mc_correct = 0
            print(f'CropNoCrop cm: {cm} \n')
            metricsi = pd.DataFrame({"Model": [f'{model_name}'],
                         #"UA": [wave(os.path.join(sub_dir,f'{model_name}_{i}.csv'), metric='UA')],
                         "UA": [cm.loc['crop','UA']],
                         "PA": [cm.loc['crop','PA']],
                         "F1": [cm.loc['crop', 'F1']],
                         "MixedCrop": [mc_correct],            
                         "1ha_UA": [wave(os.path.join(sub_dir, f'{model_name}_smallholderAcc_1ha.csv'), metric='UA')],
                         "1ha_PA": [wave(os.path.join(sub_dir, f'{model_name}_smallholderAcc_1ha.csv'), metric='PA')],
                         "1ha_F1": [wave(os.path.join(sub_dir, f'{model_name}_smallholderAcc_1ha.csv'), metric='F1')],
                         "halfha_PA": [wave(os.path.join(sub_dir, f'{model_name}_smallholderAcc_halfha.csv'), metric='PA')],
                         "halfha_UA": [wave(os.path.join(sub_dir, f'{model_name}_smallholderAcc_halfha.csv'), metric='UA')],
                         "halfha_F1": [wave(os.path.join(sub_dir, f'{model_name}_smallholderAcc_halfha.csv'), metric='F1')],
                         "Num_obs": [pixdf["LC25_name"].shape[0]],
                         "Num_sm_1ha" : [pixdf['smlhld_1ha'].sum()],                    
                         "Num_sm_halfha" : [pixdf['smlhd_halfha'].sum()]})
            
        else:
            metricsi = pd.DataFrame({"Model": [f'{model_name}'],
                         "UA": [wave(os.path.join(sub_dir,f'{model_name}_{i}.csv'), metric='UA')],
                         "PA": [wave(os.path.join(sub_dir,f'{model_name}_{i}.csv'), metric='PA')],
                         "F1": [wave(os.path.join(sub_dir,f'{model_name}_{i}.csv'), metric='F1')],
                         "Num_obs": [pixdf["LC25_name"].shape[0]]}) 
            #print(metricsi)
        
        metrics_path = os.path.join(metrics_dir,f'{i}_metrics.csv')
        if os.path.isfile(metrics_path) == False:
        ## If the output csv files do not already exist, create them:
            metricsi.to_csv(metrics_path)
        else:
            ## Get existing metrics and add new info:
            stored_metrics = pd.read_csv(metrics_path, index_col = 0)
            metrics_appended = pd.concat([stored_metrics.reset_index(drop = True), metricsi.reset_index(drop = True)], ignore_index = True)
            metrics_appended.to_csv(metrics_path)
            #print(metrics_appended)

    if binary == False:
        all_metrics_path = os.path.join(metrics_dir,'overall_metrics.csv')
        overall = overall_wave(os.path.join(metrics_dir,'cropNoCrop_metrics.csv'), 
             os.path.join(out_dir,'cropType_cms',f'{model_name}_cropType.csv'),model_name, pixdf)
        if os.path.isfile(all_metrics_path) == False:
            overall.to_csv(all_metrics_path)
            all_stored_new = overall                           
        else:
            #print(overall)
            all_stored = pd.read_csv(all_metrics_path, index_col = 0)
            all_stored_new = pd.concat([all_stored.reset_index(drop = True), overall.reset_index(drop = True)], ignore_index = True)
            all_stored_new.to_csv(all_metrics_path)          
    else:
        all_stored_new = metrics_appended
    print(all_stored_new)
    return all_stored_new

def get_stable_holdout(df_in, out_dir, thresh, ho_type, balanced_input=False, lut=None, overwrite=False):
    '''
    same holdout used for multiple models
    ho_type = smallCrop | bigCrop | noCrop
    '''
    if isinstance(df_in, pd.DataFrame):
        pixdf = df_in
    else:
        pixdf = pd.read_csv(df_in, index_col=0)
    sys.stderr.write(f'there are {pixdf.shape[0]} pts in the full data set \n')
   
    cutoff = (100 - thresh) / 100
    tr_path = os.path.join(out_dir,f'GENERAL_TRAINING.csv')
    
    if ho_type == 'smallCrop':
        sys.stderr.write('getting holdout for small crops... \n')
        pixdf_ho = pixdf[(pixdf['rand2']>cutoff) & (pixdf['LC25_name'] == 'Crops-mix')]
        sys.stderr.write(f'there are {len(pixdf_ho)} pixels in the mixed crop holdout \n')
        ho_path = os.path.join(out_dir,'GENERAL_HOLDOUT_smallCrop.csv')

    elif ho_type == 'bigCrop':
        sys.stderr.write('getting holdout for big crops... \n')
        pixdf2 = pixdf[(pixdf['rand2']>cutoff) & (pixdf['LC25_name'] != 'Crops-mix') & (pixdf['LC3_name'] == 'crop')]
        pixdf_ho = pixdf2[~(pixdf2['FieldWidth']<= 100)]
        sys.stderr.write(f'there are {len(pixdf_ho)} pixels in the big crop holdout \n')
        ho_path = os.path.join(out_dir,'GENERAL_HOLDOUT_bigCrop.csv')
        
    elif ho_type == 'noCrop':
        ## No crop is a representational sample, created with a balanced dataset with no oversampling of any class
        sys.stderr.write('getting holdout for no crop...\n')
        if balanced_input == True:
            pixdf_bal = pixdf
        else:
            pixdf_bal = balance_training_data(lut, pixdf, out_dir, 0, 1)
            
        pixdf_ho = pixdf_bal[(pixdf_bal['LC2_name'] == 'nocrop') & (pixdf_bal['rand2']>cutoff)] 
        sys.stderr.write(f'there are {len(pixdf_ho)} pixels in the no crop holdout \n')
        ho_path = os.path.join(out_dir,'GENERAL_HOLDOUT_noCrop.csv')
    
    ## make training set from remaining records
    pixdf_tr = pixdf[~pixdf.index.isin(pixdf_ho.index)]
    print(f'there are {len(pixdf_tr)} pixels left in the in the training set \n')
          
    pd.DataFrame.to_csv(pixdf_tr, tr_path, sep=',', na_rep='NaN', index=True)
    if (overwrite == True) or (os.path.isfile(ho_path) == False):
        pd.DataFrame.to_csv(pixdf_ho, ho_path, sep=',', na_rep='NaN', index=True)

    print(f'holdout sets are in {out_dir} \n')
    return ho_path, tr_path
         
def prep_test_train(df_in, out_dir, class_col, mod_name, thresh=20, stable=True):
    '''
    gets general holdout as a % {thresh} of all pixels that could be used in training
    this is called within rf_mod, so use stable=True to base HO off stable column to compare with other models 
    '''
    sys.stderr.write(f'df_in = {df_in}\n')
    if isinstance(df_in, pd.DataFrame):
        df_in = df_in
    else:
        df_in = pd.read_csv(df_in, index_col=0)
    sys.stderr.write(f'there are {df_in.shape[0]} pts in the full data set \n')
   
    # remove unknown and other entries where class is not specified a given level (i.e. crop_low if crop type is desired)
    df_in = df_in[df_in[class_col] < 98]
    #sys.stderr.write(f'there are {df_in.shape[0]} sample points after removing those without clear class. \n')
    
    #Separate training and holdout datasets to avoid confusion with numbering
    training_pix_path = os.path.join(out_dir,f'{mod_name}_TRAINING.csv')
    holdout_pix_path = os.path.join(out_dir,f'{mod_name}_HOLDOUT.csv')
    if int(thresh) == 0:
        df_train = df_in  
        df_test = None
    elif stable==True:
        ## TODO: base this on ran2 column and thresh
        df_train = df_in[df_in['TESTSET20'] == 0]
        df_test = df_in[df_in['TESTSET20'] == 1]
    else:
        ## TODO: make new ran column to get new HO each time             
        print('finish this')
    
    pd.DataFrame.to_csv(df_train, training_pix_path, sep=',', na_rep='NaN', index=False)
    
    if df_test is not None:
        pd.DataFrame.to_csv(df_test, holdout_pix_path, sep=',', na_rep='NaN', index=False)

    return(training_pix_path, holdout_pix_path)

def multiclass_rf(trainfeatures, out_dir, mod_name, lc_mod, importance_method, ran_hold, lut,runnum):
    
    df_train = pd.read_csv(trainfeatures)
    #sys.stderr.write(f'There are {df_train.shape[0]} training samples \n')

    class_col = get_class_col(lc_mod,lut)[0]
    y = df_train[class_col]
           
    vars_rf = [col for col in df_train if col.startswith('var_')]
    #sys.stderr.write(f'There are {len(vars_rf)} training features \n')
    X = df_train[vars_rf]
    
    #print (pd.Series(y).value_counts())
    X = X.values
    y = y.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = .1, random_state=ran_hold)

    ## wrap sklearn models in Pipeline to ensure they create the same predictions when loaded cold
    ##    was getting very different results without this whenever models were reloaded!
    rf_model = Pipeline([('forest', RandomForestClassifier(n_estimators=500, oob_score=True))])
    rf_model.fit(X_train,y_train)
    
    if runnum == 0:
        rfmodname = f'{mod_name}mod.joblib'
    else:
        rfmodname = f'{mod_name}mod{runnum}.joblib'
        
    dump(rf_model, os.path.join(out_dir,rfmodname))

    cm = quick_accuracy (X_test, y_test, rf_model, lc_mod, out_dir,mod_name,lut)

    if importance_method != None:
        if importance_method == "Impurity":
            var_importances = pd.Series(rf_model["forest"].feature_importances_, index=vars_rf)
            pd.Series.to_csv(var_importances,os.path.join(out_dir,f'VarImportance_{mod_name}.csv'),sep=',',index=True)
        elif importance_method == "Permutation":
            result = permutation_importance(rf_model, X_test, y_test, n_repeats=10,random_state=ran_hold, n_jobs=2)
            var_importances = pd.Series(result.importances_mean, index=vars_rf)
            pd.Series.to_csv(var_importances,os.path.join(out_dir,f'VarImportance_{mod_name}.csv'),sep=',', index=True)

    return rf_model, cm

def get_holdout_scores(holdoutpix, rf_model, class_col, out_dir,class_type=None):
    ## Save info for extra columns and drop (model is expecting only variable input columns)
    
    if isinstance(holdoutpix, pd.DataFrame):
        holdout_pix = holdoutpix
        holdout_pix.reset_index(drop=True, inplace=True)
    else:
        holdout_pix = pd.read_csv(holdoutpix)
    
    if 'entry_lev' in list(holdout_pix.columns):
        ## filter to remove less confident entries
        #holdout_pix = holdout_pix[(holdout_pix['entry_lev'] == 4) | (holdout_pix['source'].isin(['ground','GE']))]
        holdout_pix = holdout_pix[(holdout_pix['entry_lev'] > 1)]
        if class_type=='noCrop':
            ## remove mixed fields from no-crop test set, as this is ambiguous
            holdout_pix = holdout_pix[(holdout_pix['LC'] != 19) & (holdout_pix['smlhld_1ha'] == 0)]
        holdout_pix.reset_index(drop=True, inplace=True)
        
    holdout_labels = holdout_pix[class_col]
    h_IDs = holdout_pix['OID_']
    #sys.stderr.write(f'number of holdout pixels = {len(holdout_pix)} \n')
    
    ## Get list of variables to include in model:
    vars = [col for col in holdout_pix if col.startswith('var_')]
    
    holdout_fields = holdout_pix[vars]

    ## Calculate scores
    #holdout_fields_predicted = rf_model.predict_proba(holdout_fields)
    holdout_fields_predicted = rf_model.predict(holdout_fields)
    
    #holdout_fields_predicted.to_csv('~/data/ho_p_check.csv')
    
    ## Add extra columns back in
    holdout_fields = pd.concat([holdout_fields,pd.Series(holdout_fields_predicted),holdout_labels,h_IDs],axis=1)
    new_cols = [-3,-2,-1]
    new_names = ["pred","label","OID"]
    old_names = holdout_fields.columns[new_cols]
    holdout_fields.rename(columns=dict(zip(old_names, new_names)), inplace=True)
    if class_type=='noCrop':
        ## remove mixed fields from no-crop test set, as this is ambiguous
        holdout_fields = holdout_fields[(holdout_fields['label'] != 19)]

    ## Print to file
    if class_type:
        out_file = f'Holdout_predictions_{class_type}.csv'
    else:
        out_file = 'Holdout_predictions.csv'
        
    pd.DataFrame.to_csv(holdout_fields, os.path.join(out_dir,out_file), sep=',', na_rep='NaN', index=True)
   
    return holdout_fields

def get_AUC(holdoutpix, rf_model):
    '''
    Note this only works with 2-class model
    '''
                 
    from sklearn.metrics import roc_curve, auc
                 
    if isinstance(holdoutpix, pd.DataFrame):
        holdout_pix = holdoutpix
    else:
        holdout_pix = pd.read_csv(holdoutpix)
        
    ho_labels = holdout_pix['LC2']
    ho_IDs = holdout_pix['OID_']
    
    ## Isolate feature set for prediction:
    vars = [col for col in holdout_pix if col.startswith('var_')]
    holdouts = holdout_pix[vars]

    ## Calculate scores
    ho_scores = rf_model.predict_proba(holdouts)[:,1]
    #ho_scores = rf_model.predict_proba(holdouts)
    fpr, tpr, thresholds = metrics.roc_curve(ho_labels, ho_scores, pos_label=30)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, thresholds = precision_recall_curve(ho_labels, ho_scores, pos_label=30)
    precision_recall_auc = auc(recall, precision)
           
                 
    return roc_auc, precision_recall_auc

def get_binary_holdout_score(ho_path, rf_model, out_dir, lut, class_type):
    '''
    Note, this is a specific holdout set where all entries are mixed crop.
    '''
    # We are interested in binary prediciton, here, so use 'LC2' for class
    mcho_score = get_holdout_scores(ho_path, rf_model, 'LC2', out_dir, class_type)
    ## Need to rejoin with LUT and get L2 class if using any other classification system
    lut2 = pd.read_csv(lut)
    accdf = mcho_score.merge(lut2[['LC_UNQ','LC2']], left_on='pred', right_on='LC_UNQ',how='left')
    # for LC2, 0 = noCrop and 30 = Crop (but in LUT noCrop is now 98, so adjust here)
    accdf['LC2'] = np.where(accdf['LC2'] == 30, 30, 0)
   
    if class_type == 'noCrop':
        num_correct = len(accdf) - (accdf['LC2'].sum() / 30)
    else:
        num_correct = accdf['LC2'].sum() / 30
    
    per_correct = (num_correct / len(accdf)).round(3)
    
    return per_correct
                     
def getset_feature_model(feature_mod_dict,feature_model,spec_indices=None,si_vars=None,spec_indices_pheno=None,pheno_vars=None,
                         singleton_vars=None,poly_vars=None, combo_bands=None):
    
    with open(feature_mod_dict, 'r+') as feature_model_dict:
        dic = json.load(feature_model_dict)
        if feature_model in dic:
            spec_indices = dic[feature_model]['spec_indices']
            si_vars = dic[feature_model]['si_vars']
            spec_indices_pheno = dic[feature_model]['spec_indices_pheno']
            pheno_vars = dic[feature_model]['pheno_vars']
            singleton_vars = dic[feature_model]['singleton_vars']
            poly_vars = dic[feature_model]['poly_vars']
            combo_bands = dic[feature_model]['combo_bands']
            band_names = dic[feature_model]['band_names']
            print(f'using existing model: {feature_model} \n spec_indices = {spec_indices} \n si_vars = {si_vars} \n pheno_vars = {pheno_vars} on {spec_indices_pheno} \n singleton_vars={singleton_vars} \n poly_vars = {poly_vars} \n ')
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
            if spec_indices is not None and spec_indices != 'None':
                for si in spec_indices:
                    for sv in si_vars:
                        band_names.append(f'{si}_{sv}')
            if combo_bands is not None and combo_bands != 'None':
                for cb in combo_bands:
                    band_names.append(cb)
            if spec_indices_pheno is not None and spec_indices_pheno != 'None':
                for sip in spec_indices_pheno:
                    for pv in pheno_vars:
                        band_names.append(f'{sip}_{pv}')
            if singleton_vars is not None and singleton_vars != 'None':
                for sin in singleton_vars:
                    band_names.append(f'sing_{sin}')
            if poly_vars is not None and poly_vars != 'None':       
                for pv in poly_vars:
                    band_names.append(f'poly_{pv}')

            dic[feature_model]['band_names'] = band_names
            with open(feature_mod_dict, 'w') as new_feature_model_dict:
                json.dump(dic, new_feature_model_dict)
            print(f'created new model: {feature_model} \n spec_indices={spec_indices} \n si_vars={si_vars} \n pheno_vars={pheno_vars} on {spec_indices_pheno} \n singleton_vars={singleton_vars} \n poly_vars={poly_vars} \n combo_bands={ combo_bands} \n')
        
    return spec_indices,si_vars,spec_indices_pheno,pheno_vars,singleton_vars,poly_vars,combo_bands,band_names
    
def make_variable_stack(in_dir,cell_list,feature_model,start_yr,start_mo,spec_indices,si_vars,spec_indices_pheno,pheno_vars,
                        feature_mod_dict, singleton_vars=None, singleton_var_dict=None, poly_vars=None, 
                        poly_var_path=None, combo_bands=None, scratch_dir=None, out_dir=None, overwrite=False):
    
    # get model paramaters if model already exists in dict. Else create new dict entry for this model
    spec_indices, si_vars, spec_indices_pheno, pheno_vars, singleton_vars, poly_vars, combo_bands, band_names = getset_feature_model(
                                                                   feature_mod_dict, 
                                                                   feature_model, 
                                                                   spec_indices, 
                                                                   si_vars,
                                                                   spec_indices_pheno,
                                                                   pheno_vars,
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
    else:
        sys.stderr.write (f'ERR: Problem parsing {cell_list} as cell list. Needs to be list, .csv, or single int or string')
        
    for cell in cells:
        sys.stderr.write(f'working on cell: {cell}.... \n')
        cell_dir = os.path.join(in_dir,'{:06d}'.format(int(cell)))
        
        # set the path for the temporary output files prior to final stacking
        if out_dir:
            out_dir = os.path.join(out_dir,'{:06d}'.format(int(cell)),'comp')
            if scratch_dir:
                temp_dir = os.path.join(scratch_dir,'{:06d}'.format(int(cell)),'comp')
            else:
                temp_dir = out_dir
        else:
            if scratch_dir:
                out_dir = os.path.join(scratch_dir,'{:06d}'.format(int(cell)),'comp')
            else:
                out_dir = os.path.join(cell_dir,'comp')
            temp_dir = out_dir

        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        sys.stderr.write(f'making dir {temp_dir} \n')
        
        stack_exists = False
        stack_path = os.path.join(out_dir,f'{feature_model}_{start_yr}_stack.tif')
        if os.path.isfile(stack_path):
            stack_exists = True
            #sys.stderr.write(f'stack file already exists for model {feature_model} \n')
        ## Can build noPoly models from existing stacks containing polys
        elif 'NoPoly' not in feature_model:
            no_poly_model = feature_model.replace('Poly','NoPoly')
            alt_path = os.path.join(out_dir,f'{no_poly_model}_{start_yr}_stack.tif')
            if os.path.isfile(alt_path):
                stack_exists = True
                #sys.stderr.write(f'no poly stack file already exists for model {no_poly_model} \n')
        keep_running = True
        if stack_exists == True:
            sys.stderr.write('stack file already exists. \n')
            if overwrite == False:
                keep_running = False
        if keep_running == True:
            stack_paths = []
            band_names = []
            sys.stderr.write(f'making variable stack for cell {cell} \n')

            for si in spec_indices:
                ## check if all spec indices exist before going on:
                img_dir = os.path.join(cell_dir,'brdf_ts','ms',si)
                
                if not os.path.exists(img_dir):
                    sys.stderr.write(f'ERROR: missing spec index: {si} \n')
                    return False
                
            for si in spec_indices:
                    si_dir_out = os.path.join(temp_dir, si) 
                    img_dir = os.path.join(cell_dir,'brdf_ts','ms',si)
                    new_vars, new_bands = make_ts_composite(cell, img_dir, si_dir_out, start_yr, start_mo, si, si_vars)
                    try:
                        with rio.open(new_vars) as src:
                            num_bands = src.count
                    except:    
                        sys.stderr.write(f'ERROR: there is a problem with the time series for {si} \n')
                        return False

                    if num_bands < len(si_vars):
                        sys.stderr.write(f'ERROR: not all variables could be calculated for {si} \n')
                        return False
                    
                    else:
                        stack_paths.append(new_vars)
                        for b in new_bands:
                            new_band_name = f'{si}_{b}'
                            band_names.append(new_band_name)
                        sys.stderr.write(f'Added {si} with {num_bands} bands \n')
                       
            if len(stack_paths) < len(spec_indices):
                sys.stderr.write('ERROR: did not find ts data for all the requested spec_indices \n')
            else:
                if pheno_vars is not None and pheno_vars != 'None':
                    sys.stderr.write('getting pheno variables... \n')
                    for psi in spec_indices_pheno:
                        #try:
                        img_dir = os.path.join(cell_dir,'brdf_ts','ms',psi)
                        psi_dir_out = os.path.join(temp_dir, psi) 
                        new_pheno_vars, pheno_bands = make_pheno_vars(
                            cell,img_dir,psi_dir_out,start_yr,start_mo,psi,pheno_vars,500,[30,0])
                        stack_paths.append(new_pheno_vars)
                        for pb in pheno_bands:
                            new_band_name = f'{psi}_{pb}'
                            band_names.append(new_band_name)
                        #except Exception as e:
                        #    sys.stderr.write(f'ERROR: {e} \n')
                if singleton_vars is not None and singleton_vars != 'None':
                    ## Clips portion of singleton raster corresponding to gridcell 
                    ## and saves with stack files (if doesn't already exist there)
                    for sf in singleton_vars:
                        with open(singleton_var_dict, 'r+') as singleton_feat_dict:
                            dic = json.load(singleton_feat_dict)
                            if sf in dic: 
                                sf_path = dic[sf]['path']
                                sf_col = dic[sf]['col']
                                sys.stdout.write(f'getting {sf} from {sf_path}')    
                            else:
                                sys.stderr.write(f'ERROR: do not know path for {sf}. Add to singleton_var_dict and rerun \n')
                                sys.exit()

                        singleton_clipped = os.path.join(cell_dir,'comp',f'{sf}.tif')
                        if os.path.isfile(singleton_clipped):
                            stack_paths.append(singleton_clipped)
                            band_names.append(sf)
                        else:
                            comp_dir = os.path.join(cell_dir,'comp')
                            os.makedirs(comp_dir, exist_ok=True)
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
                            band_names.append(f'sing_{sf}')

                if poly_vars is not None and poly_vars != 'None':
                    sys.stdout.write('getting poly variables... \n')
                    for pv in poly_vars:
                        if pv.startswith('poly'):
                            stored_name = pv.replace('poly','pred')
                            poly_path = os.path.join(poly_var_path,f'{stored_name}_{cell}.tif')
                        else:
                            poly_path = os.path.join(poly_var_path,f'{pv}_{cell}.tif')
                        poly_comp_path = os.path.join(poly_var_path,f'pred_PY_{cell}.tif')
                        if os.path.isfile(poly_path):
                            #if pv in ['pred_APR','poly_APR']:
                            #    poly_path = os.path.join(poly_var_path,f'pred_APRef_{cell}.tif')
                            #    stack_paths.append(poly_path)
                            #    band_names.append(pv)
                            #else:
                            sys.stdout.write(f'adding {poly_path} to stack for var {pv} \n')
                            stack_paths.append(poly_path)
                            band_names.append(pv)
                        elif pv in ['NovDecGCVI_Std','poly_NovDecStd','NovDecStd']:
                            alt_poly_path = os.path.join(poly_var_path,f'AvgNovDec_FieldStd_{cell}.tif')
                            if os.path.isfile(alt_poly_path):
                                stack_paths.append(alt_poly_path)
                                band_names.append(pv)
                        elif os.path.isfile(poly_comp_path):
                            if pv in ['pred_ext','poly_ext']:
                                with rio.open(poly_comp_path) as src:
                                    vals = src.read([3])
                                    profile = src.profile
                                    profile.update(count = 1)
                                    new_file = os.path.join(temp_dir,"pred_ext.tif")
                                    with rio.open(new_file, mode="w",**profile) as new_b:
                                        new_b.write(vals)
                                stack_paths.append(new_file)
                                band_names.append(pv)           
                            elif pv in ['pred_dst','poly_dst']:
                                with rio.open(poly_comp_path) as src:
                                    vals = src.read([1])
                                    profile = src.profile
                                    profile.update(count = 1)
                                    new_file = os.path.join(temp_dir,"pred_dst.tif")
                                    with rio.open(new_file, mode="w",**profile) as new_b:
                                        new_b.write(vals)
                                stack_paths.append(new_file)
                                band_names.append(pv)    
                            elif pv in ['pred_cropbnds','poly_cropbnds']:
                                with rio.open(poly_comp_path) as src:
                                    vals = src.read([2])
                                    profile = src.profile
                                    profile.update(count = 1)
                                    new_file = os.path.join(temp_dir,"pred_bnds.tif")
                                    with rio.open(new_file, mode="w",**profile) as new_b:
                                        new_b.write(vals)
                                stack_paths.append(new_file)
                                band_names.append(pv)       
                        else:   
                            sys.stderr.write(f'variable {pv} does not exist for cell {cell} \n')
                            ## Write stack without poly variables, but change name to specify
                            if 'Poly' in feature_model and 'NoPoly' not in feature_model:
                                nop_mod = feature_model.replace('Poly', 'NoPoly')
                                stack_path = os.path.join(out_dir,f'{nop_mod}_{start_yr}_stack.tif')
                                poly_vars = None
                                      
                sys.stdout.write(f'Final stack will have {band_names} bands \n')
                sys.stdout.write('making variable stack... \n')
                #sys.stdout.write(f'All paths are {stack_paths} \n')

                output_count = 0
                indexes = []
                for path in stack_paths:
                    if 'RFVars' in os.path.basename(path) or 'Phen' in os.path.basename(path):
                        with rio.open(path, 'r') as src:
                            src_indexes = src.indexes
                            #sys.stdout.write(f'got indices {src.indexes} for path {path} \n')
                            indexes.append(src_indexes)
                            output_count += len(src_indexes)
                    else:
                        indexes.append(1)
                        output_count += 1       
                #sys.stdout.write(f'final indexes: {indexes} \n')

                with rio.open(stack_paths[0],'r') as src0:
                    profile = src0.profile
                    profile.update(count = output_count) 
                
                dst_idx = 1
                with rio.open(stack_path,'w',**profile) as dst:
                    for path, index in zip(stack_paths, indexes):
                        with rio.open(path) as src:
                            sys.stdout.write(f'inserting {path} at index {dst_idx} \n')
                            if isinstance(index, int):
                                data = src.read(index)
                                data_profile = src.profile
                                sys.stdout.write(str(data_profile))
                                dst.write(data, dst_idx)
                                dst_idx += 1
                            elif isinstance(index, Iterable):
                                #sys.stdout.write(f'inserting {path} at index {dst_idx} \n')
                                data = src.read(index)
                                dst.write(data, range(dst_idx, dst_idx + len(index)))
                                dst_idx += len(index)

                    #print(f'new stack has {dst_idx - 1} bands \n')
                    #print(f'we have {len(band_names)} band names \n')
                    dst.descriptions = tuple(band_names)
                print(f'done writing {os.path.basename(stack_path)} with {len(band_names)} bands for cell {cell} \n')     
        
def get_predictions_gdal(var_stack,rf_path,class_img_out):
    '''
    This is old method to apply random forest model to full raster using gdal.
    This is no longer in use because the xarray / geowombat method below facilitates better memory management
       and allows more flexibility in adjusting the stack input to the model
       To use this method, var_stack must be in the exact same order as the variable inputs to the random forest model
    '''
    img_ds = gdal.Open(var_stack, gdal.GA_ReadOnly)

    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
    '''
    ## Cut last 5 variables off of image stack if stack contains polygon data and using NoPoly model
    if 'NoPoly' in os.path.basename(rf_path):
        if 'Poly' in os.path.basename(var_stack) and 'NoPoly' not in os.path.basename(var_stack):
            num_bands = img.shape[2]-5
        else:
            num_bands = img.shape[2]
    '''
    num_bands = img.shape[2]
    
    for b in range(num_bands):
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

    new_shape = (img.shape[0] * img.shape[1], num_bands)
    img_as_array = img[:, :, :np.int(num_bands)].reshape(new_shape)

    sys.stdout.write(f'Reshaped from {img.shape} to {img_as_array.shape} \n')

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
                    sys.stdout.write(f'{(i*100)/(len(img_as_array))} %, derzeit: {i} \n')
                    temp = rf.predict(img_as_array[i+1:i+(slices+1),:])
                    class_preds.append(temp)

            except MemoryError as error:
                slices = slices/2
                sys.stdout.write(f'Not enought RAM, new slices = {slices} \n')

            else:
                test = False
        else:
            sys.stdout.write('Class prediction was successful without slicing! \n')

    # concatenate all slices and re-shape it to the original extent
    try:
        class_prediction = np.concatenate(class_preds,axis = 0)
    except NameError:
        sys.stdout.write('No slicing was necessary! \n')

    #sys.stdout.write(class_prediction)
    return(class_prediction)

    class_prediction = class_prediction.reshape(img[:, :, 0].shape)
    sys.stdout.write(f'Reshaped back to {class_prediction.shape} \n')
    
    cols = img.shape[1]
    rows = img.shape[0]

    class_prediction.astype(np.float16)

    driver = gdal.GetDriverByName("gtiff")
    outdata = driver.Create(class_img_out, cols, rows, 1, gdal.GDT_UInt16)
    outdata.SetGeoTransform(img_ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(img_ds.GetProjection())##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(class_prediction)
    outdata.FlushCache() ##saves to disk!!  

def get_predictions_gw(saved_stack, model_bands, rf_path, class_img_out):
    '''
    apply random forest model to full raster using xarray with geowombat wrapper for named bands and wondowed operations
    
    Parameters
    ----------
    saved_stack: path to multiband geotiff containing a band for each model variable
        The bands need to have names that match the model variables
        The stack can have extra bands; only those used in the model will be used. Likewise, the order of the 
        bands in the file does not matter, as it will be rearanged to match the model here.
    model_bands: The ordered bands used in the model. 
        This can be retrieved from 'band_names' in the feature model dictionary
    rf_path: The path to the .joblib file with the random forest model information
    class_img_out:  The path for the classified output image
    '''
    sys.stderr.write('getting predictions...\n')
    rf = load(rf_path) #this load is from joblib -- careful if there are other packages with 'Load' module
    
    chunks=256
    with rio.open(saved_stack) as src0:
        profile = dict(blockxsize=chunks,
            blockysize=chunks,
            crs=src0.crs,
            transform=src0.transform,
            driver='GTiff',
            height=src0.height,
            width=src0.width,
            nodata=0,
            count=1,
            dtype='uint8',
            compress='lzw',
            tiled=True)
        
    ## reduce stack bands to match model variables, ensuring same order as model df 
    with gw.open(saved_stack) as src0:
        #sys.stdout.write(f'{src0.attrs} \n')
        stack_bands = src0.attrs['descriptions']
        sys.stdout.write(f'bands in stack: {stack_bands} \n')
    bands_out = []
    band_names = []
    for b in model_bands:
        #sys.stdout.write(f'looking for {b}:')
        found_band = False
        for i, v in enumerate(stack_bands):
            #sys.stdout.write(f'{v}')
            if v == b:
                bands_out.append(i+1)
                band_names.append(v)
                #sys.stdout.write(' -- found! \n')
                found_band = True
                break
            elif b.startswith('sing') and v == b.split('_')[1]:
                bands_out.append(i+1)
                band_names.append(f'sing_{v}')
                #sys.stdout.write(' -- found! \n')
                found_band = True
                break
            elif b.startswith('poly') and v == b.split('_',1)[1]:
                bands_out.append(i+1)
                band_names.append(f'poly_{v}')
                #sys.stdout.write(' -- found! \n')
                found_band = True
                break
            else:
                #sys.stdout.write('...')
                pass
        if found_band == False:
            sys.stderr.write(f'ERROR: band {b} not found in stack \n')
            return False
            
    sys.stdout.write(f'bands used for model: {bands_out}')
    
    new_stack = src0.sel(band=bands_out)
    new_stack.attrs['descriptions'] = band_names
    sys.stdout.write('{}'.format(new_stack.attrs['descriptions']))

    #new_stack = new_stack.chunk({"x": len(new_stack.x), "y": len(new_stack.y)})
 
    with gw.open(saved_stack) as src:
        windows = list(src.gw.windows(row_chunks=chunks, col_chunks=chunks))
    
    for w in tqdm(windows, total=len(windows)):
        #with ExitStack() as stack:
        stackblock = new_stack[:, w.row_off:w.row_off+w.height, w.col_off:w.col_off+w.width]
        
        X = stackblock.stack(s=('y', 'x'))\
            .transpose()\
            .astype('int16')\
            .fillna(0)\
            .data\
            .rechunk((stackblock.gw.row_chunks * stackblock.gw.col_chunks, 1))

        feature_band_count = X.shape[1]
        sys.stdout.write(f'num features in df = {feature_band_count} \n')
            
        X = dask.compute(X, num_workers=4)[0]
    
        class_prediction = rf.predict(X)
        sys.stdout.write(f'class_prediction out = {class_prediction} \n')
        class_out = np.uint8(np.squeeze(class_prediction))
        class_out = class_out.reshape(w.height, w.width)
        sys.stdout.write(f'class_out = {class_out} \n')
    
        if not os.path.isfile(class_img_out):
            with rio.open(class_img_out, mode='w', **profile) as dst:
                pass
        with rio.open(class_img_out, mode='r+') as dst:
            #dst.write(class_out, window=w)
            dst.write(class_out, indexes=1, window=w)
    
    return class_prediction

def rf_model(df_in, out_dir, lc_mod, importance_method, ran_hold, lut, feature_model, samp_model, train_yrs, thresh,
             feature_mod_dict=None, update_model_dict=False, fixed_ho=False, fixed_ho_dir=None, runnum=0, existing_mod=False):

    if isinstance(df_in, pd.DataFrame):
        df = df_in
        #sys.stderr.write('reading in df as database')
    else:
        df = pd.read_csv(df_in)
        #sys.stderr.write(f'reading in df file: {df_in} \n')
        
    class_col,lut = get_class_col(lc_mod,lut)
    #sys.stderr.write(f'class_col = {class_col} \n')
    if f'{class_col}_name' in df.columns:
        df2 = df
    else:
        lutdf = pd.read_csv(lut)
        lut_cols = ['USE_NAME',f'{class_col}',f'{class_col}_name']
        filtered_lut = lutdf.filter(lut_cols)
        df2 = df.merge(filtered_lut, left_on='Class',right_on='USE_NAME', how='left')
        
    if isinstance(train_yrs, int):
        trainyrs = str(train_yrs)[-2:]
    elif len(train_yrs) == 1:
        trainyrs = str(train_yrs[0])[-2:]
    else:
        trainyrs = str(train_yrs[0])[-2:]+str(train_yrs[-1])[-2:]

    full_model_name = f'{feature_model}_{samp_model}_{class_col}_{trainyrs}_RF'
    print(f'model name = {full_model_name}')
        
    if fixed_ho == True:
        test_df_all = [os.path.join(fixed_ho_dir,f) for f in os.listdir(fixed_ho_dir) if 
                       f.split('_')[0] == feature_model and f.split('_')[-2] == 'all' and f.split('_')[-1].startswith(trainyrs)]
        if len(test_df_all)>0:
            test_all = pd.read_csv(test_df_all[0])
            smallcrops = [23,24,25,26,32,34,35,36,39]
            bigcrops = [22,31,33,37,38]
            ho_smallCrop = test_all.loc[test_all['LC_UNQ'].isin(smallcrops)] 
            ho_bigCrop = test_all.loc[(test_all['LC2'] == 30) & (test_all['LC_UNQ'].isin(bigcrops))]
            ho_noCrop = test_all.loc[(test_all['LC2'] == 98) & (test_all['LC_UNQ'] != 19)]
        elif os.path.isfile(os.path.join(fixed_ho_dir,f'{feature_model}_HOLDOUT_smallCrop_{train_yrs}.csv')):
            ho_smallCrop = pd.read_csv(os.path.join(fixed_ho_dir,f'{feature_model}_HOLDOUT_smallCrop_{train_yrs}.csv'))
            ho_bigCrop = pd.read_csv(os.path.join(fixed_ho_dir,f'{feature_model}_HOLDOUT_bigCrop_{train_yrs}.csv')) 
            ho_noCrop = pd.read_csv(os.path.join(fixed_ho_dir,f'{feature_model}_HOLDOUT_noCrop_{train_yrs}.csv'))
        else:
            sys.stderr.write(f'ERR: cannot find fixed test set')
  
    train, ho = prep_test_train(df2, out_dir, class_col, full_model_name, thresh=thresh, stable=True)

    if existing_mod:
        rf = load(existing_mod) #this load is from joblib -- careful if there are other packages with 'Load' module
        print(f'loading existing RFmod from: {existing_mod}')
    else:
        rf = multiclass_rf(train, out_dir, full_model_name, lc_mod, importance_method, ran_hold, lut, runnum)
    
    if update_model_dict == True:
        ## add columns back into feature dict to make sure they are in the right order:
        ordered_vars = [v[4:] for v in df3.columns.to_list() if v.startswith('var')]
        sys.stderr.write(f'there are {len(ordered_vars)} variables in the model \n')
        sys.stdout.write(f'model bands are: {ordered_vars} \n')
    
        with open(feature_mod_dict, 'r+') as feature_model_dict:
            dic = json.load(feature_model_dict)
            dic[feature_model].update({'band_names':ordered_vars})
        with open(feature_mod_dict, 'w') as new_feature_model_dict:
            json.dump(dic, new_feature_model_dict)
    
    if int(thresh) > 0:
        score = get_holdout_scores(ho, rf[0], class_col, out_dir)
        ## add the smallholder indication variables to the output df
        #score = pd.DataFrame(score)
        score["smalls_1ha"] = df["smlhld_1ha"]
        score["smalls_halfha"] = df["smlhd_halfha"]
        pd.DataFrame.to_csv(score, os.path.join(out_dir,f"{full_model_name}_HO_SCORES"), sep=',', na_rep='NaN', index=True)
        
    else:
        score = {}
        score["F"] = full_model_name.split('_')[0]
        score["S"] = full_model_name.split('_')[1]
        score["C"] = class_col
        score["A"] = "RF"
    if fixed_ho == True:
        ho_smallcrop_s = get_holdout_scores(ho_smallCrop, rf[0], 'LC2', out_dir, 'smallCrop')[["pred","label","OID"]]
        ho_bigcrop_s = get_holdout_scores(ho_bigCrop, rf[0], 'LC2', out_dir, 'bigCrop')[["pred","label","OID"]]
        ho_nocrop_s = get_holdout_scores(ho_noCrop, rf[0], 'LC2', out_dir, 'noCrop')[["pred","label","OID"]]
        ho = pd.concat([ho_smallcrop_s,ho_bigcrop_s,ho_nocrop_s])
        ho.to_csv(os.path.join(out_dir, 'full_ho_check.csv'))
        print(ho.head())
        cm = get_confusion_matrix(ho['pred'], ho['label'], lut, lc_mod, 'cropNoCrop', print_cm=False, out_dir=None, model_name=None)
        print(cm)
        score["recall_smallCrop"] = get_binary_holdout_score(ho_smallCrop, rf[0], out_dir, lut, 'smallCrop')
        score["recall_bigCrop"] = get_binary_holdout_score(ho_bigCrop, rf[0], out_dir, lut, 'bigCrop')
        score["recall_noCrop"] =  get_binary_holdout_score(ho_noCrop, rf[0], out_dir, lut, 'noCrop')
        score["Kappa_cnc"] = cm.at['crop','Kappa']
        score["F1_cnc"] = cm.at['crop','F1']
        score["F_5_cnc"] = cm.at['crop','F_5']
        score["F_25_cnc"] = cm.at['crop','F_25']        
        score["OA_cnc"] = cm.at['All','UA']
        if len(train_yrs)>1:
            for yr in range(train_yrs[0],train_yrs[1]+1):
                print(f'getting scores for {yr}')
                score[f'recall_smallCrop_{yr}'] = get_binary_holdout_score(ho_smallCrop[ho_smallCrop['year']==yr], rf[0], out_dir, lut, 'smallCrop')
                score[f'recall_bigCrop_{yr}'] = get_binary_holdout_score(ho_bigCrop[ho_bigCrop['year']==yr], rf[0], out_dir, lut, 'bigCrop')
                score[f'recall_noCrop_{yr}'] = get_binary_holdout_score(ho_noCrop[ho_noCrop['year']==yr], rf[0], out_dir, lut, 'noCrop')
        
    return rf, score

def rf_classification(cell_dir, cell_list, mod_dir, feature_model, samp_model, lc_mod, train_yrs, start_yr, start_mo, feature_mod_dict, 
                      rf_mod=None, img_out=None, df_in=None, singleton_var_dict=None, spec_indices=None, si_vars=None, 
                      spec_indices_pheno=None, pheno_vars=None, singleton_vars=None, poly_vars=None, poly_var_path=None, 
                      combo_bands=None, lut=None, importance_method=None, ran_hold=29, out_dir=None, scratch_dir=None):
    
    spec_indices,si_vars,spec_indices_pheno,pheno_vars,singleton_vars,poly_vars,combo_bands,band_names = getset_feature_model(
                     feature_mod_dict,feature_model, spec_indices, si_vars, spec_indices_pheno, pheno_vars, singleton_vars, poly_vars, combo_bands)
    
    ## derive model name from components
    lc_mod = 'max'
    class_col = get_class_col(lc_mod,lut)[0]
    if isinstance(train_yrs, int):
        trainyrs = str(train_yrs)[-2:]
    elif len(train_yrs) == 1:
        trainyrs = str(train_yrs[0])[-2:]
    else:
        trainyrs = str(train_yrs[0])[-2:]+str(train_yrs[1])[-2:] 
    model_name_train = f'{feature_model}_{samp_model}_{class_col}_{trainyrs}_RF'
    model_name_class = f'{feature_model}_{samp_model}_{class_col}_{trainyrs}_RF_{start_yr}'
    
    cells = []
    if isinstance(cell_list, list):
        cells = cell_list
    elif isinstance(cell_list, str) and cell_list.endswith('.csv'): 
        with open(cell_list, newline='') as cell_file:
            for row in csv.reader(cell_file):
                cells.append(row[0])
    elif isinstance(cell_list, int) or isinstance(cell_list, str): # if runing individual cells as array via bash script
        cells.append(cell_list) 
    in_dir = cell_dir
    
    for cell in cells:
        sys.stderr.write(f'working on cell {cell}... \n')
        cell_dir = os.path.join(cell_dir,'{:06d}'.format(int(cell)))
        
        stack_path = os.path.join(cell_dir,'comp',f'{feature_model}_{start_yr}_stack.tif')
        sys.stderr.write(f'looking for stack: {stack_path}... \n')
        if os.path.isfile(stack_path):
            sys.stderr.write(f'stack file already exists for model {feature_model} \n')
            var_stack = stack_path
        
        elif 'NoPoly' in feature_model:
            ## Can make a noPoly model with a Poly stack
            poly_model = feature_model.replace('NoPoly','Poly6')
            alt_path = os.path.join(cell_dir, 'comp', f'{poly_model}_{start_yr}_stack.tif')
            if os.path.isfile(alt_path):
                sys.stderr.write(f'poly stack file already exists for model {poly_model} \n')
                var_stack = alt_path
                ##TODO:Use grep for this to accept other versions of poly model (e.g. Poly4) 
        else:
            # make variable stack if it does not exist (for example for cells without sample pts)
            # -- will not be remade if a file named {feature_model}_{start_year}_stack.tif already exists in ts_dir/comp
            sys.stderr.write(f'writing variable stack for model {feature_model} ...\n') 
            var_stack = make_variable_stack(in_dir,cell,feature_model,start_yr,start_mo,spec_indices,si_vars,spec_indices_pheno,
                                        pheno_vars,feature_mod_dict,singleton_vars=None, singleton_var_dict=None, 
                                        poly_vars=None, poly_var_path=None, scratch_dir=None)
        
        if (img_out is None) or (img_out == 'None'):
            class_img_out = os.path.join(cell_dir,'comp','{:06d}_{}.tif'.format(int(cell),model_name_class))
        else:
            class_img_out = img_out

        ## Try to use existing rf model. By default, will use rf model named f'{feature_model}_{samp_mod_name}_{class_col}_{trainyrs}_RFmod.joblib'
        ##    but can force use of a different existing model with rf_mod
        if (rf_mod != None) and (rf_mod !='None'):
            if os.path.isfile(os.path.join(mod_dir, rf_mod)):
                rf_mod = os.path.isfile(os.path.join(mod_dir, rf_mod))
                sys.stderr.write(f'using existing rf model at:{rf_mod} \n')
            elif os.path.isfile(rf_mod):
                rf_mod = rf_mod
                sys.stderr.write(f'using existing rf model at:{rf_mod} \n')
            else:
                sys.stderr.write(f'cannot find existing model {rf_mod} specified with rf_mod parameter. Set rf_mod to None to create new model \n')
        else:
            default_model = os.path.join(mod_dir, f'{model_name_train}mod.joblib')
            if os.path.isfile(default_model):
                rf_mod = default_model
                sys.stderr.write(f'using existing rf model at {rf_mod} \n')
            else:
                sys.stderr.write(f'Could not find existing model at{default_model}. Creating new rf model... \n')
                rf_mod = rf_model(df_in, mod_dir, lc_mod, importance_method, ran_hold, lut, feature_model, samp_model, train_yrs, 0, feature_mod_dict)

        with open(feature_mod_dict, 'r+') as feature_model_dict:
            dic = json.load(feature_model_dict)
            model_bands = dic[feature_model]['band_names']
            sys.stdout.write(f'model bands from dict: {model_bands}: \n')
            sys.stdout.flush()
            
        ## Old gdal-based method:
        #class_prediction = get_predictions_gdal(var_stack,rf_path)
        ## geowombat / xarray method:
        class_prediction = get_predictions_gw(var_stack, model_bands, rf_mod, class_img_out)
        
        if class_prediction is not None:
            sys.stdout.write(f'Image saved to: {class_img_out} \n')    
        else:
            sys.stdout.write('got an error \n')
    return None
