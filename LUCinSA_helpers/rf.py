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
#import seaborn as sn
from joblib import dump, load
import csv
from LUCinSA_helpers.ts_composite import make_ts_composite
from LUCinSA_helpers.pheno import make_pheno_vars


## Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

def get_class_col(lc_mod,lut):
    if lc_mod == 'all':
        class_col = 'LC25'
    elif lc_mod == "trans_cats":
        class_col = 'LCTrans'
    elif lc_mod == 'cropNoCrop':
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
        print('current options for lc_mod are all, LCTrans, LC2, LC3, LC4, LC5, LC_crops and single_X with X as any category. You put {}'.format(lc_mod))
    
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

def balance_training_data(lut, pixdf, out_dir, new_name, cutoff, mix_factor):
    '''
    balances class samples based on map proportion, relative to sample size for class with max map proportion
    (this estimated map proportion is a column named "perLC25E" in the LUT )
    allows a minimum threshold to be set {cutoff} so that sample sizes are not reduced below the minimum
    allows a factor to be set for mixed (heterogeneous) classes to sample them more heavily than main classes
        (the maximum value will depend on the available samples for these classes. Current max is ~4)
    '''
    ##   first collect the sample ratios from the LUT
    lut=pd.read_csv(lut)
    ordered = lut.sort_values('perLC25E')[["perLC25E", "LC25_name"]]
    #print(ordered)
    
    ##   clean this df for NaN and repeats, check if the percents add up correctly, then divide as integer
    
    ordered = ordered.dropna()
    ordered = ordered.drop_duplicates(subset = "perLC25E") 
    #print(ordered)
    
    tot = ordered["perLC25E"].sum()
    # print(tot)     # should be close to 1
    
    ##   rescale by dividing by min proportion
    mmax = ordered["perLC25E"].max()
    ordered["perLC25E"] = ordered["perLC25E"]/mmax
    #print(ordered)
    
    ##  get sample counts for each class 
    counts = pixdf['LC25_name'].value_counts().rename_axis("LC25_name").reset_index(name="counts")
    #print(counts)
    print(f'Total sample size before balancing is: {sum(counts["counts"])}')
    
    ## join sample counts with scaled class proportions
    ratiodf = ordered.merge(counts, left_on="LC25_name", right_on="LC25_name", how='left')
    print(ratiodf)
    maxsamp = ratiodf.at[ratiodf['perLC25E'].idxmax(), 'counts']
    print(f'samp size for class with max proportion is {maxsamp}')
    ## get resample ratio based on class proportion 
    ratiodf['ratios'] = np.where(ratiodf["counts"] < cutoff, 1, 
                          np.where(ratiodf["perLC25E"] * maxsamp < ratiodf["counts"], 
                             np.maximum((cutoff / ratiodf["counts"]), (ratiodf["perLC25E"] * maxsamp / ratiodf["counts"])),   
                            1))
    mixed_classes = ["Mixed-VegEdge", "Mixed-path", "Crops-mix"]
    ratiodf['ratios'] = np.where(ratiodf["LC25_name"].isin(mixed_classes), (ratiodf['ratios'] * mix_factor), ratiodf['ratios'])
    
    pixdf_ratios_rebal = pixdf.merge(ratiodf[['LC25_name','ratios']],left_on="LC25_name", right_on="LC25_name", how='left')
    pixdf_ratios_rebal = pixdf_ratios_rebal[pixdf_ratios_rebal['rand'] < pixdf_ratios_rebal['ratios']]
    print(pixdf_ratios_rebal['LC25_name'].value_counts())
    totsamp = sum(pixdf_ratios_rebal['LC25_name'].value_counts())
    print(f'Total sample size after balancing is: {totsamp}')
    
    pixdf_path = os.path.join(out_dir,'pixdf_bal{}mix{}.csv'.format(cutoff,mix_factor))
    pd.DataFrame.to_csv(pixdf_ratios_rebal, pixdf_path)
    
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
    cm['F1'] = (2 * cm['UA'] * cm['PA'])/(cm['UA'] + cm['PA'])
 
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
        mod_path = os.path.join(out_dir,'{}_{}.csv'.format(model_name,lc_mod_acc))
        pd.DataFrame.to_csv(cm, mod_path, sep=',', index=True)
    
    return cm

def quick_accuracy(X_test, y_test, rf_model, lc_mod, out_dir,model_name,lut):
    
    predicted = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    print(f'Out-of-bag score estimate: {rf_model.oob_score_:.3}')
    print(f'Mean accuracy score: {accuracy:.3}')
    
    cm = get_confusion_matrix(predicted, y_test,lut, lc_mod, lc_mod, out_dir,model_name,lut)                    
    
    return accuracy, cm

def smallag_acc(rf_scores, lut, cutoff='1ha',print_cm=False,out_dir=None,model_name=None):
    '''
    gets confusion matrix for subset of data with just smallholder fields and noncrop classified as mixed (smallholder) ag 
    '''
    smallag = rf_scores.loc[(rf_scores['smalls_{}'.format(cutoff)] == 1) | (rf_scores['pred'] == 35)]
    print('There are {} {} smallholder samples in the holdout.'.format(smallag.shape[0], cutoff))
    smcm = get_confusion_matrix(smallag['pred'], smallag['label'],lut,'all','cropNoCrop',False,None,None)

    if print_cm == True:
        mod_path = os.path.join(out_dir,'{}_smallholderAcc_{}.csv'.format(model_name,cutoff))
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

    overall_metrics = pd.DataFrame({"Model": ["{}".format(model_name)],
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

def build_weighted_accuracy_table(out_dir,model_name,rf_scores,df_in,lut):
    '''
    take the cms and find take the averages of the accuracies and F1 scores
    '''
    if isinstance(df_in, pd.DataFrame):
        pixdf = df_in
    else:
        pixdf = pd.read_csv(df_in)
        
    metrics_dir = os.path.join(out_dir,'metrics')
    os.makedirs(metrics_dir, exist_ok=True) 
    
    types = ["cropNoCrop", "cropType", "veg", "all"] 
    
    for idx, i in enumerate(types):
        sub_dir = os.path.join(out_dir,'{}_cms'.format(i))
        os.makedirs(sub_dir, exist_ok=True) 
        cm = get_confusion_matrix(rf_scores['pred'],rf_scores['label'],lut,'all',i,print_cm=True,out_dir=sub_dir,model_name=model_name) 
        cmsm1ha = smallag_acc(rf_scores, lut, cutoff='1ha',print_cm=True,out_dir=sub_dir,model_name=model_name)
        cmsmhalfha = smallag_acc(rf_scores, lut, cutoff='halfha',print_cm=True,out_dir=sub_dir,model_name=model_name)
        
        ## Get metrics from printed confusion matrices:
        if i == 'cropNoCrop':
            print('CropNoCrop cm: {}'.format(cm))
            metricsi = pd.DataFrame({"Model": ["{}".format(model_name)],
                         "UA": [wave(os.path.join(sub_dir,'{}_{}.csv'.format(model_name,i)), metric='UA')],
                         "PA": [wave(os.path.join(sub_dir.format(i),'{}_{}.csv'.format(model_name,i)), metric='PA')],
                         "F1": [wave(os.path.join(sub_dir.format(i),'{}_{}.csv'.format(model_name,i)), metric='F1')],
                         "1ha_UA": [wave(os.path.join(sub_dir, '{}_smallholderAcc_1ha.csv'.format(model_name)), metric='UA')],
                         "1ha_PA": [wave(os.path.join(sub_dir, '{}_smallholderAcc_1ha.csv'.format(model_name)), metric='PA')],
                         "1ha_F1": [wave(os.path.join(sub_dir, '{}_smallholderAcc_1ha.csv'.format(model_name)), metric='F1')],
                         "halfha_PA": [wave(os.path.join(sub_dir, '{}_smallholderAcc_halfha.csv'.format(model_name)), metric='PA')],
                         "halfha_UA": [wave(os.path.join(sub_dir, '{}_smallholderAcc_halfha.csv'.format(model_name)), metric='UA')],
                         "halfha_F1": [wave(os.path.join(sub_dir, '{}_smallholderAcc_halfha.csv'.format(model_name)), metric='F1')],
                         "Num_obs": [pixdf["LC25_name"].shape[0]],
                         "Num_sm_1ha" : [pixdf['smlhld_1ha'].sum()],                    
                         "Num_sm_halfha" : [pixdf['smlhd_halfha'].sum()]})
        else:
            metricsi = pd.DataFrame({"Model": ["{}".format(model_name)],
                         "UA": [wave(os.path.join(sub_dir.format(i),'{}_{}.csv'.format(model_name,i)), metric='UA')],
                         "PA": [wave(os.path.join(sub_dir.format(i),'{}_{}.csv'.format(model_name,i)), metric='PA')],
                         "F1": [wave(os.path.join(sub_dir.format(i),'{}_{}.csv'.format(model_name,i)), metric='F1')],
                         "Num_obs": [pixdf["LC25_name"].shape[0]]}) 
            #print(metricsi)
        
        metrics_path = os.path.join(metrics_dir,'{}_metrics.csv'.format(i))
        if os.path.isfile(metrics_path) == False:
        ## If the output csv files do not already exist, create them:
            metricsi.to_csv(metrics_path)
        else:
            ## Get existing metrics and add new info:
            stored_metrics = pd.read_csv(metrics_path, index_col = 0)
            metrics_appended = pd.concat([stored_metrics.reset_index(drop = True), metricsi.reset_index(drop = True)], ignore_index = True)
            metrics_appended.to_csv(metrics_path)
            #print(metrics_appended)

    all_metrics_path = os.path.join(metrics_dir,'overall_metrics.csv')
    overall = overall_wave(os.path.join(metrics_dir,'cropNoCrop_metrics.csv'), 
             os.path.join(out_dir,'cropType_cms','{}_cropType.csv'.format(model_name)),model_name, pixdf)
    if os.path.isfile(all_metrics_path) == False:
        overall.to_csv(all_metrics_path)
        all_stored_new = overall                           
    else:
        #print(overall)
        all_stored = pd.read_csv(all_metrics_path, index_col = 0)
        all_stored_new = pd.concat([all_stored.reset_index(drop = True), overall.reset_index(drop = True)], ignore_index = True)
        all_stored_new.to_csv(all_metrics_path)          
    
    print(all_stored_new)
    return all_stored_new
    
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
            print('using existing model: {} \n spec_indices = {} \n si_vars = {} \n pheno_vars = {} on {} \n singleton_vars={} \n poly_vars = {} \n '
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
            if combo_bands is not None and combo_bands != 'None':
                for cb in combo_bands:
                    band_names.append(cb)
            if spec_indices_pheno is not None and spec_indices_pheno != 'None':
                for sip in spec_indices_pheno:
                    for pv in pheno_vars:
                        band_names.append('{}_{}'.format(sip,pv))
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
            print('created new model: {} \n spec_indices={} \n si_vars={} \n pheno_vars={} on {} \n singleton_vars={} \n poly_vars={} \n combo_bands={} \n'
                  .format(feature_model, spec_indices, si_vars, pheno_vars, spec_indices_pheno, singleton_vars, poly_vars, combo_bands))
        
    return spec_indices,si_vars,spec_indices_pheno,pheno_vars,singleton_vars,poly_vars,combo_bands,band_names
    
def make_variable_stack(in_dir,cell_list,feature_model,start_yr,start_mo,spec_indices,si_vars,spec_indices_pheno,pheno_vars,feature_mod_dict,
                        singleton_vars=None, singleton_var_dict=None, poly_vars=None, poly_var_path=None, combo_bands=None,                                         scratch_dir=None):
    
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
        sys.stdout.write('cell list: {} \n'.format(cells))
    for cell in cells:
        sys.stderr.write('working on cell: {}.... \n'.format(cell))
        cell_dir = os.path.join(in_dir,'{:06d}'.format(int(cell)))
        
        # set the path for the temporary output files prior to final stacking
        if scratch_dir:
            out_dir = os.path.join(scratch_dir,'{}'.format(cell))
        else:
            out_dir = os.path.join(cell_dir,'comp')        
        
        stack_exists = 0
        stack_path = os.path.join(cell_dir,'comp','{}_{}_stack.tif'.format(feature_model,start_yr))
        if os.path.isfile(stack_path):
            stack_exists = 1
            #sys.stderr.write('stack file already exists for model {} \n'.format(feature_model))
        elif 'NoPoly' not in feature_model:
            no_poly_model = feature_model.replace('Poly','NoPoly')
            alt_path = os.path.join(cell_dir,'comp','{}_{}_stack.tif'.format(no_poly_model,start_yr))
            if os.path.isfile(alt_path):
                stack_exists = 1
                #sys.stderr.write('no poly stack file already exists for model {} \n'.format(no_poly_model))
        if stack_exists == 1:
            sys.stderr.write('stack file already exists. \n')
        else:
            stack_paths = []
            band_names = []
            sys.stderr.write('making variable stack for cell {} \n'.format(cell))
            keep_going = True
            for si in spec_indices:
                ## check if all spec indices exist before going on:
                img_dir = os.path.join(cell_dir,'brdf_ts','ms',si)
                if not os.path.exists(img_dir):
                    sys.stderr.write('ERROR: missing spec index: {} \n'.format(si))
                    keep_going = False
            for si in spec_indices:
                if keep_going == True:
                    si_dir_out = os.path.join(out_dir, si) 
                    img_dir = os.path.join(cell_dir,'brdf_ts','ms',si)
                    new_vars, new_bands = make_ts_composite(cell, img_dir, si_dir_out, start_yr, start_mo, si, si_vars)
                    try:
                        with rio.open(new_vars) as src:
                            num_bands = src.count
                    except:    
                        sys.stderr.write('ERROR: there is a problem with the time series for {} \n'.format(si))
                        keep_going = False
                        continue
                    if num_bands < len(si_vars):
                        sys.stderr.write('ERROR: not all variables could be calculated for {} \n'.format(si))
                        keep_going = False
                    else:
                        stack_paths.append(new_vars)
                        for b in new_bands:
                            new_band_name = '{}_{}'.format(si,b)
                            band_names.append(new_band_name)
                        sys.stderr.write('Added {} with {} bands \n'.format(si,num_bands))
                       
            if len(stack_paths) < len(spec_indices):
                sys.stderr.write('ERROR: did not find ts data for all the requested spec_indices \n')
            else:
                if pheno_vars is not None and pheno_vars != 'None':
                    sys.stderr.write('getting pheno variables... \n')
                    for psi in spec_indices_pheno:
                        #try:
                        img_dir = os.path.join(cell_dir,'brdf_ts','ms',psi)
                        psi_dir_out = os.path.join(out_dir, psi) 
                        new_pheno_vars, pheno_bands = make_pheno_vars(
                            cell,img_dir,psi_dir_out,start_yr,start_mo,psi,pheno_vars,500,[30,0])
                        stack_paths.append(new_pheno_vars)
                        for pb in pheno_bands:
                            new_band_name = '{}_{}'.format(psi,pb)
                            band_names.append(new_band_name)
                        #except Exception as e:
                        #    sys.stderr.write('ERROR: {} \n'.format(e))
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
                                sys.stderr.write('ERROR: do not know path for {}. Add to singleton_var_dict and rerun \n'.format(sf))
                                sys.exit()

                        singleton_clipped = os.path.join(cell_dir,'comp','{}.tif'.format(sf))
                        if os.path.isfile(singleton_clipped):
                            stack_paths.append(singleton_clipped)
                            band_names.append(sf)
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
                            band_names.append('sing_{}'.format(sf))

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
                                        scaled_vals = vals * 100
                                        profile.update(dtype = 'uint16')
                                        new_area_file = os.path.join(out_dir,"pred_area_scaled.tif")
                                        with rio.open(new_area_file, mode="w",**profile) as new_area:
                                            new_area.write(scaled_vals)
                                stack_paths.append(new_area_file)
                                band_names.append(pv)
                            else:
                                stack_paths.append(poly_path)
                                band_names.append(pv)
                        else:
                            sys.stderr.write('variable {} does not exist for cell {} \n'.format(pv,cell))
                            ## Write stack without poly variables, but change name to specify
                            if 'Poly' in feature_model and 'NoPoly' not in feature_model:
                                nop_mod = feature_model.replace('Poly', 'NoPoly')
                                stack_path = os.path.join(cell_dir,'comp','{}_{}_stack.tif'.format(nop_mod,start_yr))
                                poly_vars = None
                                      
                sys.stdout.write('Final stack will have {} bands \n'.format(len(band_names)))
                sys.stdout.write('band names = {} \n'.format(band_names))
                sys.stdout.write('making variable stack... \n')
                #sys.stdout.write('All paths are {} \n'.format(stack_paths))

                output_count = 0
                indexes = []
                for path in stack_paths:
                    if 'RFVars' in os.path.basename(path) or 'Phen' in os.path.basename(path):
                        with rio.open(path, 'r') as src:
                            src_indexes = src.indexes
                            #sys.stdout.write('got indices {} for path {} \n'.format(src.indexes, path))
                            indexes.append(src_indexes)
                            output_count += len(src_indexes)
                    else:
                        indexes.append(1)
                        output_count += 1       
                #sys.stdout.write('final indexes: {} \n'.format(indexes))

                with rio.open(stack_paths[0],'r') as src0:
                    kwargs = src0.meta
                    kwargs.update(count = output_count)
                
                dst_idx = 1
                with rio.open(stack_path,'w',**kwargs) as dst:
                    for path, index in zip(stack_paths, indexes):
                        with rio.open(path) as src:
                            if isinstance(index, int):
                                data = src.read(index)
                                dst.write(data, dst_idx)
                                dst_idx += 1
                            elif isinstance(index, Iterable):
                                sys.stdout.write('inserting {} at index {} \n'.format (path,dst_idx))
                                data = src.read(index)
                                dst.write(data, range(dst_idx, dst_idx + len(index)))
                                dst_idx += len(index)

                    #print('new stack has {} bands \n'.format(dst_idx - 1))
                    #print('we have {} band names \n'.format(len(band_names)))
                    dst.descriptions = tuple(band_names)
                print('done writing {} with {} bands for cell {} \n'.format(os.path.basename(stack_path),len(band_names),cell))     
        
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

    #sys.stdout.write(class_prediction)
    return(class_prediction)

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
    sys.stderr.write('getting predictions...')
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
        #sys.stdout.write('{}'.format(src0.attrs))
        stack_bands = src0.attrs['descriptions']
    bands_out = []
    band_names = []
    for b in model_bands:
        for i, v in enumerate(stack_bands):
            if v == b:
                bands_out.append(i+1)
                band_names.append(v)
            elif b.startswith('sing') and v == b.split('_')[1]:
                bands_out.append(i+1)
                band_names.append('sing_{}'.format(v))
    sys.stdout.write('bands used for model: {}'.format(bands_out))
    
    new_stack = src0.sel(band=bands_out)
    new_stack.attrs['descriptions'] = band_names
    #sys.stdout.write(new_stack.attrs['descriptions'])

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
        #sys.stdout.write('num features in df = {}'.format(feature_band_count))
            
        X = dask.compute(X, num_workers=4)[0]
    
        class_prediction = rf.predict(X)
        #sys.stdout.write('class_prediction out = {}'.format(class_prediction))
        class_out = np.uint8(np.squeeze(class_prediction))
        class_out = class_out.reshape(w.height, w.width)
        #sys.stdout.write('class_out = {}'.format(class_out))
    
        if not os.path.isfile(class_img_out):
            with rio.open(class_img_out, mode='w', **profile) as dst:
                pass
        with rio.open(class_img_out, mode='r+') as dst:
            #dst.write(class_out, window=w)
            dst.write(class_out, indexes=1, window=w)
    
    return class_prediction

def rf_model(df_in, out_dir, lc_mod, importance_method, ran_hold, model_name, lut, feature_model, feature_mod_dict):
    if isinstance(df_in, pd.DataFrame):
        df = df_in
    else:
        df = pd.read_csv(df_in)
        
    class_col,lut = get_class_col(lc_mod,lut)
    print('class_col = {}'.format(class_col))
    if '{}_name'.format(class_col) in df.columns:
        df2 = df
    else:
        df2 = df.merge(lut[['USE_NAME','{}'.format(class_col),'{}_name'.format(class_col)]], 
                       left_on='Class',right_on='USE_NAME', how='left')
    
    df3 = df2.reindex(sorted(df2.columns), axis=1)
    
    train, ho = prep_test_train(df3, out_dir, class_col, model_name)

    rf = multiclass_rf(train, out_dir, model_name, lc_mod, importance_method, ran_hold, lut)
    
    ## add columns back into feature dict to make sure they are in the right order:
    ordered_vars = [v[4:] for v in df3.columns.to_list() if v.startswith('var')]
    print('there are {} variables in the model'.format(len(ordered_vars)))
    sys.stdout.write('model bands are: {} \n'.format(ordered_vars))
    
    with open(feature_mod_dict, 'r+') as feature_model_dict:
        dic = json.load(feature_model_dict)
        dic[feature_model].update({'band_names':ordered_vars})
    with open(feature_mod_dict, 'w') as new_feature_model_dict:
        json.dump(dic, new_feature_model_dict)
        
    score = get_holdout_scores(ho, rf[0], class_col, out_dir)
    ## add the smallholder indication variables to the output df
    #score = pd.DataFrame(score)
    score["smalls_1ha"] = df["smlhld_1ha"]
    score["smalls_halfha"] = df["smlhd_halfha"]

    
    return rf, score

def rf_classification(in_dir, cell_list, df_in, feature_model, start_yr, start_mo, samp_mod_name, 
                      feature_mod_dict, singleton_var_dict, rf_mod, img_out, spec_indices=None, si_vars=None, 
                      spec_indices_pheno=None, pheno_vars=None, singleton_vars=None, poly_vars=None, poly_var_path=None, 
                      combo_bands=None, lc_mod=None, lut=None, importance_method=None, ran_hold=29, out_dir=None, scratch_dir=None):
    
    spec_indices,si_vars,spec_indices_pheno,pheno_vars,singleton_vars,poly_vars,combo_bands,band_names = getset_feature_model(
                                                                  feature_mod_dict,
                                                                  feature_model,
                                                                  spec_indices,
                                                                  si_vars,
                                                                  spec_indices_pheno,
                                                                  pheno_vars,
                                                                  singleton_vars,
                                                                  poly_vars,
                                                                  combo_bands)
    
    model_name = '{}_{}_{}'.format(feature_model, samp_mod_name, start_yr)
    
    cells = []
    if isinstance(cell_list, list):
        cells = cell_list
    elif isinstance(cell_list, str) and cell_list.endswith('.csv'): 
        with open(cell_list, newline='') as cell_file:
            for row in csv.reader(cell_file):
                cells.append (row[0])
    elif isinstance(cell_list, int) or isinstance(cell_list, str): # if runing individual cells as array via bash script
        cells.append(cell_list) 
                
    for cell in cells:
        sys.stderr.write('working on cell {}... \n'.format(cell))
        cell_dir = os.path.join(in_dir,'{:06d}'.format(int(cell)))
        
        stack_path = os.path.join(cell_dir,'comp','{}_{}_stack.tif'.format(feature_model,start_yr))
        if os.path.isfile(stack_path):
            sys.stderr.write('stack file already exists for model {} \n'.format(feature_model))
            var_stack = stack_path
        elif 'NoPoly' in feature_model:
            poly_model = feature_model.replace('NoPoly','Poly')
            alt_path = os.path.join(cell_dir,'comp','{}_{}_stack.tif'.format(poly_model,start_yr))
            if os.path.isfile(alt_path):
                sys.stderr.write('poly stack file already exists for model {} \n'.format(poly_model))
                var_stack = alt_path
        else:
            # make variable stack if it does not exist (for example for cells without sample pts)
            # -- will not be remade if a file named {feature_model}_{start_year}_stack.tif already exists in ts_dir/comp
            var_stack = make_variable_stack(in_dir,cell,feature_model,start_yr,start_mo,spec_indices,si_vars,spec_indices_pheno,
                                        pheno_vars,feature_mod_dict,singleton_vars=None, singleton_var_dict=None, 
                                        poly_vars=None, poly_var_path=None, scratch_dir=None)
        
        #if img_out is None:
        class_img_out = os.path.join(cell_dir,'comp','{:06d}_{}.tif'.format(int(cell),model_name))
        #else:
        #    class_img_out = img_out
        
        if rf_mod != None and os.path.isfile(rf_mod):
            sys.stderr.write('using existing model... \n')
        else:
            sys.stderr.write('creating rf model... \n')
            rf_mod = rf_model(df_in, out_dir, lc_mod, importance_method, ran_hold, model_name, lut, feature_model, feature_mod_dict)

        with open(feature_mod_dict, 'r+') as feature_model_dict:
            dic = json.load(feature_model_dict)
            model_bands = dic[feature_model]['band_names']
            sys.stdout.write('model bands from dict: {}: \n'.format(model_bands))
            sys.stdout.flush()
            
        ## Old gdal-based method:
        #class_prediction = get_predictions_gdal(var_stack,rf_path)
        ## geowombat / xarray method:
        class_prediction = get_predictions_gw(var_stack, model_bands, rf_mod, class_img_out)
        
        if class_prediction is not None:
            sys.stdout.write('Image saved to: {}'.format(class_img_out))    
        else:
            print ('got an error')
    return None
