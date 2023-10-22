#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from joblib import dump, load

df_in = sys.argv[1]
lc_mod = sys.argv[2]
importance_method = sys.argv[3]
out_dir = sys.argv[4]
ran_hold = int(sys.argv[5])
model_name = sys.argv[6]
lut = sys.arg[7]

def get_class_col(lc_mod):
    if lc_mod == 'All':
        class_col = 'LC17'
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
    model names and class columns defined in get_class_col
    '''
    
    map_cat = get_class_col(lc_mod_map)
    acc_cat = get_class_col(lc_mod_acc)
    
    cmdf = pd.DataFrame()
    cmdf['obs'] = obs_col
    cmdf['pred'] = pred_col
    
    #print('getting confusion matrix based on {}...'.format(acc_cat))
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
    
    class_col = get_class_col(lc_mod)
    y = df_train[class_col]
           
    vars_rf = [col for col in df_train if col.startswith('var_')]
    X = df_train[vars_rf]
    
    #print (pd.Series(y).value_counts())
    X = X.values
    y = y.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = .01, random_state=ran_hold)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .02, random_state=ran_hold)

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

def rf_model(df_in, out_dir, lc_mod, importance_method, ran_hold, model_name, lut):
    class_col = get_class_col(lc_mod)
    print('class_col = {}'.format(class_col))
    train, ho = prep_test_train(df_in, out_dir, class_col, model_name)
    rf = multiclass_rf(train, out_dir, model_name, lc_mod, importance_method, ran_hold, lut)
    score = get_holdout_scores(ho, rf[0], class_col, out_dir)
    
    return rf, score
    