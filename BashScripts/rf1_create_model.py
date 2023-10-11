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
classification = sys.argv[2]
importance_method = sys.argv[3]
out_dir = sys.argv[4]
ran_hold = int(sys.argv[5])
model_name = sys.argv[6]

def separate_holdout(training_pix_path, out_dir):
    '''
    USE THIS WHEN WE AUGMENT BY POLYGON
    Generates separate pixel databases for training data and 20% field-level holdout
    Use this instead of generate_holdout() to fit a model to an exsisting holdout set
    '''
    #holdout_set = pd.read_csv(holdout_field_path)
    #pixels = pd.read_csv(training_pix_path)
    
    ##if there is no 'field_id' in the pixel dataset, use the following two lines (but now 'field_id' is aready in pixels)
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


def quick_accuracy (X_test, y_test, rf_model, classification, out_dir,mod_name):
    predicted = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    print(f'Out-of-bag score estimate: {rf_model.oob_score_:.3}')
    print(f'Mean accuracy score: {accuracy:.3}')
   
    confusion_matrix = pd.DataFrame()
    confusion_matrix['observed'] = y_test
    confusion_matrix['predicted'] = predicted
    cm=pd.crosstab(confusion_matrix['observed'],confusion_matrix['predicted'],margins=True)
    
    cm['correct'] = cm.apply(lambda x: x[x.name] if x.name in cm.columns else 0, axis=1)
    cm['sumcol'] = cm.apply(lambda x: cm.loc['All', x.name] if x.name in cm.columns else 0)
    cm['UA'] = cm['correct']/cm['All']
    cm['PA'] = cm['correct']/cm['sumcol']
    
    if classification == 'All':
        crop_cats = [31,32,33,34,35,36,37,38,53,55,40]
        crop_cols = [i for i in crop_cats if i in cm.columns]
        cm['Crop'] = cm[crop_cols].sum(axis=1)
        noCrop_cats = [1,2,3,7,12,13,17,51,52,56,60,65,70,80]
        noCrop_cols = [i for i in noCrop_cats if i in cm.columns]
        cm['NoCrop'] = cm[noCrop_cols].sum(axis=1)
        
    print(f'Confusion Matrix: {cm}')
    pd.DataFrame.to_csv(cm,os.path.join(out_dir,f'CM_{mod_name}.csv'), sep=',', index=True)
    
    return accuracy, cm


def prep_test_train(df_in, out_dir, classification,mod_name):
    
    if isinstance(df_in, pd.DataFrame):
        df_in = df_in
    else:
        df_in = pd.read_csv(df_in, index_col=0)
    print('there are {} pts in the full data set'.format(df_in.shape[0]))
    #df_train = df_in[df_in['TESTSET20'] == 0]
    #print('there are {} pts in the training set'.format(len(df_train)))
    
    df_in = df_in[df_in['SampMethod'] != 'CAN - unverified in GE']
    print('there are now {} pts in the training set after dropping CAN soy'.format(len(df_in)))
    
    if classification == 'All':
        class_col = 'LC17'
    elif classification == 'crop_nocrop':
        class_col = 'LC2'
    elif classification == 'crop_nocrop_medcrop':
        class_col = 'LC3'
    elif classification == 'crop_nocrop_medcrop_tree':
        class_col = 'LC4'
    elif classification == 'veg':
        class_col = 'LC5'
    elif classification == 'cropType':
        class_col = 'LC_crops'
    '''
    elif classification == 'crop_post':
        #where prediction == 1...
        df_train = df_train[df_train['LC2'] == 1]
        print('there are {} sample points after removing non-crop'.format(df_train.shape[0]))
        y = df_train['LC22']
        #TODO: need two step prediction methods
    elif classification == 'nocrop_post':
        df_train = df_train[df_train['LC2'] == 0]
        print('there are {} sample points after removing crop'.format(df_train.shape[0]))
        #TODO: need two step prediction methods
           
    elif classification in ['HighVeg', 'MedVeg', 'LowVeg', 'NoVeg']:
        df_train['label'] = 0
        df_train.loc[df_train['Class1'] == classification,'label'] = 1
        y = df_train['label']
    elif 'Crops' in classification or 'Grass' in classification:
        df_train['label'] = 0
        if 'Mate' in classification:
            df_low = df_train[(df_train['Class1'] == 'LowVeg') | (df_train['USE_NAME'] =='Crops-Yerba-Mate')]
        else:
            df_low = df_train[df_train['Class1'] == 'LowVeg']
        print('There are {} LOWVeg records'.format(str(len(df_low))))
        df_low.loc[df_low['USE_NAME'] == Class,'label'] = 1
        X = df_low[vars_RF]
        y = df_low['label']
    elif classification == 'LowVeg_Lev2':
        df_low = df_train[df_train['Class1'] == 'LowVeg']
        X = df_low[rf_vars]
        y = df_low['USE_NAME']
    '''
    
    # remove unknown and other entries where class is not specified a given level (i.e. crop_low if crop type is desired)
    df_in = df_in[df_in[class_col] < 99]
    print('there are {} sample points after removing those without clear class'.format(df_in.shape[0]))
    
    #Separate training and holdout datasets to avoid confusion with numbering
    training_pix_path = os.path.join(out_dir,f'{model_name}_TRAINING.csv')
    holdout_pix_path = os.path.join(out_dir,f'{model_name}_HOLDOUT.csv')
    df_train = df_in[df_in['TESTSET20'] == 0]
    pd.DataFrame.to_csv(df_train, training_pix_path, sep=',', na_rep='NaN', index=False)
    df_test = df_in[df_in['TESTSET20'] == 1]
    pd.DataFrame.to_csv(df_test, holdout_pix_path, sep=',', na_rep='NaN', index=False)
    
    return(training_pix_path, holdout_pix_path)

def multiclass_rf(trainfeatures, out_dir, model_name, classification, importance_method, ran_hold):
    
    df_train = pd.read_csv(trainfeatures)
    print('There are {} training features'.format(df_train.shape[0]))
    y = df_train['LC17']
           
    vars_rf = [col for col in df_train if col.startswith('var_')]
    X = df_train[vars_rf]
    
    #print (pd.Series(y).value_counts())
    X = X.values
    y = y.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = .01, random_state=ran_hold)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .02, random_state=ran_hold)

    rf = RandomForestClassifier(n_estimators=500, oob_score=True)
    rf = rf.fit(X_train,y_train)
    dump(rf, os.path.join(out_dir,'{}_RFmod.joblib'.format(model_name))

    cm = quick_accuracy (X_test, y_test, rf, classification, out_dir)

    if importance_method != None:
        if importance_method == "Impurity":
            var_importances = pd.Series(rf.feature_importances_, index=vars_rf)
            pd.Series.to_csv(var_importances,os.path.join(out_dir,'VarImportance_{}_{}.csv'.format(classification,importance_method)),sep=',',index=True)
        elif importance_method == "Permutation":
            result = permutation_importance(rf, X_test, y_test, n_repeats=10,random_state=ran_hold, n_jobs=2)
            var_importances = pd.Series(result.importances_mean, index=vars_rf)

        pd.Series.to_csv(var_importances,os.path.join(out_dir,'VarImportance_{}_{}.csv'.format(classification,importance_method)),sep=',', index=True)

    return rf, cm


def get_holdout_scores(holdoutpix, rf_model, out_dir):
    ##Save info for extra columns and drop (model is expecting only variable input columns)
    
    if isinstance(holdoutpix, pd.DataFrame):
        holdout_pix = holdoutpix
    else:
        holdout_pix = pd.read_csv(holdoutpix)
        
    #holdout_labels = holdout_pix['LC17']
    holdout_labels = holdout_pix['LC17']
    h_IDs = holdout_pix['OID_']
    print('number of holdout pixels = {}'.format(len(holdout_pix)))
    #Get list of variables to include in model:
    
    vars = [col for col in holdout_pix if col.startswith('var_')]
    holdout_fields = holdout_pix[vars]
    print(holdout_fields.head())

    ##Calculate scores
    #holdout_fields_predicted = rf_model.predict_proba(holdout_fields)
    holdout_fields_predicted = rf_model.predict(holdout_fields)
    
    ##Add extra columns back in
    holdout_fields['pred']= holdout_fields_predicted
    holdout_fields['label']= holdout_labels
    holdout_fields['OID']=h_IDs

    ##Print to file
    pd.DataFrame.to_csv(holdout_fields, os.path.join(out_dir,'Holdouts_predictions.csv'), sep=',', na_rep='NaN', index=True)
   
    return holdout_fields

def rf_model(df_in, out_dir, classification, importance_method, ran_hold, model_name):
    train, ho = prep_test_train(df_in, out_dir, classification)
    rf = multiclass_rf(train, out_dir, model_name,classification,importance_method,ran_hold)
    score = get_holdout_scores(ho, rf[0], out_dir)
    