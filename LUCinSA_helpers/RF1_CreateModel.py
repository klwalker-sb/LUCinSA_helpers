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

'''
in_dir = sys.argv[1]
classification = sys.argv[2]
importance_method = sys.argv[3]
out_dir = sys.argv[4]
ran_hold = sys.argv[5]
'''

# +
def quick_accuracy (X_test, y_test, rf_model, classification):
    predicted = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    print(f'Out-of-bag score estimate: {rf_model.oob_score_:.3}')
    print(f'Mean accuracy score: {accuracy:.3}')
   
    ConfusionMatrix = pd.DataFrame()
    ConfusionMatrix['observed'] = y_test
    ConfusionMatrix['predicted'] = predicted
    cm=pd.crosstab(ConfusionMatrix['observed'],ConfusionMatrix['predicted'],margins=True)
    
    cm['correct'] = cm.apply(lambda x: x[x.name] if x.name in cm.columns else 0, axis=1)
    cm['UA'] = cm['correct']/cm['All']
    #LUT=pd.read_csv('../Class_LUT.csv')
    #cm = cm.merge(LUT[['LC22','USE_NAME']], left_index=True, right_on='LC22', how='left')
    
    if classification == 'All':
        crop_cats = [31,32,33,34,35,36,37,38,53,55,40]
        crop_cols = [i for i in crop_cats if i in cm.columns]
        cm['Crop'] = cm[crop_cols].sum(axis=1)
        noCrop_cats = [1,2,3,7,12,13,17,51,52,56,60,65,70,80]
        noCrop_cols = [i for i in noCrop_cats if i in cm.columns]
        cm['NoCrop'] = cm[noCrop_cols].sum(axis=1)
        
    print(f'Confusion Matrix: {cm}')
    pd.DataFrame.to_csv(cm,os.path.join(out_dir,f'CM_{classification}.csv'), sep=',', index=True)
    
    return accuracy, cm


def MulticlassRF(df_in, out_dir, classification, importance_method, ran_hold):
    df_in = pd.read_csv(df_in, index_col=0)
    print('there are {} pts in the full data set'.format(df_in.shape[0]))
    df_train = df_in[df_in['TESTSET10'] == 0]
    print('there are {} pts in the training set'.format(len(df_train)))
    
    #df_train = df_train[df_train['SampMethod'] != 'CAN - unverified in GE']
    #print('there are now {} pts in the training set after dropping CAN soy'.format(len(df_train)))
    
     
    if classification == 'All':
        class_col = 'LC22'
    elif classification == 'crop_nocrop':
        class_col = 'LC2'
    elif classification == 'crop_nocrop_medcrop':
        class_col = 'LC3'
    elif classification == 'crop_nocrop_medcrop_tree':
        class_col = 'LC4'
    elif classification == 'veg':
        class_col = 'LC5'
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
        
    # remove unknown and other entries where class is not specified to LC22 (i.e. Crop_low)
    df_train = df_train[df_train[class_col] < 99]
    print('there are {} sample points after removing those without clear class'.format(df_train.shape[0]))
    y = df_train[class_col]
           
    vars_RF = [col for col in df_train if col.startswith('var_')]
    X = df_train[vars_RF]
    
    #print (pd.Series(y).value_counts())
    X = X.values
    y = y.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=ran_hold)

    rf = RandomForestClassifier(n_estimators=500, oob_score=True)
    rf = rf.fit(X_train,y_train)

    cm = quick_accuracy (X_test, y_test, rf, classification)

    if importance_method != None:
        if importance_method == "Impurity":
            var_importances = pd.Series(rf.feature_importances_, index=vars_RF)
            pd.Series.to_csv(var_importances,os.path.join(out_dir,'VarImportance_{}_{}.csv'.format(Class,importance_method)),sep=',',index=True)
        elif importance_method == "Permutation":
            result = permutation_importance(rf, X_test, y_test, n_repeats=10,random_state=ran_hold, n_jobs=2)
            var_importances = pd.Series(result.importances_mean, index=vars_RF)

        pd.Series.to_csv(var_importances,os.path.join(out_dir,'VarImportance_{}_{}.csv'.format(classification,importance_method)),sep=',', index=True)

    return rf, cm
# -

in_dir = 'D:/NasaProject/Paraguay/ClassificationModels'
df_in = os.path.join(in_dir, 'RFdf.csv')
classification = 'All'
importance_method = 'Permutation'
out_dir = 'D:/NasaProject/Paraguay/ClassificationModels/RF'
ran_hold = 29
RFmod = MulticlassRF(df_in,out_dir,classification,None,ran_hold)




