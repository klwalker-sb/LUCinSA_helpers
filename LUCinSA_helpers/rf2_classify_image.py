#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import rasterio as rio
from collections.abc import Iterable
from osgeo import gdal, ogr, gdal_array
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import seaborn as sn

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

in_dir = sys.argv[1]
#spec_indices  = sys.argv[2]
spec_indices = ['evi2','gcvi','wi','kndvi','nbr','ndmi']
#stats =  sys.argv[3]
stats = ['Max','Min','Amp','Avg','CV','Std','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
var_df = sys.argv[4]
cclass = sys.argv[5]
importance_method = sys.argv[6]
out_dir = sys.argv[7]
out_img = sys.argv[8]
class_img_out = sys.argv[9]

def multiclass_rf(df_in, out_dir, cclass, importance_method):
    dfin = pd.read_csv(df_in, index_col=0)
    rf_vars = [col for col in dfin if col.startswith('var_')]
    X = dfin[rf_vars]

    if cclass == 'All':
        y = dfin['LC2']
    elif cclass == 'Lev1':
        y = dfin['LC1']
    elif cclass in ['HighVeg', 'MedVeg', 'LowVeg', 'NoVeg']:
        dfin['label'] = 0
        dfin.loc[dfin['Class1'] == cclass,'label'] = 1
        y = dfin['label']
    elif 'Crops' in cclass or 'Grass' in cclass:
        dfin['label'] = 0
        if 'Mate' in cclass:
            dflow = dfin[(dfin['Class1'] == 'LowVeg') | (dfin['USE_NAME'] =='Crops-Yerba-Mate')]
        else:
            dflow = dfin[DFin['Class1'] == 'LowVeg']
        sys.stdout.write('There are {} LOWVeg records'.format(str(len(dflow))))
        dflow.loc[dflow['USE_NAME'] == cclass,'label'] = 1
        X = dflow[RFvars]
        y = dflow['label']
    elif cclass == 'LowVeg_Lev2':
        dflow = DFin[dfin['Class1'] == 'LowVeg']
        X = dflow[RFvars]
        y = dflow['USE_NAME']

    print (pd.Series(y).value_counts())
    X = X.values
    y = y.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=30)

    rf = RandomForestClassifier(n_estimators=500, oob_score=True)
    rf = rf.fit(X_train,y_train)

    sys.stdout.write('Our OOB prediction of accuracy is:{oob}%\n'.format(oob=rf.oob_score_ * 100))

    confusion_matrix = pd.DataFrame()
    confusion_matrix['observed'] = y
    confusion_matrix['predicted'] = rf.predict(X)
    cm=pd.crosstab(confusion_matrix['observed'],confusion_matrix['predicted'],margins=True)

    pd.DataFrame.to_csv(cm,os.path.join(out_dir,'CM.csv'), sep=',', index=True)

    if importance_method == "Impurity":
        var_importances = pd.Series(rf.feature_importances, index=rf_vars)
        pd.Series.to_csv(var_importances,os.path.join(out_dir,'VarImportance_{}_{}.csv'.format(cclass,importance_method)),sep=',',index=True)
    elif importanceMethod == "Permutation":
        result = permutation_importance(rf, X_test, y_test, n_repeats=10,random_state=42, n_jobs=2)
        var_importances = pd.Series(result.importances_mean, index=rf_vars)

        pd.Series.to_csv(var_importances,os.path.join(out_dir,'VarImportance_{}_{}.csv'.format(cclass,importance_method)),sep=',',index=True)

    return rf

def make_variable_stack(in_dir,spec_indices,stats, name_end):

    sys.stderr.write('making variable stack \n')

    stack_paths = []
    num_bands_all = 0

    for vi in spec_indices:
        for img in os.listdir(in_dir):
            if img.endswith(name_end) and vi in img:
                img_path = os.path.join(in_dir,img)
                stack_paths.append(img_path)
                with rio.open(img_path) as src:
                    num_bands = src.count
                sys.stdout.write('Found {} with {} bands \n'.format(img,num_bands))
                if num_bands < len(stats):
                    sys.stderr.write('ERROR: number of bands does not match requested number')
                    sys.exit()
                num_bands_all = num_bands_all + num_bands
    if len(stack_paths) < len(spec_indices):
        sys.stderr.write('ERROR: did not find {} files for all the requested spec_indices'.format(name_end))
        sys.exit()

    sys.stdout.write('Final stack will have {} bands\n'.format(num_bands_all))
    sys.stderr.write('made variable stack...')

    output_count = 0
    indexes = []
    for path in stack_paths:
        with rio.open(path) as src:
            src_indexes = src.indexes
            indexes.append(src_indexes)
            output_count += len(src_indexes)

    with rio.open(stack_paths[0],'r') as src0:
        kwargs = src0.meta
        kwargs.update(count = output_count)

    with rio.open(os.path.join(in_dir,'stack.tif'),'w',**kwargs) as dst:
        dst_idx = 1
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


def classify_pixels(stack,rf,class_img_out):

    img_ds = gdal.Open(stack, gdal.GA_ReadOnly)

    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
    for b in range(img.shape[2]):
        img[:, :, b] = img_ds.get_raster_band(b + 1).ReadAsArray()

    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    img_as_array = img[:, :, :np.int(img.shape[2])].reshape(new_shape)

    sys.stdout.write('Reshaped from {o} to {n} \n'.format(o=img.shape, n=img_as_array.shape))

    img_as_array = np.nan_to_num(img_as_array)

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
    outdata = driver.Create(ClassImgOut, cols, rows, 1, gdal.GDT_UInt16)
    outdata.SetGeoTransform(img_ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(img_ds.GetProjection())##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(class_prediction)
    outdata.FlushCache() ##saves to disk!!
    sys.stdout.write('Image saved to: {}'.format(class_img_out))

#MakeVariableStack(in_dir,spec_indices,stats,'RFVars.tif')
#RFmod = MulticlassRF(VarDF,OutDir,Class,importanceMethod)
#ClassifyPixels(OutImg, RFmod, ClassImgOut)
