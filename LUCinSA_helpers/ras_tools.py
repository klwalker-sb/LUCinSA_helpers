#!/usr/bin/env python
# coding: utf-8

import os
import sys
from pathlib import Path
import csv
from rasterio.enums import Resampling
import rasterio as rio
import geopandas as gpd
import pandas as pd
import rasterio.mask
import fiona
import numpy as np
import json
#import rioxarray

def make_test_patch(xpt,ypt,img_in,out_name,out_dir,nbsize,res):
    '''
    clips raster to patch of size {nbsize, nbsize} around point with coords {xpt,ypt}
    '''
    
    bigimg = rioxarray.open_rasterio(img_in)
    buf = (res + res/2)   
    #grid_bbox = box(xpt-buf,ypt-buf,xpt+buf,ypt+buf)
    
    patch = bigimg.rio.clip_box(minx=(xpt-buf),miny=(ypt-buf),maxx=(xpt+buf),maxy=(ypt+buf))
    patch.rio.to_raster(os.path.join(out_dir,out_name))

def make_test_set(ptfile, xpt,ypt,img_in,out_name,out_dir,nbsize,res):
    '''
    makes test patches of size {nbsize, nbsize} for all points in {ptfile}
    ptfile must have fields XCoord and YCoord in same input as images (here hardcoded as epsg:8858)
    '''
    if isinstance(ptfile, pd.DataFrame):
        ptsdf = ptfile
    else:
        ptsdf = pd.read_csv(ptfile, index_col=0)
    ptsgdb = gpd.GeoDataFrame(ptsdf,geometry=gpd.points_from_xy(ptsdf.XCoord,ptsdf.YCoord),crs='epsg:8858')
    for index, row in ptsgdb.iterrows():
        out_dir1 = '{}_{}pix'.format(index,nbsize)
        out_dir = os.path.join(out_main,out_dir1)
        if not os.path.exists(out_dir):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        for img in os.listdir(in_dir):
            if img.endswith('tif') and img.startswith('2020'):
                out_name = os.path.join(out_dir,'{}_{}'.format(ind,img)) 
                img_in = os.path.join(in_dir,img)
                make_test_patch(ptsgdb['geometry'].x[index],ptsgdb['geometry'].y[index],img_in,out_name,out_dir,nbsize,10)   
    
def downsample_images(cell_list, in_dir_main, local_dir, common_str, out_dir_main, new_res):

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
        cell_path = os.path.join(in_dir_main,'{:06d}'.format(int(cell)), local_dir)
        if not os.path.exists(cell_path):
            print(f'there is no {local_dir} folder for cell {cell}.')
        else:
            out_dir = os.path.join(out_dir_main,'{:06d}'.format(int(cell)), local_dir)
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            
            in_files = [f for f in os.listdir(cell_path) if f.startswith(common_str)]
            with rio.open(os.path.join(cell_path,in_files[0]), 'r') as ex_src:
                scale_factor_x = ex_src.res[0] / int(new_res)
                scale_factor_y = ex_src.res[1] / int(new_res) 
                
            for in_file in in_files:
                with rio.open(os.path.join(cell_path,in_file), 'r') as src_in:
                    profile = src_in.profile.copy()
                    data = src_in.read(
                        out_shape=(
                            src_in.count,
                            int(src_in.height * scale_factor_y),
                            int(src_in.width * scale_factor_x)
                        ),
                        resampling=Resampling.bilinear
                    )

                    transform = src_in.transform * src_in.transform.scale(
                        (1 / scale_factor_x),
                        (1 / scale_factor_y)
                    )
                    
                    profile.update({"height": data.shape[-2],
                        "width": data.shape[-1],
                        "transform": transform})

                with rio.open(os.path.join(out_dir,in_file), "w", **profile) as outdata:
                    outdata.write(data)                            


def clip_ras_to_poly(ras_in, polys, out_dir,prod_name):
    out_path = Path(os.path.join(out_dir,prod_name))
    out_path.mkdir(parents=True, exist_ok=True)

    with fiona.open(polys, "r") as poly_src:
        shapes = [feature["geometry"] for feature in poly_src]

    for i, shape in enumerate(shapes):
        with rio.open(ras_in) as src:
            out_image, out_transform = rasterio.mask.mask(src, [shape], crop=True)
            out_meta = src.meta

        out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})

        with rio.open(os.path.join(out_path,f"{i}.tif"), "w", **out_meta) as dest:
            dest.write(out_image)

def summarize_raster(ras_in,map_dict,map_product):
    with rio.open(ras_in) as ras:
        data = ras.read()

    class_dict = {}
    pix_count = 0
    mono_crop_tot = 0
    med_crop_tot = 0
    mix_crop_tot = 0
    tree_plant_tot = 0
    tree_tot = 0
    crop_tot = 0
    other_tot = 0
    
    if map_product.startswith('LUCin') or map_product.startswith('CEL'):
        mono_crop_classes = [22,31,32,33,37,38,39]
        for c in mono_crop_classes:
            c_count = np.count_nonzero(data == c)
            mono_crop_tot = mono_crop_tot + c_count
            pix_count = pix_count + c_count
        
        mix_crop_classes = [35,23,24,25,26,32,34,36]
        for mx in mix_crop_classes:
            mx_count = np.count_nonzero(data == mx)
            mix_crop_tot = mix_crop_tot + mx_count
            pix_count = pix_count + mx_count
        
        med_crop_classes = list(range(40, 49))
        for md in med_crop_classes:
            md_count = np.count_nonzero(data == md)
            med_crop_tot = med_crop_tot + md_count
            pix_count = pix_count + md_count
        
        crop_tot = mono_crop_tot + mix_crop_tot + med_crop_tot
        
        tree_plant_class = 60
        tree_plant_tot = np.count_nonzero(data == tree_plant_class)
        pix_count = pix_count + tree_plant_tot
        
        tree_classes = [65,68,70,80]
        for tc in tree_classes:
            tc_count = np.count_nonzero(data == tc)
            tree_tot = tree_tot + tc_count
            pix_count = pix_count + tc_count
        
        no_veg = list(range(1, 10))
        low_veg = list(range(10, 21))
        med_veg = list(range(50, 60))
        other_classes = no_veg + low_veg + med_veg
        for oc in other_classes:
            oc_count = np.count_nonzero(data == oc)
            other_tot = other_tot + oc_count
            pix_count = pix_count + oc_count
        
    else:
        with open(map_dict, 'r+') as map_dict_in:
            map_dict = json.load(map_dict_in)
        classes = list(map_dict[map_product]['classes'].keys())
        print(classes)
    
        if 'crop' in classes:
            crop_classes = map_dict[map_product]['classes']['crop']
            for c in crop_classes:
                c_count = np.count_nonzero(data == c)
                crop_tot = crop_tot + c_count
                pix_count = pix_count + c_count
        if 'tree' in classes:
            tree_classes = map_dict[map_product]['classes']['tree']
            for t in tree_classes:
                tree_count = np.count_nonzero(data == t)
                tree_tot = tree_tot + tree_count
                pix_count = pix_count + tree_count
        other_classes = [c for c in classes if c not in ['crop','tree']]
        for oc in other_classes:
            for o in oc:
                #val = map_dict[map_product]['classes'][oc]
                cat_tot = np.count_nonzero(data == o)
                other_tot = other_tot + cat_tot
                pix_count = pix_count + cat_tot
       
    class_dict['per_crop_mono'] = round((100 * mono_crop_tot / pix_count),1)
    class_dict['per_crop_med'] = round((100 * med_crop_tot / pix_count),1)
    class_dict['per_crop_mix'] = round((100 * mix_crop_tot / pix_count),1)
    class_dict['per_crop'] = round((100 * crop_tot / pix_count),1)
    class_dict['per_tree'] = round((100 * tree_tot / pix_count),1)
    class_dict['per_tree_plant'] = round((100 * tree_plant_tot / pix_count),1)
    class_dict['other'] = round((100 * other_tot / pix_count),1)
    class_dict['numpix'] = pix_count
          
    return class_dict
  
def summarize_zones(polys,map_dir,clip_dir,map_product,map_dict=None,out_dir=None):
    if map_dict is not None and map_dict !='None':
        with open(map_dict, 'r+') as map_dict_in:
            dict_in = json.load(map_dict_in)
        file_name = dict_in[map_product]['loc']
    else:
        file_name = f'{map_product}.tif' 
    ras_in = os.path.join(map_dir,file_name)
    clip_ras_to_poly(ras_in, polys,clip_dir,map_product)
    plys = gpd.read_file(polys)
    plys.drop(['geometry'],axis=1,inplace=True)
    for i, row in plys.iterrows():
        per_classes = summarize_raster(os.path.join(clip_dir,map_product,f'{i}.tif'),map_dict,map_product)
        for key, value in per_classes.items():
            #print(f'class={key},val={value}')
            if value > 0:
                plys.loc[i, f'{key}'] = value

    #print(plys)
   
    if out_dir is not None:
        out_path = os.path.join(out_dir,f'zone_summary_{map_product}.csv')
        pd.DataFrame.to_csv(plys, out_path, sep=',', index=True)
    
    return plys


def summarize_district_polys(polys, map_product, scratch_dir, test=True):
    
    Path(scratch_dir).mkdir(parents=True, exist_ok=True)
    if map_product.startswith('LUCin') or map_product.startswith('CEL'):
        model= map_product.split('_')[1:3]
        model_name = f'{model[0]}_{model[1]}_LC25' 
        map_dir = "/home/downspout-cel/paraguay_lc/mosaics"
        output_dir = "/home/downspout-cel/paraguay_lc/classification/RF/model_stats"
        map_dict = None
        all_scores_tab = os.path.join(output_dir,'CEL_model_scores.csv')
        all_scores_dict = os.path.join(output_dir,'CEL_model_scores_dict.json')
    else:
        map_dir = "/home/downspout-cel/paraguay_lc/lc_prods"
        output_dir = "/home/downspout-cel/paraguay_lc/lc_prods"
        map_dict = "/home/downspout-cel/paraguay_lc/lc_prods/prod_dict.json"
        all_scores_tab = os.path.join(output_dir,'other_product_scores_tab.csv')
        all_scores_dict = os.path.join(output_dir,'other_product_scores_dict.json')
        model_name = map_product
    zone_stats = summarize_zones(polys,map_dir,scratch_dir,map_product,map_dict,out_dir=None)
    zone_stats['crop_dif']=zone_stats['crop_estKW'] - zone_stats['per_crop']
    zone_stats['perCrop_found']= round(100 * zone_stats['per_crop'] / zone_stats['crop_estKW'],2)
    zone_stats['crop_dif_abs'] = np.abs(zone_stats['crop_dif'])
    zone_stats.to_csv(os.path.join(output_dir,f'district_summary_{model_name}.csv'))
    avg_crop_dif = round(np.mean(zone_stats['crop_dif']),2)
    mae_crop = round(np.mean(zone_stats['crop_dif_abs']),2)
    perCrop_found = round(np.mean(zone_stats['perCrop_found']),2)
    print(f'avg_crop_dif = {avg_crop_dif}, MAE = {mae_crop}')
    print(f'perCrop_found = {perCrop_found}')
    sm = zone_stats[zone_stats['avgFinca22']<5]
    avg_crop_dif_sm = round(np.mean(sm['crop_dif']),2)
    mae_crop_sm = round(np.mean(sm['crop_dif_abs']),2)
    perCrop_found_sm = round(np.mean(sm['perCrop_found']),2)
    print(f'avg_crop_dif_sm = {avg_crop_dif_sm}, MAE_sm = {mae_crop_sm}')
    print(f'perCrop_found_sm = {perCrop_found_sm}')
    lg = zone_stats[zone_stats['avgFinca22']>20]
    avg_crop_dif_lg = round(np.mean(lg['crop_dif']),2)
    mae_crop_lg = round(np.mean(lg['crop_dif_abs']),2)
    perCrop_found_lg = round(np.mean(lg['perCrop_found']),2)
    print(f'avg_crop_dif_lg = {avg_crop_dif_lg}, MAE_lg = {mae_crop_lg}')
    print(f'perCrop_found_lg = {perCrop_found_lg}')
    
    dict_entry = {}
    dict_entry["avg_crop_dif"] = avg_crop_dif
    dict_entry["MAE_crop"] = mae_crop
    dict_entry["avg_crop_dif_sm"] = avg_crop_dif_sm
    dict_entry["MAE_crop_sm"] = mae_crop_sm
    dict_entry["avg_crop_dif_lg"] = avg_crop_dif_lg
    dict_entry["MAE_crop_lg"] = mae_crop_lg
    dict_entry["perCrop_found_sm"] = perCrop_found_sm
    dict_entry["perCrop_found_lg"] = perCrop_found_lg
    dict_entry["perCrop_found"] = perCrop_found
    
    print(dict_entry)
    if test == False:
        with open(all_scores_dict, 'r+') as full_dict:
            dic = json.load(full_dict)
        #updated_entry = dic[model_name] | dict_entry  ##This works in Python 3.9 and above
        #entry = dic[model_name].copy()
        #entry.update(dict_entry)
        dic[model_name].update(dict_entry)
        #dic[model_name] = entry
        print(dic)

        new_scores = pd.DataFrame.from_dict(dic)
        new_scores.to_csv(all_scores_tab)

        with open(all_scores_dict, 'w') as new_dict:
            json.dump(dic, new_dict)
    
    return zone_stats

def get_variables_at_pts_external(out_dir, ras_in,ptfile,out_col,out_name):

    if isinstance(ptfile, pd.DataFrame):
        ptsdf = ptfile
    else:
        ptsdf = pd.read_csv(ptfile, index_col=0)

    ptsgdb = gpd.GeoDataFrame(ptsdf,geometry=gpd.points_from_xy(ptsdf.XCoord,ptsdf.YCoord),crs='epsg:8858')
    #pts4326 = ptsgdb.to_crs({'init': 'epsg:4326'})
    xy = [ptsgdb['geometry'].x, ptsgdb['geometry'].y]
    coords = list(map(list, zip(*xy)))
    
    with rio.open(ras_in, 'r') as comp:
        comp.np = comp.read()
        ptsgdb[out_col] = [sample[0] for sample in comp.sample(coords)]     

    pd.DataFrame.to_csv(ptsgdb,os.path.join(out_dir,out_name), sep=',', index=True)
    
    return ptsgdb

def get_confusion_matrix_generic(samp_file, pred_col, obs_col, lut_valid, lut_map, lut_colout='LC_UNQ', nodata=None, print_cm=False, out_dir=None, model_name=None):
    '''
    returns confusion matrix with optional regrouping of classes based on dataframe containing 
         <pred_col> = the column with values extracted from the map to summarize
          <obs_col> =  the column with values obeserved with the validation method (e.g. on the ground / high-res imagery)
    these are linked using the respective <lut_map> and <lut_valid> lookup tables.
    Both luts need to have a column with unique ids called <'LC_UNQ'> (which matches values in <pred_col> and <obs_col> of <samp_file>). 
    Unless these are equal, both also need a column with matching translations named <lut_colout>
    '''
    if isinstance(samp_file, pd.DataFrame):
        samp = samp_file
    else:
        samp = pd.read_csv(samp_file)
        
    if isinstance(lut_valid, pd.DataFrame):
        lut_valid = lut_valid
    else:
        lut_valid = pd.read_csv(lut_valid)
        
    if isinstance(lut_map, pd.DataFrame):
        lut_map = lut_map
    else:
        lut_map = pd.read_csv(lut_map)
    
    cmdf = pd.DataFrame()
    cmdf['pred'] = samp[pred_col]
    cmdf['obs'] = samp[obs_col]
    ## drop points that do not overlap map
    if nodata is not None:
        cmdf = cmdf[cmdf['pred'] != nodata]
    
    print(f'getting confusion matrix based on {lut_colout}...')
    ## reclass obs col using lut_valid (match on LC_UNQ, then reclass to <lut_colout>)
    cmdf2 = cmdf.merge(lut_valid[['LC_UNQ',f'{lut_colout}_name']], left_on='obs', right_on='LC_UNQ',how='left')
    cmdf2.rename(columns={f'{lut_colout}_name':'obs_reclass'}, inplace=True)
    cmdf2.drop(['LC_UNQ'],axis=1,inplace=True)
    ## reclass pred col using lut_map (match on LC_UNQ, then reclass to <lut_colout>)
    cmdf3 = cmdf2.merge(lut_map[['LC_UNQ', f'{lut_colout}_name']], left_on='pred', right_on='LC_UNQ',how='left')
    cmdf3.rename(columns={f'{lut_colout}_name':'pred_reclass'}, inplace=True)
    cmdf3.drop(['LC_UNQ'],axis=1,inplace=True)
    ## Now use crosstabs to get confusion matrix
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
    if lut_colout == 'LC2':
        cm.at['All','F1']=cm.at['crop','F1']
        TP = cm.at['crop', 'crop']
        FP = cm.at['crop', 'nocrop']
        FN = cm.at['nocrop', 'crop']
        TN = cm.at['nocrop','nocrop']
        All = TP + FP + FN + TN
        cm['Kappa'] = (2*(TP*TN - FN*FP)/((TP+FP)*(FP+TN)+(TP+FN)*(FN+TN))).round(3)
    
    #print(f'Confusion Matrix: {cm}')
    if print_cm == True:
        mod_path = os.path.join(out_dir,f'cm_{model_name}_{lut_colout}.csv')
        pd.DataFrame.to_csv(cm, mod_path, sep=',', index=True)
    
    return cm



       