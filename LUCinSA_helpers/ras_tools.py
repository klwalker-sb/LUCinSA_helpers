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
            print('there is no {} folder for cell {}.'.format(local_dir, cell))
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
        mono_crop_classes = list(range(20, 39))
        for c in mono_crop_classes:
            c_count = np.count_nonzero(data == c)
            mono_crop_tot = mono_crop_tot + c_count
            pix_count = pix_count + c_count
        
        mix_crop_class = 35
        mix_crop_tot = np.count_nonzero(data == mix_crop_class)
        pix_count = pix_count + mix_crop_tot
        
        med_crop_classes = list(range(40, 49))
        for md in med_crop_classes:
            md_count = np.count_nonzero(data == md)
            med_crop_tot = med_crop_tot + md_count
            pix_count = pix_count + md_count
        
        tree_plant_class = 60
        tree_plant_tot = np.count_nonzero(data == tree_plant_class)
        pix_count = pix_count + tree_plant_tot
        
        tree_classes = [65,68,70,80]
        for tc in tree_classes:
            tc_count = np.count_nonzero(data == tc)
            tree_tot = tree_tot + tc_count
            pix_count = pix_count + tc_count
        
        no_veg = list(range(0, 9))
        low_veg = list(range(10, 19))
        med_veg = list(range(50, 59))
        other_classes = no_veg + low_veg + med_veg
        for oc in other_classes:
            oc_count = np.count_nonzero(data == oc)
            other_tot = other_tot + oc_count
            pix_count = pix_count + oc_count
        
        crop_tot = mono_crop_tot + mix_crop_tot + med_crop_tot

    else:
        classes = list(map_dict[map_product]['classes'].keys())
        #print(classes)
    
        if 'crop' in classes:
            crop_class = map_dict[map_product]['classes']['crop']
            crop_tot = np.count_nonzero(data == crop_class)
            pix_count = pix_count + crop_tot
        if 'tree' in classes:
            tree_class = map_dict[map_product]['classes']['tree']
            tree_tot = np.count_nonzero(data == tree_class)
            pix_count = pix_count + tree_tot
        other_classes = [c for c in classes if c not in ['crop','tree']]
        for oc in other_classes:
            val = map_dict[map_product]['classes'][oc]
            cat_tot = np.count_nonzero(data == val)
            other_tot = other_tot + cat_tot
            pix_count = pix_count + cat_tot
       
    class_dict['per_crop_mono'] = mono_crop_tot / pix_count
    class_dict['per_crop_med'] = med_crop_tot / pix_count
    class_dict['per_crop_mix'] = mix_crop_tot / pix_count
    class_dict['per_crop'] = crop_tot / pix_count
    class_dict['per_tree'] = tree_tot / pix_count
    class_dict['per_tree_plant'] = tree_plant_tot / pix_count
    class_dict['other'] = other_tot / pix_count
    class_dict['numpix'] = pix_count
          
    return class_dict
  
def summarize_zones(polys,map_dir,clip_dir,map_product,map_dict=None,out_dir=None):
    if map_dict is not None and map_dict !='None':
        file_name = map_dict[map_product]['loc']
    else:
        file_name = f'{map_product}.tif' 
    ras_in = os.path.join(map_dir,file_name)
    clip_ras_to_poly(ras_in, polys,clip_dir,map_product)
    plys = gpd.read_file(polys)
    plys.drop(['geometry'],axis=1,inplace=True)
    for i, row in plys.iterrows():
        per_classes = summarize_raster(os.path.join(clip_dir,map_product,f'{i}.tif'),map_dict,map_product)
        for key, value in per_classes.items():
            print(f'class={key},val={value}')
            if value > 0:
                plys.loc[i, f'{key}'] = value

    print(plys)
   
    if out_dir is not None:
        out_path = os.path.join(out_dir,f'zone_summary_{map_product}.csv')
        pd.DataFrame.to_csv(plys, out_path, sep=',', index=True)
    
    return plys
