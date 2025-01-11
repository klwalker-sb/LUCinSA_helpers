#!/usr/bin/env python
# coding: utf-8

import os
import sys
from pathlib import Path
import csv
from rasterio.merge import merge
import rasterio as rio

def mosaic_cells(cell_list, in_dir_main, in_dir_local, common_str, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(cell_list, list):
        cells = cell_list
        output_path = os.path.join(out_dir,f'{common_str}_mosaic.tif')
    elif cell_list.endswith('.csv'):
        grouping = cell_list.split('.')[0]
        output_path = os.path.join(out_dir,f'{grouping}_{common_str}.tif')
        cells = []
        with open(cell_list, newline='') as cell_file:
            for row in csv.reader(cell_file):
                cells.append (row[0])
    else:
        print('cell_list needs to be a list or path to .csv file with list')
    print('mosaicking cells:{}'.format(cells))
    ras_list = []
    for cell in cells:
        cell_path = os.path.join(in_dir_main,'{:06d}'.format(int(cell)), in_dir_local)
        if not os.path.exists(cell_path):
            print('there is no {} folder for cell {}.'.format(in_dir_local, cell))
        else:
            matches = [f for f in os.listdir(cell_path) if common_str in f]
            if len(matches) == 0:
                print('no raster was created for cell {} for model {}.'.format (cell, common_str))
            else:
                for m in matches:
                      ras_list.append(os.path.join(cell_path,m))
            
    print(ras_list)
    
    with rio.open(ras_list[0], 'r') as src_exmp:
        output_meta = src_exmp.meta.copy()
        print('mosaicking {}-band rasters'.format(src_exmp.meta['count']))
    
    mosaic, output = merge(ras_list)
    output_meta.update(
        {"driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": output,
        })
    
    with rio.open(output_path, 'w', **output_meta) as m:
        m.write(mosaic)
        print('writing mosaic to: {}'.format(output_path))
                               
    return output_path
    