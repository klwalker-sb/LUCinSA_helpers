
##Note: This script can be copied and run directly on cluster via iPython (as long as there is 1 core free)

import os
import sys
import numpy as np
###########################

## match='brdf' if moving after brdf processing or 'landsat' if moving after downloading
match='landsat'
input_folder_name='{}'.format(match)
output_folder_name='{}_LE07_recent'.format(match)

#proj_dir="/home/sandbox-cel/paraguay_lc/stac/grid/"
proj_dir="/raid-cel/sandbox/sandbox-cel/paraguay_lc/stac/grid/"
input_folder_name='{}'.format(match)
output_folder_name='{}_LE07_recent'.format(match)

###########################
# make list of grid folders that have a brdf folder inside them with files older than 1000(sec?)
folder_list=[i for i in os.listdir(proj_dir) if (len(i)==6 and i.startswith("00"))]

grid_list = []
not_ready = [] 
for f in folder_list:
    proj_grid_dir=os.path.join(proj_dir, f)
    if match in os.listdir(proj_grid_dir):
        grid_list.append(f)
    else:
        not_ready.append(f)
print(grid_list)      
print('{} grids not ready'.format(len(not_ready)))
###########################
        
for grid in grid_list:
    proj_grid_dir = os.path.join(proj_dir, str(grid).zfill(6))
    proj_grid_in_dir = os.path.join(proj_grid_dir, input_folder_name)
    out_dir = os.path.join(proj_grid_dir, output_folder_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ## move images that have LE07 in the file name from 2016 onward out of the brdf folder and into the brdf_LE07 folder
    LE07_to_move = [i for i in os.listdir(proj_grid_in_dir) if "LE07" in i and int(i.split("_")[3]) > 20170000]
    for img in LE07_to_move:
        command = "mv " + str(os.path.join(proj_grid_in_dir, img)) + " " + str(os.path.join(out_dir, img))
        print('moving {}'.format(img))
        os.system(command)

    sys.stdout.flush()        
