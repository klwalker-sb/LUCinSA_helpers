#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o TScomp.%N.%a.%j.out # STDOUT
#SBATCH -e TScomp.%N.%a.%j.err # STDERR
#SBATCH --job-name="TScomp"
#SBATCH --array=992

# ####################################################

GRID_ID="${SLURM_ARRAY_TASK_ID}"
#GRID_ID='$(($SLURM_ARRAY_TASK_ID + 1000))'

COUNTRY='paraguay'
SPEC_INDEX='evi2'  #Note SPEC_INDEX is a folder in the IMGDIR containing the time series images
IMGDIR="/raid-cel/r/downspout-cel/${COUNTRY}_lc/raster/grids/000${GRID_ID}/brdf_ts/ms/${SPEC_INDEX}"
STARTYR=2021

##For archive:
OUT_DIR="/raid-cel/r/downspout-cel/${COUNTRY}_lc/raster/grids/000${GRID_ID}/comp"
##For bulk export (for digitizing project TODO: link straight to Google Drive):
#OUT_DIR="/raid-cel/sandbox/sandbox-cel/${COUNTRY}_lc/TO_EXPORT/000${GRID_ID}"

BANDS="[Min,Max,Amp]"
#BANDS="[Jan,Jun,Nov]"
#BANDS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ####################################################

# activate the virtual environment
source ~/.nasaenv/bin/activate

LUCinSA_helpers make_ts_composite --grid_cell $GRID_ID  --img_dir $IMGDIR --out_dir $OUT_DIR --start_yr $STARTYR --spec_index $SPEC_INDEX --bands_out $BANDS

deactivate
