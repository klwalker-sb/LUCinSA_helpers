#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o RFvars.%N.%a.%j.out # STDOUT
#SBATCH -e RFvars.%N.%a.%j.err # STDERR
#SBATCH --job-name="RFvars"
#SBATCH --array=38
################################################################

#GRID_ID="${SLURM_ARRAY_TASK_ID}"
GRID_ID="$(($SLURM_ARRAY_TASK_ID + 4000))"

COUNTRY='paraguay'
SPEC_INDEX='wi'  #Note SPEC_INDEX is a folder in the IMGDIR containing the time series images
IMGDIR="/raid-cel/r/downspout-cel/${COUNTRY}_lc/stac/grids/00${GRID_ID}/brdf_ts/ms/${SPEC_INDEX}"
STARTYR=2021

##For archive:
OUT_DIR="/raid-cel/r/downspout-cel/${COUNTRY}_lc/stac/grids/00${GRID_ID}/comp"

BANDS="[Max,Min,Amp,Avg,CV,Std,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec]"
# ####################################################

# activate the virtual environment
conda activate venv.lucinsa38_dl

LUCinSA_helpers make_ts_composite --grid_cell $GRID_ID  --img_dir $IMGDIR --out_dir $OUT_DIR --start_yr $STARTYR --spec_index $SPEC_INDEX --bands_out $BANDS

deactivate
