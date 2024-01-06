#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o RFvars.%N.%a.%j.out # STDOUT
#SBATCH -e RFvars.%N.%a.%j.err # STDERR
#SBATCH --job-name="RFvars"
#SBATCH --array=26-29
################################################################

#GRID_ID="${SLURM_ARRAY_TASK_ID}"
GRID_ID="$(($SLURM_ARRAY_TASK_ID + 4000))"

COUNTRY='paraguay'
VIs=("evi2" "gcvi" "wi" "kndvi" "nbr" "ndmi") 
#VIs=("kndvi") 
#Note VI is a folder in the IMGDIR containing the time series images
STARTYR=2021
OUT_DIR="/home/downspout-cel/${COUNTRY}_lc/stac/grids/00${GRID_ID}/comp"

BANDS="[Max,Min,Amp,Avg,CV,Std,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec]"
# ####################################################

# activate the virtual environment
conda activate venv.lucinsa38_pipe

for VI in "${VIs[@]}"
do
IMGDIR="/home/downspout-cel/${COUNTRY}_lc/stac/grids/00${GRID_ID}/brdf_ts/ms/${VI}"
LUCinSA_helpers make_ts_composite --grid_cell $GRID_ID  --img_dir $IMGDIR --out_dir $OUT_DIR --start_yr $STARTYR --spec_index $VI --bands_out $BANDS
done
conda deactivate
