#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o TScomp.%N.%a.%j.out # STDOUT
#SBATCH -e TScomp.%N.%a.%j.err # STDERR
#SBATCH --job-name="TScomp"
#SBATCH --array=26-29
################################################################

#GRID_ID="${SLURM_ARRAY_TASK_ID}"
GRID_ID="$(($SLURM_ARRAY_TASK_ID + 3000))"

COUNTRY='paraguay'
VI=("evi2") 
#Note VI is a folder in the IMGDIR containing the time series images
IMGDIR="/home/downspout-cel/${COUNTRY}_lc/stac/grids/00${GRID_ID}/brdf_ts/ms/${VI}"
STARTYR=2021
OUT_DIR="/home/downspout-cel/${COUNTRY}_lc/stac/grids/00${GRID_ID}/comp"

BANDS="[Max,Min,Amp]"
# ####################################################
# activate the virtual environment
conda activate venv.lucinsa38_pipe

LUCinSA_helpers make_ts_composite --grid_cell $GRID_ID  --img_dir $IMGDIR --out_dir $OUT_DIR --start_yr $STARTYR --spec_index $VI --bands_out $BANDS
done
conda deactivate
