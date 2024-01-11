#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 16 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o RFclass.%N.%a.%j.out # STDOUT
#SBATCH -e RFclass.%N.%a.%j.err # STDERR
#SBATCH --job-name="RSclass"
#SBATCH --array=979

################################################################
GRID_ID="$(($SLURM_ARRAY_TASK_ID + 3000))"

# Settables:
MODNAME='ClassifiedLC17_soy20new'
VARDF="/home/downspout-cel/paraguay_lc/classification/RF/pixdf_lessSoy.csv"

#################################################################
# Probably do not change:
INDIR="/home/downspout-cel/${COUNTRY}_lc/stac/grids/00${GRID_ID}/comp"
MODPATH="/home/downspout-cel/${COUNTRY}_lc/classification/RF/test/${MODNAME}_RFmod.joblib"
VARS=['evi2','gcvi','wi','kndvi','nbr','ndmi']
BANDS="['Max','Min','Amp','Avg','CV','Std','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']"
OUTIMG="${INDIR}/00${GRID_ID}_${MODNAME}.tif"

# activate the virtual environment
conda activate venv.lucinsa38_pipe

# if running from installed module:
LUCinSA_helpers rf_classification --in_dir $INDIR --df_in $VARDF --spec_indices $VARS --stats $BANDS --model_name $MODNAME --rf_mod $MODPATH --img_out $OUTIMG --classification None --importance_method None --ran_hold 0 --out_dir None

conda deactivate
