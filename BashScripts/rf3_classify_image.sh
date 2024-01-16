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
FEATUREMOD='base'
SAMPLEMOD='bal1000'
VARDF="/home/downspout-cel/paraguay_lc/classification/RF/pixdf_lessSoy.csv"
YR=2021
#################################################################
# Probably do not change:
INDIR="/home/downspout-cel/${COUNTRY}_lc/stac/grids/00${GRID_ID}/comp"
MODPATH="/home/downspout-cel/${COUNTRY}_lc/classification/RF/test/${MODNAME}_RFmod.joblib"
SINGDICT='/home/downspout-cel/paraguay_lc/singleton_var_dict.json'
MODDICT='/home/downspout-cel/paraguay_lc/Feature_Models.json'
OUTIMG="${INDIR}/00${GRID_ID}_${MODNAME}.tif"

# activate the virtual environment
conda activate venv.lucinsa38_pipe

# if running from installed module:
LUCinSA_helpers rf_classification --in_dir $INDIR --df_in $VARDF --feature_model $FEATUREMOD --start_yr $YR --sample_model $SAMPLEMOD --feature_mod_dict $MODDICT --singleton_var_dict $SINGDICT --rf_mod $MODPATH  --img_out $OUTIMG --spec_indices None --si_vars None --singleton_vars None --poly_vars None --poly_var_path None --lc_mod None --importance_method None --ran_hold 0 --out_dir None --scratch_dir None

conda deactivate