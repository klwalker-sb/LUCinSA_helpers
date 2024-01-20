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
## If a running smallish number of cells, can enter in array above and use these values as the cell input:
#CELLS="${SLURM_ARRAY_TASK_ID}"
CELLS="$(($SLURM_ARRAY_TASK_ID + 3000))"

## If running a lot of cells, can use a list. To parallelize, can split list into multiple lists in a directory:
#CELLS="/home/downspout-cel/paraguay_lc/vector/sampleData/Training_cells.csv"
#CELLS="/home/downspout-cel/paraguay_lc/vector/sampleData/TrainingCells/${SLURM_ARRAY_TASK_ID}.csv"

# Settables:
FEATUREMOD='base'
SAMPLEMOD='bal1000'
VARDF="/home/downspout-cel/paraguay_lc/classification/RF/pixdf_lessSoy.csv"
YR=2021
STARTMO=7
#################################################################
# Probably do not change:
INDIR="/home/downspout-cel/${COUNTRY}_lc/stac/grids"
MODPATH="/home/downspout-cel/${COUNTRY}_lc/classification/RF/test/${MODNAME}_RFmod.joblib"
SINGDICT='/home/downspout-cel/paraguay_lc/singleton_var_dict.json'
MODDICT='/home/downspout-cel/paraguay_lc/Feature_Models.json'
OUTIMG="None"
LUT="../Class_LUT.csv"

# activate the virtual environment
conda activate venv.lucinsa38_pipe

# if running from installed module:
LUCinSA_helpers rf_classification --in_dir $INDIR --cell_list $CELLS --df_in $VARDF --feature_model $FEATUREMOD --start_yr $YR --start_mo $STARTMO --sample_model $SAMPLEMOD --feature_mod_dict $MODDICT --singleton_var_dict $SINGDICT --rf_mod $MODPATH  --img_out $OUTIMG --spec_indices None --si_vars None --singleton_vars None --poly_vars None --poly_var_path None --lc_mod None --importance_method None --ran_hold 0 --out_dir None --scratch_dir None

conda deactivate