#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 4 # number of cores
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o RFclass.%N.%a.%j.out # STDOUT
#SBATCH -e RFclass.%N.%a.%j.err # STDERR
#SBATCH --job-name="RSclass"
#SBATCH --array=1-20%8
################################################################

## If a running smallish number of cells, can enter in array above and use these values as the cell input:
#CELLS="${SLURM_ARRAY_TASK_ID}"
CELLS="$(($SLURM_ARRAY_TASK_ID + 4000))"

## If running many cells, can use a list. To parallelize, can split list into multiple lists in a directory:
#CELLS="/home/downspout-cel/paraguay_lc/cell_lists/TrainingCells/${SLURM_ARRAY_TASK_ID}.csv"
############################################################################################## Settables:

FEATMOD='base4Poly6'
#FEATMOD='base4NoPoly'
SAMPMOD='bal300mix8'
LCMOD='all'
#LUTCOL='LC25'
TRAINYRS=2021
YR=2021
STARTMO=6

##### For all models using default time series:
CELLDIR="/home/sandbox-cel/paraguay_lc/stac/grid"
MODDIR="/home/downspout-cel/paraguay_lc/classification/RF/"

##### For models using Landsat only:
#CELLDIR="/home/sandbox-cel/paraguay_lc/stac/ts_Lonly"
#FEATMOD='base4NoPolyLonly'

##### For models using 30m res:
#CELLDIR="/home/sandbox-cel/paraguay_lc/stac/ts_30m"
#FEATMOD='base4NoPoly30m'

############## For current 2021 map
#FEATMOD='base4Poly6'
#SAMPMOD='bal300mix8'
#LCMOD='max'
#ALTMOD='currentMap_base4Poly6_bal300mix8_LC32_21_RFmod.joblib"
#############################################

##################################################################
# Probably do not change below
##################################################################
MODNAME="${FEATMOD}_${SAMPMOD}_LC25_21"
# ALTMOD is None If using default rf model named ${MODNAME}_RFmod.joblib"
ALTMOD=None
VARDF="/home/downspout-cel/paraguay_lc/classification/inputs/pixdf_${MODNAME}.csv"
SINGDICT="/home/downspout-cel/paraguay_lc/singleton_var_dict.json"
MODDICT="/home/downspout-cel/paraguay_lc/Feature_Models.json"
ALTOUT=None
#ALTOUT="${INDIR}/00${GRID_ID}_${MODNAME}.tif"
LUT="/home/klwalker/Jupyter/LUCinSA_helpers/LUCinSA_helpers/Class_LUT.csv"

#############################################
# Turn off NumPy parallelism and rely on dask
#############################################
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# This should be sufficient for OpenBlas and MKL
export OMP_NUM_THREADS=1
################################################

# activate the virtual environment
conda activate venv.lucinsa38_pipe

LUCinSA_helpers rf_classification --cell_dir $CELLDIR --cell_list $CELLS --mod_dir $MODDIR --feature_model $FEATMOD --samp_model $SAMPMOD --lc_mod $LCMOD --train_yrs $TRAINYRS --start_yr $YR --start_mo $STARTMO --feature_mod_dict $MODDICT --rf_mod $ALTMOD --img_out $ALTOUT --df_in $VARDF --singleton_var_dict $SINGDICT  --spec_indices None --si_vars None --spec_indices_pheno None  --pheno_vars None --singleton_vars None --poly_vars None --poly_var_path None --combo_bands None --lc_mod None --lut $LUT --importance_method None --ran_hold 0 --out_dir None --scratch_dir None

conda deactivate
