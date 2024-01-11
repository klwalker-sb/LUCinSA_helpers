#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o rfstack.%N.%a.%j.out # STDOUT
#SBATCH -e rfstack.%N.%a.%j.err # STDERR
#SBATCH --job-name="rfstack"
#SBATCH --array=26-29
################################################################

#GRID_ID="${SLURM_ARRAY_TASK_ID}"
GRID_ID="$(($SLURM_ARRAY_TASK_ID + 4000))"

COUNTRY='paraguay'
IMGDIR="/home/downspout-cel/${COUNTRY}_lc/stac/grids/00${GRID_ID}"
VIs="[evi2,gcvi,wi,kndvi,nbr,ndmi]"
SIVARS="[Max,Min,Amp,Avg,CV,Std,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec]"
SING="forest_strata"
MODNAME='base_noseg'
POLYVARS=''
SINGDICT='/home/downspout-cel/paraguay_lc/singleton_var_dict.json'
MODDICT='/home/downspout-cel/paraguay_lc/Feature_Models.json'
STARTYR=2021

# ####################################################

# activate the virtual environment
conda activate venv.lucinsa38_pipe

LUCinSA_helpers make_var_stack --ts_dir $IMGDIR --feature_model $MODNAME --start_yr $STARTYR --spec_indices $VIs --si_vars $SIVARS --singleton_vars $SING --poly_vars $POLYVARS --singleton_var_dict $SINGDICT --feature_mod_dict $MODDICT

conda deactivate