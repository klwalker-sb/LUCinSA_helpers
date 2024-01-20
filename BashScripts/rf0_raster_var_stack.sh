#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o rfstack.%N.%a.%j.out # STDOUT
#SBATCH -e rfstack.%N.%a.%j.err # STDERR
#SBATCH --job-name="rfstack"
##SBATCH --array=1-8
################################################################

## If a running smallish number of cells, can enter in array above and use these values as the cell input:
#CELLS="${SLURM_ARRAY_TASK_ID}"
#CELLS="$(($SLURM_ARRAY_TASK_ID + 4000))"

## If running a lot of cells, can use a list. To parallelize, can split list into multiple lists in a directory:
#CELLS="/home/downspout-cel/paraguay_lc/vector/sampleData/Training_cells.csv"
CELLS="/home/downspout-cel/paraguay_lc/vector/sampleData/TrainingCells/${SLURM_ARRAY_TASK_ID}.csv"

COUNTRY='paraguay'
TSDIR="/home/downspout-cel/${COUNTRY}_lc/stac/grids/"

MODNAME='base_noseg'
VIs="[evi2,gcvi,wi,kndvi,nbr,ndmi]"
SIVARS="[Max,Min,Amp,Avg,CV,Std,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec]"
SING="forest_strata"
POLYVARS="[pred_ext,pred_dst,pred_area,pred_APR,AvgNovDec_FieldStd]"
POLYPATH="/home/downspout-cel/paraguay_lc/Segmentations/RF_feats/"
SINGDICT="/home/downspout-cel/paraguay_lc/singleton_var_dict.json"
MODDICT="/home/downspout-cel/paraguay_lc/Feature_Models.json"
STARTYR=2021
STARTMO=7
SCRATCH=''
# ####################################################

# activate the virtual environment
conda activate venv.lucinsa38_pipe

LUCinSA_helpers make_var_stack --in_dir $TSDIR --cell_list $CELLS --feature_model $MODNAME --start_yr $STARTYR --start_mo $STARTMO --spec_indices $VIs --si_vars $SIVARS --feature_mod_dict $MODDICT --singleton_vars $SING --singleton_var_dict $SINGDICT --poly_vars $POLYVARS --poly_var_path $POLYPATH --scratch_dir=$SCRATCH

conda deactivate
