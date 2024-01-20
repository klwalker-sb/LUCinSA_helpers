#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 4 # number of cores
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o rfstack.%N.%a.%j.out # STDOUT
#SBATCH -e rfappend.%N.%a.%j.err # STDERR
#SBATCH --job-name="rfappend"
##SBATCH --array=737
################################################################

## If a running smallish number of cells, can enter in array above and use these values as the cell input:
#CELLS="${SLURM_ARRAY_TASK_ID}"
#CELLS="$(($SLURM_ARRAY_TASK_ID + 3000))"

## If running a lot of cells, can use a list. To parallelize, can split list into multiple lists in a directory:
#CELLS="/home/downspout-cel/paraguay_lc/vector/sampleData/Training_cells.csv"
#CELLS="/home/downspout-cel/paraguay_lc/vector/sampleData/TrainingCells/${SLURM_ARRAY_TASK_ID}.csv"
CELLS="[3737,3738]"
COUNTRY='paraguay'
TSDIR="/home/downspout-cel/${COUNTRY}_lc/stac/grids"
MODNAME='append'
#VIs="[evi2,gcvi,wi,kndvi,nbr,ndmi]"
VIs="[evi2]"
#SIVARS="[Max,Min,Amp,Avg,CV,Std,MaxDateCos,MinDateCos,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec]"
SIVARS="[MaxDateCos, MinDateCos]" 
SING="forest_strata"
PTFILE="/home/downspout-cel/paraguay_lc/vector/sampleData/SamplePts_Dec2023_ALL.csv"
FEATDF="/home/downspout-cel/paraguay_lc/vector/ptsgdb_Dec18.csv"
GRIDFILE="/home/sandbox-cel/paraguay_lc/vector/pry_grids.gpkg"
POLYVARS="[pred_ext,pred_dst,pred_area,pred_APR,AvgNovDec_FieldStd]"
POLYPATH="/home/downspout-cel/paraguay_lc/Segmentations/RF_feats/"
SINGDICT='/home/downspout-cel/paraguay_lc/singleton_var_dict.json'
STARTYR=2021
STARTMO=7
SCRATCH="/home/scratch-cel"
#SCRATCH="/home/scratch-cel"
# ####################################################

# activate the virtual environment
conda activate venv.lucinsa38_pipe

LUCinSA_helpers append_feature_dataframe --in_dir $TSDIR --ptfile $PTFILE --feat_df $FEATDF --cell_list $CELLS --grid_file $GRIDFILE  --start_yr $STARTYR --start_mo $STARTMO --spec_indices $VIs --si_vars $SIVARS --singleton_vars $SING --singleton_var_dict $SINGDICT --poly_vars $POLYVARS --poly_var_path $POLYPATH --scratch_dir=$SCRATCH

conda deactivate
