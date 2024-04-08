#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o resamp.%N.%a.%j.out # STDOUT
#SBATCH -e resamp.%N.%a.%j.err # STDERR
#SBATCH --job-name="resamp"
##SBATCH --array=
################################################################

## If a running smallish number of cells, can enter in array above and use these values as the cell input:
#CELLS="$(($SLURM_ARRAY_TASK_ID + 3000))"

CELLS="/home/downspout-cel/paraguay_lc/vector/sampleData/TrainingCells.csv"
## If running a lot of cells, can use a list. To parallelize, can split list into multiple lists in a directory:
#CELLS="/home/downspout-cel/paraguay_lc/vector/sampleData/TrainingCells/${SLURM_ARRAY_TASK_ID}.csv"


TSDIR='/home/downspout-cel/paraguay_lc/stac/grids'
VIs=("gcvi" "kndvi" "nbr" "ndmi")
YRS=("2021" "2022")
OUTDIR="/home/sandbox-cel/paraguay_lc/stac/ts_30m"
RESNEW=30
# ####################################################

# activate the virtual environment
conda activate venv.lucinsa38_pipe

for VI in "${VIs[@]}"
do
	DIR="brdf_ts/ms/${VI}"
	for YR in "${YRS[@]}"
	do 
		LUCinSA_helpers downsample --cell_list $CELLS --in_dir_main $TSDIR --local_dir $DIR --common_str $YR --out_dir_main $OUTDIR --new_res $RESNEW
	done
done

conda deactivate
