#!/bin/bash

#SBATCH -N 1 # number of nodes
#SBATCH -n 4 # number of cores
#SBATCH -t 0-24:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o stac_updatedb.%N.%a.%j.out # STDOUT
#SBATCH -e stac_updatedb.%N.%a.%j.err # STDERR
#SBATCH --job-name="update db"

#Settables:
DBPATH='/home/downspout-cel/paraguay_lc/cell_processing_post_test.csv'
CELLS='All'
DLDIR='/home/sandbox-cel/paraguay_lc/stac/grid'
PDIR='/home/downspout-cel/paraguay_lc/stac/grids'

#Activate the virtual environment (which relys on anaconda)
conda activate venv.lucinsa38_pipe

LUCinSA_helpers update_summary_db --status_db_path $DBPATH --cell_list $CELLS --raw_dir $DLDIR --processed_dir $PDIR

conda deactivate
