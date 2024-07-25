#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o mosaic.%N.%a.%j.out # STDOUT
#SBATCH -e mosaic.%N.%a.%j.err # STDERR
#SBATCH --job-name="mosaic"
################################################################

################################################################
CELLS='/home/downspout-cel/paraguay_lc/mosaics/lists/sm_and_P1.csv'
MAINDIR='/home//sandbox-cel/paraguay_lc/stac/grid'
LOCALDIR='comp'
MOD='base4Poly6_bal300mix3'
OUTDIR='/home/downspout-cel/paraguay_lc/mosaics'

# ####################################################

# activate the virtual environment
conda activate venv.lucinsa38_pipe

LUCinSA_helpers mosaic --cell_list $CELLS --in_dir_main $MAINDIR --in_dir_local $LOCALDIR --common_str $MOD --out_dir $OUTDIR

conda deactivate
