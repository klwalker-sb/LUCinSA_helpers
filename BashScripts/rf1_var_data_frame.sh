#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o VarDF.%N.%a.%j.out # STDOUT
#SBATCH -e VarDF.%N.%a.%j.err # STDERR
#SBATCH --job-name="VarDF"

#Settables:
COUNTRY="paraguay"
OUTDIR="/home/downspout-cel/${COUNTRY}_lc/vector"
INDIR="/home/downspout-cel/${COUNTRY}_lc/stac/grids"
GRID="/home/sandbox-cel/LUCinLA_grid_8858.gpkg"
#CELLS="[3010,3022,3023]"
CELLS="/home/downspout-cel/${COUNTRY}_lc/vector/sampleData/Sample_cells.csv"
MODEL='testing'
MODDICT="/home/downspout-cel/${COUNTRY}_lc/Feature_Models.json"
YR=2021
POLYS='None'
NEWEST=2022
OLDEST=2010
NPTS=5
SEED=88
LOADSAMP='True'
PTFILE="/home/downspout-cel/paraguay_lc/vector/sampleData/SamplePts_Dec2023_ALL.csv"

conda activate venv.lucinsa38_pipe

LUCinSA_helpers make_var_dataframe --in_dir $INDIR --out_dir $OUTDIR --grid_file $GRID --cell_list $CELLS --feature_model $MODEL --feature_mod_dict $MODDICT --start_yr $YR --polyfile $POLYS --oldest $OLDEST --newest $NEWEST --npts $NPTS --seed $SEED --load_samp $LOADSAMP --ptfile $PTFILE

conda deactivate
