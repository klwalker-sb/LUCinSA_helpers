#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o zonal.%N.%a.%j.out # STDOUT
#SBATCH -e zonal.%N.%a.%j.err # STDERR
#SBATCH --job-name="zonal"
################################################################

################################################################
POLYS="/home/downspout-cel/paraguay_lc/vector/zonal_polys/sm_districts/smSampDistricts.shp"
#MAPDIR="/home/downspout-cel/paraguay_lc/lc_prods"
MAPDIR="/home/downspout-cel/paraguay_lc/mosaics"
MAP='CEL_base4Poly6_bal300mix3_mosaic'
DICT=None
CLIPDIR="/home/downspout-cel/paraguay_lc/vector/tests"
OUTDIR="/home/downspout-cel/paraguay_lc/vector/tests"

# ####################################################

# activate the virtual environment
conda activate venv.lucinsa38_pipe

LUCinSA_helpers summarize_zones --polys $POLYS --map_dir $MAPDIR --clip_dir $CLIPDIR --map_product $MAP --out_dir $OUTDIR --map_dict $DICT

conda deactivate
