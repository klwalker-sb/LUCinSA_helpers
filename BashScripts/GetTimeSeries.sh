#!/bin/bash

#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o TS.%N.%a.%j.out # STDOUT
#SBATCH -e TS.%N.%a.%j.err # STDERR
#SBATCH --job-name="TS"

#####################################################

OUT_DIR="/home/downspout-cel/chile_lc/OutputData/TSdfs"
SPEC_INDEX='evi2'
STARTYR=2012
ENDYR=2020
IMGDIR='/home/sandbox-cel/chile_lc/raster/grids'
IMGTYPE='TS'
GRIDFILE="/home/sandbox-cel/chile_lc/chl_grids.gpkg"
CELLLIST=(1157 1158)
GROUNDPOLYS="None"
OLDEST=2018
NEWEST=2018
NPTS=2
SEED=88
LOADSAMP="True"
PTFILE="/home/sandbox-cel/chile_lc/vector/sampleData/Arauco1A_Natural2014.csv"

#####################################################

## As an array job
#GRID_ID=${SLURM_ARRAY_TASK_ID}


# activate the virtual environment
source ~/.nasaenv/bin/activate

LUCinSA_helpers GetTimeSeries --out_dir ${OUT_DIR} --spec_index ${SPEC_INDEX} --StartYr $STARTYR  --EndYr $ENDYR --img_dir ${IMGDIR} --imageType $IMGTYPE --gridFile ${GRIDFILE} --cellList ${CELLLIST} --groundPolys ${GROUNDPOLYS} --oldest $OLDEST --newest $NEWEST --npts $NPTS --seed $SEED --loadSamp $LOADSAMP --ptFile ${PTFILE}

deactivate
