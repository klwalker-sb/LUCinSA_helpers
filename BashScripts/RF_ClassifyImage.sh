#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 16 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o RFclass.%N.%a.%j.out # STDOUT
#SBATCH -e RFclass.%N.%a.%j.err # STDERR
#SBATCH --job-name="RSclass"
#SBATCH --array=40,41,59,60
##806,841-845,877,878,911-914,944-947,976-980,695-697,730-732,766-771,873-876,803,805,879


################################################################

#GRID_ID="${SLURM_ARRAY_TASK_ID}"
GRID_ID="$(($SLURM_ARRAY_TASK_ID + 4000))"

COUNTRY='paraguay'

##For archive:
INDIR="/raid-cel/r/downspout-cel/${COUNTRY}_lc/stac/grids/00${GRID_ID}/comp"
VARS=['evi2','gcvi','wi','kndvi','nbr','ndmi']
BANDS="['Max','Min','Amp','Avg','CV','Std','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']"

VARDF="/home/downspout-cel/paraguay_lc/vector/RFdf_Feb23.csv"
CLASS='All'
IMPMETH='Impurity'

OUTDIR='/home/downspout-cel/paraguay_lc/vector'
OUTIMG="/raid-cel/r/downspout-cel/${COUNTRY}_lc/stac/grids/00${GRID_ID}/comp/stack.tif"
CLASSIMG="/raid-cel/r/downspout-cel/${COUNTRY}_lc/stac/grids/00${GRID_ID}/comp/00${GRID_ID}_ClassifiedLC2.tif"

# activate the virtual environment
conda activate venv.lucinsa38_dl

python rf2_classify_image.py $INDIR $VARS $BANDS $VARDF $CLASS $IMPMETH $OUTDIR $OUTIMG $CLASSIMG

conda deactivate
