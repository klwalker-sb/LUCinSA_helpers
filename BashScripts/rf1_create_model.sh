#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o RFmod.%N.%a.%j.out # STDOUT
#SBATCH -e RFmod.%N.%a.%j.err # STDERR
#SBATCH --job-name="RFmod"
################################################################

COUNTRY='paraguay'
#VARS=['evi2','gcvi','wi','kndvi','nbr','ndmi']
#BANDS="['Max','Min','Amp','Avg','CV','Std','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']"

# CLASS= 'All'(=LC17) | 'crop_nocrop' | 'crop_nocrop_medcrop' | 'crop_nocrop_medcrop_tree'| 'veg'(=LC5) | 'cropType'(='LC_crops')
MODNAME="ClassifiedLC17_soy20"
CLASS='All'
# IMPMETH = 'Impurity' | 'Permutation' | 'None'
IMPMETH='Impurity'
VARDF="/home/downspout-cel/paraguay_lc/classification/RF/pixdf_lessSoy.csv"
RANHOLD=29
MODNAME="ClassifiedLC17_soy20"

# Probably do not change:
OUTDIR='/home/downspout-cel/paraguay_lc/classification/RF/test'
# activate the virtual environment
conda activate venv.lucinsa38_test3

# if running from rf1_create_model.py:
#python RF1_create_model.py $VARDF $CLASS $IMPMETH $OUTDIR $RANHOLD $MODNAME

# if running from installed module:
LUCinSA_helpers rf_model --df_in $VARDF --out_dir $OUTDIR --classification $CLASS --importance_method $IMPMETH --ran_hold $RANHOLD --model_name $MODNAME

conda deactivate
