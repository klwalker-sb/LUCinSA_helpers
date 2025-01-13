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

#CLASS= 'all'(=LC25) | 'crop_nocrop' | 'crop_nocrop_medcrop' | 'crop_nocrop_medcrop_tree'| 'veg'(=LC5) | 'cropType'(='LC_crops')
CLASS='all'
lUTCOL = 'LC25'
FEATMOD='base4Poly6'
#SAMPMOD='base1000'
SAMPMOD='bal200mix7'
YRS=2021

# IMPMETH = 'Impurity' | 'Permutation' | 'None'
IMPMETH='Impurity'
RANHOLD=29
THRESH=0

### Variables to create new model if feature model does not exist in FEATDICT (need to update this)
#VARS=['evi2','gcvi','wi','kndvi','nbr','ndmi']
#BANDS="['Max','Min','Amp','Avg','CV','Std','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']"

# Probably do not change:
modname = "${FEATMOD}_${SAMPMOD}_${LUTCOL}_21"
VARDF="/home/downspout-cel/paraguay_lc/classification/inputs/pixdf_${MODNAME}.csv"
OUTDIR='/home/downspout-cel/paraguay_lc/classification/RF'
LUT="/home/klwalker/Jupyter/LUCinSA_helpers/LUCinSA_helpers/Class_LUT.csv"
FEATDICT="/home/downspout-cel/paraguay_lc/Feature_Models.json"

# activate the virtual environment
conda activate venv.lucinsa38_pipe

# if running from rf1_create_model.py:
#python RF1_create_model.py $VARDF $CLASS $IMPMETH $OUTDIR $RANHOLD $MODNAME

# if running from installed module:
LUCinSA_helpers rf_model --df_in $VARDF --out_dir $OUTDIR --lc_mod $CLASS --importance_method $IMPMETH --ran_hold $RANHOLD --lut $LUT --feature_model $FEATMOD --samp_mod $SAMPMOD --train_yrs $YRS --thresh $THRESH --feature_mod_dict $FEATDICT

conda deactivate
