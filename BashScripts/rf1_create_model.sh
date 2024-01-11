#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o RFmod.%N.%a.%j.out # STDOUT
#SBATCH -e RFmod.%N.%a.%j.err # STDERR
#SBATCH --job-name="RFmod"
################################################################

COUNTRY='paraguay'
MODNAME="ClassifiedLC17_soy20"

# CLASS= 'All'(=LC25) | 'crop_nocrop' | 'crop_nocrop_medcrop' | 'crop_nocrop_medcrop_tree'| 'veg'(=LC5) | 'cropType'(='LC_crops')
CLASS='All'
# IMPMETH = 'Impurity' | 'Permutation' | 'None'
IMPMETH='Impurity'
VARDF="/home/downspout-cel/paraguay_lc/classification/RF/pixdf_lessSoy.csv"
LUT="../Class_LUT.csv"
RANHOLD=29
OUTDIR='/home/downspout-cel/paraguay_lc/classification/RF/test'

# activate the virtual environment
conda activate venv.lucinsa38_pipe

# if running from installed module:
LUCinSA_helpers rf_model --df_in $VARDF --out_dir $OUTDIR --lc_mod $CLASS --importance_method $IMPMETH --ran_hold $RANHOLD --model_name $MODNAME --lut LUT

conda deactivate
