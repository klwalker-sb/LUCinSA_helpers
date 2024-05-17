#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH -t 0-12:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o RFiter.%N.%a.%j.out # STDOUT
#SBATCH -e RFiter.%N.%a.%j.err # STDERR
#SBATCH --job-name="RFiter"
################################################################

PTS='/home/downspout-cel/paraguay_lc/vector/sampleData/SamplePts_Mar2024_ALL.csv'
MODDIR='/home/downspout-cel/paraguay_lc/classification/RF'
SCRATCH='/home/scratch-cel'
SAMPMOD=['bal100']
FEATMODS=['base4NoPoly']
#CLASSMODS="[cropNoCrop,crop_nocrop_mixcrop,crop_nocrop_medcrop,crop_nocrop_medcrop_tree,all]"
CLASSMODS=['all']
ITER=5
NEWHO='False'
LUT='../Class_LUT.csv'

# activate the virtual environment
conda activate venv.lucinsa38_pipe

# if running from installed module:
LUCinSA_helpers iterate_models --sample_pts $PTS --model_dir $MODDIR --scratch_dir $SCRATCH --lut $LUT --samp_model $SAMPMOD --feat_models $FEATMODS --class_models $CLASSMODS --get_new_hos $NEWHO

conda deactivate