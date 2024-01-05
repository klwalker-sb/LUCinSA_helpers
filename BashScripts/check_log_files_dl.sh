#!/bin/bash

#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o stac_checkDls.%N.%a.%j.out # STDOUT
#SBATCH -e stac_checkDls.%N.%a.%j.err # STDERR
#SBATCH --job-name="checkDls"

#Settables:

DBPATH='/home/downspout-cel/paraguay_lc/cell_processing_dl_test.csv'
ARCHIVE='/home/klwalker/archive/eostac_logs'
LOGS='.'
START='2000-01-01'
STOP='2022-12-31'
IGNORE=('2022-11-01,2022-12-31')

#Activate the virtual environment (which relys on anaconda)
conda activate venv.lucinsa38_pipe

LUCinSA_helpers check_dl_logs --cell_db_path $DBPATH --archive_path $ARCHIVE --log_path $LOGS --stop_date $STOP --start_date $START --ignore_dates $IGNORE

conda deactivate
