#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic 
#SBATCH -o stac_clean.%N.%a.%j.out # STDOUT
#SBATCH -e stac_clean.%N.%a.%j.err # STDERR
#SBATCH --job-name="stClean"
#SBATCH --array=152-154

#GRIDS="${SLURM_ARRAY_TASK_ID}"
GRIDS="$(($SLURM_ARRAY_TASK_ID + 3000))"
################################################

###############################################
# Directories to clean: downloads,brdf,nocoreg,processing_db
#   format:
#       list: comma, no quotes -- e.g. CLEANUP=downloads,nocoreg
#       single: brackets with quotes -- e.g. CLEANUP=['downloads']

CLEANUP=downloads,nocoreg
#CLEANUP=['downloads']

# Sensor filter:
#   SAT_SENSORS=S2A,S2B,LT05,LE07,LC08,LC09
#   can also do S2 for all sentinel only, L for all Landsat only, or 'All' for all
#   format:
#       list: comma, no quotes -- e.g. SAT_SENSORS=LT05,LE07
#       single: quotes (no brackets) -- e.g. SAT_SENSORS='LE07'
SAT_SENSORS='L'

#################################################
# Date filter:
#   format: '[YYYYMMDD, YYYYMMDD]'
#   if a single image is to be cleaned, can use '[YYYYMMDD]'
#   if all files are to be cleaned (all dates), use '[0]'

#DATES='[20100101,20110101]'
DATES='[0]'

##################################################
# Xlist:
#    can use list of images to remove
#    format is path to .csv file with one image per row (no heading) 
#    file names can be from download or brdf folder 
#         (if from download, will remove brdf as well)
#    image will be flagged in database so that it is not reprocessed.
#    default is ''

#XLIST='3491_low_quality.csv'
XLIST=''

###############################
# DO NOT MODIFY BELOW THIS LINE
###############################

# activate the virtual environment
conda activate venv.lucinsa38_pipe

CONFIG_UPDATES="grids:[${GRIDS}] res:${REF_RES} crs:${REF_CRS}
dlMehod:STAC
clean:sat_sensors:${SAT_SENSORS}
clean:remove_items:${CLEANUP}
clean:date_range:${DATES}
clean:xlist:${XLIST}
main_path:/home/sandbox-cel/paraguay_lc/stac/grid
backup_path:/home/downspout-cel/paraguay_lc/stac/grids
num_workers:${SLURM_CPUS_ON_NODE}"

tuyau clean --config-updates $CONFIG_UPDATES

conda deactivate
