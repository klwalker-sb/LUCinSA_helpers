#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH -t 0-48:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o stac_brdf_py.%N.%a.%j.out # STDOUT
#SBATCH -e stac_brdf_py.%N.%a.%j.err # STDERR
#SBATCH --job-name="brdf_py"
#SBATCH --array=283

####Run as array job:
GRID_ID=$(($SLURM_ARRAY_TASK_ID + 3000))
#GRID_ID=$SLURM_ARRAY_TASK_ID

#Set permissions on output files
umask 002

#Settables:
PROJECT_DIR="/home/sandbox-cel/paraguay_lc"
PROJECT_PATH="${PROJECT_DIR}/stac/grid/00${GRID_ID}/"
OUT_PATH="${PROJECT_DIR}/stac/grid/00${GRID_ID}/brdf"
GRID_FILE="/home/sandbox-cel/LUCinLA_grid_8858.gpkg"
COEFFS="/home/klwalker/tmp/eostac/files/coefficients"
EPSG=8858
BUFFER=100

###################################################################
### activate the virtual environment
conda activate venv.lucinsa38_test3
#conda activate venv.lucinla38_dl

TIMESTAMP0=`date "+%Y-%m-%d %H:%M:%S"`

START_YEAR=2020
END_YEAR=2023
y=$START_YEAR
while [ $y -ne $END_YEAR ]
do	
	START_DATE="${y}-1-01"
	END_DATE="${y}-12-31"
	echo  Working on $START_DATE to $END_DATE >&2 
	TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`
	echo $TIMESTAMP >&2

	eostac brdf --start-date $START_DATE --end-date $END_DATE --project-path $PROJECT_PATH --out-path $OUT_PATH --threads 8 --apply-bandpass --coeffs-path $COEFFS

	y=$(($y+1))
done

TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`
TIMETOT=$(($(date -d "$TIMESTAMP" "+%s") - $(date -d "$TIMESTAMP0" "+%s") ))
echo done at $TIMESTAMP >&2
echo full process took: $($TIMETOT/60) minutes >&2

conda deactivate
