#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 4 # number of cores
#SBATCH -t 0-20:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o stacdl1_py.%N.%a.%j.out # STDOUT
#SBATCH -e stacdl1_py.%N.%a.%j.err # STDERR
#SBATCH --job-name="stacdl1_py"
#SBATCH --array=93,98,28,29,34,35,116,292%2
##############################################
### As an array job:
GRID_ID=$(($SLURM_ARRAY_TASK_ID + 3000))
echo cell_id = $GRID_ID >&2
# Set permissions on output files
umask 002

##Settables:

PROJECT_HOME="/home/sandbox-cel/paraguay_lc/stac"
LANDSAT_DIR="${PROJECT_HOME}/grid/00${GRID_ID}/landsat"
SENTINEL_DIR="${PROJECT_HOME}/grid/00${GRID_ID}/sentinel2"
FILETYPE='.tif'
OUT_DIR="${PROJECT_HOME}/grid/00${GRID_ID}"
L7STOPYR=2017
GRID_FILE="/home/sandbox-cel/LUCinLA_grid_8858.gpkg"
EPSG=8858
BUFFER=100

################################################
### activate the virtual environment
#conda activate venv.lucinsa38_dl
conda activate venv.lucinsa38_test3

## if directories are not empty, run script to check for corrupt files
if [ -n "$LANDSAT_DIR" ]
then	
  eostac check --out-path $LANDSAT_DIR --file-type $FILETYPE
fi

if [ -n "$SENTINEL_DIR" ]
then
  eostac check --out-path $SENTINEL_DIR --file-type $FILETYPE
fi

TIMESTAMP0=`date "+%Y-%m-%d %H:%M:%S"`

START_YEAR=2000
END_YEAR=2023	
YEAR=$START_YEAR
while [ $YEAR -ne $END_YEAR ]
do
	for m in {1..11}
	do
		CURRENT_MONTH=$(printf "%02d" $m)
		NEXT_ITER=$(($m+1))
		NEXT_MONTH=$(printf "%02d" $NEXT_ITER)
		START_DATE="${YEAR}-${CURRENT_MONTH}-01"
                
                if [[ $m -eq 11 ]]
                then
                  END_DATE="${YEAR}-${NEXT_MONTH}-31"
                else
		  END_DATE="${YEAR}-${NEXT_MONTH}-01"
                fi

		echo -e \\ Working on ${START_DATE} to ${END_DATE}>&2	
		TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`
		echo $TIMESTAMP >&2

		eostac download --start-date $START_DATE --end-date $END_DATE --bounds $GRID_FILE --bounds-query UNQ==$GRID_ID --out-path $OUT_DIR --epsg $EPSG --bounds-buffer $BUFFER --l7-stop_year $L7STOPYR --max-items -1 -w 4 -t 2 
	
	done
	YEAR=$(($YEAR+1))
done

TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`
TIMETOT=$(($(date -d "$TIMESTAMP" "+%s") - $(date -d "$TIMESTAMP0" "+%s") ))
echo Done at $TIMESTAMP >&2
echo full process took: $(($TIMETOT/60)) minutes >&2
echo core used: $SLURM_NTASKS >&2
conda deactivate
