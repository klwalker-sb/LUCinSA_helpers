#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o VarDF.%N.%a.%j.out # STDOUT
#SBATCH -e VarDF.%N.%a.%j.err # STDERR
#SBATCH --job-name="VarDF"

#Settables:
COUNTRY='paraguay'
OUTDIR='/home/downspout-cel/${COUNRTY}_lc/vector'
SI="[evi2,gcvi,wi,kndvi,nbr,ndmi]"
SIVARS="[Max,Min,Amp,Avg,CV,Std,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec]"
INDIR='/home/downspout-cel/{COUNTRY}_lc/stac/grids'
GRID='/home/sandbox-cel/LUCinLA_grid_8858.gpkg'
#CELLS="[3010,3022,3023]"
CELLS='path/to/list'
POLYS=None
NEWEST=2022
OLDEST=2010
NPTS=5
SEED=88
LOADSAMP='True'
PTFILE='/home/downspout-cel/paraguay_lc/vector/sampleData/SamplePts_Dec2023_ALL.csv'

conda activate venv.lucinsa38_pipe

LUCinSA_helpers make_var_dataframe --out_dir $OUTDIR --spec_indices $SI --si_vars $SIVARS --in_dir $INDIR --grid_file $GRID --cell_list $CELLS --ground_polys $POLYS --oldest $OLDEST --newest $NEWEST --npts $NPTS --seed $SEED --load_samp $LOADSAMP --ptfile $PTFILE

conda deactivate
