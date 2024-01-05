# LUCinSA_helpers
Helper functions and notebooks to interact with data on High-Performance Computing environment, designed to be used in conjunction with processing guide for remote sensing projects on Land-Use Change in Latin America: https://klwalker-sb.github.io/LUCinLA_stac/ 

## Uses

### To summarize processing status and uncover errors amidst large numbers of files on HPC cluster
* [check download logs](#check-download-logs)
* reconstruct_db
* check_processing
* [get processing summary](#get-processing-summary)
* get_cell_status

### To quickly visualize inputs and outputs of time-series analysis for quality control and interactive troubleshooting
* get_time_series 
* make_ts_composite
* interactive notebooks

### To set and document parameter choices and compare outputs for model optimization

### Also (temporarily) hosts functions to:
####        create modelling features from smoothed time-series indices and segmentation outputs 
####        create single-year random forest classification model
* rf_model 
####        apply random forest model to gridded data to create wall-to-wall ma
* rf_classification 
* mosaic

## to install:
with your LUCinLA pipeline environment activated (see [this page of the LUCinLA_stac guide](https://klwalker-sb.github.io/LUCinLA_sta/Pipeline.html))
clone this repo into your homespace on the HPC cluster and install it: 
```
git clone https://github.com/klwalker-sb/LUCinSA_helpers
cd LUCinSA_helpers
python setup.py build && pip install .
```
Jupyter notebook tools also need to be installed into your environment if not already:
```
conda install -c conda-forge notebook ipykernel jupyter_contrib_nbextensions
```
To be able to run interactive notebook features (e.g. click on point to get time series, print thumbnails), additional modules are needed:
```
conda install -c conda-forge ipywidgets ipyleaflet localtileserver
conda install -c anaconda pillow
```

## to use:
Installed processes that can be run from command line / SLURM:
                        
Example SLURM scripts are included for these processes in the BashScripts folder.
get_time_series, rf_classification, and mosaic use significant resources and should be run through SLURM for accounting (when running 
processes through jupyter notebook, the resources are not accounted for and might result in crashing the whole HPC cluster if resources are
already maxed out and more are demanded)

Many small processes can be run through the Jupyter notebooks in the notebooks folder.
To access a Jupyter notebook via the cluster see [this page of the LUCinLA_STAC guide](https://klwalker-sb.github.io/LUCinLA_stac/)
Before running Jupyter notebooks, set the parameters in `notebook_params.ipynb` and run all cells of that notebook.
By doing so, you should not need to alter the text in any of the notebooks themselves (unless adding/modifying methods).

To save a notebook with displayed outputs, run the last cell of the notebook:
`To save an html copy of this notebook with all outputs`. This will save an html copy in the `notebooks/Outputs` folder. 
Relevant parameter settings are printed within the notebook, enabling reproducibility.

### note regarding edits/contributions:
When edits are pushed, notebook outputs are automatically cleared at commit staging, as per
[this post](https://medium.com/somosfit/version-control-on-jupyter-notebooks-6b67a0cf12a3)
This helps to keep this git repo from getting overwhelmed with notebook data.
Also make sure to use the original .gitignore file (which might be hidden) when pushing edits

## check download logs
Checks the download .err logs from eostac for errors and gaps and adds these to a database for cumulative checking. (downloading from STAC catalogs can result in dropped files and timeout errors that can kill the process. By running downloads in small time chunks (e.g. monthly) in a loop over a multi-year time period, errors should be minimal, but some months will be dropped. The resulting log files are too long to open and read individually. This function provides a way to succinctly summarize errors within them.
```
LUCinSA_helpers check_dl_logs \
    --cell_db_path '/path/to/cell_processing_dl_test.csv'
    --archive_path 'path/to/directory_for_checked_eostac_logs' 
    --log_path '/path/to/directory_with_unchecked_logs'
    --stop_date '2022-12-31'
    --start_date '2000-01-01' 
    --ignore_dates ('2022-11-01,2022-12-31')
```
example output:
![alt](/images/dl_log_check.png)

## get processing summary
![alt](/images/processing_summary.jpg)
![alt](/images/processing_summary_qual.jpg)