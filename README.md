# LUCinSA_helpers
Helper functions and notebooks to interact with data on High-Performance Computing environment, designed to be used in conjunction with processing guide for remote sensing projects on Land-Use Change in Latin America: https://klwalker-sb.github.io/LUCinLA_stac/ 

## Uses

### To summarize processing status and uncover errors amidst large numbers of files in HPC environment
#####      single cell error checking:
* [check download logs](#check-download-logs)(`check_dl_logs`)
* [check processing status for cell](#check-processing-status-for-cell)(`get_cell_status`)
* [identify external image errors](#identify-external-image-errors)(
* [identify internal image errors](#identify-internal-image-errors)
* get_cell_status
#####      multi-cell summarization and error checking
* [summarize images processed](#summarize-images-processed)(`summarize_images_multicell`)
* [get processing summary](#get-processing-summary)

### To quickly visualize inputs and outputs of time-series analysis for quality control and interactive troubleshooting
* get_time_series 
* make_ts_composite
* interactive notebooks

### To set and document parameter choices and compare outputs for model optimization

### Also (temporarily) hosts functions to:
###        create modelling features from smoothed time-series indices and segmentation outputs 
###        create single-year random forest classification model
* rf_model 
###        apply random forest model to gridded data to create wall-to-wall map
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
    --cell_db_path '/path/to/cell_processing_dl_test.csv' \
    --archive_path 'path/to/directory_for_checked_eostac_logs' \
    --log_path '/path/to/directory_with_unchecked_logs'   \
    --stop_date '2022-12-31'   \
    --start_date '2000-01-01'  \
    --ignore_dates ('2022-11-01,2022-12-31')  
```
example output:
![alt](/images/dl_log_check.png)

## check processing status for cell

The `get_cell_status` function can also be used to provide a summary for an individual cell (and is used collectively within `update_summary_db` below)
```
LUCinSA_helpers get_cell_status \
      -- raw_dir = 'path/to/main_downloading_directory' \
      -- processed_dir = 'path/to/main_processing_directory' \
      -- grid_cell =  XXXXX \
      -- yrs = [YYYY-YYYY] \
      -- print_plot = True \
      -- out_dir = path/to/local/output/directory' \
      -- data_source = 'stac' \
```
The above relies on the processing database that each cell has in its main directory named `processing.info`, which has an entry for each image encountered in the STAC catalog and data regarding its status through the downloading,brdf,and coregistration processing steps.

If `processing.info` is corrupted or deleted for a cell, it can be recreated with `reconstruct-db` (but note that it will not contain all of the detail of the original database):
```
LUCinSA_helpers reconstruct_db \
     --processing_info_path = 'path/to/cell_directory/processing.info' \
     --landsat_path = 'path/to/cell_directory/landsat'  \
     --sentinel2_path = 'path/to/cell_directory/sentinel2' \
     --brdf_path = 'path/to/cell_directory/brdf'  \
```
![alt](/images/images_processed_for_cell_by_sensor.jpg)
![alt](/images/images_processed_for_cell_by_stat.jpg)

## identify external image errors
Processing errors raised within processes are noted in the `processing.info` database.
Images not processed due to individual download failure or corrupted data are flagged with the `redownload` and `error` keys.

The Notebook: `1a_ExploreData_FileContent.ipynb` provides some methods to interact with this database to summarize processing for individual grid cells and identify registered errors.

## identify internal image errors
`check_valid_pix` will return the number of unmasked pixels in an image. This is run internally during the eostac download process and output as `numpix` in the `processing.info` database. It can be rerun after brdf/coreg steps to identify discrepancies and troubleshoot errors.
```
LUCinSA_helpers check_valid_pix \
     -- raw_dir = 'path/to/cell_directory'  \
     -- brdf_dir = 'path/to/cell_directory/brdf'  \
     -- grid_cell = XXXXXX  \
     -- image_type = 'brdf' \
     -- yrs = [YYYY-YYYY]  \
     -- data_source = 'stac'  \
```
`check_ts_windows` will check whether there is data in all of the windows for time-series outputs.
```
LUCinSA_helpers check_ts_windows \
     --processed_dir = 'path/to/main/ts_directory'  \
     -- grid_cell = XXXXXX   \
     -- spec_index = 'evi2'  \
```
## summarize images processed
`summarize_images_multicell` will summarize all images in a given processing folder (landsat downloads, sentinel2 downloads or brdf) across multiple cells and return a database (in memory or printed to .csv) with unique image names (since a single Landsat or Sentinel2 scene covers multiple grid cells). 
Note: in later stages of a project where some downloads have been cleaned out, this will only work with brdf folder.
```
LUCinSA_helpers summarize_images_multicell \
     -- full_dir = path/to/main/processing_directory \
     -- sub_dir = 'brdf' \
     -- endstring = '.nc' \
     -- print_list = False \
     -- out_dir = None \
```
Graphic smmaries can be generated in the notebook: `5a_SummarizeData_ImagesProcessed.ipynb`
![alt](/images/processing_summary.jpg)
![alt](/images/processing_summary_qual.jpg)

## get processing summary
For a more nuanced check of processing status across all cells, `update_summary_db` will...
```
LUCinSA_helpers update_summary_db \
      -- status_db_path = 'path/to/cell_processing_post.csv' 
      -- cell_list = 'All' \ 
      -- dl_dir = 'path/to/main_processing_directory' 
      -- processed_dir = 'path/to/main/ts_directory'
```