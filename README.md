# LUCinSA_helpers
Helper functions and notebooks to interact with data on High-Performance Computing environment, designed to be used in conjunction with processing guide for remote sensing projects on Land-Use Change in Latin America: https://klwalker-sb.github.io/LUCinLA_stac/ 

## Uses

### To summarize processing status, uncover errors and conduct quality control amist a large numbers of files in HPC environment
#####      single cell error checking:
* [check download logs](#check-download-logs)(`check_dl_logs`)
* [check processing status for cell](#check-processing-status-for-cell)(`get_cell_status`)
* [identify external image errors](#identify-external-image-errors)(`processing.info` with notebook `1a_ExploreData_FileContent.ipynb`)
* [identify internal image errors](#identify-internal-image-errors)(`check_valid_pix`, `check_ts_windows`)
* [generate thumbnails and select images to include/exclude](#make-thumbails-for-quality-control)

#####      multi-cell summarization and error checking
* [summarize images processed](#summarize-images-processed-for-all-cells)(`summarize_images_multicell`)
* [get processing summary](#get-processing-summary)(`update_summary_db` and notebook `5a_SummarizeData_ImagesProcessed.ipynb`)
* [mosaic rasters](#mosaic-rasters-from-multiple-cells)(`moaic`)

### To quickly visualize inputs and outputs of time-series analysis for quality control and interactive troubleshooting
* [get_time_series_at_point](#get-time-series-at-point)(`get_time_series` and notebook `2b.TimeSeriesSignatures.ipynb`)
* [make_ts_composite raster](#make_ts_composite_raster)(`make_ts_composite` with optional bash script `make_ts_composite.sh`)
* [interactively select points and view time-series data on cluster](#interactive-time-series-feedback)
 
### To set and document parameter choices and compare outputs for model optimization

### Also (temporarily) hosts functions to:

###        create modelling features from smoothed time-series indices and segmentation outputs 
* [make raster variable stack](#make-raster-variable-stack)(`make_ts_composite` within bash script `raster_var_stack.sh`)
* [make variable dataframe for sample points](#make-variable-dataframe-for-sample-points)
###        create single-year random forest classification model
* [build random forest model](#bulid-rf-model)(`rf_model`) 
###        apply random forest model to gridded data to create wall-to-wall map
* [classify_cell](#apply-rf-model-to-classify-cell)(`rf_classification`)

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
      -- raw_dir 'path/to/main_downloading_directory' \
      -- processed_dir 'path/to/main_processing_directory' \
      -- grid_cell  XXXXX \
      -- yrs [YYYY-YYYY] \
      -- print_plot True \
      -- out_dir path/to/local/output/directory' \
      -- data_source 'stac' 
```
The above relies on the processing database that each cell has in its main directory named `processing.info`, which has an entry for each image encountered in the STAC catalog and data regarding its status through the downloading,brdf,and coregistration processing steps.

If `processing.info` is corrupted or deleted for a cell, it can be recreated with `reconstruct-db` (but note that it will not contain all of the detail of the original database):
```
LUCinSA_helpers reconstruct_db \
     --processing_info_path 'path/to/cell_directory/processing.info' \
     --landsat_path 'path/to/cell_directory/landsat'  \
     --sentinel2_path 'path/to/cell_directory/sentinel2' \
     --brdf_path 'path/to/cell_directory/brdf'  
```
![alt](/images/images_processed_for_cell_by_sensor.png)
![alt](/images/images_processed_for_cell_by_stat.png)

## identify external image errors
Processing errors raised within processes are noted in the `processing.info` database.
Images not processed due to individual download failure or corrupted data are flagged with the `redownload` and `error` keys.

The Notebook: `1a_ExploreData_FileContent.ipynb` provides some methods to interact with this database to summarize processing for individual grid cells and identify registered errors.

## identify internal image errors
`check_valid_pix` will return the number of unmasked pixels in an image. This is run internally during the eostac download process and output as `numpix` in the `processing.info` database. It can be rerun after brdf/coreg steps to identify discrepancies and troubleshoot errors.
```
LUCinSA_helpers check_valid_pix \
     -- raw_dir 'path/to/cell_directory'  \
     -- brdf_dir 'path/to/cell_directory/brdf'  \
     -- grid_cell XXXXXX  \
     -- image_type 'brdf' \
     -- yrs [YYYY-YYYY]  \
     -- data_source 'stac'  
```
`check_ts_windows` will check whether there is data in all of the windows for time-series outputs.
```
LUCinSA_helpers check_ts_windows \
     --processed_dir 'path/to/main/ts_directory'  \
     -- grid_cell XXXXXX   \
     -- spec_index 'evi2'  
```

## make thumbnails for quality control

## summarize images processed for all cells
`summarize_images_multicell` will summarize all images in a given processing folder (landsat downloads, sentinel2 downloads or brdf) across multiple cells and return a database (in memory or printed to .csv) with unique image names (since a single Landsat or Sentinel2 scene covers multiple grid cells). 
Note: in later stages of a project where some downloads have been cleaned out, this will only work with brdf folder.
```
LUCinSA_helpers summarize_images_multicell \
     -- full_dir path/to/main/processing_directory \
     -- sub_dir 'brdf' \
     -- endstring '.nc' \
     -- print_list False \
     -- out_dir None 
```
Graphic smmaries can be generated in the notebook: `5a_SummarizeData_ImagesProcessed.ipynb`
![alt](/images/processing_summary.jpg)
![alt](/images/processing_summary_qual.jpg)

## get processing summary
For a more nuanced check of processing status across all cells, `update_summary_db` will...
```
LUCinSA_helpers update_summary_db \
      -- status_db_path 'path/to/cell_processing_post.csv' \
      -- cell_list 'All' \ 
      -- dl_dir 'path/to/main_processing_directory' \
      -- processed_dir 'path/to/main/ts_directory'
```
## mosaic rasters from multiple cells
```
LUCinSA_helpers mosaic \
     --cell_list `'path/to/cellList.csv'\ #no header in csv file
     -- in_dir_main 'path/to/main_ts_directory'\
     -- in_dir_local 'folder_within_cell_directory_with_item_to_mosaic'\
     -- common_str 'common_string_to_identify_image_within_dir`\
     -- out_dir 'path/to/directory_for_final_mosaic'\
```        
## get time series at point

```
LUCinSA_helpers get_time_series \
    --out_dir \
    --spec_index 'evi2'\
    --start_yr '2010'
    --end_yr '2022'                        
    --img_dir 'path/to/main_ts_dir'
    --image_type  
    --grid_file  
    --cell_list 
    --ground_polys = args.ground_polys,
    --oldest args.oldest,
    --newest args.newest,
    --npts 2
    --seed 0
    --load_samp 'True'
    --ptfile 
```

## make time series composite raster
The bash script: `make_ts_composite.sh` allows the `make_ts_composite` function to be run over multiple grid cells in parallel (as an array).
```
LUCinSA_helpers make_ts_composite \
     --grid_cell XXXX  \   
     --spec_index 'evi2' \
     --img_dir 'path/to/ts_directory_for/cell/and/index'  \
     --out_dir 'path/to/output_variable_directory/for/cell' \
     --start_yr YYYY \
     --bands_out '[Max,Min,Amp]'
```
## interactive time-series feedback
The notebook, `2b_ViewTimeSeriesComposite.ipynb` allows a user to build and view a time-series composite for given cell and year without the need to transfer any data from the HPC cluster. Both smoothed and raw time series can be viewed for up to four points selected interactively from the image. This is useful for troubleshooting as well as quick feedback on variables and useful for mapping targeted classes and the effect of different smoothing functions on the final dataset.

![alt](/images/ts_example.jpg)
           
## make raster variable stack
Once a set of raster variables has been selected for a model, a variable stack can be produced for each index using the `make_ts_composite` function with more than three variables. The bash script: `raster_var_stack.sh` contains the configuration to run the function in a loop over all desired spectral indices and in parallel for multiple grid cells (as an array).
```
LUCinSA_helpers make_ts_composite \
     --grid_cell XXXX ## (run as array in bash script) \   
     --spec_index 'evi2' ## but run from loop over multiple indices in bash script \
     --img_dir 'path/to/ts_directory_for/cell/and/index' ##(cell and index are informed by --grid-cell and --spec_index in bash script) \
     --out_dir 'path/to/output_variable_directory/for/cell' ##(cell is informed by --grid-cell in bash script) \
     --start_yr YYYY \
     --bands_out '[Max,Min,Amp,Avg,CV,Std,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec]'
```
## make variable dataframe for sample points
input point file is in the form of .csv with 'XCoord' and 'YCoord' columns in same coord system as grid_file
```
LUCinSA_helpers make_var_dataframe \
    --out_dir 'path/to/directory_for_final_dataframe' \ 
    --spec_indices '[evi2,gcvi,wi,kndvi,nbr,ndmi]' \
    --si_vars '[Max,Min,Amp,Avg,CV,Std,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec]' \
    --in_dir 'path/to/main_ts_directory' \
    --grid_file  'path/to/gridcell_geojson' \
    --cell_list '[XXXX,XXXX,XXXX]' #or 'path/to/SampleCells.csv'
    --ground_polys '' \
    --oldest 2020 \
    --newest 2022 \
    --npts 2\
    --seed 888 \
    --load_samp 'True'
    --ptfile 'path/to/file/with/samplepts'
```    
## build rf model
```
LUCinSA_helpers rf_model \
   -- out_dir 'path/to/directory_for_rf_model' \
   -- classification \
   -- importance_method \
   -- ran_hold \
   -- model_name \
```
## classified raster
```
LUCinSA_helpers rf_classification \
    -- in_dir   \
    -- df_in  \
    -- spec_indices \
    -- stats \
    -- rf_mod \
    -- img_out \
    -- classification \
    -- importance_method \
    -- ran_hold \
    -- out-dir 'path/to/directory_for_classified_raster'\
```      