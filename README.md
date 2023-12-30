# LUCinSA_helpers
Helper functions and notebooks to interact with data on High-Performance Computing environment, designed to be used in conjunction with processing guide for remote sensing projects on Land-Use Change in Latin America: https://klwalker-sb.github.io/LUCinLA_stac/ 

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
* version 
* get_time_series 
* make_ts_composite 
* check_processing 
* rf_model 
* rf_classification 
* mosaic

Example SLURM scripts are included for these processes in the BashScripts folder.
get_time_series, rf_classification, and mosaic use significant resources and should be run through SLURM for accounting (when running 
processes through jupyter notebook, the resources are not accounted for and might result in crashing the whole HPC cluster if resources are
already maxed out and more are demanded)

Many small processes can be run through the Jupyter notebooks in the notebooks folder.
To access a Jupyter notebook via the cluster see [this page of the LUCinLA_STAC guide](https://klwalker-sb.github.io/LUCinLA_stac/)
Before running Jupyter notebooks, set the parameters in `notebook_params.ipynb` and run all cells of that notebook.
By doing so, you should not need to alter the text in any of the notebooks themselves (unless adding/modifying methods).

To save a notebook with displayed outputs, run the last cell of the notebook:
`To save an html copy of this notebook with all outputs`. This will save an html copy in the  `notebooks/Outputs` folder. 
Relevant parameter settings are printed within the notebook, so it should be reproducible.

### note regarding contributions:
When edits are pushed, notebook outputs are automatically cleared at commit staging, as per
[this post](https://medium.com/somosfit/version-control-on-jupyter-notebooks-6b67a0cf12a3)
This helps to keep this git repo from getting overwhelmed with notebook data.
Also make sure to use the original .gitignore file (which might be hidden) when pushing edits