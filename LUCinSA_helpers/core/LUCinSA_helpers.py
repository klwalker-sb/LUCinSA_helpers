#!/usr/bin/env python
import argparse
from LUCinSA_helpers.check_log_files_dl import check_dl_logs
from LUCinSA_helpers.ts_profile import get_timeseries_for_pts_multicell
from LUCinSA_helpers.ts_composite import make_ts_composite
from LUCinSA_helpers.file_checks import reconstruct_db, get_cell_status, check_valid_pixels, update_cell_status_db
from LUCinSA_helpers.file_checks import print_files_in_multiple_directories
from LUCinSA_helpers.var_dataframe import make_var_dataframe
from LUCinSA_helpers.rf import make_variable_stack, rf_model, rf_classification
from LUCinSA_helpers.mosaic import mosaic_cells
from LUCinSA_helpers.version import __version__

def main():
       
    def check_for_list(arg_input):
        '''lists passed from Bash will not be parsed correctly. 
           This function converts variables in Bash scripts that are 
           entered as strings "[X,X,X]" into lists in Python.
        '''
        if isinstance(arg_input, list):
            arg_input = arg_input
        elif arg_input.startswith('['):
            arg_input = arg_input[1:-1].split(',')
            try:
                arg_input = list(map(int, arg_input))
            except:
                pass
        return arg_input

    parser = argparse.ArgumentParser(description='scripts to augment info for data exploration in notebooks',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(dest='process')

    available_processes = [
                           'version',
                           'check_dl_logs',
                           'get_cell_status',
                           'update_summary_db',
                           'get_time_series', 
                           'make_ts_composite', 
                           'check_valid_pix',
                           'reconstruct_db',
                           'check_ts_windows',
                           'summarize_images_multicell',
                           'make_var_stack',
                           'make_var_dataframe',
                           'rf_model', 
                           'rf_classification', 
                           'mosaic'
                          ]

    for process in available_processes:
        subparser = subparsers.add_parser(process)
        if process == 'version':
            continue
        if process == 'check_dl_logs':
            subparser.add_argument('--cell_db_path', dest = 'cell_db_path', help='path to master downloading database') 
            subparser.add_argument('--archive_path', dest = 'archive_path', 
                                   help='path to location to store log files after processing')
            subparser.add_argument('--stop_date', dest = 'stop_date', 
                                   help='last date for gap search, YYYY-MM-DD', default='2022-12-31')
            subparser.add_argument('--start_date', dest = 'start_date', 
                                   help='first date for gap search, YYYY-MM-DD', default='2000-01-01') 
            subparser.add_argument('--ignore_dates', dest = 'ignore_dates', 
                                   help='dates to ignore errors for. YYYY-MM-DD,YYYY-MM-DD',default=None)
            subparser.add_argument('--log_path', dest = 'log_path', 
                                   help='location of log files before processing',default=None)
        if process in ['get_cell_status','check_processing','update_summary_db']:
            subparser.add_argument('--raw_dir', dest ='raw_dir', help='directory containing downloaded images')
            subparser.add_argument('--processed_dir', dest ='processed_dir', 
                                   help='directory with processed images -- brdf for check_processing, smooth ts for get_cell_status')
        if process in ['get_cell_status','check_processing']:                           
            subparser.add_argument('--grid_cell', dest ='grid_cell', help='cell to process')
            subparser.add_argument('--yrs', dest ='yrs', help='Years to process, [YYYY,YYYY]. or all if None',default=None)
            subparser.add_argument('--data_source', dest ='data_source', 
                                   help='stac or GEE', default='stac')
        if process == 'update_summary_db':
            subparser.add_argument('--status_db_path', dest = 'status_db_path', 
                                   help='path to master processing database') 
            subparser.add_argument('--cell_list', dest ='cell_list', 
                                   help='list of cells to process. If All, processes all in raw_dir', default='All')
        if process == 'check_valid_pix':
            subparser.add_argument('--image_type', dest ='image_type', 
                                   help='Type of image to process (Landsat(5,7,8,9), Sentinel, or All', default='All')
        if process == 'get_cell_status':
            subparser.add_argument('--print_plot', dest='print_plot', help='whether to generate plot graphics', default=False)
            subparser.add_argument('--out_dir', dest='out_dir', help='out directory for plot graphics', default=None)
        if process == 'reconstruct_db':
            subparser.add_argument('--processing_info_path', dest ='processing_info_path', 
                                   help='path to processing.info file for cell')
            subparser.add_argument('--landsat_path', dest ='landsat_path', help='path to landsat download folder')
            subparser.add_argument('--sentinel2_path', dest ='sentinel2_path', help='path to sentinel2 download folder')
            subparser.add_argument('--brdf_path', dest ='brdf_path', help='path to brdf folder')
        
        if process == 'summarize_images_multicell':
            subparser.add_argument('--full_dir', dest='full_dir', help='path to main directory containing all cells')
            subparser.add_argument('--sub_dir', dest='sub-dir', help='name of subdirectory with images to summarize',default='brdf')
            subparser.add_argument('--endstring', dest='endstring', help='endstring of files to be included in sumary',default='.nc')
            subparser.add_argument('--print_list', dest='print_list', help='True if list is to be printed to file',default=False)
            subparser.add_argument('--out_dir', dest='out_dir', help='directory to print list to if print_list==True', default=None)
            
        if process == 'check_ts_windows':
            subparser.add_argument('--processed_dir', dest='processed_dir', help='path to main directory with ts indices') 
            subparser.add_argument('-- grid_cell', dest='grid_cell', help='grid cell # (1 to 6 digits)')
            subparser.add_argument('-- spec_index', dest='spec-index', help='spectral index to be processed', default='evi2')
        
        if process == 'mosaic':
            subparser.add_argument('--cell_list', dest='cell_list', help='list of cells to mosiac', default=None)
            subparser.add_argument('--in_dir_main', dest='in_dir_main', help='overarching directory with all cells', default=None)
            subparser.add_argument('--in_dir_local;', dest='in_dir_local', help='local folder name or path with raster', default=None)
            subparser.add_argument('--common_str', dest='common_str', help='unique string in file name for mosaic', default=None)
            subparser.add_argument('--out_dir', dest='out_dir', help='out directory for processed outputs', default=None)
        
        if process in ['get_time_series','make_ts_composite','rf_model','rf_classification']:
            subparser.add_argument('--out_dir', dest='out_dir', help='out directory for processed outputs', default=None)
            subparser.add_argument('--img_dir', dest ='img_dir', help='directory containing images')
            subparser.add_argument('--start_yr', dest ='start_yr', help='start year', default=2010, type=int)
            subparser.add_argument('--spec_index', dest='spec_index', help='Spectral index to explore. options are...', default='evi2')

        if process == 'get_time_series':
            subparser.add_argument('--end_yr', dest ='end_yr', help='end year', default=2020, type=int)
            subparser.add_argument('--image_type', dest ='image_type', help='.nc or TS currently supported', default='TS')
        if process in['get_time_series','make_var_dataframe']:
            subparser.add_argument('--grid_file', dest ='grid_file', help='path to grid file')
            subparser.add_argument('--cell_list', dest ='cell_list',
                                   help='list of cells to look for points/poys in. (list or path to .csv file with list, no header' )
            subparser.add_argument('--ground_polys', dest='ground_polys',
                                   help='path to polygons to sample from; only needed if load_samp =False')
            subparser.add_argument('--oldest', dest ='oldest', help='if using ground_polys, oldest poly to use', default=2010)
            subparser.add_argument('--newest', dest ='newest', help='if using groun_Polys, oldest poly to use', default=2020)
            subparser.add_argument('--npts', dest ='npts', help='if using ground_olys, number of pts per poly to sample', default=2)
            subparser.add_argument('--seed', dest ='seed', help='if using ground_olys, seed for random sampling within', default=888)
            subparser.add_argument('--load_samp', dest ='load_samp',
                                   help='whether to load point sample directly instead of sampling from polygons', 
                                   type=bool, default=True)
            subparser.add_argument('--ptfile', dest ='ptfile', 
                                   help='Path to file containing points, if load_samp=True', default=None)

        if process == 'make_ts_composite':
            subparser.add_argument('--bands_out', dest ='bands_out', help='bands to create')
            subparser.add_argument('--grid_cell', dest='grid_cell', help='cell being processed')
  
        if process in ['make_var_dataframe','make_var_stack']:
            subparser.add_argument('--out_dir', dest='out_dir',help='out directory for final dataframes')
            subparser.add_argument('--spec_indices', dest='spec_indices', help='',
                                  default = ['evi2','gcvi','wi','kndvi','nbr','ndmi'])
            subparser.add_argument('--si_vars', dest='si_vars', help = '',
                                   default='[Max,Min,Amp,Avg,CV,Std,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec]')
            subparser.add_argument('--in_dir', dest='in_dir', help='')
        
        if process == 'make_var_stack':
            subparser.add_argument('--feature_model', dest = 'feature_model', 
                                   help='unique name for variable combo (start_Yr (spec_indices*si_vars + singleton_vars + poly_vars))')
            subparser.add_argument('--start_yr', dest='start_yr', help='year to model (first year if using split calendar)')
            subparser.add_argument('--singleton_vars', dest='singleton_vars', help='list of singleton variables to include', default=None)
            subparser.add_argument('--poly_vars', dest='poly_vars', help='list of polygon-level variables to include', default=None)
            subparser.add_argument('--singleton_var_dict', dest='singleton_var_dict', default=None,
                                   help='path to json describing singleton vars. (see example in main folder of this repo)')
            subparser.add_argument('--feature_mod_dict', dest='feature_mod_dict', default=None,
                                   help='path to dict defining variable model names. (see example in main folder of this repo)')
            
        if process in ['rf_model','rf_classification']: 
            subparser.add_argument('--df_in', dest ='df_in', help='path to sample dataframe with extracted variable data')
            subparser.add_argument('--classification', dest ='classification', help='Permultation | Inference | None')
            subparser.add_argument('--importance_method', dest ='importance_method', help="All(=LC17),...", default='All')
            subparser.add_argument('--ran_hold', dest ='ran_hold', 
                                   help='fixed random number, for repetition of same dataset', type=int, default=0)
            subparser.add_argument('--model_name', dest ='model_name', help='name of model')
                                  
        if process == 'rf_classification':
            subparser.add_argument('--ts_dir', dest='ts_dir',help='path to time series variables (within cell)')
            subparser.add_argument('--rf_mod', dest='rf_mod',
                                   help='path to existing random forest model, or None if model is to be created')
            subparser.add_argument('--img_out', dest='img_out',help='Full path name of classified image to be created')
            subparser.add_argument('--spec_indices', dest ='spec_indices',
                                   help='indices used.')
            subparser.add_argument('--stats', dest ='stats', help='raster variables used')

    args = parser.parse_args()

    if args.process == 'version':
      print(__version__)
      return
                                   
    if args.process == 'check_dl_logs':
        check_dl_logs(cell_db_path = args.cell_db_path,
                      archive_path = args.archive_path,
                      log_path = args.log_path,
                      stop_date = args.stop_date,
                      start_date = args.start_date,
                      ignore_dates = args.ignore_dates)
        
    if args.process == 'update_summary_db':
        update_cell_status_db(status_db_path = args.status_db_path, 
                              cell_list = args.cell_list, 
                              dl_dir = args.raw_dir, 
                              processed_dir = args.processed_dir)
        
    if args.process == 'summarize_images_multicell':
        print_files_in_multiple_directories(full_dir = args.full_dir,
                                            sub_dir = args.sub_dir,
                                            endstring = args.endstring,
                                            print_list = args.print_list,
                                            out_dir = args.out_dir)    
    if args.process == 'get_time_series':
        get_timeseries_for_pts_multicell(out_dir = args.out_dir,
                             spec_index = args.spec_index,
                             start_yr = args.start_yr,
                             end_yr = args.end_yr,
                             img_dir = args.img_dir,
                             image_type = args.image_type,
                             grid_file = args.grid_file,
                             cell_list = check_for_list(args.cell_list),
                             ground_polys = args.ground_polys,
                             oldest = args.oldest,
                             newest = args.newest,
                             npts = args.npts,
                             seed = args.seed,
                             load_samp = args.load_samp,
                             ptfile = args.ptfile)
                                   
    if args.process == 'mosaic':
        moasic_cells(cell_list = args.cell_list,
                     in_dir_main = args.in_dir_main,
                     in_dir_local = args.in_dir_local,
                     common_str = args.common_str,
                     out_dir = args.out_dir)
        
    if args.process == 'make_ts_composite':
        make_ts_composite(grid_cell = args.grid_cell,
                        img_dir = args.img_dir,
                        out_dir = args.out_dir,
                        start_yr = args.start_yr,
                        spec_index = args.spec_index,
                        bands_out = check_for_list(args.bands_out))

    if args.process == 'check_valid_pix':
        check_valid_pixels(raw_dir = args.raw_dir,
                         brdf_dir = args.processed_dir,
                         grid_cell = args.grid_cell,
                         image_type = args.image_type,
                         yrs = args.yrs,
                         data_source = args.data_source)
    
    if args.process == 'get_cell_status':
        get_cell_status(dl_dir  = args.raw_dir,
                        processed_dir = args.processed_dir, 
                        grid_cell = args.grid_cell, 
                        yrs = args.yrs, 
                        print_plot = arge.print_plot,
                        out_dir = args.out_dir, 
                        data_source = args.data_source)
        
    if args.process == 'reconstruct_db':           
        reconstruct_db(processing_info_path = args.processing_info_path,
                 landsat_path = args.landsat_path,
                 sentinel2_path = args.sentinel2_path,
                 brdf_path = args.brdf_path)
        
    if args.process ==  'make_var_stack':
        make_variable_stack (in_dir = args.in_dir,
                             feature_model = args.feature_model,
                             start_yr = args.start_yr,
                             spec_indices = check_for_list(args.spec_indices),
                             si_vars = check_for_list(args.si_vars),
                             feature_mod_dict = args.feature_mod_dict,
                             singleton_vars = check_for_list(args.singleton_vars),
                             poly_vars = check_for_list(args.poly_vars),
                             singleton_var_dict = args.singleton_var_dict)      
        
    if args.process == 'make_var_dataframe':
        make_var_dataframe (out_dir = args.out_dir, 
                            spec_indices = check_for_list(args.spec_indices),
                            si_vars = check_for_list(args.si_vars),
                            in_dir = args.in_dir,
                            grid_file = args.grid_file,
                            cell_list = check_for_list(args.cell_list),
                            ground_polys = args.ground_polys,
                            oldest = args.oldest,
                            newest = args.newest,
                            npts = args.npts,
                            seed = args.seed,
                            load_samp = args.load_samp,
                            ptfile = args.ptfile)
        
    if args.process == 'rf_model':
        rf_model(df_in = args.df_in, 
                 out_dir = args.out_dir,
                 classification = args.classification,
                 importance_method = args.importance_method,
                 ran_hold = args.ran_hold,
                 model_name = args.model_name)
                                   
    if args.process == 'rf_classification':
        rf_classification(ts_dir = args.ts_dir,
                 df_in = args.df_in,
                 spec_indices = args.spec_indices,
                 stats = args.stats,
                 model_name = args.model_name,
                 rf_mod = args.rf_mod,
                 img_out = args.img_out,
                 classification = args.classification,
                 importance_method = args.importance_method,
                 ran_hold = args.ran_hold,
                 out_dir = args.out_dir)
        
    #if process == 'check_ts_windows':
    #    check_ts_windows(processed_dir = args.processed_dir,
    #                     grid_cell = args.grid_cell,
    #                     spec_index = args.spec_index)
        
if __name__ == '__main__':
    main()
