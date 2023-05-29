#!/usr/bin/env python
import argparse
from LUCinSA_helpers.ts_profile import get_timeseries_for_pts_multicell
from LUCinSA_helpers.ts_composite import make_ts_composite
from LUCinSA_helpers.file_checks import check_valid_pixels
from LUCinSA_helpers.version import __version__


def main():

    ##Setup to parse lists from Bash script (need to enter as string in Bash)
    ## NOTE: THIS IS PROBABLY NOT NEEDED ANYMORE
    def check_for_list(arg_input):
        if arg_input.startswith('['):
            arg_input = arg_input[1:-1].split(',')
            try:
                arg_input = list(map(int, arg_input))
            except:
                pass
        return arg_input

    parser = argparse.ArgumentParser(description='scripts to augment info for data exploration in notebooks',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(dest='process')

    available_processes = ['version', 'get_time_series', 'make_ts_composite', 'check_processing']

    for process in available_processes:
        subparser = subparsers.add_parser(process)

        if process == 'version':
            continue

        if process == 'check_processing':
            subparser.add_argument('--raw_dir', dest ='raw_dir', help='directory containing downloaded images')
            subparser.add_argument('--brdf_dir', dest ='brdf_dir', help='directory containing brdf images')
            subparser.add_argument('--grid_cell', dest ='grid_cell', help='cell to process')
            subparser.add_argument('--image_type', dest ='image_type', help='Type of image to process (Landsat(5,7,8,9), Sentinel, or All', default='All')
            subparser.add_argument('--yrs', dest ='yrs', help='Years to process, [YYYY,YYYY]. or all if None',default=None)
            subparser.add_argument('--data_source', dest ='data_source', help='stac or GEE', default='stac')

        else:
            subparser.add_argument('--out_dir', dest='out_dir', help='out directory for processed outputs', default=None)
            subparser.add_argument('--img_dir', dest ='img_dir', help='directory containing images')
            subparser.add_argument('--start_yr', dest ='start_yr', help='start year', default=2010, type=int)
            subparser.add_argument('--spec_index', dest='spec_index', help='Spectral index to explore. options are...', default='evi2')

        if process == 'get_time_series':
            subparser.add_argument('--end_yr', dest ='end_yr', help='end year', default=2020, type=int)
            subparser.add_argument('--image_type', dest ='image_type', help='.nc or TS currently supported', default='TS')
            subparser.add_argument('--grid_file', dest ='grid_file', help='path to grid file')
            subparser.add_argument('--cell_list', dest ='cell_list', help='list of cells to process', type=int, nargs='+')
            subparser.add_argument('--ground_polys', dest='ground_polys',help='path to polygons to sample from; only needed if loadSamp =False')
            subparser.add_argument('--oldest', dest ='oldest', help='if using groundPolys, oldest poly to use', default=2010)
            subparser.add_argument('--newest', dest ='newest', help='if using groundPolys, oldest poly to use', default=2020)
            subparser.add_argument('--npts', dest ='npts', help='if using groundPolys, number of pts per poly to sample', default=2)
            subparser.add_argument('--seed', dest ='seed', help='if using groundPolys, seed for random sampling within', default=888)
            subparser.add_argument('--load_samp', dest ='load_samp',help='whether to load point sample directly instead of sampling from polygons', type=bool, default=True)
            subparser.add_argument('--pt_file', dest ='pt_file', help='Path to file containing points, if load_samp=True', default=None)

        if process == 'make_ts_composite':
            subparser.add_argument('--bands_out', dest ='bands_out', help='bands to create. Currently only 3 allowed. Current options are Max,Min,Amp,Avg,CV,Std,MaxDate,MaxDateCos,MinDate,MinDateCos,Jan,Apr,Jun,Aug,Nov')
            subparser.add_argument('--grid_cell', dest='grid_cell', help='cell being processed')
    args = parser.parse_args()

    if args.process == 'version':
      print(__version__)
      return

    if args.process == 'get_time_series':
        get_timeseries_for_pts_multicell(out_dir = args.out_dir,
                             spec_index = args.spec_index,
                             start_yr = args.start_yr,
                             end_yr = args.end_yr,
                             img_dir = args.img_dir,
                             image_type = args.image_type,
                             grid_file = args.grid_file,
                             cell_list = args.cell_list,
                             ground_polys = args.ground_polys,
                             oldest = args.oldest,
                             newest = args.newest,
                             npts = args.npts,
                             seed = args.seed,
                             load_samp = args.load_samp,
                             ptfile = args.ptfile)

    if args.process == 'make_ts_composite':
        make_ts_composite(grid_cell = args.grid_cell,
                        img_dir = args.img_dir,
                        out_dir = args.out_dir,
                        start_yr = args.start_yr,
                        spec_index = args.spec_index,
                        bands_out = args.bands_out)

    if args.process == 'check_processing':
        check_valid_pixels(raw_dir = args.raw_dir,
                         brdf_dir = args.brdf_dir,
                         grid_cell = args.grid_cell,
                         image_type = args.image_type,
                         yrs = args.yrs,
                         data_source = args.data_source)


if __name__ == '__main__':
    main()

