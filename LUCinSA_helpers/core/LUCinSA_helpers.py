#!/usr/bin/env python

import argparse
from LUCinSA_helpers.TSprofile import GetTimeSeriesForPts_MultiCell
from LUCinSA_helpers.version import __version__


def main():

    ##TODO: figure out how to apply this to celllist
    ##Setup to parse lists from Bash script (need to enter as string in Bash)
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

    available_processes = ['version',
                            'GetTimeSeries'
                            ]

    for process in available_processes:
        subparser = subparsers.add_parser(process)

        if process == 'version':
            continue

        subparser.add_argument('--out_dir', dest='out_dir', help='out directory for processed outputs', default=None)

        if process == 'GetTimeSeries':
            subparser.add_argument('--spec_index', dest='spec_index', help='Spectral index to explore. options are...', default='evi2')
            subparser.add_argument('--StartYr', dest ='StartYr', help='start year', default=2010, type=int)
            subparser.add_argument('--EndYr', dest ='EndYr', help='end year', default=2020, type=int)
            subparser.add_argument('--img_dir', dest ='img_dir', help='directory contining smoothed images')
            subparser.add_argument('--imageType', dest ='imageType', help='.nc or TS currently supported', default='TS')
            subparser.add_argument('--gridFile', dest ='gridFile', help='path to grid file')
            subparser.add_argument('--cellList', dest ='cellList', help='list of cells to process', type=int, nargs='+')
            #subparser.add_argument('--cellList', dest ='cellList', help='list of cells to process')
            subparser.add_argument('--groundPolys', dest='groundPolys',help='path to polygons to sample from; only needed if loadSamp =False')
            subparser.add_argument('--oldest', dest ='oldest', help='if using groundPolys, oldest poly to use', default=2010)
            subparser.add_argument('--newest', dest ='newest', help='if using groundPolys, oldest poly to use', default=2020)
            subparser.add_argument('--npts', dest ='npts', help='if using groundPolys, number of pts per poly to sample', default=2)
            subparser.add_argument('--seed', dest ='seed', help='if using groundPolys, seed for random sampling within', default=888)
            subparser.add_argument('--loadSamp', dest ='loadSamp',help='whether to load point sample directly instead of sampling from polygons', type=bool, default=True)
            subparser.add_argument('--ptFile', dest ='ptFile', help='Path to file containing points, if loadSamp=True', default=None)

    args = parser.parse_args()

    if args.process == 'version':
      print(__version__)
      return

    if args.process == 'GetTimeSeries':
        GetTimeSeriesForPts_MultiCell(out_dir = args.out_dir,
                             spec_index = args.spec_index,
                             StartYr = args.StartYr,
                             EndYr = args.EndYr,
                             img_dir = args.img_dir,
                             imageType = args.imageType,
                             gridFile = args.gridFile,
                             cellList = args.cellList,
                             groundPolys = args.groundPolys,
                             oldest = args.oldest,
                             newest = args.newest,
                             npts = args.npts,
                             seed = args.seed,
                             loadSamp = args.loadSamp,
                             ptFile = args.ptFile)

if __name__ == '__main__':
    main()

