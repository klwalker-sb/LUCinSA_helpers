#!/usr/bin/env python
# coding: utf-8

'''
checks log files for errors (.err files produced from download process).
(download process must be run with specific SLURM script that prints info)
logs errors and gaps in time sequence to master database
updates errors and gaps with subsequent logs for same cell
archives all log files once there are no errors in 'dl_to_fix'
note: there still may be gaps and unfixable errors that can be seen in database
deletes all .out files (they contain no information)
.out file from this script contains info on this current batch of cells for
quick reference.
'''

import os
import sys
import datetime
import pandas as pd
from pathlib import Path
import ast
import shutil

cell_db_path = sys.argv[1]
archive_path = sys.argv[2]
ignore_dates = sys.argv[3]

ignore = [d for d in ignore_dates.split(',')]
ignore_dt = [datetime.datetime.strptime(d,'%Y-%m-%d').date() for d in ignore]

def find_gaps(ranges, start='2000-01-01', stop='2022-12-31'):
    if len(ranges) <= 1:
        return []
    gaps = []
    # check for gap at the beginning of range:
    startd = datetime.datetime.strptime(start,'%Y-%m-%d').date()
    if ranges[0][0] > startd:
        gaps.append([startd,ranges[0][0]])
    # Set marker at the end of the first range
    current = ranges[0][1]
    # Iterate through ranges, ignoring the first range
    for pair in ranges:
        # if next start time is before current end time, keep going until we find a gap
        # if next start time is after current end time, found the first gap
        if pair[0] > current:
            # ignore gaps between 31-Dec and 1-Jan:
            if pair[0].day==1 and pair[0].month==1 and current.day==31 and current.month==12:
                pass
            else:
                gaps.append([current,pair[0]])
        # advance "current" if the next end time is past the current end time
        current = max(pair[1], current)
    # check for gap at the end of range:
    stopd = datetime.datetime.strptime(stop,'%Y-%m-%d').date()
    if ranges[-1][1] < stopd:
        gaps.append([ranges[-1][1],stopd])
    return gaps

def check_logfile_dl(logfile, cell_dict,ignore_dates):
    cell_id=None
    core_requested=None
    runtime=None
    periods = []
    errors=[]
    with open(logfile) as f:
        for line in f:
            if 'cell_id' in line:
                cell_id = int(line.split(' ')[2])
            if 'Working on' in line:
                period=[line.split(' ')[3][:10],line.split(' ')[5][:10]]
                periods.append(period)
            if 'TimeoutError' in line or 'pystac_client.exceptions.APIError' in line or 'urllib.error.HTTPError' in line:
                errors.append(period)
            if 'full process took:' in line:
                runtime = int(line.split(' ')[3])
            if 'core used' in line:
                core_requested = int(line.split(' ')[2])
    ## core used was not always in download log. Use 4 if value is not known:
    if core_requested == None or core_requested >90:
        core_requested = 4
    ## also, cell id in logfile title is only 3 digits (because it is array id).
    ## Started printing id within file, but need this for old files that didn't have that printed.
    if cell_id == None:
        cell_id3 = int(logfile.split('.')[2])
        if cell_id3 < 101:
            cell_id = cell_id3 + 4000
        else:
            cell_id = cell_id3 + 3000
    if runtime == None:
        runtime = 0
    print(f'working on {cell_id} from logfile: {logfile}')
    print(f'cores used:{core_requested}')

    if len(periods)==0:
        print('this log file contains no info')
    else:
        dates = [[datetime.date(int(x[:4]),int(x[5:7]),int(x[8:10])) for x in p] for p in periods]
        ranges = sorted(dates)
        date_range = [ranges[0][0], ranges[-1][1]]
        print (f'downloaded from {date_range}')

        if cell_id in cell_dict:
            print('updating cell info...')
            ## note: if dict has been saved as dataframe and reconstructed as dict, entries will be strings
            ## update start and end value
            if isinstance(cell_dict[cell_id]['dllog_start'],str):
                old_start = datetime.datetime.strptime(cell_dict[cell_id]['dllog_start'],'%Y-%m-%d').date()
            else:
                old_start = cell_dict[cell_id]['dllog_start']
            if ranges[0][0] < old_start:
                cell_dict[cell_id]['dllog_start']=ranges[0][0]
            if isinstance(cell_dict[cell_id]['dllog_end'],str):
                old_end = datetime.datetime.strptime(cell_dict[cell_id]['dllog_end'],'%Y-%m-%d').date()
            else:
                old_end = cell_dict[cell_id]['dllog_end']
            if ranges[-1][1] > old_end:
                cell_dict[cell_id]['dllog_end']=ranges[-1][1]
            ## add former errors to current dllog_errors if time period not recorded in new ranges (without error)
            new_ranges = [[x.strftime('%Y-%m-%d') for x in r] for r in ranges]
            if isinstance(cell_dict[cell_id]['dllog_errors'],str):
                old_errors =  ast.literal_eval(cell_dict[cell_id]['dllog_errors'])
            else:
                old_errors = cell_dict[cell_id]['dllog_errors']
            unresolved_errors = [e for e in old_errors if e in errors or e not in new_ranges]
            new_errors = [e for e in errors if
            datetime.datetime.strptime(e[0],'%Y-%m-%d').date() < old_start or datetime.datetime.strptime(e[1],'%Y-%m-%d').date() > old_end]
            if len(unresolved_errors) > 0:
                new_errors.extend(unresolved_errors)
            cell_dict[cell_id]['dllog_errors']=new_errors
            ## error in most recent period (specifiec with ignore param) probably cant be fixed and should be ignored for now:
            if ignore in new_errors:
                errors_to_fix = new_errors.remove(ignore)
            elif ignore_dt in errors:
                errors_to_fix = new_errors.remove(ignore_dt)
            else:
                errors_to_fix = new_errors
            cell_dict[cell_id]['dl_fix_now']=errors_to_fix
            ## get new gap sequence:
            orig_range = [old_start,old_end]
            ranges.append(orig_range)
            date_gaps=find_gaps(ranges)
            ## add runtime:
            cell_dict[cell_id]['dltime']=int(cell_dict[cell_id]['dltime'])+runtime
            cell_dict[cell_id]['dlcoremin']=int(cell_dict[cell_id]['dlcoremin'])+runtime*core_requested
        else:
            print('adding cell to processing db...')
            if ignore in errors:
                errors_to_fix = errors.remove(ignore)
            if ignore_dt in errors:
                errors_to_fix = errors.remove(ignore_dt)
            else:
                errors_to_fix = errors
            date_gaps = find_gaps(ranges)
            new_dict_entry={cell_id:{'dllog_start':ranges[0][0],'dllog_end':ranges[-1][1],'dllog_gaps':date_gaps,'dllog_errors':errors,'dltime':runtime,'dlcoremin':runtime*core_requested,'dl_fix_now':errors_to_fix}}
            cell_dict.update(new_dict_entry)

        if len(date_gaps) == 0:
            print('found no gaps')
        else:
            print(f'found gaps: {date_gaps}')

    return cell_id, cell_dict

def archive_logfile(logfile,cell_dict,archive_path):
    cell_id = None
    with open(logfile) as f:
        for line in f:
            if 'cell_id' in line:
                cell_id = int(line.split(' ')[2])
                break
    if cell_id == None:
        cell_id3 = int(logfile.split('.')[2])
        if cell_id3 < 101:
            cell_id = cell_id3 + 4000
        else:
            cell_id = cell_id3 + 3000
    if isinstance(cell_dict[cell_id]['dl_fix_now'],str):
        if cell_dict[cell_id]['dl_fix_now'] == '[]':
            shutil.move(f'./{logfile}', os.path.join(archive_path,logfile))
    elif cell_dict[cell_id]['dl_fix_now'] == None:
        shutil.move(f'./{logfile}', archive_path)
    elif len(cell_dict[cell_id]['dl_fix_now']) == 0:
        shutil.move(f'./{logfile}', archive_path)

## get the existing cell database as a dictionary
if Path(cell_db_path).is_file():
    cell_dict = pd.read_csv(Path(cell_db_path),index_col=[0]).to_dict(orient='index')
else:
    cell_dict = {}
## update records based on curent logfiles
cell_batch = set([])
logfiles = [f for f in os.listdir('.') if f.startswith('stacdl')and f.endswith('.err')]
for logfile in logfiles:
    processed = check_logfile_dl(logfile, cell_dict, ignore_dates)
    cell_batch.add(processed[0])
## These output files contain no information; can just remove
outfiles = [f for f in os.listdir('.') if f.startswith('stacdl')and f.endswith('.out')]
for outfile in outfiles:
    os.remove(outfile)
## Archive logfiles after running all because errors in some may be removed by subsequent files
for logfile in logfiles:
    archive_logfile(logfile,cell_dict,archive_path)
## Save full updated database
new_processing_info = pd.DataFrame.from_dict(cell_dict,orient='index')
new_processing_info.rename_axis('cell_id', axis=1, inplace=True)
pd.DataFrame.to_csv(new_processing_info, cell_db_path, index='cell_id')
## Print just the current batch of cells to logfile for easy error checking
print(f'all cells prcessed in this batch:{cell_batch}')
batch_status = new_processing_info[new_processing_info.index.isin(cell_batch)].sort_index()
pd.set_option("max_columns", None)
pd.set_option('display.max_rows', 500)
print(batch_status)
