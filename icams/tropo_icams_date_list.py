#! /usr/bin/env python
#################################################################
###  This program is part of icams  v1.0                     ### 
###  Copy Right (c): 2020, Yunmeng Cao                        ###  
###  Author: Yunmeng Cao                                      ###                                                          
###  Email : ymcmrs@gmail.com                                 ### 
#################################################################

import h5py
import numpy as np
import sys
import os
import argparse
#from bs4 import BeautifulSoup
#import requests

from icams import _utils as ut

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def parallel_process(array, function, n_jobs=16, use_kwargs=False):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return [function(**a) if use_kwargs else function(a) for a in tqdm(array[:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[:]]
        else:
            futures = [pool.submit(function, a) for a in array[:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return out

def hour2seconds(hour0):
    
    hh = hour0.split(':')[0]
    mm = hour0.split(':')[1]
    
    seconds = float(hh)*3600 + float(mm)*60
    
    return seconds

def read_region(STR):
    WEST = STR.split('/')[0]
    EAST = STR.split('/')[1].split('/')[0]
    
    SOUTH = STR.split(EAST+'/')[1].split('/')[0]
    NORTH = STR.split(EAST+'/')[1].split('/')[1]
    
    WEST =float(WEST)
    SOUTH=float(SOUTH)
    EAST=float(EAST)
    NORTH=float(NORTH)
    return WEST,SOUTH,EAST,NORTH

def get_meta_corner(meta):
    if 'Y_FIRST' in meta.keys():
        length = int(meta['LENGTH'])
        width = int(meta['WIDTH'])
        lat0 = float(meta['Y_FIRST'])
        lon0 = float(meta['X_FIRST'])
        lat_step = float(meta['Y_STEP'])
        lon_step = float(meta['X_STEP'])
        lat1 = lat0 + lat_step * (length - 1)
        lon1 = lon0 + lon_step * (width - 1)
        
        NORTH = lat0
        SOUTH = lat1  
        WEST = lon0
        EAST = lon1    
        
    else:       
        lats = [float(meta['LAT_REF{}'.format(i)]) for i in [1,2,3,4]]
        lons = [float(meta['LON_REF{}'.format(i)]) for i in [1,2,3,4]]
        
        lat0 = np.max(lats[:])
        lat1 = np.min(lats[:])
        lon0 = np.min(lons[:])
        lon1 = np.max(lons[:])
        
        NORTH = lat0 + 0.1
        SOUTH = lat1 + 0.1
        WEST = lon0 + 0.1
        EAST = lon1 + 0.1
    
    return WEST,SOUTH,EAST,NORTH

def sort_unique_list(numb_list):
    list_out = sorted(set(numb_list))
    return list_out

def read_hdf5(fname, datasetName=None, box=None):
    # read hdf5
    with h5py.File(fname, 'r') as f:
        data = f[datasetName][:]
        atr = dict(f.attrs)
        
    return data, atr

def read_attr(fname):
    # read hdf5
    with h5py.File(fname, 'r') as f:
        atr = dict(f.attrs)
        
    return atr


def check_existed_file_unr(date,stationList,path):
    year,doy = ut.yyyymmdd2yyyyddd(date)
    stationDownload = []
    for i in range(len(stationList)):
        k0 = path + '/' + stationList[i].upper()+doy + '0.' + year[2:4] + 'zpd.gz' 
        if not os.path.isfile(k0):
            stationDownload.append(stationList[i])
    return stationDownload

def check_existed_file_unavco(date_list,path):
    return

def datelist2yearlist(datelist):
    yearlist = []
    for a0 in datelist:
        str0 = a0[0:4]
        if str0 not in yearlist:
            yearlist.append(str0)
    return yearlist

def tropo_icams_date(data0):
    date0, region_str, time_str, ref_str, dem_str, res_str = data0
    call_str= 'tropo_icams_date.py ' + date0 + dem_str + region_str + time_str + ref_str + res_str
    os.system(call_str)
    
    return
###################################################################################################

INTRODUCTION = '''

Generating high-resolution tropospheric delay map using icams.'
    
'''

EXAMPLE = '''EXAMPLES:

    tropo_icams_date_list.py --date-list-txt date_list_txt --region " 172.8/173.8/-42.6/-41.6 " --imaging-time 11:24 
    tropo_icams_date_list.py --date-list 20180101 20181002 20190304 --reference-file velocity.h5
    tropo_icams_date_list.py --date-list-txt date_list_txt --reference-file velocity.h5 --resolution 100.0 
    tropo_icams_date_list.py --date-list-txt date_list_txt --reference-file velocity.h5 --dem-tif /Download/dem.tif
    tropo_icams_date_list.py --date-list 20180101 20181002 20190304 --reference-file velocity.h5 --parallel 4
'''    
    


def cmdLineParse():
    parser = argparse.ArgumentParser(description='Generating high-resolution tropospheric delay map using icams.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)
    parser.add_argument('--ts-file',dest='ts_file', help='Timeseries file used for getting date list.')
    parser.add_argument('--date-list',dest='date_list',nargs='*',help='Date list for downloading trop list.')
    parser.add_argument('--date-list-txt',dest='date_list_txt',help='Text file of date list for downloading trop list.')
    parser.add_argument('--region','-r',dest='region', help='research area in degree. w/e/n/s')
    parser.add_argument('--imaging-time','-t',dest='time', type = str, help='interested SAR-imaging UTC-time. e.g., 12:30')
    parser.add_argument('--reference-file', dest='reference_file', help = 'Reference file used to provide region/imaging-time information. e.g., velocity.h5 from MintPy') 
    parser.add_argument('--resolution',dest='resolution',type = float, default =100.0, help='resolution (m) of delay map.')
    parser.add_argument('--dem-tif', dest='dem_tif', help = 'DEM file (e.g., dem.tif). The coverage should be bigger or equal to the region of interest, if no DEM file provided, it will be downloaded automatically.') 
    parser.add_argument('--parallel', dest='parallelNumb', type=int, default=1, 
                        help='Enable parallel processing and Specify the number of processors.[default: 1]')

    inps = parser.parse_args()
    
    #if not inps.date_list and not inps.date_list_txt:
    #    parser.print_usage()
    #    sys.exit(os.path.basename(sys.argv[0])+': error: date_list or date_list_txt should provide at least one.')

    return inps

    
####################################################################################################
def main(argv):
    
    inps = cmdLineParse()
    root_path = os.getcwd()
    icams_dir = root_path + '/icams'
    if not os.path.isdir(icams_dir): os.mkdir(icams_dir)
    era5_dir = icams_dir + '/ERA5'
    if not os.path.isdir(era5_dir): os.mkdir(era5_dir)
    era5_raw_dir = era5_dir  + '/raw'
    if not os.path.isdir(era5_raw_dir): os.mkdir(era5_raw_dir)
    era5_sar_dir =  era5_dir  + '/sar'
    if not os.path.isdir(era5_sar_dir): os.mkdir(era5_sar_dir)
    dem_dir = icams_dir + '/dem'
    
    # Get date list
    if inps.ts_file:
        date_list = read_hdf5(inps.ts_file,datasetName='date')[0]
        date_list = date_list.astype('U13')
        date_list = list(date_list)
    else:
        date_list = []
        
    if inps.date_list:
        date_list = inps.date_list
    if inps.date_list_txt:
        date_list0 = ut.read_txt2list(inps.date_list_txt)      
        for i in range(len(date_list0)):
            k0 = date_list0[i]
            if k0 not in date_list:
                date_list.append(k0)
    
    date_list = sort_unique_list(date_list)
    nn = len(date_list)
    
    #########
    if inps.reference_file:
        ref0 = inps.reference_file
        attr = ut.read_attr(ref0)
        w,s,e,n = get_meta_corner(attr)
        wsen = (w,s,e,n)   
        if 'CENTER_LINE_UTC' in attr:
            cent_time = attr['CENTER_LINE_UTC']
        else:
            cent_time = '43200'   # 12:00
    else:
        region0 = inps.region
        w,s,e,n = read_region(region0)
        wsen = (w,s,e,n)   
        cent_time = hour2seconds(inps.time)
    
    
    if inps.dem_tif:
        dem0 = inps.dem_tif
    else:
        if not os.path.isdir(dem_dir):
            os.mkdir(dem_dir)
        os.chdir(dem_dir)
        print('Start to download 30m SRTM ...')
        call_str = 'eio --product SRTM1 clip -o dem.tif --bounds ' + str(w - 0.1) + ' ' + str(s-0.1) + ' ' + str(e+0.1) + ' ' + str(n+0.1)
        print(call_str)
        os.system(call_str)
        dem0 = dem_dir + '/dem.tif'
    
    os.chdir(root_path)
    dem_str = ' --dem-tif ' + dem0    
    if inps.region:
        region_str = ' --region ' + inps.region
    else:
        region_str = '' 
   
    if inps.time:
        time_str = ' --imaging-time ' + inps.time
    else:
        time_str = ''
     
            
    if inps.reference_file:
        ref_str = ' --reference-file ' + inps.reference_file
    else:
        ref_str = ''
        
    res_str = ' --resolution ' + str(inps.resolution)   
        
    data_para = []
    for i in range(nn):
        str0 = era5_sar_dir + '/' + date_list[i] + '_tot.h5'
        if not os.path.isfile(str0):
            print('Add >>> ' + date_list[i])
            data0 = [date_list[i], region_str, time_str, ref_str, dem_str, res_str]
            data_para.append(data0)

    parallel_process(data_para, tropo_icams_date, n_jobs=inps.parallelNumb, use_kwargs=False)               

if __name__ == '__main__':
    main(sys.argv[1:])

