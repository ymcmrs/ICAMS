#! /usr/bin/env python
#################################################################
###  This program is part of ICAMS   v1.0                     ### 
###  Copy Right (c): 2020, Yunmeng Cao                        ###  
###  Author: Yunmeng Cao                                      ###                                                          
###  Contact : ymcmrs@gmail.com                               ### 
#################################################################
import os
import sys
import numpy as np
import argparse
from pyrite import _utils as ut
import glob

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False  

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

def unitdate(DATE):
    LE = len(str(int(DATE)))
    DATE = str(int(DATE))
    if LE==5:
        DATE = '200' + DATE  
    
    if LE == 6:
        YY = int(DATE[0:2])
        if YY > 80:
            DATE = '19' + DATE
        else:
            DATE = '20' + DATE
    return DATE

def read_txt_line(txt):
    # Open the file with read only permit
    f = open(txt, "r")
    # use readlines to read all lines in the file
    # The variable "lines" is a list containing all lines in the file
    lines = f.readlines()
    lines0 = [line.strip() for line in lines] 
    f.close()
    
    # remove empty lines
    lines_out = [] 
    for line in lines0:
        if not len(line) ==0:
            lines_out.append(line)
    return lines_out

def get_lack_datelist(date_list, date_list_exist):
    date_list0 = []
    for k0 in date_list:
        if k0 not in date_list_exist:
            date_list0.append(k0)
    
    return date_list0

def era5_time(research_time0):
    
    research_time =  round(float(research_time0) / 3600)
    if len(str(research_time)) == 1:
        time0 = '0' + str(research_time)
    else:
        time0 = str(research_time)
    
    return time0

def ceil_to_5(x):
    """Return the closest number in multiple of 5 in the larger direction"""
    assert isinstance(x, (int, np.int16, np.int32, np.int64)), 'input number is not int: {}'.format(type(x))
    if x % 5 == 0:
        return x
    return x + (5 - x % 5)

def floor_to_5(x):
    """Return the closest number in multiple of 5 in the lesser direction"""
    assert isinstance(x, (int, np.int16, np.int32, np.int64)), 'input number is not int: {}'.format(type(x))
    return x - x % 5

def floor_to_1(x):
    """Return the closest number in multiple of 5 in the lesser direction"""
    assert isinstance(x, (int, np.int16, np.int32, np.int64)), 'input number is not int: {}'.format(type(x))
    return x - x % 1

def ceil_to_1(x):
    """Return the closest number in multiple of 5 in the larger direction"""
    assert isinstance(x, (int, np.int16, np.int32, np.int64)), 'input number is not int: {}'.format(type(x))
    if x % 1 == 0:
        return x
    return x + (1 - x % 1)

def floor_to_2(x):
    """Return the closest number in multiple of 5 in the lesser direction"""
    assert isinstance(x, (int, np.int16, np.int32, np.int64)), 'input number is not int: {}'.format(type(x))
    return x - x % 2

def ceil_to_2(x):
    """Return the closest number in multiple of 5 in the larger direction"""
    assert isinstance(x, (int, np.int16, np.int32, np.int64)), 'input number is not int: {}'.format(type(x))
    if x % 2 == 0:
        return x
    return x + (2 - x % 2)

def get_snwe(wsen, min_buffer=0.5, multi_1=True):
    # get bounding box
    lon0, lat0, lon1, lat1 = wsen
    # lat/lon0/1 --> SNWE
    S = np.floor(min(lat0, lat1) - min_buffer).astype(int)
    N = np.ceil( max(lat0, lat1) + min_buffer).astype(int)
    W = np.floor(min(lon0, lon1) - min_buffer).astype(int)
    E = np.ceil( max(lon0, lon1) + min_buffer).astype(int)

    # SNWE in multiple of 5
    if multi_1:
        S = floor_to_1(S)
        W = floor_to_1(W)
        N = ceil_to_1(N)
        E = ceil_to_1(E)
    return (S, N, W, E)


def get_fname_list(date_list,area,hr):
    
    flist = []
    for k0 in date_list:
        f0 = 'ERA-5{}_{}_{}.grb'.format(area, k0, hr)
        flist.append(f0)
        
    return flist
    
def snwe2str(snwe):
    """Get area extent in string"""
    if not snwe:
        return None

    area = ''
    s, n, w, e = snwe

    if s < 0:
        area += '_S{}'.format(abs(s))
    else:
        area += '_N{}'.format(abs(s))

    if n < 0:
        area += '_S{}'.format(abs(n))
    else:
        area += '_N{}'.format(abs(n))

    if w < 0:
        area += '_W{}'.format(abs(w))
    else:
        area += '_E{}'.format(abs(w))

    if e < 0:
        area += '_W{}'.format(abs(e))
    else:
        area += '_E{}'.format(abs(e))
    return area


def cmdLineParse():
    parser = argparse.ArgumentParser(description='Extract the SAR synchronous GPS tropospheric products.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('--region',dest='region', help='research area in degree. w/e/n/s')
    parser.add_argument('--region-file',dest='region_file', help='mintPy formatted h5 file, which contains corner infomration.')
    # region and region-file should provide at least one
    parser.add_argument('--time',dest='time', help='interested research UTC-time in seconds. e.g., 49942')
    parser.add_argument('--time-file',dest='time_file', help='mintPy formatted h5 file, which contains CENTER_LINE_UTC')
    # time and time-file should provide at least one
    parser.add_argument('--date-list', dest='date_list', nargs='*',help='date list to extract.')
    parser.add_argument('--date-txt', dest='date_txt',help='date list text to extract.')
    parser.add_argument('--date-file', dest='date_file',help='mintPy formatted h5 file, which contains date infomration.')
    # date-list, date-txt, date-file should provide at least one.
    
    inps = parser.parse_args()

    if (not (inps.region or inps.region_file)) or (not (inps.time or inps.time_file)) or (not (inps.date_list or inps.date_txt or inps.date_file)):
        parser.print_usage()
        sys.exit(os.path.basename(sys.argv[0])+': error: region, time, and date information should be provided.')
    
    return inps

###################################################################################################

INTRODUCTION = '''
    Download ERA-5 re-analysis-datasets from the latest ECMWF platform: 
    'https://cds.climate.copernicus.eu/api/v2'
   
       -------------------------------------------------------------------------------
       coverage   temporal_resolution  spatial_resolution      latency     analysis

        Global          Hourly          0.25 deg (~31 km)       3-month      4D-var
       -------------------------------------------------------------------------------
    
'''

EXAMPLE = '''Examples:

    download_era5.py --region 122/124/33/35 --time 49942 --date-list 20180101 20180102
    download_era5.py --region 122/124/33/35 --time 49942 --date-txt ~/date_list.txt
    
    download_era5.py --region-file velocity.h5 --time-file velocity.h5 --date-txt ~/date_list.txt
    download_era5.py --region-file timeseries.h5 --time-file velocity.h5 --date-file timeseries.h5
    download_era5.py --region-file timeseries.h5 --time-file velocity.h5 --date-list 20180101 20180102
    
    download_era5.py --region-file 122/124/33/35 --time-file velocity.h5 --date-file timeseries.h5
    download_era5.py --region-file 122/124/33/35 --time 49942 --date-file timeseries.h5
    
'''    
##################################################################################################   

def main(argv):
    
    inps = cmdLineParse()
    root_dir = os.getcwd()
    
    pyrite_dir = root_dir + '/pyrite'
    era5_dir = pyrite_dir + '/ERA5'
    raw_dir = era5_dir + '/raw'
    sar_dir = era5_dir + '/sar'
    
    
    if not os.path.isdir(pyrite_dir):
        print('pyrite folder is not found under the corrent directory.')
        print('Generate folder: %s' % pyrite_dir)
        os.mkdir(pyrite_dir)
    if not os.path.isdir(era5_dir):
        print('ERA5 folder is not found under the corrent directory.')
        print('Generate folder: %s' % era5_dir)
        os.mkdir(era5_dir)
    if not os.path.isdir(raw_dir):
        print('Generate raw-data folder: %s' % raw_dir)
        os.mkdir(raw_dir)
           
    # Get research region (w, s, e, n)
    if inps.region: 
        w,s,e,n = read_region(inps.region)
    elif inps.region_file: 
        meta_region = ut.read_attr(inps.region_file)
        w,s,e,n = get_meta_corner(meta_region)
        
    wsen = (w,s,e,n)    
    snwe = get_snwe(wsen, min_buffer=0.5, multi_1=True)
    area = snwe2str(snwe) 
    
    # Get research time (utc in seconds)
    if inps.time: 
        research_time = inps.time
    elif inps.time_file: 
        meta_time = ut.read_attr(inps.time_file)
        research_time = meta_time['CENTER_LINE_UTC']
    hour = era5_time(research_time)
    
    # Get date list  
    date_list = []
    if inps.date_list: date_list = inps.date_list
        
    if inps.date_txt:
        date_list2 = read_txt_line(inps.date_txt)
        for list0 in date_list2:
            if (list0 not in date_list) and is_number(list0):
                date_list.append(list0)
                
    if inps.date_file: 
        date_list3 = ut.read_hdf5(inps.date_file, datasetName='date')[0]
        date_list3 = date_list3.astype('U13')
        for list0 in date_list3:
            if (list0 not in date_list) and is_number(list0):
                date_list.append(list0)
    
    flist = get_fname_list(date_list,area,hour)
    
    date_list = list(map(int, date_list))
    date_list = sorted(date_list)
    date_list = list(map(str, date_list))
    
    flist_era5_exist = [os.path.basename(x) for x in glob.glob(raw_dir + '/ERA-5*')]
    flist_download = []
    date_list_download = []
    for f0 in flist: 
        if not f0 in flist_era5_exist:
            f00 = os.path.join(raw_dir, f0) # add basedir
            flist_download.append(f00)
            date_list_download.append(f0.split('_')[5])
    
    print('')
    print('Total number of ERA5-data need to be downloaded: %s' % str(len(flist)))
    print('Existed number of ERA5-data : %s' % str(len(flist)-len(flist_download)))
    print('Number of the to be downloaded ERA5-data : %s' % str(len(flist_download)))
    print('')
    
    ut.ECMWFdload(date_list_download, hour, era5_dir, model='ERA5', snwe=snwe, flist = flist_download)
    
    #s,n,w,e = snwe
    #if not os.path.isfile('geometry_era5.h5'):
    #    print('')
    #    print('Start to generate the geometry file over the EAR5-data region...')
    #    AREA = '" ' + str(w - 0.1) + '/'  + str(e + 0.1) + '/' + str(s - 0.1) + '/' + str(n + 0.1) + ' "'
    #    #print(AREA)
    #    call_str = 'generate_geometry_era5.py --region ' + AREA + ' --resolution 10000 -o geometry_era5.h5'
    #    os.system(call_str)
    #else:
    #    print('geometry_ear5.h5 exist.')
    #    print('skip the step of generating the geometry file.')
    
    print('Done.')     
    sys.exit(1)        

if __name__ == '__main__':
    main(sys.argv[1:])


