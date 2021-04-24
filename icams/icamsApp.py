#! /usr/bin/env python
#################################################################
###  ICAMS: Python module for InSAR troposphere correction    ### 
###  Copy Right (c): 2020, Yunmeng Cao                        ###  
###  Author: Yunmeng Cao                                      ###                                                          
###  Email : ymcmrs@gmail.com                                 ###   
#################################################################

import sys
import os
import re
import subprocess
import argparse
import numpy as np
import h5py
from skimage import io
import glob

from icams import _utils as ut

###############################################################
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

def get_sufix(STR):
    n = len(STR.split('.'))
    SUFIX = STR.split('.')[n-1]
   
    return SUFIX 

def get_dataNames(FILE):
    with h5py.File(FILE, 'r') as f:
        dataNames = []
        for k0 in f.keys():
            dataNames.append(k0)
    return dataNames

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


def write_h5(datasetDict, out_file, metadata=None, ref_file=None, compression=None):
    
    if os.path.isfile(out_file):
        print('delete exsited file: {}'.format(out_file))
        os.remove(out_file)

    print('create HDF5 file: {} with w mode'.format(out_file))
    with h5py.File(out_file, 'w') as f:
        for dsName in datasetDict.keys():
            data = datasetDict[dsName]
            ds = f.create_dataset(dsName,
                              data=data,
                              compression=compression)
        
        for key, value in metadata.items():
            f.attrs[key] = str(value)
            #print(key + ': ' +  value)
    print('finished writing to {}'.format(out_file))
        
    return out_file

def generate_datelist_txt(date_list,txt_name):
    
    if os.path.isfile(txt_name): os.remove(txt_name)
    for list0 in date_list:
        call_str = 'echo ' + list0 + ' >>' + txt_name 
        os.system(call_str)
        
    return

def cmdLineParse():
    parser = argparse.ArgumentParser(description='InSAR troposphere correction using ICAMS.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)


    parser.add_argument('corners', help='research area in degree. w/e/n/s')
    parser.add_argument('sar_par', help='SLC_par file for providing orbit state paramters.')
    parser.add_argument('--date-list', dest='date_list', nargs='*',help='date list to extract.')
    parser.add_argument('--date-txt', dest='date_txt',help='date list text to extract.')
    parser.add_argument('--date-file', dest='date_file',help='mintPy formatted h5 file, which contains date infomration.')
    
    parser.add_argument('--project', dest='project', choices = {'zenith','los'},default = 'los',help = 'project method for calculating the accumulated delays. [default: los]')
    parser.add_argument('--method', dest='method', choices = {'sklm','linear','cubic'},default = 'sklm',help = 'method used to interp the high-resolution map. [default: kriging]')
    parser.add_argument('--sklm-points-numb', dest='sklm_points_numb', type=int, default=20, help='Number of the closest points used for sklm interpolation. [default: 20]')
    parser.add_argument('--lalo-rescale', dest='lalo_rescale', type=int, default=5, help='oversample rate of the lats and lons [default: 5]')

    inps = parser.parse_args()

    return inps


INTRODUCTION = '''
##################################################################################
   Copy Right(c): 2020, Yunmeng Cao   @icams v1.0
   
   InSAR tropospheric correction based on ICAMS.
'''

EXAMPLE = """example:
  
  icamsApp.py 122/124/33/35 Hawaii.slc.par --date-list 20180101 20180501 --lalo-rescale 10 
  icamsApp.py 122/124/33/35 Hawaii.slc.par --date-list 20180101 20180501 --method sklm
  icamsApp.py 122/124/33/35 Hawaii.slc.par --date-list 20180101 --project los --method sklm
  icamsApp.py 122/124/33/35 Hawaii.slc.par --date-list 20180101 --project zenith --lalo-rescale 10 

###################################################################################
"""


def main(argv):
    
    inps = cmdLineParse() 
    slc_par = inps.sar_par
    
    root_dir = os.getcwd()
    icams_dir = root_dir + '/icams'
    atm_dir = root_dir + '/icams/ERA5'
    atm_raw_dir = root_dir + '/icams/ERA5/raw'
    atm_sar_dir = root_dir + '/icams/ERA5/sar'
    raw_dir = atm_raw_dir
    era5_dir = raw_dir
    
    T0 = ut.read_gamma_par(inps.sar_par, 'read', 'center_time:')
    research_time = T0.split('s')[0]
    hour = era5_time(research_time)
    
    if not os.path.isdir(icams_dir): os.mkdir(icams_dir)
    if not os.path.isdir(atm_dir): os.mkdir(atm_dir)
    if not os.path.isdir(atm_raw_dir): os.mkdir(atm_raw_dir)
    if not os.path.isdir(atm_sar_dir): os.mkdir(atm_sar_dir)

     # Get research region (w, s, e, n)
    w,s,e,n = read_region(inps.corners)   
    wsen = (w,s,e,n)    
    snwe = get_snwe(wsen, min_buffer=0.5, multi_1=True)
    area = snwe2str(snwe) 
    
    #meta = read_attr(ts_file)
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
    
    #if not os.path.isdir(gps_dir): 
    #    os.mkdir(gps_dir)
    #    print('Generate GPS directory: %s' % gps_dir)
    #if not os.path.isdir(atm_dir): 
    #    os.mkdir(atm_dir)
    #    print('Generate GPS-atm directory: %s' % atm_dir)
    
    #date_list_exist = [os.path.basename(x).split('_')[5] for x in glob.glob(atm_raw_dir + '/ERA-5*.grb')]
   
    ## Download data
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
    #######
    
    ### Download DEM 
    #call_str='wget -q -O dem.tif "http://ot-data1.sdsc.edu:9090/otr/getdem?north=%f&south=%f&east=%f&west=%f&demtype=SRTMGL1"' % (n,s,e,w)
    call_str = 'eio --product SRTM1 clip -o dem.tif --bounds ' + str(w) + ' ' + str(s) + ' ' + str(e) + ' ' + str(n)
    print(call_str)
    os.system(call_str)    
    DEM = 'dem.tif'

    SUFIX = get_sufix(DEM)
    SS ='.' + SUFIX
    DTIF = DEM.replace(SS,'.tif')

    if not SUFIX == 'tif':
        call_str = 'gdal_translate ' + DEM + ' -of GTiff ' + DTIF
        os.system(call_str)

    DEM = DTIF
    call_str = 'gdalinfo ' + DEM + ' >ttt'     
    os.system(call_str)
    
    f = open('ttt')    
    for line in f:
        if 'Origin =' in line:
            STR1 = line
            AA = STR1.split('Origin =')[1]
            Corner_LON = AA.split('(')[1].split(',')[0]
            Corner_LAT = AA.split('(')[1].split(',')[1].split(')')[0]
        elif 'Pixel Size ' in line:
            STR2 = line
            AA = STR2.split('Pixel Size =')[1]
            Post_LON = AA.split('(')[1].split(',')[0]
            Post_LAT = AA.split('(')[1].split(',')[1].split(')')[0]
        
        elif 'Size is' in line:
            STR3 = line
            AA =STR3.split('Size is')[1]
            WIDTH = AA.split(',')[0]
            FILE_LENGTH = AA.split(',')[1]
    f.close()
    
    heis = io.imread(DEM); heis = np.asarray(heis, dtype = float)
    
    WIDTH = int(WIDTH); LENGTH = int(FILE_LENGTH)
    xnumb = np.arange(WIDTH); ynumb = np.arange(LENGTH)
    xx,yy = np.meshgrid(xnumb,ynumb)
    
    lats = yy*float(Post_LAT) + float(Corner_LAT)
    lons = xx*float(Post_LON) + float(Corner_LON)
    
    DEMH5 = 'geometry.h5'
    datasetDict = dict()
    datasetDict['height'] = heis
    datasetDict['latitude'] = lats
    datasetDict['longitude'] = lons
    
    meta0 = dict()
    meta0['X_FIRST'] = Corner_LON
    meta0['Y_FIRST'] = Corner_LAT
    meta0['X_STEP'] = Post_LAT
    meta0['Y_STEP'] = Post_LON
    meta0['LENGTH'] = LENGTH
    meta0['WIDTH'] = WIDTH
    
    ut.write_h5(datasetDict, DEMH5, metadata= meta0, ref_file=None, compression=None)
    
    
    print('')
    print('Start to generate SAR delays...')
    date_generate = []
    for i in range(len(date_list)):
        out0 = atm_sar_dir + '/' + date_list[i] + '_' + inps.project + '_' + inps.method + '.h5'
        if not os.path.isfile(out0):
            date_generate.append(date_list[i])
    print('Total number of data set: %s' % str(len(date_list)))          
    print('Exsit number of data set: %s' % str(len(date_list)-len(date_generate))) 
    print('Number of high-resolution maps need to be generated: %s' % str(len(date_generate)))  
    
    
    for i in range(len(date_generate)):
        
        print('Date: ' + date_generate[i] + ' (' + str(i+1) + '/' + str(len(date_generate)) + ')')
        call_str = 'tropo_icams_sar.py ' + DEMH5 + ' ' + inps.sar_par + ' --date ' + date_generate[i] + ' --method ' + inps.method + ' --project ' + inps.project + ' --lalo-rescale ' + str(inps.lalo_rescale) + ' --sklm-points-numb ' + str(inps.sklm_points_numb)
        os.system(call_str)        
    
    
    print('Done.')

    sys.exit(1)


###############################################################

if __name__ == '__main__':
    main(sys.argv[:])

