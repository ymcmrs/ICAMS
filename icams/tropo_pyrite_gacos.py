#! /usr/bin/env python
#################################################################
###  This program is part of PyRite  v1.0                     ### 
###  Copy Right (c): 2019, Yunmeng Cao                        ###  
###  Author: Yunmeng Cao                                      ###                                                          
###  Email : ymcmrs@gmail.com                                 ###
###  Univ. : King Abdullah University of Science & Technology ###   
#################################################################

import sys
import os
import re
import subprocess
import argparse
import numpy as np
import h5py
import glob

from pyrite import _utils as ut
###############################################################

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
    parser = argparse.ArgumentParser(description='Correcting InSAR tropospheric delays using PyRite based on GACOS data.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('ts_file',help='input InSAR time-series file name (e.g., timeseries.h5).')
    parser.add_argument('geo_file',help='input geometry file name (e.g., geometryRadar.h5).')
    parser.add_argument('--gacos-folder', dest='gacos_folder', metavar='FILE',help='folder of the gacos raw dada. [Default: gacos/raw ]')
    #parser.add_argument('--incAngle', dest='incAngle', metavar='FILE',help='incidence angle file for projecting zenith to los, for case of PROJECT = ZENITH when geo_file does not include [incidenceAngle].')


    inps = parser.parse_args()

    return inps


INTRODUCTION = '''
##################################################################################
   Copy Right(c): 2019, Yunmeng Cao   @PyRite v1.0
   
   Correcting InSAR tropospheric delays using PyRite based on GACOS data.
'''

EXAMPLE = """example:
  
  tropo_pyrite_gacos.py timeseries.h5 geometryRadar.h5 
  tropo_pyrite_gacos.py timeseries.h5 geometryRadar.h5 --gacos-folder gacos/raw
  
###################################################################################
"""


def main(argv):
    
    inps = cmdLineParse() 
    ts_file = inps.ts_file
    geo_file = inps.geo_file
    inc,atr = ut.read_hdf5(geo_file, datasetName = 'incidenceAngle')
    LENGTH = int(atr['LENGTH'])
    WIDTH = int(atr['WIDTH'])
    
    root_dir = os.getcwd()
    gacos_dir = root_dir + '/gacos'
    raw_gacos_dir = root_dir + '/gacos/raw'
    sar_gacos_dir = root_dir + '/gacos/sar'
    
    
    if not os.path.isdir(gacos_dir): os.mkdir(gacos_dir)
    #if not os.path.isdir(raw_gacos_dir): os.mkdir(raw_gacos_dir)
    if not os.path.isdir(sar_gacos_dir): os.mkdir(sar_gacos_dir)
    
    if inps.gacos_folder: raw_gacos_dir = inps.gacos_folder
    else: raw_gacos_dir = root_dir + '/gacos/raw'
    
    meta = read_attr(ts_file)
    
    #if not os.path.isdir(gps_dir): 
    #    os.mkdir(gps_dir)
    #    print('Generate GPS directory: %s' % gps_dir)
    #if not os.path.isdir(atm_dir): 
    #    os.mkdir(atm_dir)
    #    print('Generate GPS-atm directory: %s' % atm_dir)
    
    #date_list_exist = [os.path.basename(x).split('_')[5] for x in glob.glob(atm_raw_dir + '/ERA-5*.grb')]
   
    date_list = read_hdf5(ts_file,datasetName='date')[0]
    date_list = date_list.astype('U13')
    date_list = list(date_list)
    
    date_list_exist0 = [os.path.basename(x).split('.')[0] for x in glob.glob(raw_gacos_dir + '/*.ztd')]
    
    date_list_exist = []
    for k0 in date_list_exist0:
        if k0 in date_list:
            date_list_exist.append(k0)
    
    download_list = []
    for k0 in date_list:
        if k0 not in date_list_exist0:
            download_list.append(k0)
    
    if len(download_list) > 0:
        print('Not available GACOS data list:')
        for k0 in download_list:
            print(k0)
    
    print('Total date list number: %s' % str(len(date_list)))
    print('Exist GACOS data number: %s' % str(len(date_list_exist)))
    
    print('')
    print('----------- Step 1 --------------')
    print('Start to generate gacos-based SAR tropospheric maps ...' )
    sar_list_exist0 = [os.path.basename(x).split('_')[0] for x in glob.glob(sar_gacos_dir + '/*_tzd.h5')]
    sar_list_exist = []
    for k0 in sar_list_exist0:
        if k0 in date_list:
            sar_list_exist.append(k0)
    
    N = len(date_list) - len(sar_list_exist)   
    generate_list = []
    for k0 in date_list:
        if k0 not in sar_list_exist:
            if k0 in date_list_exist:
                generate_list.append(k0)
    
    print('Exist gacos-sar delay map: %s' % str(len(sar_list_exist)))
    print('To be generated gacos-sar delay map: %s' % str(len(generate_list)))
    
    for i in range(len(generate_list)):
        k0 = generate_list[i]
        ut.print_progress(i+1, len(generate_list), prefix='GACOS-sar delay map generated:', suffix='complete', decimals=1, barLength=50, elapsed_time=None)
        call_str = 'gacos2sar.py ' + raw_gacos_dir + '/' + k0 + '.ztd' + ' ' + raw_gacos_dir + '/' + k0 + '.ztd.rsc ' + ' ' + geo_file
        os.system(call_str)
    
    print('')
    print('----------- Step 2 --------------')
    print('Start to generate time-series of gacos-based SAR tropospheric maps ...' )
    
    gacos_list = [os.path.basename(x).split('_')[0] for x in glob.glob(sar_gacos_dir + '/*_tzd.h5')]
    
    ts_data_list = []
    for k0 in gacos_list:
        if k0 in date_list:
            ts_data_list.append(k0)
    
    print('To be used number of gacos-sar delay map: %s' % str(len(gacos_list)))
    ts_gacos = np.zeros((len(ts_data_list),LENGTH,WIDTH),dtype = np.float32)
    ts_data_list = sorted(ts_data_list)
    
    ff = np.cos(inc/180*np.pi)
    for i in range(len(ts_data_list)):
        file0 = sar_gacos_dir + '/' + ts_data_list[i] + '_tzd.h5'
        data0 = ut.read_hdf5(file0,datasetName = 'aps_sar')[0]
        data0 = data0/ff
        ts_gacos [i,:,:] = data0
    
    out_ts_gacos = 'timeseries_gacos_los.h5' 
    meta = ut.read_attr(ts_file)
    datasetDict = dict()
    datasetDict['timeseries'] = ts_gacos
    datasetDict['date'] = np.asarray(ts_data_list,dtype = np.string_)
    #datasetDict['bperp'] = bperp
    meta['FILE_TYPE'] ='timeseries'
    meta['UNIT'] = 'm'
    ut.write_h5(datasetDict, out_ts_gacos, metadata=meta, ref_file=None, compression=None)
    print('Done.')   

    print('')
    print('----------- Step 3 --------------')
    print('Start to correct TS-InSAR results using GACOS data...' )
    
    call_str = 'diff_pyrite.py ' + ts_file + ' ' + ' timeseries_gacos_los.h5 --add  -o timeseries_gacosCor.h5'
    os.system(call_str)
    print('Done.')

    sys.exit(1)


###############################################################

if __name__ == '__main__':
    main(sys.argv[:])

