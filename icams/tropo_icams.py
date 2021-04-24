#! /usr/bin/env python
#################################################################
###  This program is part of icams  v1.0                      ### 
###  Copy Right (c): 2020, Yunmeng Cao                        ###  
###  Author: Yunmeng Cao                                      ###                                                          
###  Email : ymcmrs@gmail.com                                 ###   
#################################################################
### this programs can be used for MintPy outputs

import sys
import os
import re
import subprocess
import argparse
import numpy as np
import h5py
import glob


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
    parser = argparse.ArgumentParser(description='Check common busrts for TOPS data.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('ts_file',help='input InSAR time-series file name (e.g., timeseries.h5).')
    parser.add_argument('geo_file',help='input geometry file name (e.g., geometryRadar.h5).')
    parser.add_argument('sar_par', help='SLC_par file for providing orbit state paramters.')
    parser.add_argument('--project', dest='project', choices = {'zenith','los'},default = 'los',help = 'project method for calculating the accumulated delays. [default: los]')
    parser.add_argument('--incAngle', dest='incAngle', metavar='FILE',help='incidence angle file for projecting zenith to los, for case of PROJECT = ZENITH when geo_file does not include [incidenceAngle].')
    parser.add_argument('--method', dest='method', choices = {'sklm','linear','cubic'},default = 'sklm',help = 'method used to interp the high-resolution map. [default: sklm]')
    parser.add_argument('--sklm-points-numb', dest='sklm_points_numb', type=int, default=20, help='Number of the closest points used for sklm interpolation. [default: 20]')
    parser.add_argument('--lalo-rescale', dest='lalo_rescale', type=int, default=5, help='oversample rate of the lats and lons [default: 5]')

    inps = parser.parse_args()

    return inps


INTRODUCTION = '''
##################################################################################
   Copy Right(c): 2020, Yunmeng Cao   @icams v1.0
   
   Correcting InSAR tropospheric delays using icams based on ERA5 reanalysis data.
'''

EXAMPLE = """example:
  
  tropo_icams.py timeseries.h5 geometryRadar.h5 SAR.slc.par --lalo-rescale 10 
  tropo_icams.py timeseries.h5 geometryRadar.h5 SAR.slc.par --project zenith
  tropo_icams.py timeseries.h5 geometryRadar.h5 SAR.slc.par --project los --lalo-rescale 5
  tropo_icams.py timeseries.h5 geometryRadar.h5 SAR.slc.par --project los --method linear
  tropo_icams.py timeseries.h5 geometryRadar.h5 SAR.slc.par --project los --method sklm

  
###################################################################################
"""


def main(argv):
    
    inps = cmdLineParse() 
    ts_file = inps.ts_file
    geo_file = inps.geo_file
    slc_par = inps.sar_par
    
    root_dir = os.getcwd()
    icams_dir = root_dir + '/icams'
    atm_dir = root_dir + '/icams/ERA5'
    atm_raw_dir = root_dir + '/icams/ERA5/raw'
    atm_sar_dir = root_dir + '/icams/ERA5/sar'
    
    if not os.path.isdir(icams_dir): os.mkdir(icams_dir)
    if not os.path.isdir(atm_dir): os.mkdir(atm_dir)
    if not os.path.isdir(atm_raw_dir): os.mkdir(atm_raw_dir)
    if not os.path.isdir(atm_sar_dir): os.mkdir(atm_sar_dir)

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
    
    #date_list_download = []
    #for i in range(len(date_list)):
    #    if not date_list[i] in date_list_exist:
    #        date_list_download.append(date_list[i])
            
    #print('---------------------------------------')
    #print('Total number of gps data: ' + str(len(date_list)))
    #print('Exist number of gps data: ' + str(len(date_list)-len(date_list_download)))
    #print('Number of date to download: ' + str(len(date_list_download)))
    
    #if len(date_list_download) > 0:
    #    #print('Start to download gps data...')
    txt_download = 'datelist_download.txt'
    generate_datelist_txt(date_list,txt_download)
    print('----------- Step 1 --------------')
    print('Start to download ERA-5 data...')
    call_str = 'download_era5.py --region-file ' + ts_file + ' --time-file ' + ts_file + ' --date-txt ' + txt_download
    os.system(call_str)
    
    #if inps.type =='tzd': 
    #    Stype = 'Tropospheric delays'
    #else:  
    #    Stype = 'Atmospheric water vapor'
    print('')
    print('----------- Step 2 --------------')
    print('Start to generate high-resolution tropospheric maps ...' )
    #print('Type of the tropospheric products: %s' % Stype)
    #print('Elevation model: %s' % inps.elevation_model)
    #print('Method used for interpolation: %s' % inps.interp_method)
    if inps.method =='sklm':
        print('Variogram model: Spherical')
        print('Number of bins: 50')
        #$print('Max-length used for variogram-modeling: %s' % str(inps.max_length))
        
    print('')
    
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
        call_str = 'tropo_icams_sar.py ' + geo_file + ' ' + slc_par + ' --date ' + date_generate[i] + ' --method ' + inps.method + ' --project ' + inps.project + ' --lalo-rescale ' + str(inps.lalo_rescale) + ' --sklm-points-numb ' + str(inps.sklm_points_numb)
        os.system(call_str)        
   
    print('')
    print('----------- Step 3 --------------')
    print('Start to generate time-series of tropospheric data ...' )
    txt_list = 'date_list_atm.txt'
    generate_datelist_txt(date_list,txt_list)
    date_list0 = [os.path.basename(k0).split('_')[0] for k0 in glob.glob(atm_sar_dir+'/*' + inps.project + '_' + inps.method + '*')]
    date_list = sorted(set(date_list0))
    generate_datelist_txt(date_list,txt_list)
    
    call_str =  'icams_timeseries.py --date-txt ' + txt_list + ' --project ' + inps.project + ' --method ' + inps.method 
    os.system(call_str)
    
    print('Done.')
    
    print('')
    print('----------- Step 4 --------------')

    print('')
    print('Start to correct InSAR time-series tropospheric delays ...' )
    call_str = 'diff_icams.py ' + inps.ts_file + ' ' + ' timeseries_icams_' + inps.project + '_' + inps.method + '.h5' + ' --add  -o timeseries_icamsCor' + '_' + inps.project + '_' + inps.method + '.h5'
    os.system(call_str)
    print('Done.')

    sys.exit(1)


###############################################################

if __name__ == '__main__':
    main(sys.argv[:])

