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


    
def cmdLineParse():
    parser = argparse.ArgumentParser(description='Generate high-resolution tropospheric product map for a list of SAR acquisitions',formatter_class=argparse.RawTextHelpFormatter,epilog=INTRODUCTION+'\n'+EXAMPLE)
    parser.add_argument('--date-txt',dest='date_list', help='text of date list.')
    parser.add_argument('--data', dest='data', choices = {'turb','aps','hgt','trend','sigma'}, default = 'aps',help = 'type of the high-resolution tropospheric map.[default: aps]')
    parser.add_argument('--type', dest='type', choices = {'tzd','wzd'}, default = 'tzd',help = 'type of the high-resolution tropospheric map.[default: tzd]')
    #parser.add_argument('--absolute', dest='absolute', action='store_true',help='generate absolute value based tropospheric map. ')
    parser.add_argument('--atm-dir',dest='atm_dir', help='directory of the atmospheric data. [default: ./gigpy/atm]')
    parser.add_argument('-o','--out', dest='out', help='Name of the output file.')

       
    inps = parser.parse_args()

    return inps


INTRODUCTION = '''
##################################################################################
   Copy Right(c): 2019, Yunmeng Cao   @PyRite v1.0
   
   Generate time-series of high-resolution ERA5-based tropospheric maps (include maps of turb, atmospheric delay, trend, elevation-correlated components).
'''

EXAMPLE = """Example:
  
  generate_timeseries_era5.py --date-txt date_list --data aps --atm-dir /Yunmeng/SCRATCH/pyrite/ERA5/sar
  generate_timeseries_era5.py --date-txt date_list --data turb --type wzd
  generate_timeseries_era5.py --date-txt date_list --data hgt
  generate_timeseries_era5.py --date-txt date_list --data trend --type tzd -o timeseries_gps_trend.h5
  
###################################################################################
"""

###############################################################

def main(argv):
    
    inps = cmdLineParse()
    #date_list_txt = inps.date_list      
    #date_list = np.loadtxt(date_list_txt,dtype=np.str)
    #date_list = date_list.tolist()
    #N=len(date_list)
    
    path0 = os.getcwd()
    if inps.atm_dir: atm_dir = inps.atm_dir
    else: atm_dir = path0 + '/pyrite/ERA5/sar'
        
    if inps.date_list:
        date_list = np.loadtxt(inps.date_list,dtype=np.str)
        date_list = date_list.tolist()
    else:
        print('')
        print('Search the available era5_sar data from folder:')
        print(atm_dir)
        date_list = [os.path.basename(x).split('_')[0] for x in glob.glob(atm_dir + '/*_' + inps.type + '.h5')]
        #date_list = date_list.tolist()
    date_list = list(map(int, date_list))
    date_list = sorted(date_list)
    date_list = list(map(str, date_list))
    
    N = len(date_list)
    print('')
    print('-----------------------------')
    print('Date list of the time-series:')
    data_list = []
    for i in range(N):
        print(date_list[i])
        file0 = atm_dir + '/' + date_list[i] + '_' + inps.type + '.h5' 
        data_list.append(file0)
    
    meta = ut.read_attr(data_list[0])
    WIDTH = int(meta['WIDTH'])
    LENGTH = int(meta['LENGTH'])
    #REF_X = int(meta['REF_X'])
    #REF_Y = int(meta['REF_Y'])
    print('')
    print('---------------------------------')
    
    if inps.data =='aps': S0 = 'aps_sar'       # total tropospheric map (tropospheric delay & atmospheric water vapor)
    elif inps.data =='turb': S0 ='turb_sar'    # turbulent tropospheric map
    elif inps.data=='hgt': S0 ='hgt_sar'      # elevation-correlated tropospheric map
    elif inps.data =='trend': S0 ='trend_sar'  # spatial trend/ramp of the tropospheric map
    elif inps.data =='sigma': S0 ='sigma_sar'  # uncertainty map of the turbulent tropospheric map
    
    ts_era5 = np.zeros((len(date_list),LENGTH,WIDTH),dtype = np.float32)
    for i in range(len(date_list)):
        file0 = atm_dir + '/' + date_list[i] + '_' + inps.type + '.h5'
        data0 = ut.read_hdf5(file0,datasetName = S0)[0]
        #if not inps.absolute:
        #    ts_gps[i,:,:] = data0 - data0[REF_Y,REF_X]
        #else:
        #    ts_gps[i,:,:] = data0
        
        ts_era5[i,:,:] = data0
    #ts0 = ts_gps[2,:,:]
    # relative to the first date
    #if not inps.absolute:
     #   for i in range(len(date_list)):
     #       ts_gps[i,:,:] = ts_gps[i,:,:] - ts0
    
    #if not inps.absolute:
    #    ts_gps -=ts_gps[0,:,:]
    #else:
    #    del meta['REF_X']
    #    del meta['REF_Y']
        
    if inps.out: out_ts_era5 = inps.out
    else: out_ts_era5 = 'timeseries_pyrite_' + inps.type + '_'+ inps.data + '.h5'
    
    datasetDict = dict()
    datasetDict['timeseries'] = ts_era5
    #date_list = read_hdf5(ts_file,datasetName='date')[0]
    #bperp = read_hdf5(ts_file,datasetName='bperp')[0]
    datasetDict['date'] = np.asarray(date_list,dtype = np.string_)
    #datasetDict['bperp'] = bperp
    meta['FILE_TYPE'] ='timeseries'
    meta['UNIT'] = 'm'
    ut.write_h5(datasetDict, out_ts_era5, metadata=meta, ref_file=None, compression=None)
    print('Done.')   

    sys.exit(1)
    
###############################################################

if __name__ == '__main__':
    main(sys.argv[:])
