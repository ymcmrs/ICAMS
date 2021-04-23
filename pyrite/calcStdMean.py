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
from pyrite import _utils as ut

def cmdLineParse():
    parser = argparse.ArgumentParser(description='Calculate standard deviations and the mean values of a specific data',formatter_class=argparse.RawTextHelpFormatter,epilog=INTRODUCTION+'\n'+EXAMPLE)
    parser.add_argument('file', help='mintPy/PyRite formatted h5 data file')
    parser.add_argument('--data',dest = 'data_type', help='data type, e.g., turb, aps')
    parser.add_argument('-m','--mask',dest = 'mask', help='mask file used to mask the input data. [default: non-zero values]')
    parser.add_argument('-o','--out', dest='out',help = 'Name of the output statistic results file.')
      
    inps = parser.parse_args()

    return inps


INTRODUCTION = '''
##################################################################################
   Copy Right(c): 2019, Yunmeng Cao   @PyRite v1.0
   
   Calculate standard deviations and the mean values of a specific data
'''

EXAMPLE = """Example:
  
  calcStdMean.py timeseries_pyriteCor.h5
  calcStdMean.py timeseries_pyriteCor.h5 -m mask.h5 -o timeseries_pyriteCor.stat
  calcStdMean.py diff_gps_pyrite.h5 --data aps_sar -m mask.h5
  calcStdMean.py geometryRadar.h5 --data height
  
###################################################################################
"""

###############################################################

def main(argv):
    
    inps = cmdLineParse()
    
    h5_file = inps.file
    meta = ut.read_attr(h5_file)
    if inps.mask: mask = ut.read_hdf5(inps.mask,datasetName = 'mask')[0]
    else: mask = np.ones((int(meta['LENGTH']),int(meta['WIDTH'])),dtype = np.float32)
    
    if inps.out: OUT = inps.out
    else: OUT = os.path.basename(h5_file).split('.')[0] + '.stat'
        
    if os.path.isfile(OUT):
        os.remove(OUT)
    
    WIDTH = int(meta['WIDTH'])
    LENGTH = int(meta['LENGTH'])
    if meta['FILE_TYPE'] == 'timeseries':
        date_list = ut.read_hdf5(h5_file,datasetName = 'date')[0]
        date_list = date_list.astype('U13')
        date_list = date_list.tolist()
        data_all = ut.read_hdf5(h5_file,datasetName = 'timeseries')[0]
        for i in range(1,len(date_list)):
            data0 = data_all[i,:,:]
            data0 = data0.reshape(LENGTH,WIDTH)
            data0 = data0 * mask
            where_are_NaNs = np.isnan(data0)
            data0[where_are_NaNs] = 0
            std0 = np.std(data0[data0!=0])
            mean0 = np.mean(data0[data0!=0])
            if meta['UNIT'] == 'm':
                std0 = round(std0*10000)/100
                mean0 = round(mean0*10000)/100
            
            print('{}: std {}  mean {}'.format(date_list[i], str(std0), str(mean0)))
            
            S0 = date_list[i] + ' ' + str(std0) + ' ' + str(mean0)
            call_str = 'echo ' + S0 + ' >>' + OUT
            os.system(call_str)
    
    else:
        
        data0 = ut.read_hdf5(h5_file,datasetName = inps.data_type)[0]
        data0 = data0.reshape(LENGTH,WIDTH)
        data0 = data0 * mask
        where_are_NaNs = np.isnan(data0)
        data0[where_are_NaNs] = 0
        std0 = np.std(data0[data0!=0])
        mean0 = np.mean(data0[data0!=0])
        if meta['UNIT'] == 'm':
                std0 = round(std0*10000)/100
                mean0 = round(mean0*10000)/100
            
        print('{}: std {}  mean {}'.format(inps.data_type, str(std0), str(mean0))) 
        S0 = str(inps.data_type) + ' ' + str(std0) + ' ' + str(mean0)
        call_str = 'echo ' + S0 + ' >>' + OUT
        os.system(call_str)
        
    print('Done.')   

    sys.exit(1)
    
###############################################################

if __name__ == '__main__':
    main(sys.argv[:])
