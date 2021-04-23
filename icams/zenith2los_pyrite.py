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

def get_dataNames(FILE):
    with h5py.File(FILE, 'r') as f:
        dataNames = []
        for k0 in f.keys():
            dataNames.append(k0)
    return dataNames
    
def cmdLineParse():
    parser = argparse.ArgumentParser(description='Transfer the products from zenith direction to the los direction, or inverse',formatter_class=argparse.RawTextHelpFormatter,epilog=INTRODUCTION+'\n'+EXAMPLE)
    parser.add_argument('data_file', help='the input data file.')
    parser.add_argument('geo_file',help='the geometry file with incidence angle')
    parser.add_argument('--type', dest='type', choices = {'turb','aps','hgt','trend','sigma'}, default = 'aps',help = 'type of the high-resolution tropospheric map.[default: aps]')
    parser.add_argument('--inverse', dest='inverse', action='store_true',help='transform from los to zenith')
    parser.add_argument('-o','--out', dest='out',help = 'Name of the output file.')
      
    inps = parser.parse_args()

    return inps


INTRODUCTION = '''
##################################################################################
   Copy Right(c): 2019, Yunmeng Cao   @PyRite v2.0
   
   Transfer the measurements from zenith direction to the LOS direction, or inverse.
'''

EXAMPLE = """Example:
  
  zenith2los_pyrite.py 20190101_aps.h5 geometryRadar.h5 --data aps_sar -o 20190120_aps_los.h5
  zenith2los_pyrite.py timeseries_pyrite_tzd.h5 geometryRadar.h5 -o  timeseries_pyrite_tzd_los.h5
  zenith2los_pyrite.py timeseries.h5 geometryRadar.h5 --inverse -o  timeseries_zenith.h5
  
###################################################################################
"""

def main(argv):
    
    inps = cmdLineParse()
    data_file = inps.data_file
    geo_file = inps.geo_file
    meta = read_attr(data_file)
    datasetNames = get_dataNames(data_file)
    datasetDict = dict()
    
    INC = read_hdf5(geo_file, datasetName = 'incidenceAngle')[0]
    INC = INC/180*np.pi # degree to radian
    if inps.inverse: 
        ff = 1/(np.cos(INC))
        S0 = 'zenith'
    else: 
        ff = np.cos(INC)
        S0 = 'los'
    
    print(meta['FILE_TYPE'])
    if meta['FILE_TYPE'] == 'timeseries':
        data_type = 'timeseries'
        data0 = read_hdf5(data_file, datasetName = 'timeseries')[0]
        data = data0
            
        for i in range(data0.shape[0]):
            data[i, :, :] = data0[i, :, :] / ff
            
        for datasetName in datasetNames:
            datasetDict[datasetName] = read_hdf5(data_file, datasetName = datasetName)[0]
        
        datasetDict['timeseries'] = data
            
    else:
        
        data_type = inps.type + '_sar'
        data0 = read_hdf5(data_file, datasetName = data_type)[0]

        data = data0 * ff
        datasetDict[data_type] = data
        
    if inps.out: OUT = inps.out
    else: OUT = (os.path.basename(data_file)).split('.')[0] + '_' + S0 + '.h5'
        
    write_h5(datasetDict, OUT, metadata=meta, ref_file=None, compression=None)
    print('Done.')   

    sys.exit(1)
    
###############################################################

if __name__ == '__main__':
    main(sys.argv[:])