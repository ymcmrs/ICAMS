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

from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator as RGI
###############################################################
def write_h5(datasetDict, out_file, metadata=None, ref_file=None, compression=None):

    if os.path.isfile(out_file):
        #print('delete exsited file: {}'.format(out_file))
        os.remove(out_file)

    #print('create HDF5 file: {} with w mode'.format(out_file))
    dt = h5py.special_dtype(vlen=np.dtype('float64'))

    
    with h5py.File(out_file, 'w') as f:
        for dsName in datasetDict.keys():
            data = datasetDict[dsName]
            ds = f.create_dataset(dsName,
                              data=data,
                              compression=compression)
        
        for key, value in metadata.items():
            f.attrs[key] = str(value)
            #print(key + ': ' +  value)
    #print('finished writing to {}'.format(out_file))
        
    return out_file 

def read_gacos_tzd(gacos_tzd, gacos_rsc):
    rscDict = read_rsc(gacos_rsc,delimiter=' ', standardize=None)
    WIDTH = int(rscDict['WIDTH'])
    LENGTH = int(rscDict['FILE_LENGTH'])
    
    X_STEP = float(rscDict['X_STEP'])
    Y_STEP = float(rscDict['Y_STEP'])
    
    X_FIRST = float(rscDict['X_FIRST'])
    Y_FIRST = float(rscDict['Y_FIRST'])
    
    xv = np.arange(0,WIDTH)
    yv = np.arange(0,LENGTH)
    
    xvv, yvv = np.meshgrid(xv, yv)
    
    lonvv = X_FIRST + xvv*X_STEP
    latvv = Y_FIRST + yvv*Y_STEP
    
    lonvv = lonvv.astype(np.float32)
    latvv = latvv.astype(np.float32)
    
    
    data = np.fromfile(gacos_tzd, dtype='float32', count=WIDTH*LENGTH).reshape((LENGTH, WIDTH))
    
    return data, lonvv, latvv
    

def read_rsc(fname,delimiter=' ', standardize=None):
    """Read ROI_PAC style RSC file.
    Parameters: fname : str.
                    File path of .rsc file.
    Returns:    rscDict : dict
                    Dictionary of keys and values in RSC file.
    Examples:
        from mintpy.utils import readfile
        atr = readfile.read_roipac_rsc('filt_101120_110220_c10.unw.rsc')
    """
    # read .rsc file
    with open(fname, 'r') as f:
        lines = f.readlines()

    # convert list of str into dict
    rscDict = {}
    for line in lines:
        c = [i.strip() for i in line.strip().replace('\t',' ').split(delimiter, 1)]
        key = c[0]
        value = c[1].replace('\n', '').strip()
        rscDict[key] = value

    if standardize:
        rscDict = standardize_metadata(rscDict)
    return rscDict


def cmdLineParse():
    parser = argparse.ArgumentParser(description='CDonvert gacos delay to sar coord.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('gacos_tzd', help='gacos tzd data.')
    parser.add_argument('gacos_tzd_rsc', help='gacos tzd rsc file.')
    parser.add_argument('geo_file',help='input geometry file name (e.g., geometryRadar.h5).')
    
    parser.add_argument('-o','--out', dest='out', metavar='FILE',help='name of the prefix of the output file')
    inps = parser.parse_args()

    return inps


INTRODUCTION = '''
##################################################################################
   Copy Right(c): 2019, Yunmeng Cao   @PyRite v1.0
   
   Resample GACOS products to an interested coord (e.g., SAR-coord).
'''

EXAMPLE = """Example:
  
  gacos2sar.py 20190101.tzd 20190101.tzd.rsc geometryRadar.h5 
  gacos2sar.py 20190101.tzd 20190101.tzd.rsc geometryRadar.h5 -o 20190101_tzd.h5
  gacos2sar.py 20190101.tzd 20190101.tzd.rsc geometryGeo.h5 -o 20190101_tzd.h5
  
###################################################################################
"""

###############################################################

def main(argv):
    
    inps = cmdLineParse() 
    gacos_file = inps.gacos_tzd
    rsc_file = inps.gacos_tzd_rsc
    geo_file = inps.geo_file
    
    gacos_tzd = gacos_file
    gacos_rsc = rsc_file
   
    root_path = os.getcwd()
    gacos_folder = root_path + '/gacos'
    gacos_sar_folder = root_path + '/gacos/sar'
    if not os.path.isdir(gacos_folder):
        os.mkdir(gacos_folder)
    if not os.path.isdir(gacos_sar_folder):
        os.mkdir(gacos_sar_folder)    
        
    
    if inps.out: OUT  = gacos_sar_folder + '/' + inps.out
    else: OUT = gacos_sar_folder + '/' + os.path.basename(inps.gacos_tzd).split('.')[0] + '_tzd.h5'
    
    data, lonvv, latvv = read_gacos_tzd(gacos_tzd, gacos_rsc)
        
    datasetNames = ut.get_dataNames(inps.geo_file)
    meta = ut.read_attr(geo_file)
    if 'latitude' in datasetNames: 
        lats = ut.read_hdf5(geo_file,datasetName='latitude')[0]
        lons = ut.read_hdf5(geo_file,datasetName='longitude')[0]
    else:
        lats,lons = ut.get_lat_lon(meta)     
    #print(data.shape)
    
    lonk = lonvv.flatten()
    latk = latvv.flatten()
    datak = data.flatten()

    #lonks = lonk[data~=0]
    #latks = latk[data~=0]
    #datas = data[data~=0]
    
    #points = np.zeros((len(lonks),2))
    #poins[:,0] = lonks
    #points[:,1] = latks

    data[data==0] = np.nan
    latv = latvv[:,0]
    lonv = lonvv[0,:]

    rgi_func = RGI((latv[::-1], lonv), data[::-1,:])
    #print(rgi_func((34.5,-130)))
    
    #pts = np.zeros((len(lats.flatten()),2), dtype = np.float32)
    pts = (lats.flatten(), lons.flatten())
    #pts[:,0] = lats.flatten()
    #pts[:,1] = lons.flatten()
    
    lats0 = lats.flatten()
    lons0 = lons.flatten()
    
    latss = lats0[~np.isnan(lats0)]
    lonss = lons0[~np.isnan(lats0)]
    data_sar = rgi_func(np.vstack((latss, lonss)).T)
    
    data_sar0 = np.zeros((lats.shape),dtype = np.float32)
    data_sar0 = data_sar0.flatten()
    data_sar0[~np.isnan(lats0)] = data_sar
    
    data_sar0 = data_sar0.reshape(lats.shape)
    
    datasetDict = dict()
    datasetDict['aps_sar'] = data_sar0.astype(np.float32)
  
    meta['UNIT'] = 'm'   
    write_h5(datasetDict, OUT, metadata=meta, ref_file=None, compression=None)
    
    sys.exit(1)
###############################################################

if __name__ == '__main__':
    main(sys.argv[:])
