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
from pyrite import _utils as ut0
from pyint import _utils as ut

from mintpy.utils import readfile

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

    parser.add_argument('gacos_folder', help='gacos tzd data folder.')
    parser.add_argument('projectName', help='Name of project.')
    parser.add_argument('Mdate', help='Mdate.')
    parser.add_argument('Sdate', help='Sdate.')
    #parser.add_argument('geo_file',help='input geometry file name (e.g., geometryRadar.h5).')
    
    parser.add_argument('-r','--reference',dest='ref_file',help='reference file.')
    parser.add_argument('--ref_x',dest='ref_x',help='reference x.')
    parser.add_argument('--ref_y',dest='ref_y',help='reference y.')
    
    parser.add_argument('-o','--out', dest='out', metavar='FILE',help='name of the prefix of the output file')
    inps = parser.parse_args()

    return inps


INTRODUCTION = '''
##################################################################################
   Copy Right(c): 2019, Yunmeng Cao   @PyRite v1.0
   
   Resample GACOS products to an interested coord (e.g., SAR-coord).
'''

EXAMPLE = """Example:
  
  gacos2insar.py gacos_folder projectName  Mdate Sdate -r ref_file 
  gacos2insar.py /Yunmeng/GacosChile ChileMS1A 20190101 20190203 -r velocity.h5 
  gacos2insar.py /Yunmeng/GacosChile ChileMS1A 20190101 20190203 -r temporalCoh.h5 -o 20190101-20190203_gacos.h5
    
###################################################################################
"""

###############################################################

def main(argv):
    
    inps = cmdLineParse() 
    raw_folder = inps.gacos_folder
    
    projectName = inps.projectName
    Mdate = inps.Mdate
    Sdate = inps.Sdate
    scratchDir0 = os.getenv('SCRATCHDIR0')
    scratchDir = os.getenv('SCRATCHDIR')
    templateDir = os.getenv('TEMPLATEDIR')
    templateFile = templateDir + "/" + projectName + ".template"
    templateDict=ut.update_template(templateFile)
    rlks = templateDict['range_looks']
    azlks = templateDict['azimuth_looks']
    masterDate = templateDict['masterDate']

    processDir = scratchDir + '/' + projectName + "/PROCESS"
    slcDir     = scratchDir + '/' + projectName + "/SLC"
    rslcDir    = scratchDir + '/' + projectName + '/RSLC'
    ifgDir     = scratchDir + '/' + projectName + '/ifgrams'

    #if not inps.slc_par:
    #    slc_par = slcDir  + '/' + masterDate + '/' + masterDate + '.slc.par'
    #else:
    #    slc_par = inps.slc_par

    Pair = Mdate + '-' + Sdate
    workDir = ifgDir + '/' + Pair

    unw_file = workDir + '/' + Pair + '_' + rlks + 'rlks.diff_filt.unw'
    cor_file = workDir + '/' + Pair + '_' + rlks + 'rlks.diff_filt.cor'
    insar_data0, atr = readfile.read(unw_file)
    coherence0, atr = readfile.read(cor_file)

    if inps.ref_x:
        ref_x = inps.ref_x
        ref_y = inps.ref_y
    elif inps.ref_file:
        atr0 = readfile.read_attribute(inps.ref_file)
        ref_x = atr0['REF_X']
        ref_y = atr0['REF_Y']

    atr['REF_X'] = ref_x
    atr['REF_Y'] = ref_y

    root_path = scratchDir0 + '/' + projectName + '/inputs'
    geo_file = root_path + '/geometryRadar.h5'
    INC,atr0 = ut0.read_hdf5(geo_file,datasetName='incidenceAngle')

    if inps.out: OUT = root_path + '/' + inps.out
    else: OUT = root_path + '/' + Mdate + '-' + Sdate + '_gacos.h5'

    sar_folder = root_path + '/gacos'
    Mgacos = sar_folder + '/' + Mdate + '_tzd.h5'
    Sgacos = sar_folder + '/' + Sdate + '_tzd.h5'
    
    Mraw = raw_folder + '/' + Mdate + '.ztd'
    Sraw = raw_folder + '/' + Sdate + '.ztd'
    
    Mraw_rsc = raw_folder + '/' + Mdate + '.ztd.rsc'
    Sraw_rsc = raw_folder + '/' + Sdate + '.ztd.rsc'

    if not os.path.isfile(Mgacos):
        call_str = 'gacos2sar.py ' + Mraw + ' ' + Mraw_rsc + ' ' + geo_file
        os.system(call_str)
    
    if not os.path.isfile(Sgacos):
        call_str = 'gacos2sar.py ' + Sraw + ' ' + Sraw_rsc + ' ' + geo_file
        os.system(call_str)
    
    insar_data = -insar_data0/(4*np.pi)*float(atr['WAVELENGTH'])
    
    Mzenith_data,atr0 = ut0.read_hdf5(Mgacos,datasetName='aps_sar')
    Szenith_data,atr0 = ut0.read_hdf5(Sgacos,datasetName='aps_sar')
    Mdlos_data = Mzenith_data / np.cos(INC/180*np.pi)
    Sdlos_data = Szenith_data / np.cos(INC/180*np.pi)

    insar_aps = Mdlos_data - Sdlos_data
    insar_aps = insar_aps - insar_aps[int(atr['REF_Y']),int(atr['REF_X'])]
    insar_data = insar_data - insar_data[int(atr['REF_Y']),int(atr['REF_X'])]
    insar_cor = insar_data -  insar_aps


    datasetDict = dict()
    datasetDict['insar_obs'] = insar_data
    datasetDict['insar_aps'] = insar_aps
    datasetDict['insar_cor'] = insar_cor
    datasetDict['coherence'] = coherence0
    atr['UNIT'] = 'm'
    ut.write_h5(datasetDict, OUT, metadata= atr, ref_file=None, compression=None)

    print("Correcting atmospheric delays for interferogram %s is done." %Pair)
    sys.exit(1)

###############################################################

if __name__ == '__main__':
    main(sys.argv[:])
