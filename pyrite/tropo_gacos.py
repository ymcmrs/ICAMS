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
import argparse
import numpy as np
import h5py
import glob

from pyrite import _utils as ut
###############################################################
def generate_datelist_txt(date_list,txt_name):
    
    if os.path.isfile(txt_name): os.remove(txt_name)
    for list0 in date_list:
        call_str = 'echo ' + list0 + ' >>' + txt_name 
        os.system(call_str)
        
    return

def cmdLineParse():
    parser = argparse.ArgumentParser(description='Correcting InSAR troposphere using GACOS.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('ts_file', help='mintPy formatted timeseries file.')
    parser.add_argument('geo_file',help='input geometry file name (e.g., geometryRadar.h5).')
    parser.add_argument('--gacos-dir',dest='gacos_dir', help='directory of the gacos data. [default: currect dir]')
    inps = parser.parse_args()

    return inps


INTRODUCTION = '''
##################################################################################
   Copy Right(c): 2019, Yunmeng Cao   @PyRite v1.0
   
   Correcting InSAR troposphere using GACOS tropospheric products.
'''

EXAMPLE = """Example:
  
  tropo_gacos.py timeseries.h5 geometryRadar.h5 
  tropo_gacos.py timeseries.h5 geometryRadar.h5 --gacos-dir /Yunmeng/SCRATCH/losAngeles_gacos
  
###################################################################################
"""

###############################################################

def main(argv):
    
    inps = cmdLineParse() 
    ts_file = inps.ts_file
    geo_file = inps.geo_file
    if inps.gacos_dir: gacos_dir = inps.gacos_dir
    else: gacos_dir = os.getcwd()
    root_path = os.getcwd()
    
    # change dir
    #os.chdir(gacos_dir)
    date_list = ut.read_hdf5(ts_file,datasetName='date')[0]
    date_list = date_list.astype('U13')
    date_list = list(date_list)
    
    date_list_exist = [os.path.basename(x).split('_')[0] for x in glob.glob(gacos_dir + '/*_tzd.h5')]
    date_list_resamp = []
    for k0 in date_list:
        if not k0 in date_list_exist:
            date_list_resamp.append(k0)
    
    print('Start to resample GACOS data ...')
    print('Total number: %s' % str(len(date_list)))
    print('Existed number: %s' % str(len(date_list) - len(date_list_resamp)))
    print('To be resampled number: %s' % str(len(date_list_resamp)))
      
    txt_gacos = 'gacos.txt'
    generate_datelist_txt(date_list,txt_gacos)
    
    if len(date_list_resamp) > 0:
        txt_resamp = 'resamp.txt'
        generate_datelist_txt(date_list_resamp,txt_resamp)
    
        call_str = 'gacos2sar_list.py ' + txt_resamp + ' ' + geo_file + ' --gacos-dir ' + gacos_dir
        os.system(call_str)
    
    print('')
    print('Start to generate timeseries of GACOS data ...')
    call_str = 'generate_timeseries_era5.py --date-txt ' + txt_gacos + ' --atm-dir ' + gacos_dir + ' -o timeseries_gacos.h5'
    os.system(call_str)
    
    print('')
    print('Start to correct InSAR troposphere using GACOS products ...')
    call_str = 'diff_pyrite.py ' + ts_file + ' timeseries_gacos.h5 --add -o timeseries_gacosCor.h5'
    os.system(call_str)
    print('done')
    
    
    sys.exit(1)
###############################################################

if __name__ == '__main__':
    main(sys.argv[:])