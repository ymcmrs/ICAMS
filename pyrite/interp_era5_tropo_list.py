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
from pyrite import elevation_models

from pykrige import OrdinaryKriging

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
###############################################################
    
def cmdLineParse():
    parser = argparse.ArgumentParser(description='Generate high-resolution tropospheric product map for a list of SAR acquisitions',formatter_class=argparse.RawTextHelpFormatter,epilog=INTRODUCTION+'\n'+EXAMPLE)
    parser.add_argument('date_list', help='SAR acquisition date.')
    parser.add_argument('era5_file',help='input file name (e.g., era5_sample_variogramModel.h5).')
    parser.add_argument('geo_file',help='input geometry file name (e.g., geometryRadar.h5).')
    parser.add_argument('--type', dest='type', choices = {'tzd','wzd'}, default = 'tzd',help = 'type of the high-resolution tropospheric map.[default: tzd]')
    parser.add_argument('--method', dest='method', choices = {'kriging','weight_distance'},default = 'kriging',help = 'method used to interp the high-resolution map. [default: kriging]')
    parser.add_argument('--parallel', dest='parallelNumb', type=int, default=1, help='Enable parallel processing and Specify the number of processors.')
    parser.add_argument('--kriging-points-numb', dest='kriging_points_numb', type=int, default=20, help='Number of the closest points used for Kriging interpolation. [default: 20]')
       
    inps = parser.parse_args()

    return inps


INTRODUCTION = '''
##################################################################################
   Copy Right(c): 2019, Yunmeng Cao   @GigPy v1.0
   
   Generate high-resolution GPS-based tropospheric maps (delays & water vapor) for InSAR Geodesy & meteorology.
'''

EXAMPLE = """Example:
  
  interp_era5_tropo_list.py date_list era5_file geometryRadar.h5 --type tzd
  interp_era5_tropo_list.py date_list era5_file geometryRadar.h5 --type wzd --parallel 8
  interp_era5_tropo_list.py date_list era5_file geometryRadar.h5 --method kriging
  interp_era5_tropo_list.py date_list era5_file geometryRadar.h5 --method kriging --kriging-points-numb 15
  interp_era5_tropo_list.py date_list era5_file geometryRadar.h5 --method weight_distance
  interp_era5_tropo_list.py date_list era5_file geometryRadar.h5 --type tzd 
  interp_era5_tropo_list.py date_list era5_file geometryRadar.h5 --type wzd --parallel 4
  
###################################################################################
"""

###############################################################

def main(argv):
    
    inps = cmdLineParse()
    date_list_txt = inps.date_list
    gps_file = inps.era5_file
    geo_file = inps.geo_file
      
    date_list = np.loadtxt(date_list_txt,dtype=np.str)
    date_list = date_list.tolist()
    N=len(date_list)

    for i in range(N):
        out0 = date_list[i] + '_' + inps.type + '.h5'
        if not os.path.isfile(out0):
            #print('-------------------------------------------------------------')
            #print('Start to interpolate high-resolution map for date: %s' % date_list[i])
            call_str = 'interp_era5_tropo.py ' + date_list[i] + ' era5_sample_variogramModel.h5 ' + inps.geo_file + ' --type ' + inps.type + ' --method ' + inps.method + '  --kriging-points-numb ' + str(inps.kriging_points_numb) + ' --parallel ' + str(inps.parallelNumb)
            os.system(call_str)
    
    sys.exit(1)
    
###############################################################

if __name__ == '__main__':
    main(sys.argv[:])