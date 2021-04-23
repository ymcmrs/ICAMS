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


###############################################################
def read_txt_line(txt):
    with open(txt) as f:
        lines = f.readlines()
    f.close()
    list0 = []
    for ll0 in lines:    
        if len(ll0) > 0:
            list0.append(ll0)
    return list0
    
def cmdLineParse():
    parser = argparse.ArgumentParser(description='Generate high-resolution tropospheric product map for a list of SAR acquisitions',formatter_class=argparse.RawTextHelpFormatter,epilog=INTRODUCTION+'\n'+EXAMPLE)
    parser.add_argument('date_list_txt', help='SAR acquisition date list.')
    parser.add_argument('geo_file',help='input geometry file name (e.g., geometryRadar.h5).')
    parser.add_argument('--type', dest='type', choices = {'tzd','wzd'}, default = 'tzd',help = 'type of the high-resolution tropospheric map.[default: tzd]')
    parser.add_argument('--variogram-model', dest='variogram_model', 
                        choices = {'spherical','gaussian','exponential','linear'},
                        default = 'spherical',help = 'variogram model used for mdoeling. [default: spherical]')
    parser.add_argument('--elevation-model', dest='elevation_model', 
                        choices = {'linear','onn','exp','onn_linear','exp_linear'},
                        default = 'exp',help = 'elevation model used for model the elevation-dependent component. [default: exp]') 
    parser.add_argument('--method', dest='method', choices = {'kriging','weight_distance'},default = 'kriging',help = 'method used to interp the high-resolution map. [default: kriging]')
    parser.add_argument('--kriging-points-numb', dest='kriging_points_numb', type=int, default=20, help='Number of the closest points used for Kriging interpolation. [default: 20]')
    parser.add_argument('--max-length', dest='max_length', type=float, default=250.0, help='max length used for modeling the variogram model [default: 250]')
    parser.add_argument('--bin-numb', dest='bin_numb', type=int, default=50, help='bin number used for kriging variogram modeling [default: 50]')
    parser.add_argument('--hgt-rescale', dest='hgt_rescale', type=int, default=10, help='oversample rate of the height levels [default: 10]')
    parser.add_argument('--max-altitude', dest='max_altitude', type=float, default=5000.0, help='max altitude used for height levels [default: 5000]')
    parser.add_argument('--min-altitude', dest='min_altitude', type=float, default= -200.0, help='min altitude used for height levels [default: -200]')
    parser.add_argument('--lalo-rescale', dest='lalo_rescale', type=int, default=5, help='oversample rate of the lats and lons [default: 5]')
       
    inps = parser.parse_args()

    return inps


INTRODUCTION = '''
##################################################################################
   Copy Right(c): 2019, Yunmeng Cao   @GigPy v1.0
   
   Generate high-resolution GPS-based tropospheric maps (delays & water vapor) for InSAR Geodesy & meteorology.
'''

EXAMPLE = """Example:
  
  interp_era5_pyrite_list.py date_list geometryRadar.h5 --lalo-rescale 10 --elevation-model exp
  interp_era5_pyrite_list.py date_list geometryRadar.h5 --lalo-rescale 10 --elevation-model exp --variogram-model spherical
  interp_era5_pyrite_list.py date_list geometryRadar.h5 --type wzd 
  interp_era5_pyrite_list.py date_list geometryRadar.h5 --method kriging
  interp_era5_pyrite_list.py date_list geometryRadar.h5 --method kriging --kriging-points-numb 15
  interp_era5_pyrite_list.py date_list geometryRadar.h5 --method weight_distance
  interp_era5_pyrite_list.py date_list geometryRadar.h5 --type tzd 
  interp_era5_pyrite_list.py date_list geometryRadar.h5 --type wzd 
  
###################################################################################
"""

###############################################################

def main(argv):
    
    inps = cmdLineParse()
    date_list_txt = inps.date_list_txt
    geo_file = inps.geo_file
      
    date_list = np.loadtxt(date_list_txt,dtype=np.str)
    date_list = date_list.tolist()
    #date_list = read_txt_line(date_list_txt)
    N=len(date_list)

    for i in range(N):
        out0 = date_list[i] + '_' + inps.type + '.h5'
        if not os.path.isfile(out0):
            print('-------------------------------------------------------------')
            #print('{}/{}'.format(str(i+1),str(N)))
            print('Start to interpolate high-resolution map for date: {}  {}/{}'.format(date_list[i],str(i+1),str(N)))
            call_str = 'interp_era5_pyrite.py ' + date_list[i] + ' ' + inps.geo_file + ' --type ' + inps.type + ' --method ' + inps.method + '  --kriging-points-numb ' + str(inps.kriging_points_numb) + ' --variogram-model ' + str(inps.variogram_model) + ' --elevation-model ' + str(inps.elevation_model) + ' --max-length ' + str(inps.max_length) + ' --min-altitude ' + str(inps.min_altitude) + ' --max-altitude ' + str(inps.max_altitude) +  ' --bin-numb ' + str(inps.bin_numb) +  ' --hgt-rescale ' + str(inps.hgt_rescale) +  ' --lalo-rescale ' + str(inps.lalo_rescale)
            os.system(call_str)
    
    sys.exit(1)
    
###############################################################

if __name__ == '__main__':
    main(sys.argv[:])