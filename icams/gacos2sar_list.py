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
from mintpy.utils import ptime

###############################################################



def cmdLineParse():
    parser = argparse.ArgumentParser(description='CDonvert gacos delay to sar coord.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('date_list_txt', help='date list text file.')
    parser.add_argument('geo_file',help='input geometry file name (e.g., geometryRadar.h5).')
    parser.add_argument('--gacos-dir',dest='gacos_dir', help='directory of the gacos data. [default: correct dir]')
    
    inps = parser.parse_args()

    return inps


INTRODUCTION = '''
##################################################################################
   Copy Right(c): 2019, Yunmeng Cao   @PyRite v1.0
   
   Resample GACOS products to an interested coord (e.g., SAR-coord).
'''

EXAMPLE = """Example:
  
  gacos2sar_list.py date_txt geometryRadar.h5 
  
###################################################################################
"""

###############################################################

def main(argv):
    
    inps = cmdLineParse() 
    date_list_txt = inps.date_list_txt
    date_list = np.loadtxt(date_list_txt,dtype=np.str)
    date_list = date_list.tolist()
    if inps.gacos_dir: gacos_dir = inps.gacos_dir
    else: gacos_dir = os.getcwd()
        
    prog_bar = ptime.progressBar(maxValue=len(date_list))
    for i in range(len(date_list)):
        k0 = date_list[i]
        tzd0 = gacos_dir + '/' + k0 + '.ztd'
        rsc0 = gacos_dir + '/' + k0 + '.ztd.rsc'
        
        call_str = 'gacos2sar.py ' + tzd0 + ' ' + rsc0 + ' ' + inps.geo_file + ' -o ' + gacos_dir + '/' + k0 + '_tzd.h5'
        os.system(call_str)
        prog_bar.update(i+1, every=1, suffix='{}/{} slices'.format(i+1, len(date_list)))
    prog_bar.close()   
    
    sys.exit(1)
###############################################################

if __name__ == '__main__':
    main(sys.argv[:])