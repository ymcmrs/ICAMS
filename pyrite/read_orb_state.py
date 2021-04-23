#! /usr/bin/env python
#################################################################
###  This program is part of PyINT  v2.1                      ### 
###  Copy Right (c): 2019, Yunmeng Cao                        ###  
###  Author: Yunmeng Cao                                      ###                                                          
###  Contact : ymcmrs@gmail.com                               ###
#################################################################
import numpy as np
import os
import sys  
import subprocess
import getopt
import time
import glob
import argparse

from pyint import _utils as ut


#########################################################################

INTRODUCTION = '''
#############################################################################
   Copy Right(c): 2019, Yunmeng Cao   @PyINT v2.0   
   
   Read orbit state parameters from parameter file.
     
'''

EXAMPLE = '''
    Usage:
            read_orb_state.py slc_par
 
    Examples:
            read_orb_state.py 20180101.slc.par

##############################################################################
'''


def cmdLineParse():
    parser = argparse.ArgumentParser(description='Generate SLC for ERS raw data with ENVISAT format.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('slc_par',help='slc parameter file.')
    parser.add_argument('-o',dest='out_put',help='Out put text file.')
    
    inps = parser.parse_args()
    return inps

################################################################################    


def main(argv):
      
    inps = cmdLineParse()
    slc_par = inps.slc_par
    
    if inps.out_put: out_put = inps.out_put
    else: out_put = 'orb_state.txt'
    
    if os.path.isfile(out_put):
        os.remove(out_put)
    
    with open(slc_par) as f:
        Lines = f.readlines()
    
    out0 = slc_par + '_out'
    if os.path.isfile(out0):
        os.remove(out0)
    
    first_state = ut.read_gamma_par(slc_par,'read', 'time_of_first_state_vector')
    first_state = str(float(first_state.split('s')[0]))
    first_sar = ut.read_gamma_par(slc_par,'read', 'start_time')
    first_sar = str(float(first_sar.split('s')[0]))
    
    earth_R = ut.read_gamma_par(slc_par,'read', 'earth_radius_below_sensor')
    earth_R = float(earth_R.split('m')[0])
    
    end_sar = ut.read_gamma_par(slc_par,'read', 'end_time')
    end_sar = str(float(end_sar.split('s')[0]))
    tot_sar = float(end_sar) - float(first_sar)
    
    intv_state = ut.read_gamma_par(slc_par,'read', 'state_vector_interval')
    intv_state = str(float(intv_state.split('s')[0]))
    
    with open(out0, 'a') as fo:
        k0 = 0
        for Line in Lines:
            if 'state_vector_position' in Line:
                
                kk = float(first_state) + float(intv_state)*k0 - float(first_sar)
                fo.write(str(float(first_state) + float(intv_state)*k0) +' ' + str(round(kk*10)/10) + ' ' + str(round(tot_sar*10)/10) + ' ' + str(round(earth_R*10)/10)  +' ' + Line)
                k0 = k0+1
                
    call_str = "awk '{print $1,$2,$3,$4,$6,$7,$8}' " + out0 + " >" + out_put
    os.system(call_str)
    
    if os.path.isfile(out0):
        os.remove(out0)
      
    sys.exit(1)
    
if __name__ == '__main__':
    main(sys.argv[:])

    
