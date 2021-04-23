#! /usr/bin/env python
#################################################################
###  This program is part of PyRite  v 1.0                    ### 
###  Copy Right (c): 2019, Yunmeng Cao                        ###  
###  Author: Yunmeng Cao                                      ###                                                          
###  Contact : ymcmrs@gmail.com                               ###  
#################################################################

import numpy as np
import os
import sys  
import argparse

from pyint import _utils as ut
from mintpy.utils import readfile

INTRODUCTION = '''
-------------------------------------------------------------------  
       Correcting atmospheric delays for interferogram using PyRite.
   
'''

EXAMPLE = '''
    Usage: 
            tropo_pyrite_pyint.py projectName Mdate Sdate -r velocity.h5
            tropo_pyrite_pyint.py projectName Mdate Sdate -r velocity.h5 --slc_par master.slc.par
            tropo_pyrite_pyint.py projectName Mdate Sdate -r velocity.h5 --project los  --method kriging
            tropo_pyrite_pyint.py projectName Mdate Sdate -r velocity.h5 --lalo-rescale 10
            tropo_pyrite_pyint.py projectName Mdate Sdate  --ref_x 100 --ref_y 100 --lalo-rescale 10 --kriging-points-numb 20
            
-------------------------------------------------------------------  
'''

def cmdLineParse():
    parser = argparse.ArgumentParser(description='Correct InSAR tropospheric delays using GAM for pyint products.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('projectName',help='Name of the project.')
    parser.add_argument('Mdate',help='M date of the interferogram.')
    parser.add_argument('Sdate', help='S date of the interferogram.')  
    parser.add_argument('--slc_par',dest='slc_par',help='slc par file.')
    parser.add_argument('-r','--reference',dest='ref_file',help='reference file.')
    parser.add_argument('--ref_x',dest='ref_x',help='reference x.')
    parser.add_argument('--ref_y',dest='ref_y',help='reference y.')

    parser.add_argument('--method', dest='method', choices = {'kriging','linear','cubic'},default = 'kriging',help = 'method used to interp the high-resolution map. [default: kriging]')
    parser.add_argument('--project', dest='project', choices = {'zenith','los'},default = 'los',help = 'project method for calculating the accumulated delays. [default: los]')
    parser.add_argument('--incAngle', dest='incAngle', metavar='FILE',help='incidence angle file for projecting zenith to los, for case of PROJECT = ZENITH when geo_file does not include [incidenceAngle].')
    parser.add_argument('--lalo-rescale', dest='lalo_rescale', type=int, default=5, help='oversample rate of the lats and lons [default: 5]')
    parser.add_argument('--kriging-points-numb', dest='kriging_points_numb', type=int, default=15, help='Number of the closest points used for Kriging interpolation. [default: 15]')
    inps = parser.parse_args()
    return inps


def main(argv):
    
    inps = cmdLineParse()
    
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
    
    if not inps.slc_par:
        slc_par = slcDir  + '/' + masterDate + '/' + masterDate + '.slc.par'
    else:
        slc_par = inps.slc_par
    
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
    
    path_dir = scratchDir0 + '/' + projectName + '/inputs'
    os.chdir(path_dir)
    
    insar_data = -insar_data0/(4*np.pi)*float(atr['WAVELENGTH'])
    
    pyrite_dir = path_dir + '/pyrite'
    era5_dir = pyrite_dir + '/ERA5'
    era5_raw_dir = era5_dir  + '/raw'
    era5_sar_dir =  era5_dir  + '/sar'
    
    geo_file = path_dir + '/geometryRadar.h5'
    
    if not os.path.isdir(pyrite_dir): os.mkdir(pyrite_dir)
    if not os.path.isdir(era5_dir): os.mkdir(era5_dir)
    if not os.path.isdir(era5_raw_dir): os.mkdir(era5_raw_dir)
    if not os.path.isdir(era5_sar_dir): os.mkdir(era5_sar_dir)
    
    Mdlos = era5_sar_dir + '/' + Mdate + '_' + inps.project + '_' + inps.method + '.h5'
    Sdlos = era5_sar_dir + '/' + Sdate + '_' + inps.project + '_' + inps.method + '.h5'
    
    OUT = Mdate + '-' + Sdate + '_' + inps.project + '_' + inps.method + '.h5'
    
    if not os.path.isfile(Mdlos):
        call_str = 'tropo_pyrite_sar.py ' + geo_file + ' ' + slc_par + ' --date ' + Mdate + ' --method ' + inps.method + ' --project ' + inps.project + ' --lalo-rescale ' + str(inps.lalo_rescale) + ' --kriging-points-numb ' + str(inps.kriging_points_numb)
        os.system(call_str)
    
    if not os.path.isfile(Sdlos):
        call_str = 'tropo_pyrite_sar.py ' + geo_file + ' ' + slc_par + ' --date ' + Sdate + ' --method ' + inps.method + ' --project ' + inps.project + ' --lalo-rescale ' + str(inps.lalo_rescale) + ' --kriging-points-numb ' + str(inps.kriging_points_numb)
        os.system(call_str)
        
    Mdlos_data,atr0 = ut.read_hdf5(Mdlos,datasetName='tot_delay')
    Sdlos_data,atr0 = ut.read_hdf5(Sdlos,datasetName='tot_delay')
    
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
if __name__ == '__main__':
    main(sys.argv[:])
