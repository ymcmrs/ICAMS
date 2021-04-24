#! /usr/bin/env python
#################################################################
###  This program is part of icams   v 1.0                    ### 
###  Copy Right (c): 2019, Yunmeng Cao                        ###  
###  Author: Yunmeng Cao                                      ###                                                          
###  Contact : ymcmrs@gmail.com                               ###  
#################################################################

import numpy as np
import os
import sys  
import argparse

from pyint import _utils as ut


INTRODUCTION = '''
-------------------------------------------------------------------  
       Correcting atmospheric delays for interferogram using icams.
   
'''

EXAMPLE = '''
    Usage: 
            tropo_icams_insar.py unwrapIfgram.h5 geometryRadar.h5 master.slc.par -n 1
            tropo_icams_insar.py unwrapIfgram.h5 geometryRadar.h5 master.slc.par --project los 
            tropo_icams_insar.py unwrapIfgram.h5 geometryRadar.h5 master.slc.par --method sklm
            tropo_icams_insar.py unwarpIfgram.h5 geometryRadar.h5 master.slc.par --lalo-rescale 10
            tropo_icams_insar.py unwarpIfgram.h5 geometryRadar.h5 master.slc.par --sklm-points-numb 20
            
-------------------------------------------------------------------  
'''

def cmdLineParse():
    parser = argparse.ArgumentParser(description='Loading the unwrapIfg and the coherence into a h5 file.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('unwrapIfgram',help='MintPy unwrapIfgram h5 file.')
    parser.add_argument('geo_file',help='input geometry file name (e.g., geometryRadar.h5).')
    parser.add_argument('sar_par', help='SLC_par file for providing orbit state paramters.')  
    parser.add_argument('--numb',dest='numb',type=int, default=0, help='number of the interferogram.')
    parser.add_argument('--all', dest='all', action='store_true', help='Correction for all of the interferogram.')
    parser.add_argument('--method', dest='method', choices = {'sklm','linear','cubic'},default = 'sklm',help = 'method used to interp the high-resolution map. [default: sklm]')
    parser.add_argument('--project', dest='project', choices = {'zenith','los'},default = 'los',help = 'project method for calculating the accumulated delays. [default: los]')
    parser.add_argument('--incAngle', dest='incAngle', metavar='FILE',help='incidence angle file for projecting zenith to los, for case of PROJECT = ZENITH when geo_file does not include [incidenceAngle].')
    parser.add_argument('--lalo-rescale', dest='lalo_rescale', type=int, default=5, help='oversample rate of the lats and lons [default: 5]')
    parser.add_argument('--sklm-points-numb', dest='sklm_points_numb', type=int, default=15, help='Number of the closest points used for sklm interpolation. [default: 15]')
    inps = parser.parse_args()
    return inps


def main(argv):
    
    inps = cmdLineParse()
    unwFile = inps.unwrapIfgram
    numb = inps.numb
    
    date_list,atr = ut.read_hdf5(unwFile,datasetName='date')
    ds,atr = ut.read_hdf5(unwFile,datasetName='unwrapPhase')
    coh,atr = ut.read_hdf5(unwFile,datasetName='coherence')
    insar_data0 = ds[numb,:,:]
    coherence0 = coh[numb,:,:]
    insar_data = -insar_data0/(4*np.pi)*float(atr['WAVELENGTH'])
    
    
    MSdate = date_list[numb]
    Mdate = MSdate[0].astype(str)
    Sdate = MSdate[1].astype(str)
    
    pair = Mdate + '-' + Sdate
    root_path = os.getcwd()
    icams_dir = root_path + '/icams'
    era5_dir = icams_dir + '/ERA5'
    era5_raw_dir = era5_dir  + '/raw'
    era5_sar_dir =  era5_dir  + '/sar'

    if not os.path.isdir(icams_dir): os.mkdir(icams_dir)
    if not os.path.isdir(era5_dir): os.mkdir(era5_dir)
    if not os.path.isdir(era5_raw_dir): os.mkdir(era5_raw_dir)
    if not os.path.isdir(era5_sar_dir): os.mkdir(era5_sar_dir)
    
    Mdlos = era5_sar_dir + '/' + Mdate + '_' + inps.project + '_' + inps.method + '.h5'
    Sdlos = era5_sar_dir + '/' + Sdate + '_' + inps.project + '_' + inps.method + '.h5'
    
    OUT = Mdate + '-' + Sdate + '_' + inps.project + '_' + inps.method + '.h5'
    
    if not os.path.isfile(Mdlos):
        call_str = 'tropo_icams_sar.py ' + inps.geo_file + ' ' + inps.sar_par + ' --date ' + Mdate + ' --method ' + inps.method + ' --project ' + inps.project + ' --lalo-rescale ' + str(inps.lalo_rescale) + ' --sklm-points-numb ' + str(inps.sklm_points_numb)
        os.system(call_str)
    
    if not os.path.isfile(Sdlos):
        call_str = 'tropo_icams_sar.py ' + inps.geo_file + ' ' + inps.sar_par + ' --date ' + Sdate + ' --method ' + inps.method + ' --project ' + inps.project + ' --lalo-rescale ' + str(inps.lalo_rescale) + ' --sklm-points-numb ' + str(inps.sklm_points_numb)
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

    print("Correcting atmospheric delays for interferogram %s is done." %pair)
    sys.exit(1)

if __name__ == '__main__':
    main(sys.argv[:])
