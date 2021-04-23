#! /usr/bin/env python
#################################################################
###  This program is part of PyRite  v 1.0                    ### 
###  Copy Right (c): 2019, Yunmeng Cao                        ###  
###  Author: Yunmeng Cao                                      ###                                                          
###  Contact : ymcmrs@gmail.com                               ###  
#################################################################
from numpy import *
import sys
import os
import re
import subprocess
import argparse
import numpy as np
import h5py
import glob

from scipy.interpolate import griddata
import scipy.interpolate as intp
from scipy.optimize import leastsq
from scipy.stats.stats import pearsonr

from pyrite import elevation_models
from pyrite import _utils as ut
from pykrige import OrdinaryKriging
from pykrige import variogram_models


from pyrite import _utils as ut

def get_fname_list(date_list,area,hr):
    
    flist = []
    for k0 in date_list:
        f0 = 'ERA-5{}_{}_{}.grb'.format(area, k0, hr)
        flist.append(f0)
        
    return flist

def get_meta_corner(meta):
    if 'Y_FIRST' in meta.keys():
        lat0 = float(meta['Y_FIRST'])
        lon0 = float(meta['X_FIRST'])
        lat_step = float(meta['Y_STEP'])
        lon_step = float(meta['X_STEP'])
        lat1 = lat0 + lat_step * (length - 1)
        lon1 = lon0 + lon_step * (width - 1)
        
        NORTH = lat0
        SOUTH = lat1  
        WEST = lon0
        EAST = lon1    
        
    else:       
        lats = [float(meta['LAT_REF{}'.format(i)]) for i in [1,2,3,4]]
        lons = [float(meta['LON_REF{}'.format(i)]) for i in [1,2,3,4]]
        
        lat0 = np.max(lats[:])
        lat1 = np.min(lats[:])
        lon0 = np.min(lons[:])
        lon1 = np.max(lons[:])
        
        NORTH = lat0 + 0.1
        SOUTH = lat1 + 0.1
        WEST = lon0 + 0.1
        EAST = lon1 + 0.1
    
    return WEST,SOUTH,EAST,NORTH


def era5_time(research_time0):
    
    research_time =  round(float(research_time0) / 3600)
    if len(str(research_time)) == 1:
        time0 = '0' + str(research_time)
    else:
        time0 = str(research_time)
    
    return time0

def ceil_to_5(x):
    """Return the closest number in multiple of 5 in the larger direction"""
    assert isinstance(x, (int, np.int16, np.int32, np.int64)), 'input number is not int: {}'.format(type(x))
    if x % 5 == 0:
        return x
    return x + (5 - x % 5)

def floor_to_5(x):
    """Return the closest number in multiple of 5 in the lesser direction"""
    assert isinstance(x, (int, np.int16, np.int32, np.int64)), 'input number is not int: {}'.format(type(x))
    return x - x % 5

def floor_to_1(x):
    """Return the closest number in multiple of 5 in the lesser direction"""
    assert isinstance(x, (int, np.int16, np.int32, np.int64)), 'input number is not int: {}'.format(type(x))
    return x - x % 1

def ceil_to_1(x):
    """Return the closest number in multiple of 5 in the larger direction"""
    assert isinstance(x, (int, np.int16, np.int32, np.int64)), 'input number is not int: {}'.format(type(x))
    if x % 1 == 0:
        return x
    return x + (1 - x % 1)

def floor_to_2(x):
    """Return the closest number in multiple of 5 in the lesser direction"""
    assert isinstance(x, (int, np.int16, np.int32, np.int64)), 'input number is not int: {}'.format(type(x))
    return x - x % 2

def ceil_to_2(x):
    """Return the closest number in multiple of 5 in the larger direction"""
    assert isinstance(x, (int, np.int16, np.int32, np.int64)), 'input number is not int: {}'.format(type(x))
    if x % 2 == 0:
        return x
    return x + (2 - x % 2)

def get_snwe(wsen, min_buffer=0.5, multi_1=True):
    # get bounding box
    lon0, lat0, lon1, lat1 = wsen
    # lat/lon0/1 --> SNWE
    S = np.floor(min(lat0, lat1) - min_buffer).astype(int)
    N = np.ceil( max(lat0, lat1) + min_buffer).astype(int)
    W = np.floor(min(lon0, lon1) - min_buffer).astype(int)
    E = np.ceil( max(lon0, lon1) + min_buffer).astype(int)

    # SNWE in multiple of 5
    if multi_1:
        S = floor_to_2(S)
        W = floor_to_2(W)
        N = ceil_to_2(N)
        E = ceil_to_2(E)
    return (S, N, W, E)


def get_fname_list(date_list,area,hr):
    
    flist = []
    for k0 in date_list:
        f0 = 'ERA-5{}_{}_{}.grb'.format(area, k0, hr)
        flist.append(f0)
        
    return flist
    
def snwe2str(snwe):
    """Get area extent in string"""
    if not snwe:
        return None

    area = ''
    s, n, w, e = snwe

    if s < 0:
        area += '_S{}'.format(abs(s))
    else:
        area += '_N{}'.format(abs(s))

    if n < 0:
        area += '_S{}'.format(abs(n))
    else:
        area += '_N{}'.format(abs(n))

    if w < 0:
        area += '_W{}'.format(abs(w))
    else:
        area += '_E{}'.format(abs(w))

    if e < 0:
        area += '_W{}'.format(abs(e))
    else:
        area += '_E{}'.format(abs(e))
    return area

def read_par_orb(slc_par):
    
    date0 = ut.read_gamma_par(slc_par,'read', 'date')
    date = date0.split(' ')[0] + date0.split(' ')[1] + date0.split(' ')[2]
    
    first_sar = ut.read_gamma_par(slc_par,'read', 'start_time')
    first_sar = str(float(first_sar.split('s')[0]))

    first_state = ut.read_gamma_par(slc_par,'read', 'time_of_first_state_vector')
    first_state = str(float(first_state.split('s')[0]))
    
    intv_state = ut.read_gamma_par(slc_par,'read', 'state_vector_interval')
    intv_state = str(float(intv_state.split('s')[0]))
    
    out0 = date + '_orbit0'
    out = date + '_orbit'
    
    if os.path.isfile(out0): 
        os.remove(out0)
    if os.path.isfile(out): 
        os.remove(out)
    
    with open(slc_par) as f:
        Lines = f.readlines()
    
    with open(out0, 'a') as fo:
        k0 = 0
        for Line in Lines:
            if 'state_vector_position' in Line:
                kk = float(first_state) + float(intv_state)*k0
                fo.write(str(kk - float(first_sar)) +' ' + Line)
                k0 = k0+1
                
    call_str = "awk '{print $1,$3,$4,$5}' " + out0 + " >" + out
    os.system(call_str)
    
    orb_data = ut.read_txt2array(out)
    #t_orb = Orb[:,0]
    #X_Orb = Orb[:,1]
    #Y_Orb = Orb[:,2]
    #Z_Orb = Orb[:,3]
    return orb_data


def get_delay_compare(date, geo_file, slc_par):

    datasetNames = ut.get_dataNames(geo_file)
    meta = ut.read_attr(geo_file) 
    if 'latitude' in datasetNames: 
        lats = ut.read_hdf5(geo_file,datasetName='latitude')[0]
        lons = ut.read_hdf5(geo_file,datasetName='longitude')[0]
    else:
        lats,lons = ut.get_lat_lon(meta)  
    heis = ut.read_hdf5(geo_file,datasetName='height')[0]    
    
    attr = ut.read_attr(geo_file)
    orb_data = read_par_orb(slc_par)
    #date0 = ut.read_gamma_par(slc_par,'read', 'date')
    date0 = date
    #date = str(int(date0.split(' ')[0])) + str(int(date0.split(' ')[1])) + str(int(date0.split(' ')[2]))
      
    start_sar = ut.read_gamma_par(slc_par,'read', 'start_time')
    earth_R = ut.read_gamma_par(slc_par,'read', 'earth_radius_below_sensor')
    end_sar = ut.read_gamma_par(slc_par,'read', 'end_time')
    cent_time = ut.read_gamma_par(slc_par,'read', 'center_time')
    
    attr['EARTH_RADIUS'] = str(float(earth_R.split('m')[0]))
    attr['END_TIME'] = str(float(end_sar.split('s')[0]))
    attr['START_TIME'] = str(float(start_sar.split('s')[0]))
    attr['DATE'] = date
    attr['CENTER_TIME'] = str(float(cent_time.split('s')[0]))

    root_path = os.getcwd()
    pyrite_dir = root_path + '/pyrite'
    era5_dir = pyrite_dir + '/ERA5'
    era5_raw_dir = era5_dir  + '/raw'
    era5_sar_dir =  era5_dir  + '/sar'

    if not os.path.isdir(pyrite_dir): os.mkdir(pyrite_dir)
    if not os.path.isdir(era5_dir): os.mkdir(era5_dir)
    if not os.path.isdir(era5_raw_dir): os.mkdir(era5_raw_dir)
    if not os.path.isdir(era5_sar_dir): os.mkdir(era5_sar_dir)
    
    cdic = ut.initconst()
    fname_list = glob.glob(era5_raw_dir + '/ERA*' + date0 + '*')    
    w,s,e,n = get_meta_corner(attr)
    wsen = (w,s,e,n)    
    snwe = get_snwe(wsen, min_buffer=0.5, multi_1=True)
    area = snwe2str(snwe)
    hour = era5_time(attr['CENTER_TIME'])

    fname = get_fname_list([date],area,hour)
    if len(fname_list) < 1:
        fname0 = os.path.join(era5_raw_dir, fname[0])
        ut.ECMWFdload([date],hour,era5_raw_dir,model='ERA5',snwe=snwe,flist=[fname0])
    
    ERA5_file = era5_raw_dir + '/' + fname[0]
    ##### step2 : read ERA5 data
    lvls,latlist,lonlist,gph,tmp,vpr = ut.get_ecmwf('ERA5',ERA5_file,cdic, humidity='Q')
    mean_lon = np.mean(lonlist.flatten())
    if mean_lon > 180:    
        lonlist = lonlist - 360.0 # change to insar format lon [-180, 180]
   
    lats0 = lats.flatten()
    lons0 = lons.flatten()
    mean_geoid = ut.get_geoid_point(np.nanmean(lats0),np.nanmean(lons0))
    #mean_geoid = 0
    print('Average geoid heigh: ' + str(mean_geoid))
    heis = heis + mean_geoid # geoid correct
    heis0 = heis.flatten()
    
    maxdem = max(heis0)
    mindem = min(heis0)
    if mindem < -200:
        mindem = -100.0
    
    hh = heis0[~np.isnan(lats0)]
    lala = lats0[~np.isnan(lats0)]
    lolo = lons0[~np.isnan(lats0)]
    
    
    sar_wet = np.zeros((heis.shape),dtype = np.float32)   
    sar_wet0 = sar_wet.flatten()
    
    sar_dry = np.zeros((heis.shape),dtype = np.float32)   
    sar_dry0 = sar_dry.flatten()
    
    hgtlvs = [ -200, 0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400
              ,2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000
              ,5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 11000, 12000, 13000
              ,14000, 15000, 16000, 17000, 18000, 19000, 20000, 25000, 30000, 35000, 40000]
    hgtlvs = np.asarray(hgtlvs)

    Presi,Tempi,Vpri = ut.intP2H(lvls,hgtlvs,gph,tmp,vpr,cdic,verbose=False)
    
    ## Method 1:  PyAPS
    delay_h5 = era5_sar_dir + '/' + date + '_zenith_linear.h5'
    if os.path.isfile(delay_h5):
        delay_zenith_linear = ut.read_hdf5(delay_h5,datasetName='tot_delay')[0]
    else:
        ddry_zenith,dwet_zenith = ut.PTV2del(Presi,Tempi,Vpri,hgtlvs,cdic)
        dwet_intp,ddry_intp, lonvv, latvv, hgtuse = ut.pyrite_griddata_zenith(lonlist,latlist,hgtlvs,ddry_zenith,dwet_zenith, attr, maxdem, mindem, 10,'linear',10)
        
        latv = latvv[:,0]
        lonv = lonvv[0,:]
        
        delay_tot = dwet_intp + ddry_intp
        linearint_dry = intp.RegularGridInterpolator((latv[::-1], lonv, hgtuse), ddry_intp[::-1,:,:], method='linear', bounds_error=False, fill_value = 0.0)
        linearint_wet = intp.RegularGridInterpolator((latv[::-1], lonv, hgtuse), dwet_intp[::-1,:,:], method='linear', bounds_error=False, fill_value = 0.0)
        
        sar_wet00  = linearint_wet(np.vstack((lala, lolo, hh)).T)
        sar_wet0[~np.isnan(lats0)] = sar_wet00
        sar_wet0 = sar_wet0.reshape(lats.shape)

        sar_dry00  = linearint_dry(np.vstack((lala, lolo, hh)).T)
        sar_dry0[~np.isnan(lats0)] = sar_dry00
        sar_dry0 = sar_dry0.reshape(lats.shape)
        
        inc = ut.read_hdf5(geo_file,datasetName='incidenceAngle')[0]
        
        sar_wet0 = sar_wet0/np.cos(inc/180*np.pi)
        sar_dry0 = sar_dry0/np.cos(inc/180*np.pi)
    
        delay_zenith_linear = sar_wet0 + sar_dry0
        
        datasetDict = dict()
        datasetDict['tot_delay'] = delay_zenith_linear
        datasetDict['wet_delay'] = sar_wet0
        datasetDict['dry_delay'] = sar_dry0
    
        ut.write_h5(datasetDict, delay_h5, metadata= attr, ref_file=None, compression=None)
    
    ## Method 2: D-LOS
    delay_h5 = era5_sar_dir + '/' + date + '_los_linear.h5'
    if os.path.isfile(delay_h5):
        delay_los_linear = ut.read_hdf5(delay_h5,datasetName='tot_delay')[0]
    else:
        sar_wet = np.zeros((heis.shape),dtype = np.float32)   
        sar_wet0 = sar_wet.flatten()
    
        sar_dry = np.zeros((heis.shape),dtype = np.float32)   
        sar_dry0 = sar_dry.flatten()
    
        lat_intp, lon_intp, los_intp = ut.get_LOS3D_coords(latlist,lonlist,hgtlvs, orb_data, attr)
        LosP,LosT,LosV = ut.get_LOS_parameters(latlist,lonlist,Presi,Tempi,Vpri,lat_intp, lon_intp,'linear',15)
        ddrylos,dwetlos = ut.losPTV2del(LosP,LosT,LosV,los_intp ,cdic,verbose=False)
        # interp grid delays
        dwet_intp,ddry_intp, lonvv, latvv, hgtuse = ut.pyrite_griddata_los(lon_intp,lat_intp,hgtlvs,ddrylos,dwetlos,attr, maxdem, mindem, 10,'linear',10)
        
        delay_tot = dwet_intp + ddry_intp

        latv = latvv[:,0]
        lonv = lonvv[0,:]
        linearint_dry = intp.RegularGridInterpolator((latv[::-1], lonv, hgtuse), ddry_intp[::-1,:,:], method='linear', bounds_error=False, fill_value = 0.0)
        linearint_wet = intp.RegularGridInterpolator((latv[::-1], lonv, hgtuse), dwet_intp[::-1,:,:], method='linear', bounds_error=False, fill_value = 0.0)
        
        sar_wet00  = linearint_wet(np.vstack((lala, lolo, hh)).T)
        sar_wet0[~np.isnan(lats0)] = sar_wet00
        sar_wet0 = sar_wet0.reshape(lats.shape)
        
        sar_dry00  = linearint_dry(np.vstack((lala, lolo, hh)).T)
        sar_dry0[~np.isnan(lats0)] = sar_dry00
        sar_dry0 = sar_dry0.reshape(lats.shape)

        delay_los_linear = sar_wet0 + sar_dry0
    
        datasetDict = dict()
        datasetDict['tot_delay'] = delay_los_linear
        datasetDict['wet_delay'] = sar_wet0
        datasetDict['dry_delay'] = sar_dry0
    
        ut.write_h5(datasetDict, delay_h5, metadata= attr, ref_file=None, compression=None)
    
    ## METHOD 3: LOS + Kriging
     # interp grid delays
    delay_h5 = era5_sar_dir + '/' + date + '_los_kriging.h5'
    if os.path.isfile(delay_h5):
        delay_los_kriging = ut.read_hdf5(delay_h5,datasetName='tot_delay')[0]
    else:    
        sar_wet = np.zeros((heis.shape),dtype = np.float32)   
        sar_wet0 = sar_wet.flatten()
    
        sar_dry = np.zeros((heis.shape),dtype = np.float32)   
        sar_dry0 = sar_dry.flatten()
    
        LosP,LosT,LosV = ut.get_LOS_parameters(latlist,lonlist,Presi,Tempi,Vpri,lat_intp, lon_intp,'kriging',15)
        ddrylos,dwetlos = ut.losPTV2del(LosP,LosT,LosV,los_intp ,cdic,verbose=False)
        dwet_intp,ddry_intp, lonvv, latvv, hgtuse = ut.pyrite_griddata_los(lon_intp,lat_intp,hgtlvs,ddrylos,dwetlos,attr, maxdem, mindem, 10,'kriging',15)
        
        delay_tot = dwet_intp + ddry_intp

        latv = latvv[:,0]
        lonv = lonvv[0,:]
        linearint_dry = intp.RegularGridInterpolator((latv[::-1], lonv, hgtuse), ddry_intp[::-1,:,:], method='linear', bounds_error=False, fill_value = 0.0)
        linearint_wet = intp.RegularGridInterpolator((latv[::-1], lonv, hgtuse), dwet_intp[::-1,:,:], method='linear', bounds_error=False, fill_value = 0.0)
        
        sar_wet00  = linearint_wet(np.vstack((lala, lolo, hh)).T)
        sar_wet0[~np.isnan(lats0)] = sar_wet00
        sar_wet0 = sar_wet0.reshape(lats.shape)
        
        sar_dry00  = linearint_dry(np.vstack((lala, lolo, hh)).T)
        sar_dry0[~np.isnan(lats0)] = sar_dry00
        sar_dry0 = sar_dry0.reshape(lats.shape)

        delay_los_kriging = sar_wet0 + sar_dry0
        
        datasetDict = dict()
        datasetDict['tot_delay'] = delay_los_kriging
        datasetDict['wet_delay'] = sar_wet0
        datasetDict['dry_delay'] = sar_dry0
    
        ut.write_h5(datasetDict, delay_h5, metadata= attr, ref_file=None, compression=None)
        
    return delay_zenith_linear, delay_los_linear, delay_los_kriging
    

INTRODUCTION = '''
-------------------------------------------------------------------  
       Correcting atmospheric delays for interferogram using PyRite.
   
'''

EXAMPLE = '''
    Usage: 
            tropo_pyrite_insar.py unwrapIfgram.h5 geometryRadar.h5 master.slc.par -n 1
            tropo_pyrite_insar.py unwrapIfgram.h5 geometryRadar.h5 master.slc.par --project los 
            tropo_pyrite_insar.py unwrapIfgram.h5 geometryRadar.h5 master.slc.par --method kriging
            tropo_pyrite_insar.py unwarpIfgram.h5 geometryRadar.h5 master.slc.par --lalo-rescale 10
            tropo_pyrite_insar.py unwarpIfgram.h5 geometryRadar.h5 master.slc.par --kriging-points-numb 20
            
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
    parser.add_argument('--incAngle', dest='incAngle', metavar='FILE',help='incidence angle file for projecting zenith to los, for case of PROJECT = ZENITH when geo_file does not include [incidenceAngle].')
    parser.add_argument('--lalo-rescale', dest='lalo_rescale', type=int, default=5, help='oversample rate of the lats and lons [default: 5]')
    parser.add_argument('--kriging-points-numb', dest='kriging_points_numb', type=int, default=15, help='Number of the closest points used for Kriging interpolation. [default: 15]')
    inps = parser.parse_args()
    return inps


def main(argv):
    
    inps = cmdLineParse()
    unwFile = inps.unwrapIfgram
    geo_file = inps.geo_file
    slc_par = inps.sar_par
    numb = inps.numb
    
    date_list,atr = ut.read_hdf5(unwFile,datasetName='date')
    ds,atr = ut.read_hdf5(unwFile,datasetName='unwrapPhase')
    cs,atr = ut.read_hdf5(unwFile,datasetName='coherence')
    insar_data0 = ds[numb,:,:]
    coh = cs[numb,:,:]
    insar_data = -insar_data0/(4*np.pi)*float(atr['WAVELENGTH'])
    
    
    MSdate = date_list[numb]
    Mdate = MSdate[0].astype(str)
    Sdate = MSdate[1].astype(str)
    
    print(Mdate)
    print(Sdate)
    pair = Mdate + '-' + Sdate
    
    Mdate_zenith_linear, Mdate_los_linear, Mdate_los_kriging = get_delay_compare(Mdate, geo_file, slc_par)
    Sdate_zenith_linear, Sdate_los_linear, Sdate_los_kriging = get_delay_compare(Sdate, geo_file, slc_par)
    
    
    insar_data = insar_data - insar_data[int(atr['REF_Y']),int(atr['REF_X'])]
    
    aps_zenit_linear = Mdate_zenith_linear - Sdate_zenith_linear
    aps_los_linear = Mdate_los_linear - Sdate_los_linear
    aps_los_kriging = Mdate_los_kriging - Sdate_los_kriging
    
    aps_zenit_linear = aps_zenit_linear - aps_zenit_linear[int(atr['REF_Y']),int(atr['REF_X'])]
    aps_los_linear = aps_los_linear - aps_los_linear[int(atr['REF_Y']),int(atr['REF_X'])]
    aps_los_kriging = aps_los_kriging - aps_los_kriging[int(atr['REF_Y']),int(atr['REF_X'])]
    
    cor_zenith_linear = insar_data - aps_zenit_linear
    cor_los_linear = insar_data - aps_los_linear
    cor_los_kriging = insar_data - aps_los_kriging
    diff_los_zenith = cor_zenith_linear - cor_los_linear
    
    datasetDict = dict()
    datasetDict['insar'] = insar_data
    datasetDict['coherence'] = coh 
    datasetDict['zenith_linear_cor'] = cor_zenith_linear
    datasetDict['los_linear_cor'] = cor_los_linear
    datasetDict['los_kriging_cor'] = cor_los_kriging
    datasetDict['diff_los_zenith'] = diff_los_zenith
    atr['UNIT'] = 'm'
    OUT = pair + '_pyrite_compare.h5'
    ut.write_h5(datasetDict, OUT, metadata= atr, ref_file=None, compression=None)

    print("Correcting atmospheric delays for interferogram %s is done." %pair)
    sys.exit(1)

if __name__ == '__main__':
    main(sys.argv[:])
