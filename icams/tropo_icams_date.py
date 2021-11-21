#! /usr/bin/env python
#################################################################
###  This program is part of icams  v1.0                     ### 
###  Copy Right (c): 2020, Yunmeng Cao                        ###  
###  Author: Yunmeng Cao                                      ###                                                          
###  Email : ymcmrs@gmail.com                                 ### 
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

from icams import elevation_models
from icams import _utils as ut
from pykrige import OrdinaryKriging
from pykrige import variogram_models
#import matlab.engine # using matlab to estimate the variogram parameters

###############################################################

model_dict = {'linear': elevation_models.linear_elevation_model,
                      'onn': elevation_models.onn_elevation_model,
                      'onn_linear': elevation_models.onn_linear_elevation_model,
                      'exp': elevation_models.exp_elevation_model,
                      'exp_linear': elevation_models.exp_linear_elevation_model}

residual_dict = {'linear': elevation_models.residuals_linear,
                      'onn': elevation_models.residuals_onn,
                      'onn_linear': elevation_models.residuals_onn_linear,
                      'exp': elevation_models.residuals_exp,
                      'exp_linear': elevation_models.residuals_exp_linear}

initial_dict = {'linear': elevation_models.initial_linear,
                      'onn': elevation_models.initial_onn,
                      'onn_linear': elevation_models.initial_onn_linear,
                      'exp': elevation_models.initial_exp,
                      'exp_linear': elevation_models.initial_exp_linear}

para_numb_dict = {'linear': 2,
                  'onn' : 3,
                  'onn_linear':4,
                  'exp':2,
                  'exp_linear':3}


variogram_dict = {'linear': variogram_models.linear_variogram_model,
                      'power': variogram_models.power_variogram_model,
                      'gaussian': variogram_models.gaussian_variogram_model,
                      'spherical': variogram_models.spherical_variogram_model,
                      'exponential': variogram_models.exponential_variogram_model,
                      'hole-effect': variogram_models.hole_effect_variogram_model}

def get_fname_list(date_list,area,hr):
    
    flist = []
    for k0 in date_list:
        f0 = 'ERA-5{}_{}_{}.grb'.format(area, k0, hr)
        flist.append(f0)
        
    return flist

def get_meta_corner(meta):
    if 'Y_FIRST' in meta.keys():
        length = int(meta['LENGTH'])
        width = int(meta['WIDTH'])
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

def read_region(STR):
    WEST = STR.split('/')[0]
    EAST = STR.split('/')[1].split('/')[0]
    
    SOUTH = STR.split(EAST+'/')[1].split('/')[0]
    NORTH = STR.split(EAST+'/')[1].split('/')[1]
    
    WEST =float(WEST)
    SOUTH=float(SOUTH)
    EAST=float(EAST)
    NORTH=float(NORTH)
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

def hour2seconds(hour0):
    
    hh = hour0.split(':')[0]
    mm = hour0.split(':')[1]
    
    seconds = float(hh)*3600 + float(mm)*60
    
    return seconds

def resamp_dem(dem_data,dem_attr,resolution,wsen):
    
    dr = float(resolution)/1000
    lat1 = float(dem_attr['Y_FIRST']); lon1 = float(dem_attr['X_FIRST'])
    lat2 = lat1 + 0.1; lon2 = lon1; dy = ut.haversine(lon1, lat1, lon2, lat2); y_step =  -dr/dy*0.1
    lat2 = lat1; lon2 = lon1+0.1; dx = ut.haversine(lon1, lat1, lon2, lat2); x_step =  dr/dx*0.1 
    
    w,s,e,n = wsen; lala = np.arange(n-y_step, s, y_step); lolo = np.arange(w+y_step, e, x_step)
    lons,lats = np.meshgrid(lolo,lala);  lats0,lons0 = ut.get_lat_lon(dem_attr); lons0 =lons0.flatten(); lats0 = lats0.flatten()
    points_all = np.zeros((len(lons0),2),dtype='float32'); points_all[:,0] = lons0; points_all[:,1] = lats0
    
    dems = griddata(points_all, dem_data.flatten(), (lons, lats), method='nearest')
    return dems, lats, lons, y_step, x_step
    
def cmdLineParse():
    parser = argparse.ArgumentParser(description='Generating high-resolution tropospheric delay map using icams.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    
    parser.add_argument('date', type= str, help = 'SAR acquisition time for generating the delay maps. e.g., 20201024')
    parser.add_argument('--region','-r',dest='region', help='research area in degree. w/e/n/s')
    parser.add_argument('--imaging-time','-t',dest='time', type = str, help='interested SAR-imaging UTC-time. e.g., 12:30')
    parser.add_argument('--reference-file', dest='reference_file', help = 'Reference file used to provide region/imaging-time information. e.g., velocity.h5 from MintPy') 
    parser.add_argument('--resolution',dest='resolution',type = float, default =100.0, help='resolution (m) of delay map.')
    parser.add_argument('--dem-tif', dest='dem_tif', help = 'DEM file (e.g., dem.tif). The coverage should be bigger or equal to the region of interest, if no DEM file provided, it will be downloaded automatically.') 
    parser.add_argument('-o','--out', dest='out_file', metavar='FILE',help='name of the prefix of the output file')
       
    inps = parser.parse_args()

    return inps


INTRODUCTION = '''
##################################################################################
   Copy Right(c): 2021, Yunmeng Cao   @icams v2.0
   
   Generate InSAR tropospheric delays using ERA5 based on icams.
'''

EXAMPLE = """Example:
  
  tropo_icams_date.py 20201024 --region " 172.8/173.8/-42.6/-41.6 " --imaging-time 11:24 --resolution 200.0 
  tropo_icams_date.py 20201024 --region " 172.8/173.8/-42.6/-41.6 " --imaging-time 13:06 --dem-tif /Download/dem.tif
  tropo_icams_date.py 20201024 --reference-file velocity.h5 --dem-tif /Download/dem.tif
  

###################################################################################
"""

###############################################################

def main(argv):
    
    inps = cmdLineParse()
    date = inps.date
    
    root_path = os.getcwd()
    icams_dir = root_path + '/icams'
    if not os.path.isdir(icams_dir): os.mkdir(icams_dir)
    era5_dir = icams_dir + '/ERA5'
    if not os.path.isdir(era5_dir): os.mkdir(era5_dir)
    era5_raw_dir = era5_dir  + '/raw'
    if not os.path.isdir(era5_raw_dir): os.mkdir(era5_raw_dir)
    era5_sar_dir =  era5_dir  + '/sar'
    if not os.path.isdir(era5_sar_dir): os.mkdir(era5_sar_dir)
    dem_dir = icams_dir + '/dem'
    
    if inps.reference_file:
        ref0 = inps.reference_file
        attr = ut.read_attr(ref0)
        w,s,e,n = get_meta_corner(attr)
        wsen = (w,s,e,n)   
        if 'CENTER_LINE_UTC' in attr:
            cent_time = attr['CENTER_LINE_UTC']
        else:
            cent_time = '43200'   # 12:00
    else:
        region0 = inps.region
        w,s,e,n = read_region(region0)
        wsen = (w,s,e,n)   
        cent_time = hour2seconds(inps.time)
    
    
    if inps.dem_tif:
        dem0 = inps.dem_tif
    else:
        if not os.path.isdir(dem_dir):
            os.mkdir(dem_dir)
        os.chdir(dem_dir)
        call_str = 'eio --product SRTM1 clip -o dem.tif --bounds ' + str(w - 0.1) + ' ' + str(s-0.1) + ' ' + str(e+0.1) + ' ' + str(n+0.1)
        print(call_str)
        os.system(call_str)
        dem0 = dem_dir + '/dem.tif'
    
    dem_data, dem_attr = ut.read_demtif(dem0)
    heis, lats, lons, lat_step, lon_step = resamp_dem(dem_data,dem_attr,inps.resolution,wsen)
    LENGTH,WIDTH = heis.shape
    
    cdic = ut.initconst()
    fname_list = glob.glob(era5_raw_dir + '/ERA*' + date + '*')    
    snwe = get_snwe(wsen, min_buffer=0.5, multi_1=True)
    area = snwe2str(snwe)
    hour = era5_time(cent_time)
    
    fname = get_fname_list([date],area,hour)
    print(fname)
    fname0 = os.path.join(era5_raw_dir, fname[0])
    ERA5_file = era5_raw_dir + '/' + fname[0]
    print(ERA5_file)
    cdic = ut.initconst()
    
    ##### step1 : check ERA5 file
    if not os.path.isfile(ERA5_file):
        fname = get_fname_list([date],area,hour)
        fname0 = os.path.join(era5_raw_dir, fname[0])
        ut.ECMWFdload([date],hour,era5_raw_dir,model='ERA5',snwe=snwe,flist=[fname0])
    
       
    ##### step2 : read ERA5 data
    lvls,latlist,lonlist,gph,tmp,vpr = ut.get_ecmwf('ERA5',ERA5_file,cdic, humidity='Q')
    mean_lon = np.mean(lonlist.flatten())
    if mean_lon > 180:    
        lonlist = lonlist - 360.0 # change to insar format lon [-180, 180]
  
    lats0 = lats.flatten(); Y_FIRST = np.max(lats0)
    lons0 = lons.flatten(); X_FIRST = np.min(lons0)
    mean_geoid = ut.get_geoid_point(np.nanmean(lats0),np.nanmean(lons0))
    print('Average geoid height: ' + str(mean_geoid))
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
    
    lat_intp_binary = era5_dir + '/lat_intp.npy'
    lon_intp_binary = era5_dir + '/lon_intp.npy'
    los_intp_binary = era5_dir + '/los_intp.npy'
    
    meta = dict()
    meta['EARTH_RADIUS'] = str(6371)
    meta['DATE'] = date
    meta['CENTER_TIME'] = cent_time
    meta['WIDTH'] = WIDTH
    meta['LENGTH'] = LENGTH
    meta['X_FIRST'] = X_FIRST
    meta['Y_FIRST'] = Y_FIRST
    meta['X_STEP'] = lon_step
    meta['Y_STEP'] = lat_step
    
    ddry_zenith,dwet_zenith = ut.PTV2del(Presi,Tempi,Vpri,hgtlvs,cdic)
    dwet_intp,ddry_intp, lonvv, latvv, hgtuse = ut.icams_griddata_zenith(lonlist,latlist,hgtlvs,ddry_zenith,dwet_zenith, meta, maxdem, mindem, 5,'sklm',20)
        
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
       
    sar_delay = sar_wet0 + sar_dry0
    
    OUT_dry = era5_sar_dir + '/' + date + '_dry.h5'
    OUT_wet = era5_sar_dir + '/' + date + '_wet.h5'
    OUT_tot = era5_sar_dir + '/' + date + '_tot.h5'
        
    datasetDict = dict()
    datasetDict['tot_delay'] = sar_delay    
    ut.write_h5(datasetDict, OUT_tot, metadata= meta, ref_file=None, compression=None)
    
    datasetDict = dict()
    datasetDict['dry_delay'] = sar_dry0  
    ut.write_h5(datasetDict, OUT_dry, metadata= meta, ref_file=None, compression=None)
    
    datasetDict = dict()
    datasetDict['wet_delay'] = sar_wet0   
    ut.write_h5(datasetDict, OUT_wet, metadata= meta, ref_file=None, compression=None)
    os.chdir(root_path)
    sys.exit(1)
###############################################################

if __name__ == '__main__':
    main(sys.argv[:])
