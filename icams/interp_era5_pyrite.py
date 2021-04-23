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
import glob

from scipy.interpolate import griddata
import scipy.interpolate as intp
from scipy.optimize import leastsq
from scipy.stats.stats import pearsonr

from pyrite import elevation_models
from pyrite import _utils as ut
from pykrige import OrdinaryKriging
from pykrige import variogram_models
#import matlab.engine # using matlab to estimate the variogram parameters
from mintpy.utils import ptime
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

def remove_ramp(lat,lon,data):
    # mod = a*x + b*y + c*x*y
    lat = lat/180*np.pi
    lon = lon/180*np.pi  
    lon = lon*np.cos(lat) # to get isometrics coordinates
    
    p0 = [0.0001,0.0001,0.0001,0.0000001]
    plsq = leastsq(residual_trend,p0,args = (lat,lon,data))
    para = plsq[0]
    data_trend = data - func_trend(lat,lon,para)
    corr, _ = pearsonr(data, func_trend(lat,lon,para))
    return data_trend, para, corr

def func_trend(lat,lon,p):
    a0,b0,c0,d0 = p
    
    return a0 + b0*lat + c0*lon +d0*lat*lon

def residual_trend(p,lat,lon,y0):
    a0,b0,c0,d0 = p 
    return y0 - func_trend(lat,lon,p)

def func_trend_model(lat,lon,p):
    lat = lat/180*np.pi
    lon = lon/180*np.pi  
    lon = lon*np.cos(lat) # to get isometrics coordinates
    a0,b0,c0,d0 = p
    
    return a0 + b0*lat + c0*lon +d0*lat*lon

def OK_function(data0):
    OK,lat0,lon0,np = data0
    z0,s0 = OK.execute('points', lon0, lat0,n_closest_points= np,backend='loop')
    return z0,s0

def dist_weight_interp(data0):
    lat0,lon0,z0,lat1,lon1 = data0
    
    lat0 = np.asarray(lat0)
    lon0 = np.asarray(lon0)
    z0 = np.asarray(z0)
    
    if len(z0)==1:
        z0 = z0[0]
    nn = len(lat1)
    data_interp = np.zeros((nn,))
    weight_all = []
    for i in range(nn):
        dist0 = latlon2dis(lat0,lon0,lat1[i],lon1[i])
        weight0 = (1/dist0)**2
        if len(weight0) ==1:
            weight0 = weight0[0]
        weight = weight0/sum(weight0[:])
        data_interp[i] = sum(z0*weight)
        weight_all.append(weight)
    return data_interp,weight_all

def latlon2dis(lat1,lon1,lat2,lon2,R=6371):
    
    lat1 = np.array(lat1)*np.pi/180.0
    lat2 = np.array(lat2)*np.pi/180.0
    dlon = (lon1-lon2)*np.pi/180.0

    # Evaluate trigonometric functions that need to be evaluated more
    # than once:
    c1 = np.cos(lat1)
    s1 = np.sin(lat1)
    c2 = np.cos(lat2)
    s2 = np.sin(lat2)
    cd = np.cos(dlon)

    # This uses the arctan version of the great-circle distance function
    # from en.wikipedia.org/wiki/Great-circle_distance for increased
    # numerical stability.
    # Formula can be obtained from [2] combining eqns. (14)-(16)
    # for spherical geometry (f=0).

    dist =  R*np.arctan2(np.sqrt((c2*np.sin(dlon))**2 + (c1*s2-s1*c2*cd)**2), s1*s2+c1*c2*cd)

    return dist
    
def cmdLineParse():
    parser = argparse.ArgumentParser(description='Interpolate high-resolution tropospheric product map in a 3D perspective.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('date', help='SAR acquisition date.')
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
    parser.add_argument('-o','--out', dest='out_file', metavar='FILE',help='name of the prefix of the output file')
       
    inps = parser.parse_args()

    return inps


INTRODUCTION = '''
##################################################################################
   Copy Right(c): 2019, Yunmeng Cao   @PyRite v1.0
   
   Generate high-resolution GPS-based tropospheric maps (delays & water vapor) for InSAR Geodesy & meteorology.
'''

EXAMPLE = """Example:
  
  interp_era5_pyrite.py 20190101 geometryRadar.h5 --type tzd --max-altitude 3500 --min-altitude -200 --lalo-rescale 10
  interp_era5_pyrite.py 20190101 geometryRadar.h5 --type wzd --parallel 8
  interp_era5_pyrite.py 20190101 geometryRadar.h5 --elevation-model exp --type tzd 
  interp_era5_pyrite.py 20190101 geometryRadar.h5 --variogram-model spherical --method kriging -o 20190101_gps_aps_kriging.h5
  interp_era5_pyrite.py 20190101 geometryRadar.h5 --elevation-model exp --max-length 250 --variogram-model spherical 
  interp_era5_pyrite.py 20190101 geometryRadar.h5 --method kriging --kriging-points-numb 15
  interp_era5_pyrite.py 20190101 geometryRadar.h5 --method weight_distance
  interp_era5_pyrite.py 20190101 geometryRadar.h5 --type tzd 
  interp_era5_pyrite.py 20190101 geometryRadar.h5 --type wzd --parallel 4
  
###################################################################################
"""

###############################################################

def main(argv):
    
    inps = cmdLineParse() 
    geo_file = inps.geo_file
    date0 = inps.date
    
    # determine out put file name
    root_path = os.getcwd()
    pyrite_dir = root_path + '/pyrite'
    era5_dir = pyrite_dir + '/ERA5'
    era5_raw_dir = era5_dir  + '/raw'
    era5_sar_dir =  era5_dir  + '/sar'
    
    if not os.path.isdir(era5_sar_dir):
        os.mkdir(era5_sar_dir)
    
    if inps.out_file: OUT = inps.out_file
    else: OUT = date0 + '_' + inps.type + '.h5'
   
    OUT = era5_sar_dir + '/' + OUT
    
    # Get raw values from ERA5
    cdic = ut.initconst()
    fname0 = glob.glob(era5_raw_dir + '/ERA*' + date0 + '*')[0]
    lvls,latlist,lonlist,gph,tmp,vpr = ut.get_ecmwf('ERA5',fname0,cdic, humidity='Q')
    mean_lon = np.mean(lonlist.flatten())
    if mean_lon > 180:    
        lonlist = lonlist - 360.0 # change to insar format lon [-180, 180]
        lonlist = np.asarray(lonlist)
    
    lon = lonlist.flatten() 
    lat = latlist.flatten() 
    
    lonStep = lonlist[0,1] - lonlist[0,0]
    latStep = latlist[1,0] - latlist[0,0]
    
    maxlon = max(lonlist[0,:])
    minlon = min(lonlist[0,:])
    
    maxlat = max(latlist[:,0])
    minlat = min(latlist[:,0])
    
    
    Rescale = inps.lalo_rescale # default 5  around ~ 6km
    lonStep1 = lonStep/Rescale
    latStep1 = latStep/Rescale
    
    lonv = np.arange(minlon,maxlon,lonStep1)
    latv = np.arange(maxlat,minlat,latStep1)
    
    lonvv,latvv = np.meshgrid(lonv,latv)
    
    # Make a height scale
    hgt = np.linspace(cdic['minAltP'], gph.max().round(), cdic['nhgt'])
    # Interpolate pressure, temperature and Humidity of hgt
    [Pi,Ti,Vi] = ut.intP2H(lvls, hgt, gph, tmp, vpr, cdic)
    # Calculate the delays
    [DDry,DWet] = ut.PTV2del(Pi,Ti,Vi,hgt,cdic)
    
    #max_altitude_model = inps.max_altitude + 200
    hgt0 = hgt.copy()
    idx0 = np.where(((inps.min_altitude<hgt0) & (hgt0 < inps.max_altitude)))
    hgt00 = hgt0[idx0]
    
    hgt0 = list(hgt0)

    dd0 = DDry.copy()
    dw0 = DWet.copy()
    dt0 = dd0 + dw0
    
    if inps.type =='tzd': data0 = dt0
    elif inps.type =='wzd': data0 = dw0
    
    nk = len(hgt00)
    
    mdd = np.zeros((nk,),dtype = np.float32)
    mdw = np.zeros((nk,),dtype = np.float32)
    mdt = np.zeros((nk,),dtype = np.float32)
    
    for i in range(nk):
        mdd[i] = np.mean(dd0[:,:,hgt0.index(hgt00[i])])
        mdw[i] = np.mean(dw0[:,:,hgt0.index(hgt00[i])])
        mdt[i] = np.mean(dt0[:,:,hgt0.index(hgt00[i])])
    
    
    R = 6371    
    BIN_NUMB = inps.bin_numb
    max_length = inps.max_length
    range0 = max_length/2
    model = 'spherical'
    #eng = matlab.engine.start_matlab()
    row,col = lonvv.shape
    Delf_dense = np.zeros((nk,row,col),dtype = np.float32)
    
    hx = hgt00.copy()
    rescale_h = inps.hgt_rescale # default 10  around ~ 15 m/level
    hx_step = hx[1] - hx[0]
    hx_step2 = hx_step/rescale_h
    hgt_dense = np.arange(min(hx),max(hx)-0.5,hx_step2) 
    # minus 0.5 to make sure all of the interpolated locations are inside the function region.
    
    #print(hgt_dense)
    #print(hx)
    ## interp horizontal 
    prog_bar = ptime.progressBar(maxValue=nk)
    
    def resi_func(m,d,y):
        variogram_function =variogram_dict[inps.variogram_model] 
        return  y - variogram_function(m,d)
    
    for i in range(nk):
    #for i in range(1):
        k0 = i
        tzd = data0[:,:,hgt0.index(hgt00[k0])]
        tzd = tzd.flatten()
        
        if inps.method =='kriging':
            tzd0, para, corr= remove_ramp(lat,lon,tzd)
            trend = func_trend_model(latvv,lonvv,para)
            #trend = trend.reshape(row,col)
            uk = OrdinaryKriging(lon, lat, tzd0, coordinates_type = 'geographic', nlags=BIN_NUMB)
            Semivariance_trend = 2*(uk.semivariance)    
            x0 = (uk.lags)/180*np.pi*R
            y0 = Semivariance_trend
    
            LL0 = x0[x0< max_length]
            SS0 = y0[x0< max_length]
            sill0 = max(SS0)
            sill0 = sill0.tolist()
    
            p0 = [sill0, range0, 0.0001]    
            vari_func = variogram_dict[inps.variogram_model]        
            tt, _ = leastsq(resi_func,p0,args = (LL0,SS0))   
            corr, _ = pearsonr(SS0, vari_func(tt,LL0))
            if tt[2] < 0:
                tt[2] =0
            para = tt
            
            #LLm = matlab.double(LL0.tolist())
            #SSm = matlab.double(SS0.tolist()) 
            #tt = eng.variogramfit(LLm,SSm,range0,sill0,[],'nugget',0.00001,'model',inps.variogram_model)
            #variogram_para0 = tt[0]
            #para = variogram_para0[0:3]
            
            para[1] = para[1]/R/np.pi*180
            #print(para)
            uk.variogram_model_parameters = para
            z0,s0 = uk.execute('grid', lonv, latv, n_closest_points = inps.kriging_points_numb, backend='loop')
            #ax.plot(x0,y0,'r.')
            Delf_dense[i,:,:] = z0 + trend
        else:
            # using scipy.interpolate.griddata
            points = np.zeros((len(lon),2),dtype = np.float32)
            points[:,0] = lon
            points[:,1] = lat
            tzd = tzd.reshape(len(tzd),)
            
            grid_tzd = griddata(points, tzd, (lonvv, latvv), method='linear')
            Delf_dense[i,:,:] = grid_tzd
        
        prog_bar.update(i+1, every=1, suffix='{}/{} slices'.format(i+1, nk))
    prog_bar.close()
    
    ## interp latitude
    nk2 = len(hgt_dense)
    Delf_dense2 = np.zeros((nk2,row,col),dtype = np.float32)
    
    initial_function = initial_dict[inps.elevation_model]
    elevation_function = model_dict[inps.elevation_model]
    residual_function = residual_dict[inps.elevation_model]
    
    
    ## elevation model and remove
    if inps.type =='tzd': y0 = mdt
    elif inps.type =='wzd': y0 = mdw
    x0 = hgt00
    p0 = initial_function(x0,y0)
    plsq = leastsq(residual_function,p0,args = (x0,y0))
    plsq = leastsq(residual_function,plsq[0],args = (x0,y0))
    plsq = leastsq(residual_function,plsq[0],args = (x0,y0))
    plsq = leastsq(residual_function,plsq[0],args = (x0,y0))
    
    turb_dense = Delf_dense.copy()
    for i in range(nk):
        turb_dense[i,:,:] = Delf_dense[i,:,:] - elevation_function(plsq[0],hgt00[i])

    prog_bar = ptime.progressBar(maxValue=row)
    for i in range(row):
        for j in range(col):
            hy = turb_dense[:,i,j]
            #print(hy)
            tck = intp.interp1d(hx,hy,kind='cubic')
            temp = tck(hgt_dense)
            Delf_dense2[:,i,j] = temp
        prog_bar.update(i+1, every=5, suffix='{}/{} slices'.format(i+1, row))
    prog_bar.close()
    
    #print(temp)
    #fnc = make3dintp(Delf_dense2,lonvv,latvv,hgt_dense,1)
    
    linearint = intp.RegularGridInterpolator((hgt_dense,latv[::-1], lonv), Delf_dense2[:,::-1,:], method='linear', bounds_error=False, fill_value = 0.0)
    
    #eng.quit()
    
    ## interp InSAR
    
    #geo_file = '/Users/caoy0a/Documents/SCRATCH/LosAngelesT71F479D/geometryRadar.h5'
    datasetNames = ut.get_dataNames(inps.geo_file)
    meta = ut.read_attr(geo_file)
    if 'latitude' in datasetNames: 
        lats = ut.read_hdf5(geo_file,datasetName='latitude')[0]
        lons = ut.read_hdf5(geo_file,datasetName='longitude')[0]
    else:
        lats,lons = ut.get_lat_lon(meta)
    
    heis = ut.read_hdf5(geo_file,datasetName='height')[0]
    val_turb = np.zeros((heis.shape),dtype = np.float32)   
    val_turb0 = val_turb.flatten()
    
    heis0 = heis.flatten()
    lats0 = lats.flatten()
    lons0 = lons.flatten()
    
    hh = heis0[~np.isnan(lats0)]
    lala = lats0[~np.isnan(lats0)]
    lolo = lons0[~np.isnan(lats0)]
    
    val_turb00 = linearint(np.vstack((hh, lala, lolo)).T)
    val_turb0[~np.isnan(lats0)] = val_turb00

    row0,col0 = lats.shape

    val_turb = val_turb0.reshape(row0,col0)
    val_topo = elevation_function(plsq[0],heis)
    
    val_total = val_turb + val_topo
    
    #####
    meta['UNIT'] = 'm'
    meta['FILE_TYPE'] = 'delay'
    
    datasetDict = dict()
    datasetDict['turb_sar'] = val_turb
    datasetDict['hgt_sar'] = val_topo
    datasetDict['aps_sar'] = val_total
    ut.write_h5(datasetDict, OUT, metadata=meta, ref_file=None, compression=None)

    
    sys.exit(1)
###############################################################

if __name__ == '__main__':
    main(sys.argv[:])