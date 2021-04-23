#! /usr/bin/env python
#################################################################
###  This program is part of PyRite  v1.0                     ### 
###  Copy Right (c): 2019, Yunmeng Cao                        ###  
###  Author: Yunmeng Cao                                      ###                                                          
###  Contact : ymcmrs@gmail.com                               ###
###  Univ. : King Abdullah University of Science & Technology ###   
#################################################################


import numpy as np
import getopt
import sys
import os
import h5py
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.stats.stats import pearsonr
from pyrite import _utils as ut

from pyrite import elevation_models
#### define dicts for model/residual/initial values ###############

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
    
def cmdLineParse():
    parser = argparse.ArgumentParser(description='Analyze different components of ERA5 tropospheric delays.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('era5_file',help='era5_h5 file.')
    parser.add_argument('-m','--model', dest='model',default = 'onn_linear',choices= ['linear','onn','onn_linear','exp','exp_linear'],help='elevation models')
    parser.add_argument('-o', dest='out', help='output file.')
    
    inps = parser.parse_args()
    
    return inps


INTRODUCTION = '''
################################################################################################
    Modeling for the elevation-correlated tropospheric products: tropospheric delays, atmospehric water vapor ...    
    Supported models: Linear, Onn, Onn-linear, Exponential, Exponential-linear.
    
    See alse:   search_gps.py, download_gps_atm.py, extract_sar_atm.py, 
'''


EXAMPLE = '''EXAMPLES:
    elevation_correlation_era.py era5_sample.h5 -m linear
    elevation_correlation_era.py era5_sample.h5 -m onn
    elevation_correlation_era.py era5_sample.h5 -m onn_linear 
    elevation_correlation_era.py era5_sample.h5 -m exp
    
################################################################################################
'''


def main(argv):
    
    inps = cmdLineParse()
    era5_file = inps.era5_file 
    FILE = era5_file 
    
    if inps.out: OUT = inps.out
    else: OUT = 'era5_sample_hgtCor.h5'
        
    model = inps.model
    date_list, meta =  ut.read_hdf5(FILE,datasetName='date')
    wdelay_all = ut.read_hdf5(FILE,datasetName='wetDelay')[0]
    tdelay_all = ut.read_hdf5(FILE,datasetName='totDelay')[0]
    
    wzd_model_parameters = np.zeros((len(date_list),para_numb_dict[model]),dtype = np.float32)
    tzd_model_parameters = np.zeros((len(date_list),para_numb_dict[model]),dtype = np.float32)
    
    wzd_trend_parameters = np.zeros((len(date_list),4),dtype = np.float32)
    tzd_trend_parameters = np.zeros((len(date_list),4),dtype = np.float32)
    
    tzd_turb_trend = np.zeros((np.shape(wdelay_all)),dtype = np.float32)
    wzd_turb_trend = np.zeros((np.shape(wdelay_all)),dtype = np.float32)
    
    tzd_turb = np.zeros((np.shape(wdelay_all)),dtype = np.float32)
    wzd_turb = np.zeros((np.shape(wdelay_all)),dtype = np.float32)
    
    initial_function = initial_dict[model]
    elevation_function = model_dict[model]
    residual_function = residual_dict[model]
    
    
    hei =  ut.read_hdf5(FILE,datasetName='height')[0]
    lat = ut.read_hdf5(FILE,datasetName='latitude')[0]
    lon = ut.read_hdf5(FILE,datasetName='longitude')[0]
    row,col = hei.shape
    hei = hei.flatten()
    lat = lat.flatten()
    lon = lon.flatten()
    
    for i in range(len(date_list)):
        #hei, wzd, tzd = adjust_aps(FILE,epoch = i)
        #hei,lat,lon = adjust_aps_lat_lon(FILE,epoch = i)
        wzd = wdelay_all[i,:,:]
        wzd = wzd.flatten()
        
        tzd = tdelay_all[i,:,:]
        tzd = tzd.flatten()
        
        p0 = initial_function(hei,wzd)
        plsq = leastsq(residual_function,p0,args = (hei,wzd))
        wzd_model_parameters[i,:] = plsq[0]
        res0 = wzd - elevation_function(plsq[0],hei)
        wzd_turb_trend[i,:,:] = res0.reshape(row,col)
        corr_wzd, _ = pearsonr(wzd, elevation_function(plsq[0],hei))
        
        y0 = wzd_turb_trend[i,:,:]
        y0 = y0.flatten()
        wzd_turb0,para0,corr_trend0 = remove_ramp(lat,lon,y0)
        wzd_turb[i,:,:] = wzd_turb0.reshape(row,col)
        wzd_trend_parameters[i,:] = para0
        
        p0 = initial_function(hei,tzd)
        plsq = leastsq(residual_function,p0,args = (hei,tzd))
        tzd_model_parameters[i,:] = plsq[0]
        res0 = tzd - elevation_function(plsq[0],hei)
        tzd_turb_trend[i,:,:] = res0.reshape(row,col)
        corr_tzd, _ = pearsonr(tzd, elevation_function(plsq[0],hei))
        
        y0 = tzd_turb_trend[i,:,:]
        y0 = y0.flatten()
        tzd_turb0,para0,corr0 = remove_ramp(lat,lon,y0)
        tzd_turb[i,:,:] = tzd_turb0.reshape(row,col)
        tzd_trend_parameters[i,:] = para0
        
        print(str(date_list[i].decode("utf-8")) + ' : ' + str(round(corr_wzd*1000)/1000) + ' ' + str(round(corr_tzd*10000)/10000) + ' ' + str(round(corr_trend0*1000)/1000) + ' ' + str(round(corr0*1000)/1000))
        

    datasetNames =['height','latitude','longitude','wetDelay','totDelay','dryDelay','date']
    datasetDict = dict()
    
  
    for dataName in datasetNames:
        datasetDict[dataName] = ut.read_hdf5(FILE,datasetName=dataName)[0]
    
    datasetDict['tzd_turb_trend'] = tzd_turb_trend
    datasetDict['wzd_turb_trend'] = wzd_turb_trend    
        
    datasetDict['tzd_turb'] = tzd_turb
    datasetDict['wzd_turb'] = wzd_turb
    
    datasetDict['tzd_elevation_parameter'] = tzd_model_parameters
    datasetDict['wzd_elevation_parameter'] = wzd_model_parameters
    
    datasetDict['tzd_trend_parameter'] = tzd_trend_parameters
    datasetDict['wzd_trend_parameter'] = wzd_trend_parameters
    
    meta['elevation_model'] = model
    
    ut.write_h5(datasetDict, OUT, metadata=meta, ref_file=None, compression=None)
            
if __name__ == '__main__':
    main(sys.argv[1:])