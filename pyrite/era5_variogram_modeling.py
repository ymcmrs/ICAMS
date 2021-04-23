#! /usr/bin/env python
#################################################################
###  This program is part of PyRite  v1.0                     ### 
###  Copy Right (c): 2019, Yunmeng Cao                        ###  
###  Author: Yunmeng Cao                                      ###                                                          
###  Email : ymcmrs@gmail.com                                 ###
###  Univ. : King Abdullah University of Science & Technology ###   
#################################################################

import numpy as np
import os
import sys  
import subprocess
import getopt
import time
import glob
import argparse
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
import random

import pykrige
from pykrige import OrdinaryKriging
from pykrige import variogram_models
from scipy.optimize import leastsq
from scipy.stats.stats import pearsonr

from pyrite import _utils as ut
#import matlab.engine
#######################################################
#residual_variogram_dict = {'linear': variogram_models.linear_variogram_model_residual,
#                      'power': variogram_models.power_variogram_model_residual,
#                      'gaussian': variogram_models.gaussian_variogram_model_residual,
#                      'spherical': variogram_models.spherical_variogram_model_residual,
#                      'exponential': variogram_models.exponential_variogram_model_residual,
#                      'hole-effect': variogram_models.hole_effect_variogram_model_residual}


variogram_dict = {'linear': variogram_models.linear_variogram_model,
                      'power': variogram_models.power_variogram_model,
                      'gaussian': variogram_models.gaussian_variogram_model,
                      'spherical': variogram_models.spherical_variogram_model,
                      'exponential': variogram_models.exponential_variogram_model,
                      'hole-effect': variogram_models.hole_effect_variogram_model}


 
#########################################################################

INTRODUCTION = '''
#############################################################################
   Copy Right(c): 2019, Yunmeng Cao   @PyRite v1.0
   
   Variogram model estimation of the ERA-5 tropospheric measurements.
'''

EXAMPLE = '''
    Usage:
            era5_variogram_modeling.py era5_sample_variogram.h5 
            era5_variogram_modeling.py era5_sample_variogram.h5  --model gaussian
            ear5_variogram_modeling.py era5_sample_variogram.h5  --max-length 150 --model spherical

##############################################################################
'''


def cmdLineParse():
    parser = argparse.ArgumentParser(description='Check common busrts for TOPS data.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('input_file',help='input file name (e.g., gps_aps_variogram.h5).')
    parser.add_argument('-m','--model', dest='model', default='spherical',
                      help='variogram model used to fit the variance samples')
    parser.add_argument('--max-length', dest='max_length',type=float, metavar='NUM',
                      help='used bin ratio for mdeling the structure model.')
    parser.add_argument('-o','--out_file', dest='out_file', metavar='FILE',
                      help='name of the output file')

    inps = parser.parse_args()

    return inps

################################################################################    
    
    
def main(argv):
    
    inps = cmdLineParse() 
    FILE = inps.input_file
    
    date,meta = ut.read_hdf5(FILE, datasetName='date')
    variance_tzd = ut.read_hdf5(FILE, datasetName='Semivariance')[0]
    variance_wzd = ut.read_hdf5(FILE, datasetName='Semivariance_wzd')[0]
 
    Lag = ut.read_hdf5(FILE, datasetName='Lags')[0]
    
    if inps.max_length:
        max_lag = inps.max_length
    else:
        max_lag = max(lag) + 0.001
    meta['max_length'] = max_lag
    r0 = np.asarray(1/2*max_lag)
    range0 = r0.tolist()
    
    datasetDict = dict()
    datasetDict['Lags'] = Lag
    
    if inps.out_file: OUT = os.path.out_file
    else:  OUT = 'era5_sample_variogramModel.h5'
    
    #eng = matlab.engine.start_matlab()
    
    row,col = variance_tzd.shape
    model_parameters = np.zeros((row,4),dtype='float32')   # sill, range, nugget, Rs
    model_parameters_wzd = np.zeros((row,4),dtype='float32')   # sill, range, nugget, Rs
    
    def resi_func(m,d,y):
        variogram_function =variogram_dict[inps.model] 
        return  y - variogram_function(m,d)
    
    for i in range(row):
        print(date[i])
        LL0 = Lag[Lag < max_lag]     
        S0 = variance_tzd[i,:]
        SS0 = S0[Lag < max_lag]
        sill0 = max(SS0)
        sill0 = sill0.tolist()
        
        p0 = [sill0, range0, 0.0001]    
        vari_func = variogram_dict[inps.model]        
        tt, _ = leastsq(resi_func,p0,args = (LL0,SS0))   
        corr, _ = pearsonr(SS0, vari_func(tt,LL0))
        if tt[2] < 0:
            tt[2] =0
        model_parameters[i,0:3] = tt
        model_parameters[i,3] = corr   
        #print(tt)
    
        #LLm = matlab.double(LL0.tolist())
        #SSm = matlab.double(SS0.tolist()) 
        #tt = eng.variogramfit(LLm,SSm,range0,sill0,[],'nugget',0.00001,'model',inps.model)
        #model_parameters[i,:] = np.asarray(tt)
        #print(model_parameters[i,:])
        
        #S0 = variance_wzd[i,:]
        #SS0 = S0[Lag < max_lag]
        #sill0 = max(SS0)
        #sill0 = sill0.tolist()
        
        #p0 = [sill0, range0, 0.0001]      
        #vari_func = variogram_dict[inps.model]
        #tt, _ = leastsq(resi_func,p0,args = (LL0,SS0))   
        #corr, _ = pearsonr(SS0, vari_func(tt,LL0))
        #if tt[2] < 0:
        #    tt[2] =0
        #model_parameters_wzd[i,0:3] = tt
        #model_parameters_wzd[i,3] = corr
        
        #LLm = matlab.double(LL0.tolist())
        #SSm = matlab.double(SS0.tolist())

        #print(model_parameters_wzd[i,:])
        #tt = eng.variogramfit(LLm,SSm,range0,sill0,[],'nugget',0.00001,'model',inps.model)
        #model_parameters_wzd[i,:] = np.asarray(tt)
        
    meta['variogram_model'] = inps.model
    #meta['elevation_model'] = meta['elevation_model']    
    #del meta['model']
    
    datasetNames = ut.get_dataNames(FILE)
    
    datasetDict = dict()
    for dataName in datasetNames:
        datasetDict[dataName] = ut.read_hdf5(FILE,datasetName=dataName)[0]
    
    datasetDict['tzd_variogram_parameter'] = model_parameters  
    datasetDict['wzd_variogram_parameter'] = model_parameters_wzd  
    #eng.quit()
    ut.write_h5(datasetDict, OUT, metadata=meta, ref_file=None, compression=None) 
    
    sys.exit(1)

if __name__ == '__main__':
    main(sys.argv[:])
