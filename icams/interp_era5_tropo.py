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

from pyrite import elevation_models
from pyrite import _utils as ut
from pykrige import OrdinaryKriging

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
###############################################################

model_dict = {'linear': elevation_models.linear_elevation_model,
                      'onn': elevation_models.onn_elevation_model,
                      'onn_linear': elevation_models.onn_linear_elevation_model,
                      'exp': elevation_models.exp_elevation_model,
                      'exp_linear': elevation_models.exp_linear_elevation_model}


def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=4):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out

def function_trend(lat,lon,para):
    # mod = a*x + b*y + c*x*y
    lat = lat/180*np.pi
    lon = lon/180*np.pi  
    lon = lon*np.cos(lat) # to get isometrics coordinates
    
    a0,b0,c0,d0 = para
    data_trend = a0 + b0*lat + c0*lon +d0*lat*lon
    
    return data_trend

def split_lat_lon_kriging(nn, processors = 4):

    dn = round(nn/int(processors))
    
    idx = []
    for i in range(processors):
        a0 = i*dn
        b0 = (i+1)*dn
        
        if i == (processors - 1):
            b0 = nn
        
        if not a0 > b0:
            idx0 = np.arange(a0,b0)
            #print(idx0)
            idx.append(idx0)
            
    return idx

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
    parser = argparse.ArgumentParser(description='Interpolate high-resolution tropospheric product map.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('date', help='SAR acquisition date.')
    parser.add_argument('era5_file',help='input file name (e.g., era5_sample_variogramModel.h5).')
    parser.add_argument('geo_file',help='input geometry file name (e.g., geometryRadar.h5).')
    parser.add_argument('--type', dest='type', choices = {'tzd','wzd'}, default = 'tzd',help = 'type of the high-resolution tropospheric map.[default: tzd]')
    parser.add_argument('--method', dest='method', choices = {'kriging','weight_distance'},default = 'kriging',help = 'method used to interp the high-resolution map. [default: kriging]')
    parser.add_argument('-o','--out', dest='out_file', metavar='FILE',help='name of the prefix of the output file')
    parser.add_argument('--parallel', dest='parallelNumb', type=int, default=1, help='Enable parallel processing and Specify the number of processors.')
    parser.add_argument('--kriging-points-numb', dest='kriging_points_numb', type=int, default=20, help='Number of the closest points used for Kriging interpolation. [default: 20]')
       
    inps = parser.parse_args()

    return inps


INTRODUCTION = '''
##################################################################################
   Copy Right(c): 2019, Yunmeng Cao   @PyRite v1.0
   
   Generate high-resolution GPS-based tropospheric maps (delays & water vapor) for InSAR Geodesy & meteorology.
'''

EXAMPLE = """Example:
  
  interp_era5_tropo.py 20190101 era5_file geometryRadar.h5 --type tzd
  interp_era5_tropo.py 20190101 era5_file geometryRadar.h5 --type wzd --parallel 8
  interp_era5_tropo.py 20190101 era5_file geometryRadar.h5 --method kriging -o 20190101_gps_aps_kriging.h5
  interp_era5_tropo.py 20190101 era5_file geometryRadar.h5 --method kriging --kriging-points-numb 15
  interp_era5_tropo.py 20190101 era5_file geometryRadar.h5 --method weight_distance
  interp_era5_tropo.py 20190101 era5_file geometryRadar.h5 --type tzd 
  interp_era5_tropo.py 20190101 era5_file geometryRadar.h5 --type wzd --parallel 4
  
###################################################################################
"""

###############################################################

def main(argv):
    
    inps = cmdLineParse() 
    era5_file = inps.era5_file
    geom_file = inps.geo_file
    date = inps.date
    
    meta0 = ut.read_attr(era5_file)
    meta = ut.read_attr(geom_file)
    WIDTH = int(meta['WIDTH'])
    LENGTH = int(meta['LENGTH'])
    date_list = ut.read_hdf5(era5_file,datasetName='date')[0]
    date_list = date_list.astype('U13')
    date_list = list(date_list)

    # Get sample coord and values
    k0 = date_list.index(date)
    wzd_turb_all = ut.read_hdf5(era5_file,datasetName='wzd_turb')[0]
    tzd_turb_all = ut.read_hdf5(era5_file,datasetName='tzd_turb')[0]
    lat_samp = ut.read_hdf5(era5_file,datasetName='latitude')[0]
    lon_samp = ut.read_hdf5(era5_file,datasetName='longitude')[0]

    k0 = date_list.index(date)
    tzd_turb = tzd_turb_all[k0,:,:]
    tzd_turb = tzd_turb.flatten()
    lat0 = lat_samp.flatten()
    lon0 = lon_samp.flatten()
    
    wzd_turb = wzd_turb_all[k0,:,:]
    wzd_turb = wzd_turb.flatten()
    
    
    root_path = os.getcwd()
    pyrite_dir = root_path + '/pyrite'
    era5_dir = pyrite_dir + '/ERA5'
    era5_raw_dir = era5_dir  + '/raw'
    era5_sar_dir =  era5_dir  + '/sar'
    
    if not os.path.isdir(era5_sar_dir):
        os.mkdir(era5_sar_dir)
    
    if inps.out_file: OUT = inps.out_file
    else: OUT = date + '_' + inps.type + '.h5'
   
    OUT = era5_sar_dir + '/' + OUT


    dem = ut.read_hdf5(geom_file,datasetName='height')[0]
    #inc = read_hdf5(geom_file,datasetName='incidenceAngle')[0]
    
    dataNames = ut.get_dataNames(geom_file)
    if 'latitude' in dataNames:
        grid_lat = ut.read_hdf5(geom_file,datasetName='latitude')[0]
        grid_lon = ut.read_hdf5(geom_file,datasetName='longitude')[0]
    else:
        grid_lat, grid_lon = ut.get_lat_lon(meta)
    
    lats = grid_lat.flatten()
    lons = grid_lon.flatten()
    #print(grid_lat.shape)


    #print(inps.type)
    if inps.type == 'tzd': 
        S0 = 'tzd'
        turb =tzd_turb
    elif inps.type == 'wzd':
        S0 = 'wzd'
        turb = wzd_turb
    
    #lat0, lon0, turb0 = remove_numb(lat,lon,turb,int(meta0['remove_numb']))
    #lon0 = lon0 - 360
    turb0 = turb
    
    variogram_para = ut.read_hdf5(era5_file,datasetName = S0 + '_variogram_parameter')[0]
    trend_para = ut.read_hdf5(era5_file,datasetName = S0 + '_trend_parameter')[0]
    elevation_para = ut.read_hdf5(era5_file,datasetName = S0 + '_elevation_parameter')[0]
    
    variogram_para0 = variogram_para[k0,:]
    trend_para0 = trend_para[k0,:]
    elevation_para0 = elevation_para[k0,:]
    
    #lons0 = lons + 360 # change to gps longitude format
    lons0 = lons
    lats0 = lats
    Trend0 = function_trend(lats0,lons0,trend_para0)
    Trend0 = Trend0.reshape(LENGTH,WIDTH)
    
    hei = dem.flatten()
    elevation_function = model_dict[meta0['elevation_model']]
    Dem_cor0 = elevation_function(elevation_para0,hei)
    Dem_cor0 = Dem_cor0.reshape(LENGTH,WIDTH)

    OK = OrdinaryKriging(lon0, lat0, turb0, variogram_model=meta0['variogram_model'], verbose=False,enable_plotting=False)
    para = variogram_para0[0:3]
    para[1] = para[1]/6371/np.pi*180
    #print(para)
    OK.variogram_model_parameters = para

    Numb = inps.parallelNumb
    zz = np.zeros((len(lats),))
    zz_sigma = np.zeros((len(lats),))
    #print(len(lats))
    n_jobs = inps.parallelNumb
    
    # split dataset into multiple subdata
    split_numb = 2000
    idx_list = split_lat_lon_kriging(len(lats),processors = split_numb)
    #print(idx_list[split_numb-1])
    
    print('------------------------------------------------------------------------------')
    if inps.type =='tzd':
        print('Start to interpolate the high-resolution tropospheric delay for SAR acquisition: ' + date)
    elif inps.type =='wzd':
        print('Start to interpolate the high-resolution atmospheric water vapor for SAR acquisition: ' + date)
        
    if inps.method =='kriging':
        np0 = inps.kriging_points_numb
        data = []
        for i in range(split_numb):
            data0 = (OK,lats[idx_list[i]],lons[idx_list[i]],np0)
            data.append(data0)
        future = parallel_process(data, OK_function, n_jobs=Numb, use_kwargs=False, front_num=2)
        
    elif inps.method =='weight_distance':
        data = []
        for i in range(split_numb):
            data0 = (lat0,lon0,turb0,lats[idx_list[i]],lons[idx_list[i]])
            data.append(data0)
        future = parallel_process(data, dist_weight_interp, n_jobs=Numb, use_kwargs=False, front_num=2)
    
    for i in range(split_numb):
        id0 = idx_list[i]
        gg = future[i]
        zz[id0] = gg[0]
        
    turb = zz.reshape(LENGTH,WIDTH)
    datasetDict =dict()
    
    aps = Dem_cor0 + Trend0 + turb
    datasetDict['hgt_sar'] = Dem_cor0
    datasetDict['trend_sar'] = Trend0
    datasetDict['turb_sar'] = turb
    datasetDict['aps_sar'] = aps
    
    if inps.method == 'kriging':
        for i in range(split_numb):
            id0 = idx_list[i]
            gg = future[i]
            zz_sigma[id0] = gg[1]
        sigma = zz_sigma.reshape(LENGTH,WIDTH)
        datasetDict['turb_sar_sigma'] = sigma
    
    meta['DATA_TYPE'] = inps.type
    meta['interp_method'] = inps.method
    
    for key, value in meta.items():
        meta0[key] = str(value)
    
    meta0['UNIT'] = 'm'   
    ut.write_h5(datasetDict, OUT, metadata=meta0, ref_file=None, compression=None)
    
    sys.exit(1)
###############################################################

if __name__ == '__main__':
    main(sys.argv[:])