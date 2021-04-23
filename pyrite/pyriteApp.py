#! /usr/bin/env python
#################################################################
###  Project: PyRite                                          ###
###  Purpose: PYthon package of Reducing InSAR Trop Effects   ###
###  Copy Right (c): 2019, Yunmeng Cao                        ###                                                           
###  Contact : ymcmrs@gmail.com                               ###
###  Inst. : King Abdullah University of Science & Technology ###   
#################################################################

import sys
import os
import re
import subprocess
import argparse
import numpy as np
import h5py
import glob

from pyrite import _utils as ut

###############################################################
def check_variable_name(path):
    s=path.split("/")[0]
    if len(s)>0 and s[0]=="$":
        p0=os.getenv(s[1:])
        path=path.replace(path.split("/")[0],p0)
    return path


def read_cfg(File, delimiter='='):
    '''Reads the gigpy-configure file into a python dictionary structure.
    Input : string, full path to the template file
    Output: dictionary, gigpy configure content
    Example:
        tmpl = read_cfg(LosAngelse.cfg)
        tmpl = read_cfg(R1_54014_ST5_L0_F898.000.pi, ':')
    '''
    cfg_dict = {}
    for line in open(File):
        line = line.strip()
        c = [i.strip() for i in line.split(delimiter, 1)]  #split on the 1st occurrence of delimiter
        if len(c) < 2 or line.startswith('%') or line.startswith('#'):
            next #ignore commented lines or those without variables
        else:
            atrName  = c[0]
            atrValue = str.replace(c[1],'\n','').split("#")[0].strip()
            atrValue = check_variable_name(atrValue)
            cfg_dict[atrName] = atrValue
    return cfg_dict

def generate_datelist_txt(date_list,txt_name):
    
    if os.path.isfile(txt_name): os.remove(txt_name)
    for list0 in date_list:
        call_str = 'echo ' + list0 + ' >>' + txt_name 
        os.system(call_str)
        
    return

def cmdLineParse():
    parser = argparse.ArgumentParser(description='Generate high-resolution maps of GPS tropospheric measurements.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('cfg_file',help='name of the input configure file')
    parser.add_argument('-g',  dest='generate_cfg' ,help='generate an example of configure file.')

    inps = parser.parse_args()

    return inps


INTRODUCTION = '''
##################################################################################
   Copy Right(c): 2019, Yunmeng Cao   @PyRite v1.0
   
   PyRite: PYthon package of Reducing InSAR Tropospheric Effects.
   
'''

EXAMPLE = """example:

                 pyriteApp.py -h
                 pyriteApp.py -g
                 pyriteApp.py LosAngels.cfg
  
###################################################################################
"""


def main(argv):
    
    inps = cmdLineParse() 
    templateContents = read_cfg(inps.cfg_file)
    
    if 'process_dir' in templateContents: root_dir = templateContents['process_dir']
    else: root_dir = os.getcwd()
    
    data_source = 'ERA-5 Reanalysis data of ECMWF'
    print('')
    print('---------------- Basic Parameters ---------------------')
    print('Working directory: %s' % root_dir)
    print('Data source : %s' % data_source)
    
    os.chdir(root_dir)
    pyrite_dir = root_dir + '/pyrite'
    era5_dir = root_dir + '/pyrite/ERA5'
    era5_raw_dir = root_dir + '/pyrite/ERA5/raw'
    era5_sar_dir = root_dir + '/pyrite/ERA5/sar'

    # Get the interested data type
    if 'interested_type' in templateContents: interested_type = templateContents['interested_type']
    else: interested_type = 'delay'
    print('Interested data type: %s' % interested_type)
    
    
    if interested_type == 'delay':  inps_type = 'tzd'
    elif interested_type == 'pwv': inps_type = 'wzd'
    
    if not os.path.isdir(pyrite_dir): os.mkdir(pyrite_dir)
    if not os.path.isdir(era5_dir): os.mkdir(era5_dir)
    if not os.path.isdir(era5_raw_dir): os.mkdir(era5_raw_dir)
    if not os.path.isdir(era5_sar_dir): os.mkdir(era5_sar_dir)

    # ------------ Get research time in seconds
    if 'research_time' in templateContents: research_time = templateContents['research_time']
    elif 'research_time_file' in templateContents: 
        research_time_file = templateContents['research_time_file']
        meta = ut.read_attr(research_time_file)
        research_time = meta['CENTER_LINE_UTC']
    print('Research UTC-time (s): %s' % str(research_time))
    print('')
    
    # Get date_list
    date_list = []
    if 'date_list' in templateContents: 
        date_list0 = templateContents['date_list']
        date_list1 = date_list0.split(',')[:]
        for d0 in date_list1: 
            date_list.append(d0)
    
    if 'date_list_txt' in templateContents: 
        date_list_txt = templateContents['date_list_txt']
        date_list2 = np.loadtxt(date_list_txt,dtype=np.str)
        date_list2 = date_list2.tolist()
        for d0 in date_list2: 
            date_list.append(d0)
    
    if 'date_list_file' in templateContents: 
        date_list_file = templateContents['date_list_file']
        date_list3 = ut.read_hdf5(date_list_file, datasetName = 'date')[0]
        date_list3 = date_list3.astype('U13')
        date_list3 = date_list3.tolist()
        for d0 in date_list3: 
            date_list.append(d0)
    
    date_list = set(date_list)
    date_list = list(map(int, date_list))
    date_list = sorted(date_list)
    date_list = list(map(str, date_list))
 
    print('Interested date list:')
    print('')
    for k0 in date_list:
        print(k0)
    
    # ------------ Get model parameters
    if 'elevation_model' in templateContents: elevation_model = templateContents['elevation_model']
    else: elevation_model = 'onn_linear'
    
    if 'bin_numb' in templateContents: bin_numb = templateContents['bin_numb']
    else: bin_numb = 50    
        
    if 'variogram_model' in templateContents: variogram_model = templateContents['variogram_model']
    else: variogram_model = 'spherical'    
        
    if 'max_length' in templateContents: max_length = templateContents['max_length']
    else: max_length  = 250 
          
    # ------------ Get interpolate parameters    
    if 'interp_method'  in templateContents: interp_method = templateContents['interp_method']
    else: interp_method  = 'kriging'  
        
    if 'kriging_points_numb'  in templateContents: kriging_points_numb = templateContents['kriging_points_numb']
    else: kriging_points_numb  = 20
    
    
    # check geometry file
    if 'geometry_file' in templateContents: 
        geometry_file = templateContents['geometry_file']
    else: 
        print('')
        print('geometry_pyrite.h5 will be used as the geometry file.')
        geometry_file = 'geometry_pyrite.h5'
        if not os.path.isfile(geometry_file):
            print('Start to generate geometry_pyrite.h5')
            if 'research_area_file' in templateContents: 
                research_area_file = templateContents['research_area_file'] 
                call_str = 'generate_geometry_pyrite.py --ref ' + research_area_file + ' --resolution 90 -o geometry_pyrite.h5'
                os.system(call_str)
            elif 'research_area' in templateContents: 
                region_area = templateContents['research_area'] 
                call_str = 'generate_geometry_pyrite.py --region ' + region_area + ' --resolution 90 -o geometry_pyrite.h5'
                os.system(call_str)
        else:
            print('geometry_pyrite.h5 exists, skip to generate geometry file.')
            print('')
    # check min_altitude/ max_altitude
    if 'min_altitude' in templateContents: min_altitude = templateContents['min_altitude'] 
    else: min_altitude = -200
    
    if 'max_altitude' in templateContents: max_altitude = templateContents['max_altitude'] 
    else: max_altitude = 5000
        
    # check rescale values
    
    if 'hgt_rescale' in templateContents: hgt_rescale = templateContents['hgt_rescale'] 
    else: hgt_rescale = 10
    
    if 'lalo_rescale' in templateContents: lalo_rescale = templateContents['lalo_rescale'] 
    else: lalo_rescale = 10 
  
    print('----------- Step 1 --------------')
    print('Start to download ERA-5 data...')
    
    txt_download = 'download_datelist.txt'
    generate_datelist_txt(date_list,txt_download)
    
    if 'research_area_file' in templateContents: 
        region_file = templateContents['research_area_file'] 
        call_str = 'download_era5.py --region-file ' + region_file + ' --time ' + research_time + ' --date-txt ' + txt_download
        os.system(call_str)
    elif 'research_area' in templateContents: 
        region_area = templateContents['research_area'] 
        call_str = 'download_era5.py --region ' + region_area + ' --time ' + research_time + ' --date-txt ' + txt_download
        os.system(call_str)     
    
    print('----------- Step 2 --------------')
    print('Start to generate high-resolution tropospheric maps ...' )
    print('Type of the tropospheric products: %s' % interested_type)
    print('Minimum altitude used: %s' % str(min_altitude))
    print('Maximum altitude used: %s' % str(max_altitude))
    print('Height Rescale: %s' % str(hgt_rescale))
    print('lalo rescale: %s' % str(lalo_rescale))
    print('')
        
    print('Elevation model: %s' % elevation_model)
    print('Method used for interpolation: %s' % interp_method)
    if interp_method =='kriging':
        print('Variogram model: %s' % variogram_model)
        print('Number of bins: %s' % str(bin_numb))
        print('Max-length used for variogram-modeling: %s' % str(max_length))
    print('')
    
    
    
    date_generate = []
    for i in range(len(date_list)):
        out0 = era5_sar_dir + '/' + date_list[i] + '_' + inps_type + '.h5'
        if not os.path.isfile(out0):
            date_generate.append(date_list[i])
    print('Total number of data set: %s' % str(len(date_list)))          
    print('Exsit number of data set: %s' % str(len(date_list)-len(date_generate))) 
    print('Number of high-resolution maps need to be interpolated: %s' % str(len(date_generate)))  
    
    if len(date_generate) > 0 :
        txt_generate = 'datelist_generate.txt'
        generate_datelist_txt(date_generate,txt_generate)
        call_str = 'interp_era5_pyrite_list.py ' + txt_generate  + ' ' + geometry_file + ' --type ' + inps_type + ' --method ' + interp_method + '  --kriging-points-numb ' + str(kriging_points_numb) + ' --variogram-model ' + str(variogram_model) + ' --elevation-model ' + str(elevation_model) + ' --max-length ' + str(max_length) + ' --min-altitude ' + str(min_altitude) + ' --max-altitude ' + str(max_altitude) +  ' --bin-numb ' + str(bin_numb) +  ' --hgt-rescale ' + str(hgt_rescale) +  ' --lalo-rescale ' + str(lalo_rescale)

        #call_str = 'interp_era5_pyrite_list.py ' + txt_generate + ' era5_sample_variogramModel.h5 ' + inps.geo_file + ' --type ' + inps.type + ' --method ' + inps.interp_method + '  --kriging-points-numb ' + str(inps.kriging_points_numb) + ' --parallel ' + str(inps.parallelNumb)
        os.system(call_str)
        
        
    print('')
    print('----------- Step 3 --------------')
    print('Start to generate time-series of tropospheric data ...' )
    txt_list = 'date_list.txt'
    generate_datelist_txt(date_list,txt_list)
    call_str = 'generate_timeseries_era5.py --date-txt ' + txt_list + ' --type ' + inps_type
    os.system(call_str)
    
    print('Done.')
    
    #if inps.type =='tzd':
    #    print('')
    #    print('----------- Step 4 --------------')
    #    print('Transfer the atmospheric delays from zenith direction to Los direction ...')
    #    call_str = 'zenith2los_pyrite.py timeseries_pyrite_tzd_aps.h5 ' + geometry_file
    #    os.system(call_str)
    #    print('Done.')
    #    print('')
    #    print('Start to correct InSAR time-series tropospheric delays ...' )
    #    call_str = 'diff_pyrite.py ' + inps.ts_file + ' ' + ' timeseries_pyrite_tzd_aps_los.h5 --add  -o timeseries_pyriteCor.h5'
    #    os.system(call_str)
    #    print('Done.')

    sys.exit(1)
###############################################################

if __name__ == '__main__':
    main(sys.argv[:])

