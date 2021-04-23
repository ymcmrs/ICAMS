#! /usr/bin/env python
###########################################
# Program is part of PyRite               #
# Copyright 2019, Yunmeng Cao             #
# Contact: ymcmrs@gmail.com               #
###########################################
import os
import sys
import numpy as np
import argparse
from pyrite import _utils as ut
import glob


def get_lack_datelist(date_list, date_list_exist):
    date_list0 = []
    for k0 in date_list:
        if k0 not in date_list_exist:
            date_list0.append(k0)
    
    return date_list0


def cmdLineParse():
    parser = argparse.ArgumentParser(description='Extract and calculate useful parameters from raw ERA5 datasets.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('--date-list', dest='date_list', nargs='*',help='date list to extract.')
    parser.add_argument('--date-txt', dest='date_txt',help='date list text to extract.')
    parser.add_argument('--date-file', dest='date_file',help='mintPy formatted h5 file, which contains date infomration.')
    # date-list, date-txt, date-file should provide at least one.
    
    inps = parser.parse_args()
    
    if (not inps.date_list) and (not inps.date_txt) and (not inps.date_file) :
        parser.print_usage()
        sys.exit(os.path.basename(sys.argv[0])+': error: date-list, date-txt, or date-file should provide at least one.')
    
    return inps

###################################################################################################

INTRODUCTION = '''
   Calculate and extract useful parameters from original ERA-5 datasets. 
    
'''

EXAMPLE = '''Examples:

    extract_era5.py --date-list 20180101 20180102
    extract_era5.py --date-txt ~/date_list.txt
    extract_era5.py --date-file timeseries.h5
    
'''    
##################################################################################################   

def main(argv):
    
    inps = cmdLineParse()
    root_dir = os.getcwd()
    
    pyrite_dir = root_dir + '/pyrite'
    era5_dir = pyrite_dir + '/ERA5'
    raw_dir = era5_dir + '/raw'
    
    # Get date list  
    date_list = []
    if inps.date_list: date_list = inps.date_list
        
    if inps.date_txt:
        date_list2 = read_txt_line(inps.date_txt)
        for list0 in date_list2:
            if (list0 not in date_list) and is_number(list0):
                date_list.append(list0)
                
    if inps.date_file: 
        date_list3 = ut.read_hdf5(inps.date_file, datasetName='date')[0]
        #print(date_list3)
        date_list3 = date_list3.astype('U13')
        #print(date_list3)
        for list0 in date_list3:
            if (list0 not in date_list) and ut.is_number(list0):
                date_list.append(list0)
    
    date_list = list(map(int, date_list))
    date_list = sorted(date_list)
    date_list = list(map(str, date_list))
    
    print('Number of the to be extracted ERA-5 dataset: %s' % str(len(date_list)))
    print('Date list of the ERA-5 datasets to be extracted:')
    print('')
        
    cdic = ut.initconst()
    
    # get height at each lat/lon coordinates
    k0 = date_list[0]
    fname0 = glob.glob(raw_dir + '/ERA*' + k0 + '*')[0]
    lvls,latlist,lonlist,gph,tmp,vpr = ut.get_ecmwf('ERA5',fname0,cdic, humidity='Q') 
       
    lonlist = lonlist - 360.0   # change to the SAR formatted (GAMMA/ISCE) longitude  
    meta_geo = ut.read_attr('geometry_era5.h5')
    hei_geo = ut.read_hdf5('geometry_era5.h5',datasetName = 'height')[0]
    X_FIRST = float(meta_geo['X_FIRST'])
    Y_FIRST = float(meta_geo['Y_FIRST'])
    X_STEP = float(meta_geo['X_STEP'])
    Y_STEP = float(meta_geo['Y_STEP'])

    rows = np.round((latlist - Y_FIRST)/Y_STEP)
    cols = np.round((lonlist - X_FIRST)/X_STEP)
    
    rows = rows.astype(int)
    cols = cols.astype(int)
    heis = hei_geo[rows,cols]
    
    row,col = latlist.shape
    # read data
    
    wdel_all = np.zeros((len(date_list),row,col),dtype = np.float32)
    ddel_all = np.zeros((len(date_list),row,col),dtype = np.float32)
    tdel_all = np.zeros((len(date_list),row,col),dtype = np.float32)
    
    Pi_all = np.zeros((len(date_list),row,col),dtype = np.float32)
    Ti_all = np.zeros((len(date_list),row,col),dtype = np.float32)
    Vi_all = np.zeros((len(date_list),row,col),dtype = np.float32)
    
    datasetDict =dict()
    datasetDict['latitude'] = latlist.astype(np.float32)
    datasetDict['longitude'] = lonlist.astype(np.float32)
    datasetDict['height'] = heis.astype(np.float32)
    
    mdd_all = []
    mdw_all = []
    mdt_all = []
    hgt_all = []
    nk0 = np.zeros((len(date_list),))
    
    for i in range(len(date_list)): 
    #for i in range(1): 
        k0 = date_list[i]
        ut.print_progress(i+1, len(date_list), prefix='Date: ', suffix=k0)
        
        fname0 = glob.glob(raw_dir + '/ERA*' + k0 + '*')[0]
        #print(fname0)
        lvls,latlist,lonlist,gph,tmp,vpr = ut.get_ecmwf('ERA5',fname0,cdic, humidity='Q')     
        # Make a height scale
        hgt = np.linspace(cdic['minAltP'], gph.max().round(), cdic['nhgt'])
        # Interpolate pressure, temperature and Humidity of hgt
        [Pi,Ti,Vi] = ut.intP2H(lvls, hgt, gph, tmp, vpr, cdic)
        # Calculate the delays
        [DDry,DWet] = ut.PTV2del(Pi,Ti,Vi,hgt,cdic)
        hgt0 = hgt.copy()
        dd0 = DDry.copy()
        dw0 = DWet.copy()
        #print(hgt0.shape)
        #print(dd0.shape)
        #print(dw0.shape)
        idx0 = np.where(((-100<hgt0) & (hgt0 < 5000)))
        hgt00 = hgt0[idx0]
        hgt0 = list(hgt0)
        
        dd0 = DDry.copy()
        dw0 = DWet.copy()
        dt0 = dd0 + dw0
     
        nk = len(hgt00)
        nk0[i] = nk
        mdd = np.zeros((nk,),dtype = np.float32)
        mdw = np.zeros((nk,),dtype = np.float32)
        mdt = np.zeros((nk,),dtype = np.float32)
        
        for i in range(nk):
            mdd[i] = np.mean(dd0[:,:,hgt0.index(hgt00[i])])
            mdw[i] = np.mean(dw0[:,:,hgt0.index(hgt00[i])])
            mdt[i] = np.mean(dt0[:,:,hgt0.index(hgt00[i])])
        mdd_all.append(mdd)
        mdw_all.append(mdw)
        mdt_all.append(mdt)
        hgt_all.append(hgt00)
        # Get delay samples at each coords (lat,lon)
        wdels = ut.get_delay_sample(hgt, DWet,heis)
        ddels = ut.get_delay_sample(hgt, DDry,heis)
        
        Pis = ut.get_delay_sample(hgt, Pi, heis)
        Tis = ut.get_delay_sample(hgt, Ti, heis)
        Vis = ut.get_delay_sample(hgt, Vi, heis)
        
        wdel_all[i,:,:] = wdels
        ddel_all[i,:,:] = ddels
        tdel_all[i,:,:] = wdels + ddels
        
        Pi_all[i,:,:] = Pis
        Ti_all[i,:,:] = Tis
        Vi_all[i,:,:] = Vis
            
    
    mdd_all0 = np.zeros((len(date_list),int(max(nk0.flatten()))),dtype = np.float32)
    mdw_all0 = np.zeros((len(date_list),int(max(nk0.flatten()))),dtype = np.float32)
    mdt_all0 = np.zeros((len(date_list),int(max(nk0.flatten()))),dtype = np.float32)
    hgt_all0 = np.zeros((len(date_list),int(max(nk0.flatten()))),dtype = np.float32)
    
    for i in range(len(date_list)):
        mdd_all0[i,0:int(nk0[i])] = mdd_all[i]
        mdw_all0[i,0:int(nk0[i])] = mdw_all[i] 
        mdt_all0[i,0:int(nk0[i])] = mdt_all[i] 
        hgt_all0[i,0:int(nk0[i])] = hgt_all[i]
    
    datasetDict['wetDelay_all'] = mdw_all0
    datasetDict['dryDelay_all'] = mdd_all0
    datasetDict['totDelay_all'] = mdt_all0
    datasetDict['height_all'] = hgt_all0
    
    datasetDict['wetDelay'] = wdel_all
    datasetDict['dryDelay'] = ddel_all
    datasetDict['totDelay'] = tdel_all
    
    datasetDict['Pi'] = Pi_all
    datasetDict['Ti'] = Ti_all
    datasetDict['Vi'] = Vi_all
    
    date_list = np.asarray(date_list,dtype = np.string_)
    datasetDict['date'] = date_list
    
    meta = {}
    meta['WIDTH'] = str(col)
    meta['LENGTH'] = str(row)
    meta['FILE_TYPE'] = 'ERA5'
    
    
    #print(heis.shape)
    ut.write_h5(datasetDict, 'era5_sample.h5', metadata=meta, ref_file=None, compression=None)
   
    datasetDict0 =dict() 
    datasetDict0['timeseries'] = ddel_all
    datasetDict0['date'] = date_list
    ut.write_h5(datasetDict0, 'era5_drydelay_sample.h5', metadata=meta, ref_file=None, compression=None)
    
    datasetDict0 =dict() 
    datasetDict0['timeseries'] = wdel_all
    datasetDict0['date'] = date_list
    ut.write_h5(datasetDict0, 'era5_wetdelay_sample.h5', metadata=meta, ref_file=None, compression=None)
    
    datasetDict0 =dict() 
    datasetDict0['timeseries'] = tdel_all
    datasetDict0['date'] = date_list
    ut.write_h5(datasetDict0, 'era5_totdelay_sample.h5', metadata=meta, ref_file=None, compression=None)

    print('Done.')     
    sys.exit(1)        

if __name__ == '__main__':
    main(sys.argv[1:])











