############################################################
# Program is part of PyRite                                #
# Copyright 2019 Yunmeng Cao                               #
# Contact: ymcmrs@gmail.com                                #
############################################################
# This program is modified from some sub-scripts of PyAPS                      
# Copyright 2012, by the California Institute of Technology
# Contact: earthdef@gps.caltech.edu                        
# Modified by A. Benoit and R. Jolivet 2019                
# Contact: insar@geologie.ens.fr                           

import configparser as ConfigParser
import sys
import os.path
import cdsapi
# disable InsecureRequestWarning message from cdsapi
import urllib3
urllib3.disable_warnings()

import scipy.interpolate as intp
import scipy.integrate as intg
import scipy.spatial as ss

from scipy.optimize import leastsq
from scipy.stats.stats import pearsonr

from pyrite import elevation_models
from pykrige import OrdinaryKriging
from pykrige import variogram_models

from scipy.interpolate import RegularGridInterpolator as RGI
from scipy import interpolate

import pygrib
import numpy as np
import h5py

import pyproj

from pyrite.geoid import GeoidHeight as gh # calculating geoid height

######Set up variables in model.cfg before using
dpath = os.path.dirname(__file__)
config = ConfigParser.RawConfigParser(delimiters='=')
config.read('%s/user.cfg'%(dpath))

variogram_dict = {'linear': variogram_models.linear_variogram_model,
                      'power': variogram_models.power_variogram_model,
                      'gaussian': variogram_models.gaussian_variogram_model,
                      'spherical': variogram_models.spherical_variogram_model,
                      'exponential': variogram_models.exponential_variogram_model,
                      'hole-effect': variogram_models.hole_effect_variogram_model}

def ECMWFdload(bdate,hr,filedir,model='ERA5',datatype='fc',humidity='Q',snwe=None,flist=None):
    '''
    ECMWF-ERA5 data downloading.

    Args:
        * bdate     : date to download (str)
        * hr        : hour to download (str)
        * filedir   : files directory (str)
        * model     : weather model ('ERA5', 'ERAINT', 'HRES')
        * datatype  : reanalysis data type (an)
        * snwe      : area extent (tuple of int)
        * humidity  : humidity
    '''
    
    #-------------------------------------------
    # Initialize

    # Check data
    assert model in ('ERA5', 'ERAINT', 'HRES'), 'Unknown model for ECMWF: {}'.format(model)

    # Infos for downloading
    if model in 'ERA5':
        print('Download ERA-5 re-analysis-datasets from the latest ECMWF platform:')
        print('https://cds.climate.copernicus.eu/api/v2')
    else:
        print('WARNING: you are downloading from the old ECMWF platform. '
              'ERA-Interim is deprecated, use ERA-5 instead.')
    #-------------------------------------------
    # Define parameters

    # Humidity
    assert humidity in ('Q','R'), 'Unknown humidity field for ECMWF'
    
    if humidity in 'Q':
        humidparam = 'specific_humidity'
    elif humidity in 'R':
        humidparam = 'relative_humidity'

    #-------------------------------------------
    # file name
    if not flist:
        flist = []
        for k in range(len(bdate)):
            day = bdate[k]
            # Only for ERA-5
            fname = os.path.join(filedir, 'ERA-5_{}_{}.grb'.format(day, hr))
            flist.append(fname)

    # Iterate over dates
    for k in range(len(bdate)):
        day = bdate[k]
        fname = flist[k]
        
        #-------------------------------------------
        # Request for CDS API client (new ECMWF platform, for ERA-5)    
        url = 'https://cds.climate.copernicus.eu/api/v2'
        key = config.get('CDS', 'key')

        # Contact the server
        c = cdsapi.Client(url=url, key=key)

        # Pressure levels
        pressure_lvls = ['1','2','3','5','7','10','20','30','50', 
                            '70','100','125','150','175','200','225',
                            '250','300','350','400','450','500','550',
                            '600','650','700','750','775','800','825',
                            '850','875','900','925','950','975','1000']

        # Dictionary
        indict = {'product_type'   :'reanalysis',
                  'format'         :'grib',
                  'variable'       :['geopotential','temperature','{}'.format(humidparam)],
                  'pressure_level' : pressure_lvls,
                  'year'           :'{}'.format(day[0:4]),
                  'month'          :'{}'.format(day[4:6]),
                  'day'            :'{}'.format(day[6:8]),
                  'time'           :'{}:00'.format(hr)}

        # download a geographical area subset
        if snwe is not None:
            s, n, w, e = snwe
            indict['area'] = '/'.join(['{:.2f}'.format(x) for x in [n, w, s, e]])

        # Assert grib file not yet downloaded
        if not os.path.exists(fname):
            print('Downloading %d of %d: %s '%(k+1,len(bdate), fname))
            print(indict)

            # Make the request
            c.retrieve('reanalysis-{}-pressure-levels'.format(model.lower()),indict,target=fname)

    return flist

######################## calculate delay #############################
def initconst():
    '''Initialization of various constants needed for computing delay.

    Args:
        * None

    Returns:
        * constdict (dict): Dictionary of constants'''
    constdict = {}
    constdict['k1'] = 0.776   #(K/Pa)
    constdict['k2'] = 0.716   #(K/Pa)
    constdict['k3'] = 3750    #(K^2.Pa)
    constdict['g'] = 9.81     #(m/s^2)
    constdict['Rd'] = 287.05  #(J/Kg/K)
    constdict['Rv'] = 461.495 #(J/Kg/K)
    constdict['mma'] = 29.97  #(g/mol)
    constdict['mmH'] = 2.0158 #(g/mol)
    constdict['mmO'] = 16.0   #(g/mol)
    constdict['Rho'] = 1000.0 #(kg/m^3)

    constdict['a1w'] = 611.21 # hPa
    constdict['a3w'] = 17.502 #
    constdict['a4w'] = 32.19  # K
    constdict['a1i'] = 611.21 # hPa
    constdict['a3i'] = 22.587 #
    constdict['a4i'] = -0.7   # K
    constdict['T3'] = 273.16  # K
    constdict['Ti'] = 250.16  # K
    constdict['nhgt'] = 300   # Number of levels for interpolation (OLD 151)
    constdict['minAlt'] = -200.0
    constdict['maxAlt'] = 50000.0
    constdict['minAltP'] = -200.0
    constdict['Rearth'] = 6371 # km
    return constdict    

def cc_era(tmp,cdic):
    '''Clausius Clayperon law used by ERA Interim.

    Args:
        * tmp  (np.array) : Temperature.
        * cdic (dict)     : Dictionary of constants

    Returns:
        * esat (np.array) : Water vapor saturation partial pressure.'''


    a1w = cdic['a1w']
    a3w = cdic['a3w']
    a4w = cdic['a4w']
    a1i = cdic['a1i']
    a3i = cdic['a3i']
    a4i = cdic['a4i']
    T3  = cdic['T3']
    Ti  = cdic['Ti'] 

    esatw = a1w*np.exp(a3w*(tmp-T3)/(tmp-a4w))
    esati = a1i*np.exp(a3i*(tmp-T3)/(tmp-a4i))
    esat = esati.copy()
    for k in range(len(tmp)):
        if (tmp[k] >= T3):
            esat[k] = esatw[k]
        elif (tmp[k] <= Ti):
            esat[k] = esati[k]
        else:
            wgt = (tmp[k]-Ti)/(T3-Ti)
            esat[k] = esati[k] + (esatw[k]-esati[k])*wgt*wgt

    return esat

########Read in ERA data from a given ERA Interim file##################

def get_ecmwf(model,fname,cdic, humidity='Q',minlat= -90,maxlat = 90,minlon = 0,maxlon = 360,verbose=False):
    '''Read data from ERA Interim, ERA-5 or HRES grib file. Note that Lon values should be between [0-360].
    Modified by A. Benoit, January 2019.

    Args:
        * model       (str):  Model used (ERA5, ERAINT or HRES)
        * fname       (str):  Path to the grib file
        * minlat (np.float):  Minimum latitude
        * maxlat (np.float):  Maximum latitude
        * minlon (np.float):  Minimum longitude
        * maxlon (np.float):  Maximum longitude
        * cdic   (np.float):  Dictionary of constants
    
    Kwargs:
        * humidity    (str): Specific ('Q') or relative humidity ('R').

    Returns:
        * lvls   (np.array): Pressure levels
        * latlist(np.array): Latitudes of the stations
        * lonlist(np.array): Longitudes of the stations
        * gph    (np.array): Geopotential height
        * tmp    (np.array): Temperature
        * vpr    (np.array): Vapor pressure

    .. note::
        Uses cc_era by default.
        '''

    assert humidity in ('Q','R'), 'Undefined humidity field in get_era.'
    assert model in ('ERA5', 'ERAINT','HRES'), 'Model not recognized.'
    if verbose:
        print('PROGRESS: READING GRIB FILE')
    if model in 'HRES':
        if verbose:
            print('INFO: USING PRESSURE LEVELS OF HRES DATA')
        lvls = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 150, 
                         200, 250, 300, 400, 500, 600, 700,
                         800, 850, 900, 925, 950, 1000])
    else:
        if verbose:
            print('INFO: USING PRESSURE LEVELS OF ERA-INT OR ERA-5 DATA')
        lvls = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 
                         200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775,
                         800, 825, 850, 875, 900, 925, 950, 975, 1000])
    nlvls = len(lvls)

    alpha = cdic['Rv']/cdic['Rd']
    gphind = np.arange(nlvls)*3

    grbs = pygrib.open(fname)
    grbs.seek(gphind[0])
    grb = grbs.read(1)[0]
    lats,lons = grb.latlons()
    if model == 'ERA5':
        lons[lons < 0.] += 360.
    g = cdic['g']
    
    mask = ((lats > minlat) & (lats < maxlat)) & ((lons > minlon) & (lons < maxlon))
    #extrqct indices 
    uu = [i for i in list(range(np.shape(mask)[0])) if any(mask[i,:])]
    vv = [j for j in list(range(np.shape(mask)[1])) if any(mask[:,j])]
    
    latlist = lats[uu,:][:,vv]
    lonlist = lons[uu,:][:,vv]
    #nstn = len(lat.flatten())
    nlat, nlon = latlist.shape

    ####Create arrays for 3D storage
    gph = np.zeros((nlvls, nlat, nlon))   #Potential height
    tmp = gph.copy()                  #Temperature
    vpr = gph.copy()                  #Vapor pressure
    if verbose:
        print('INFO: IMAGE DIMENSIONS: {} LATITUDES AND {} LONGITUDES'.format(nlat, nlon))

    lvls = 100.0*lvls              #Conversion to absolute pressure
    for i in range(nlvls):
        grbs.seek(gphind[i])   #Reading potential height.
        grb = grbs.read(3)
        gph[i,:,:] = grb[0].values[uu,:][:,vv]/g

        #Reading temperature
        temp = grb[1].values[uu,:][:,vv]
        tmp[i,:,:] = temp

        if humidity in ('R'):   # Relative humidity
            esat = cc_era(temp,cdic)       
            temp = grb[2].values[uu,:][:,vv]/100.0
            vpr[i,:,:] = temp*esat
            
        elif humidity in ('Q'):
            val = grb[2].values  #Specific humidity
            temp = grb[2].values[uu,:][:,vv]
            vpr[i,:,:] = temp*lvls[i]*alpha/(1+(alpha - 1)*temp)
            
        else:
            assert 1==0, 'Undefined Humidity in get_ecmwf().'     

    return lvls,latlist,lonlist,gph,tmp,vpr
###############Completed GET_ECMWF########################################



###############Completed the list of constants################

##########Interpolating to heights from Pressure levels###########
def intP2H(lvls,hgt,gph,tmp,vpr,cdic,verbose=False):
    '''Interpolates the pressure level data to altitude.

    Args:
        * lvls (np.array) : Pressure levels.
        * hgt  (np.array) : Height values for interpolation.
        * gph  (np.array) : Geopotential height.
        * tmp  (np.array) : Temperature.
        * vpr  (np.array) : Vapor pressure.
        * cdic (dict)     : Dictionary of constants.

    .. note::
        gph,tmp,vpr are of size (nstn,nlvls).

    Returns:
        * Presi (np.array) : Interpolated pressure.
        * Tempi (np.array) : Interpolated temperature.
        * Vpri  (np.array) : Interpolated vapor pressure.

    .. note::
        Cubic splines are used to convert pressure level data to height level data.'''

    minAlt = cdic['minAlt']      #Hardcoded parameter.
    maxAlt = gph.max().round()

    if verbose:
        print('PROGRESS: INTERPOLATING FROM PRESSURE TO HEIGHT LEVELS')
    nlat = gph.shape[1]           #Number of stations
    nlon = gph.shape[2]
    nhgt = len(hgt)               #Number of height points
    Presi = np.zeros((nlat,nlon,nhgt))
    Tempi = np.zeros((nlat,nlon,nhgt))
    Vpri  = np.zeros((nlat,nlon,nhgt))

    for i in range(nlat):
        for j in range(nlon):
            temp = gph[:,i,j]       #Obtaining height values
            hx = temp.copy()
            sFlag = False
            eFlag = False
            if (hx.min() > minAlt):          #Add point at start
                sFlag = True
                hx = np.concatenate((hx,[minAlt-1]),axis = 0) #changed from 1 to 0 (-1 should also work), CL

            if (hx.max() < maxAlt):        #Add point at end
                eFlag = True
                hx = np.concatenate(([maxAlt+1],hx),axis=0) #changed from 1 to 0 (-1 should also work), CL

            hx = -hx             #Splines needs monotonically increasing.    
    
            hy = lvls.copy()     #Interpolating pressure values
            if (sFlag == True):
                val = hy[-1] +(hx[-1] - hx[-2])* (hy[-1] - hy[-2])/(hx[-2]-hx[-3])
                hy = np.concatenate((hy,[val]),axis=0) #changed from 1 to 0 (-1 should also work), CL
            if (eFlag == True):
                val = hy[0] - (hx[0] - hx[1]) * (hy[0] - hy[1])/(hx[1]-hx[2]) 
                hy = np.concatenate(([val],hy),axis=0) #changed from 1 to 0 (-1 should also work), CL

            tck = intp.interp1d(hx,hy,kind='cubic')
            temp = tck(-hgt)      #Again negative for consistency with hx
            Presi[i,j,:] = temp.copy()
            del temp

            temp = tmp[:,i,j]        #Interpolating temperature
            hy = temp.copy()
            if (sFlag == True):
                val = hy[-1] +(hx[-1] - hx[-2])* (hy[-1] - hy[-2])/(hx[-2]-hx[-3])
                hy = np.concatenate((hy,[val]),axis=0) #changed from 1 to 0 (-1 should also work), CL
            if (eFlag == True):
                val = hy[0] - (hx[0] - hx[1]) * (hy[0] - hy[1])/(hx[1]-hx[2])
                hy = np.concatenate(([val],hy),axis=0) #changed from 1 to 0 (-1 should also work), CL

            tck = intp.interp1d(hx,hy,kind='cubic')
            temp = tck(-hgt)
            Tempi[i,j,:] = temp.copy()
            del temp
    
            temp = vpr[:,i,j]          #Interpolating vapor pressure
            hy = temp.copy()
            if (sFlag == True):
                val = hy[-1] +(hx[-1] - hx[-2])* (hy[-1] - hy[-2])/(hx[-2]-hx[-3])
                hy = np.concatenate((hy,[val]),axis=0) #changed from 1 to 0 (-1 should also work), CL
            if (eFlag == True):
                val = hy[0] - (hx[0] - hx[1]) * (hy[0] - hy[1])/(hx[1]-hx[2])
                hy = np.concatenate(([val],hy),axis=0) #changed from 1 to 0 (-1 should also work), CL
            tck = intp.interp1d(hx,hy,kind='cubic')
            temp = tck(-hgt)
            Vpri[i,j,:] = temp.copy()
            del temp

    ## Save into files for plotting (test for era5 interpolation)
    #np.savetxt('/home/angel/Tests/test_era5interpolation/Outputs/alt_20141023_20141210_era5.txt', gph, fmt=' '.join(['%s']*380))       # Save altitude (of lvls)
    #np.savetxt('/home/angel/Tests/test_era5interpolation/Outputs/lvls_20141023_20141210_era5.txt', lvls.T, fmt='%s')                   # Save pressure
    #np.savetxt('/home/angel/Tests/test_era5interpolation/Outputs/tmp_20141023_20141210_era5.txt', tmp, fmt=' '.join(['%s']*380))       # Save temperature
    #np.savetxt('/home/angel/Tests/test_era5interpolation/Outputs/vpr_20141023_20141210_era5.txt', vpr, fmt=' '.join(['%s']*380))       # Save vapor
    #np.savetxt('/home/angel/Tests/test_era5interpolation/Outputs/Alti_20141023_20141210_era5.txt', hgt.T, fmt='%s')                    # Save interpo altitude
    #np.savetxt('/home/angel/Tests/test_era5interpolation/Outputs/Presi_20141023_20141210_era5.txt', Presi.T, fmt=' '.join(['%s']*380)) # Save interpo pressure
    #np.savetxt('/home/angel/Tests/test_era5interpolation/Outputs/Tempi_20141023_20141210_era5.txt', Tempi.T, fmt=' '.join(['%s']*380)) # Save interpo temperature
    #np.savetxt('/home/angel/Tests/test_era5interpolation/Outputs/Vpri_20141023_20141210_era5.txt', Vpri.T, fmt=' '.join(['%s']*380))   # Save interpo vapor

    return Presi,Tempi,Vpri
###########Completed interpolation to height levels #####################


###########Computing the delay function ###############################
def PTV2del(Presi,Tempi,Vpri,hgt,cdict,verbose=False):
    '''Computes the delay function given Pressure, Temperature and Vapor pressure.

    Args:
        * Presi (np.array) : Pressure at height levels.
        * Tempi (np.array) : Temperature at height levels.
        * Vpri  (np.array) : Vapor pressure at height levels.
        * hgt   (np.array) : Height levels.
        * cdict (np.array) : Dictionary of constants.

    Returns:
        * DDry2 (np.array) : Dry component of atmospheric delay.
        * DWet2 (np.array) : Wet component of atmospheric delay.

    .. note::
        Computes refractive index at each altitude and integrates the delay using cumtrapz.'''

    if verbose:
        print('PROGRESS: COMPUTING DELAY FUNCTIONS')
    nhgt = len(hgt)            #Number of height points
    nlat = Presi.shape[0]        #Number of stations
    nlon = Presi.shape[1]
    WonT = Vpri/Tempi
    WonT2 = WonT/Tempi

    k1 = cdict['k1']
    Rd = cdict['Rd']
    Rv = cdict['Rv']
    k2 = cdict['k2']
    k3 = cdict['k3']
    g = cdict['g']

    #Dry delay
    DDry2 = np.zeros((nlat,nlon,nhgt))
    DDry2[:,:,:] = k1*Rd*(Presi[:,:,:] - Presi[:,:,-1][:,:,np.newaxis])*1.0e-6/g

    #Wet delay
    S1 = intg.cumtrapz(WonT,x=hgt,axis=-1)
    val = 2*S1[:,:,-1]-S1[:,:,-2]
    val = val[:,:,None]
    S1 = np.concatenate((S1,val),axis=-1)
    del WonT

    S2 = intg.cumtrapz(WonT2,x=hgt,axis=-1)
    val = 2*S2[:,:,-1]-S2[:,:,-2]
    val = val[:,:,None]
    S2 = np.concatenate((S2,val),axis=-1)
    DWet2 = -1.0e-6*((k2-k1*Rd/Rv)*S1+k3*S2)
    
    for i in range(nlat):
        for j in range(nlon):
            DWet2[i,j,:]  = DWet2[i,j,:] - DWet2[i,j,-1]

    return DDry2,DWet2

############# consider LOS direction & horizontal vartiation ##########
def PTV2del_Imp(Presi,Tempi,Vpri,hgt,cdict,verbose=False):
    '''Computes the delay function given Pressure, Temperature and Vapor pressure.

    Args:
        * Presi (np.array) : Pressure at height levels.
        * Tempi (np.array) : Temperature at height levels.
        * Vpri  (np.array) : Vapor pressure at height levels.
        * hgt   (np.array) : Height levels.
        * cdict (np.array) : Dictionary of constants.

    Returns:
        * DDry2 (np.array) : Dry component of atmospheric delay.
        * DWet2 (np.array) : Wet component of atmospheric delay.

    .. note::
        Computes refractive index at each altitude and integrates the delay using cumtrapz.'''

    if verbose:
        print('PROGRESS: COMPUTING DELAY FUNCTIONS')
    nhgt = len(hgt)            #Number of height points
    nlat = Presi.shape[0]        #Number of stations
    nlon = Presi.shape[1]
    WonT = Vpri/Tempi
    WonT2 = WonT/Tempi

    k1 = cdict['k1']
    Rd = cdict['Rd']
    Rv = cdict['Rv']
    k2 = cdict['k2']
    k3 = cdict['k3']
    g = cdict['g']

    #Dry delay
    DDry2 = np.zeros((nlat,nlon,nhgt))
    DDry2[:,:,:] = k1*Rd*(Presi[:,:,:] - Presi[:,:,-1][:,:,np.newaxis])*1.0e-6/g

    #Wet delay
    S1 = intg.cumtrapz(WonT,x=hgt,axis=-1)
    val = 2*S1[:,:,-1]-S1[:,:,-2]
    val = val[:,:,None]
    S1 = np.concatenate((S1,val),axis=-1)
    del WonT

    S2 = intg.cumtrapz(WonT2,x=hgt,axis=-1)
    val = 2*S2[:,:,-1]-S2[:,:,-2]
    val = val[:,:,None]
    S2 = np.concatenate((S2,val),axis=-1)
    DWet2 = -1.0e-6*((k2-k1*Rd/Rv)*S1+k3*S2)
    
    for i in range(nlat):
        for j in range(nlon):
            DWet2[i,j,:]  = DWet2[i,j,:] - DWet2[i,j,-1]

    return DDry2,DWet2


## 3D interpolate
def make3dintp(Delfn,lonlist,latlist,hgt,hgtscale):
    '''Returns a 3D interpolation function that can be used to interpolate using llh coordinates.

    Args:
        * Delfn    (np.array) : Array of delay values.
        * lonlist  (np.array) : Array of station longitudes.
        * latlist  (np.array) : Array of station latitudes.
        * hgt      (np.array) : Array of height levels.
        * hgtscale (np.float) : Height scale factor for interpolator.

    Returns:
        * fnc  (function) : 3D interpolation function.

    .. note::
        We currently use the LinearNDInterpolator from scipy.
        '''
    ##Delfn   = Ddry + Dwet. Delay function.
    ##lonlist = list of lons for stations. / x
    ##latlist = list of lats for stations. / y
    nstn = Delfn.shape[0]
    nhgt = Delfn.shape[1]
    xyz = np.zeros((nstn*nhgt,3))
    Delfn = np.reshape(Delfn,(nstn*nhgt,1))
    print('3D interpolation')
    count = 0
    for m in range(nstn):
        for n in range(nhgt):
            xyz[count,0] = lonlist[m]
            xyz[count,1] = latlist[m]
            xyz[count,2] = hgt[n]/hgtscale     #For same grid spacing as lat/lon
            count += 1

    #xyz[:,2] = xyz[:,2] #+ 1e-30*np.random.rand((nstn*nhgt))/hgtscale #For unique Delaunay    
    del latlist
    del lonlist
    del hgt
    if verbose:
        print('PROGRESS: BUILDING INTERPOLATION FUNCTION')
    fnc = intp.LinearNDInterpolator(xyz,Delfn)

    return fnc
###################  Computing the delay samples: lats, lons, heights ##################

def get_delay_sample(height_level, delay_level, heigh_samples):
    '''Computes the delay samples at a given altitude, given [heigh_level, delay_level]

    Args:
        * hgt   2D (np.array) : Height levels.
        * delay 2D (np.array) : delay levels.
        * hgts  1D (np.array) : altitude sample at each lon/lat

    Returns:
        * dels  1D (np.array) : delay samples at the corresponding hgts
'''    
    #nstn = delay_level.shape[0]   # samples of horizontal level  (lat, lon)
    #nhgt = delay_level.shape[1]   # samples of vertical level    (altitude)
    #delay_sample = np.zeros((nstn,), dtype = np.float32)
    
    r0 = delay_level.shape[0]
    c0 = delay_level.shape[1]
    
    delay_sample = np.zeros((r0,c0), dtype = np.float32)
    
    for i in range(r0):
        for j in range(c0):
            hy = delay_level[i,j,:]
            hx = height_level
            tck = intp.interp1d(hx,hy,kind='cubic')
            delay_sample[i,j] = tck(heigh_samples[i,j])
            
    
    #for i in range(nstn):
    #    hy = delay_level[i,:]
    #    hx = height_level[i,:]
    #    tck = intp.interp1d(hx,hy,kind='cubic')
    #    delay_sample[i] = tck(heigh_samples[i])
    
    return delay_sample

def read_gamma_par(inFile, task, keyword):
    if task == "read":
        f = open(inFile, "r")
        while 1:
            line = f.readline()
            if not line: break
            if line.count(keyword) == 1:
                strtemp = line.split(":")
                value = strtemp[1].strip()
                return value
        print("Keyword " + keyword + " doesn't exist in " + inFile)
        f.close

def read_attr(fname):
    # read hdf5
    with h5py.File(fname, 'r') as f:
        atr = dict(f.attrs)
        
    return atr

def read_hdf5(fname, datasetName=None, box=None):
    # read hdf5
    with h5py.File(fname, 'r') as f:
        data = f[datasetName][:]
        atr = dict(f.attrs)
        
    return data, atr

def read_txt_line(txt):
    # Open the file with read only permit
    f = open(txt, "r")
    # use readlines to read all lines in the file
    # The variable "lines" is a list containing all lines in the file
    lines = f.readlines()
    lines0 = [line.strip() for line in lines] 
    f.close()
    
    # remove empty lines
    lines_out = [] 
    for line in lines0:
        if not len(line) ==0:
            lines_out.append(line)
    return lines_out

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False  


def write_h5(datasetDict, out_file, metadata=None, ref_file=None, compression=None):

    if os.path.isfile(out_file):
        print('delete exsited file: {}'.format(out_file))
        os.remove(out_file)

    print('create HDF5 file: {} with w mode'.format(out_file))
    dt = h5py.special_dtype(vlen=np.dtype('float64'))

    
    with h5py.File(out_file, 'w') as f:
        for dsName in datasetDict.keys():
            data = datasetDict[dsName]
            ds = f.create_dataset(dsName,
                              data=data,
                              compression=compression)
        
        for key, value in metadata.items():
            f.attrs[key] = str(value)
            #print(key + ': ' +  value)
    print('finished writing to {}'.format(out_file))
        
    return out_file  

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

def print_progress(iteration, total, prefix='calculating:', suffix='complete', decimals=1, barLength=50, elapsed_time=None):
    """Print iterations progress - Greenstick from Stack Overflow
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int) 
        barLength   - Optional  : character length of bar (Int) 
        elapsed_time- Optional  : elapsed time in seconds (Int/Float)
    
    Reference: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    if elapsed_time:
        sys.stdout.write('%s [%s] %s%s    %s    %s secs\r' % (prefix, bar, percents, '%', suffix, int(elapsed_time)))
    else:
        sys.stdout.write('%s [%s] %s%s    %s\r' % (prefix, bar, percents, '%', suffix))
    sys.stdout.flush()
    if iteration == total:
        print("\n")

    '''
    Sample Useage:
    for i in range(len(dateList)):
        print_progress(i+1,len(dateList))
    '''
    return


def get_dataNames(FILE):
    with h5py.File(FILE, 'r') as f:
        dataNames = []
        for k0 in f.keys():
            dataNames.append(k0)
    return dataNames

def get_bounding_box(meta):
    """Get lat/lon range (roughly), in the same order of data file
    lat0/lon0 - starting latitude/longitude (first row/column)
    lat1/lon1 - ending latitude/longitude (last row/column)
    """
    length, width = int(meta['LENGTH']), int(meta['WIDTH'])
    if 'Y_FIRST' in meta.keys():
        # geo coordinates
        lat0 = float(meta['Y_FIRST'])
        lon0 = float(meta['X_FIRST'])
        lat_step = float(meta['Y_STEP'])
        lon_step = float(meta['X_STEP'])
        lat1 = lat0 + lat_step * (length - 1)
        lon1 = lon0 + lon_step * (width - 1)
    else:
        # radar coordinates
        lats = [float(meta['LAT_REF{}'.format(i)]) for i in [1,2,3,4]]
        lons = [float(meta['LON_REF{}'.format(i)]) for i in [1,2,3,4]]
        lat0 = np.mean(lats[0:2])
        lat1 = np.mean(lats[2:4])
        lon0 = np.mean(lons[0:3:2])
        lon1 = np.mean(lons[1:4:2])
    return lat0, lat1, lon0, lon1

def get_lat_lon(meta):
    """Get 2D array of lat and lon from metadata"""
    length, width = int(meta['LENGTH']), int(meta['WIDTH'])
    lat0, lat1, lon0, lon1 = get_bounding_box(meta)
    if 'Y_FIRST' in meta.keys():
        lat, lon = np.mgrid[lat0:lat1:length*1j, lon0:lon1:width*1j]
    else:
        lat, lon = get_lat_lon_rdc(meta)
    return lat, lon

def get_lat_lon_rdc(meta):
    """Get 2D array of lat and lon from metadata"""
    length, width = int(meta['LENGTH']), int(meta['WIDTH'])
    lats = [float(meta['LAT_REF{}'.format(i)]) for i in [1,2,3,4]]
    lons = [float(meta['LON_REF{}'.format(i)]) for i in [1,2,3,4]]

    lat = np.zeros((length,width),dtype = np.float32)
    lon = np.zeros((length,width),dtype = np.float32)

    for i in range(length):
        for j in range(width):
            lat[i,j] = lats[0] + j*(lats[1] - lats[0])/width + i*(lats[2] - lats[0])/length
            lon[i,j] = lons[0] + j*(lons[1] - lons[0])/width + i*(lons[2] - lons[0])/length
            
    return lat, lon

def get_geoid(lat,lon):
    
    row,col = lat.shape
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    lat0 = lat.flatten()
    lon0 = lon.flatten()  
    obj = gh()
    geoidH = []
    for i in range(len(lat0)):
        N0 = gh.get(obj, lat0[i], lon0[i], cubic=True)
        geoidH.append(N0)    
    geoidH = np.asarray(geoidH)
    geoidH = geoidH.reshape(row,col)
    return geoidH

def get_geoid_point(lat,lon):
    obj = gh()
    geoidH = []
    N0 = gh.get(obj, lat, lon, cubic=True)
    return N0

def lla_to_ecef(lat, lon, alt):
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)
    return x, y, z

def ecef_to_lla(X, Y, Z):
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    lon, lat, alt = pyproj.transform(ecef, lla, X, Y, Z, radians=False)
    return lon, lat, alt

def get_uLOS(SatCoord,GroundLLH):
    # GroundLLH = (lat, lon, alt)
    lat, lon, alt = GroundLLH
    
    sX,sY,sZ = SatCoord
    gX,gY,gZ = lla_to_ecef(lat, lon, alt)
    
    uLOS0 = ((sX - gX),(sY - gY),(sZ - gZ))
    dLOS = np.sqrt((sX - gX)**2 + (sY - gY)**2 + (sZ - gZ)**2)
    uLOS = uLOS0/dLOS
    
    return uLOS,uLOS0

def latlon2rgaz_coarse(lat0,lon0,attr):
    
    width = int(attr['WIDTH'])
    length = int(attr['LENGTH'])
    ref_lat1 = float(attr['LAT_REF1'])
    ref_lat2 = float(attr['LAT_REF2'])
    ref_lat3 = float(attr['LAT_REF3'])
    ref_lon1 = float(attr['LON_REF1'])
    ref_lon2 = float(attr['LON_REF2'])
    ref_lon3 = float(attr['LON_REF3'])
    dl_Lat =(ref_lat3 - ref_lat1)/length 
    dw_Lat = (ref_lat2 - ref_lat1)/width 
    
    dl_Lon = (ref_lon3 - ref_lon1)/length 
    dw_Lon = (ref_lon2 - ref_lon1)/width 
    
    A = np.zeros((2,2))
    A[0,:] = [dl_Lat,dw_Lat]
    A[1,:] = [dl_Lon,dw_Lon]
    
    
    L = [(lat0 - ref_lat1), (lon0 - ref_lon1)]
    L =np.asarray(L)
    L = L.reshape((2,1))
    
    kk = np.dot(np.linalg.inv(A),L)
    #print(kk)
    azimuth0 = round(kk[0,0] + 1) 
    range0 = round(kk[1,0] + 1) 
    
    return range0,azimuth0

def read_txt2array(txt):
    A = np.loadtxt(txt,dtype=np.float32)
    if np.array(A).size ==1:
        A = [A]
    if isinstance(A[0],bytes):
        A = A.astype(str)
    #A = list(A)    
    return A

def orb_state_lalo_uLOS(lat0,lon0,orb_data,attr):
    
    width = int(attr['WIDTH'])
    length = int(attr['LENGTH'])
    ref_lat1 = float(attr['LAT_REF1'])
    ref_lat2 = float(attr['LAT_REF2'])
    ref_lat3 = float(attr['LAT_REF3'])
    ref_lon1 = float(attr['LON_REF1'])
    ref_lon2 = float(attr['LON_REF2'])
    ref_lon3 = float(attr['LON_REF3'])
    # orb_state_file from read_orb_state.py   
    ra0,az0 = latlon2rgaz_coarse(lat0,lon0,attr)
    #Orb = read_txt2array(orb_state_file)
    t_orb = orb_data[:,0]
    X_Orb = orb_data[:,1]
    Y_Orb = orb_data[:,2]
    Z_Orb = orb_data[:,3]
    
    t_tot = float(attr['END_TIME'])  - float(attr['START_TIME']) 
    
    t0 = az0/length*t_tot
    
    tck = intp.interp1d(t_orb,X_Orb,kind='cubic',bounds_error =False, fill_value='extrapolate')
    X0 = tck(t0)      #Again negative for consistency with hx
    
    tck = intp.interp1d(t_orb,Y_Orb,kind='cubic',bounds_error =False, fill_value='extrapolate')
    Y0 = tck(t0)      #Again negative for consistency with hx
    
    tck = intp.interp1d(t_orb,Z_Orb,kind='cubic',bounds_error =False, fill_value='extrapolate')
    Z0 = tck(t0)      #Again negative for consistency with hx
    
    
    SatCoord = (X0,Y0,Z0)
    GroundLLH = (lat0,lon0,0)
    uLOS,uLOS0 = get_uLOS(SatCoord,GroundLLH)

    return uLOS

def get_sar_area(attr):
    
    length, width = int(attr['LENGTH']), int(attr['WIDTH'])
    if 'Y_FIRST' in attr.keys():
        # geo coordinates
        lat0 = float(meta['Y_FIRST'])
        lon0 = float(meta['X_FIRST'])
        lat_step = float(meta['Y_STEP'])
        lon_step = float(meta['X_STEP'])
        lat1 = lat0 + lat_step * (length - 1)
        lon1 = lon0 + lon_step * (width - 1)
        maxlon = max(lon0,lon1)
        minlon = min(lon0,lon1)
        maxlat = max(lat0,lat1)
        minlat = min(lat0,lat1)
        
    else:
        ref_lat1 = float(attr['LAT_REF1'])
        ref_lat2 = float(attr['LAT_REF2'])
        ref_lat3 = float(attr['LAT_REF3'])
        ref_lat4 = float(attr['LAT_REF3'])
            
        ref_lon1 = float(attr['LON_REF1'])
        ref_lon2 = float(attr['LON_REF2'])
        ref_lon3 = float(attr['LON_REF3'])
        ref_lon4 = float(attr['LON_REF4'])
    
        maxlon = max((ref_lon1,ref_lon2,ref_lon3,ref_lon4))
        minlon = min((ref_lon1,ref_lon2,ref_lon3,ref_lon4))
    
        maxlat = max((ref_lat1,ref_lat2,ref_lat3,ref_lat4))
        minlat = min((ref_lat1,ref_lat2,ref_lat3,ref_lat4))
    
    return minlon,maxlon,minlat,maxlat      # wesn

def get_hgt_index(hgtlvs,mindem,maxdem):
    
    idxmin = 0
    nn = len(hgtlvs)
    for i in range(nn-1):
        if (hgtlvs[i] < mindem) & (mindem < hgtlvs[i+1]):
            minkk = i
        if (hgtlvs[i] < maxdem) & (maxdem < hgtlvs[i+1]):
            maxkk = i+1
            
    return minkk, maxkk

# get the location of the points along the LOS direction
def get_uLOS_llh(lat0,lon0,hgt_lvs,uLOS):
    
    hgt_lvs = sorted(hgt_lvs)
    #print(hgt_lvs)
    nn = len(hgt_lvs)
    deltaH = np.asarray(hgt_lvs[1:nn]) - np.asarray(hgt_lvs[0:(nn-1)])
 
    minh = min(hgt_lvs)    # hgt_lvs[0]
    #inc = inc/180*np.pi
    
    gX,gY,gZ = lla_to_ecef(lat0, lon0, minh)
    gX1,gY1,gZ1 = lla_to_ecef(lat0, lon0, minh+1000)
    uZTD = ((gX1-gX)/1000,(gY1-gY)/1000,(gZ1-gZ)/1000)
    
    dH0 = np.sqrt((uLOS[0]*uZTD[0])**2 + (uLOS[1]*uZTD[1])**2 + (uLOS[2]*uZTD[2])**2)
    dH = uLOS[0]*uZTD[0] + uLOS[1]*uZTD[1] + uLOS[2]*uZTD[2]   
    #dLOS_lvs0 = deltaH/np.cos(inc)   

    losX = np.zeros((nn,1));losY = np.zeros((nn,1));losZ = np.zeros((nn,1))
    latH = np.zeros((nn,));lonH = np.zeros((nn,));losH = np.zeros((nn,))
    
    losX[0] = gX;   losY[0]=gY;     losZ[0]=gZ
    latH[0] = lat0; lonH[0] = lon0; losH[0]=0
    
    N = len(deltaH)
    dLOS_lvs = np.zeros((N,1))
    for i in range(N):
        dLOS_lvs[i] = deltaH[i]/dH
        losX[i+1] = losX[i] + uLOS[0]*dLOS_lvs[i]
        losY[i+1] = losY[i] + uLOS[1]*dLOS_lvs[i]
        losZ[i+1] = losZ[i] + uLOS[2]*dLOS_lvs[i]
        lo, la, alt = ecef_to_lla(losX[i],losY[i],losZ[i])
        latH[i+1] = la
        lonH[i+1] = lo
        losH[i+1] = losH[i] + dLOS_lvs[i]
    
    return losX,losY,losZ, latH, lonH, losH

    
def get_LOS3D_coords(latlist,lonlist,hgt_lvs, orb_data,attr):
    
    width = int(attr['WIDTH'])
    length = int(attr['LENGTH'])
    ref_lat1 = float(attr['LAT_REF1'])
    ref_lat2 = float(attr['LAT_REF2'])
    ref_lat3 = float(attr['LAT_REF3'])
    ref_lon1 = float(attr['LON_REF1'])
    ref_lon2 = float(attr['LON_REF2'])
    ref_lon3 = float(attr['LON_REF3'])
    
    row,col = latlist.shape
    nh = len(hgt_lvs)
    
    lat_intp = np.zeros((row,col,nh))
    lon_intp = np.zeros((row,col,nh))
    los_intp = np.zeros((row,col,nh))
    
    #latlist = latlist.flatten()
    #lonlist = lonlist.flatten()
    #nn = len(latlist)
    
    for i in range(row):
        print_progress(i+1, row, prefix='LOS locations calculating:', suffix='complete', decimals=1, barLength=50, elapsed_time=None)
        for j in range(col):   
            lat0 = latlist[i,j]
            lon0 = lonlist[i,j]
            uLOS = orb_state_lalo_uLOS(lat0,lon0,orb_data,attr)
            losX,losY,losZ, latH, lonH, losH = get_uLOS_llh(lat0,lon0,hgt_lvs,uLOS)
            lat_intp[i,j,:] = latH
            lon_intp[i,j,:] = lonH
            los_intp[i,j,:] = losH
    
    if np.mean(lon_intp) > 150:
        lon_intp[lon_intp<0] = lon_intp[lon_intp<0] + 360
    
    return lat_intp, lon_intp, los_intp

def get_LOS_parameters(latlist,lonlist,Presi,Tempi,Vpri,latH, lonH, method,kriging_points_numb):
    R = 6371
    row,col,nh = Presi.shape
    latv = latlist[:,0]
    lonv = lonlist[0,:]
    
    lat00 = latlist.flatten()
    lon00 = lonlist.flatten()
    
    LosP = np.zeros((row,col,nh))
    LosT = np.zeros((row,col,nh))
    LosV = np.zeros((row,col,nh))
    
    
    def resi_func(m,d,y):
        variogram_function =variogram_dict['spherical'] 
        return  y - variogram_function(m,d)
    
    for i in range(nh):
        print_progress(i+1, nh, prefix='Calculating LOS parameters:', suffix='complete', decimals=1, barLength=50, elapsed_time=None)
        lat = latH[:,:,i]
        lon = lonH[:,:,i]
        lat = lat.flatten()
        lon = lon.flatten()
        
        points = np.zeros((len(lat),2))
        points[:,0] = lat
        points[:,1] = lon
        
        Presi0 = Presi[:,:,i]
        Tempi0 = Tempi[:,:,i]
        Vpri00 = Vpri[:,:,i]
        Vpri0 = Vpri00.flatten()
        
        if method =='kriging':
            Vpri0_cor, para, corr= remove_ramp(lat00,lon00,Vpri0)
            trend = func_trend_model(lat,lon,para)
        
            uk = OrdinaryKriging(lon, lat, Vpri0_cor, coordinates_type = 'geographic', nlags=50)
            Semivariance_trend = 2*(uk.semivariance)    
            x0 = (uk.lags)/180*np.pi*R
            y0 = Semivariance_trend
            max_length = 2/3*max(x0)
            range0 = max_length/2
            LL0 = x0[x0< max_length]
            SS0 = y0[x0< max_length]
            
            sill0 = max(SS0)
            sill0 = sill0.tolist()

            p0 = [sill0, range0, 0.0001]    
            vari_func = variogram_dict['spherical']        
            tt, _ = leastsq(resi_func,p0,args = (LL0,SS0))   
            corr, _ = pearsonr(SS0, vari_func(tt,LL0))
            if tt[2] < 0:
                tt[2] =0
            para = tt
            para[1] = para[1]/R/np.pi*180

            uk.variogram_model_parameters = para
            z0,s0 = uk.execute('points', lon, lat, n_closest_points = kriging_points_numb, backend='loop')
            z0 = z0 + trend
        else:
            Vpri0 = Vpri0.reshape((latlist.shape))
            fV = RGI((latv[::-1], lonv), Vpri00[::-1,:],method=method,bounds_error=False,fill_value=None)
            z0 = fV(points)
            
        LosV[:,:,i] = z0.reshape((latlist.shape))
        ######### interp template and pressure
        Presi0 = Presi0.reshape((latlist.shape))
        Tempi0 = Tempi0.reshape((latlist.shape))

        fP = RGI((latv[::-1], lonv), Presi0[::-1,:],method='linear',bounds_error=False,fill_value=None)
        fT = RGI((latv[::-1], lonv), Tempi0[::-1,:],method='linear',bounds_error=False,fill_value=None)

        Pintp = fP(points)
        Tintp = fT(points)
        LosT[:,:,i] = Tintp.reshape((latlist.shape))
        LosP[:,:,i] = Pintp.reshape((latlist.shape))
    
    return LosP,LosT,LosV

def kriging_levels(lonlos,latlos,hgtlvs,dwetlos,attr, maxdem, mindem, Rescale,kriging_points_numb):
    R = 6371 
    # Get interp grids    
    lonStep = lonlos[0,1,0] - lonlos[0,0,0]
    latStep = latlos[1,0,0] - latlos[0,0,0]
    
    minlon,maxlon,minlat,maxlat = get_sar_area(attr)
    
    minlon = minlon - 0.5 #extend 0.5 degree
    maxlon = maxlon + 0.5
    minlat = minlat - 0.5
    maxlat = maxlat + 0.5
    
    lonStep = lonStep/Rescale
    latStep = latStep/Rescale
    
    lonv = np.arange(minlon,maxlon,lonStep)
    latv = np.arange(maxlat,minlat,latStep)
    
    lonvv,latvv = np.meshgrid(lonv,latv)
    
    lonvv_all = lonvv.flatten()
    latvv_all = latvv.flatten()
    
    # Get index of the useful hgtlvs
    mindex,maxdex = get_hgt_index(hgtlvs,mindem,maxdem)
    kl = np.arange(mindex,(maxdex+1))
    hgtuseful = hgtlvs[kl]
    row,col = lonvv.shape
    nh = len(kl)
    
    dwet_intp = np.zeros((row,col,nh))

    def resi_func(m,d,y):
        variogram_function =variogram_dict['spherical'] 
        return  y - variogram_function(m,d)
    
    for i in range(len(kl)):
        #print_progress(i+1, nh, prefix='Kriging wet-delay levels:', suffix='complete', decimals=1, barLength=50, elapsed_time=None)
        lat = latlos[:,:,kl[i]]
        lon = lonlos[:,:,kl[i]]
        lat = lat.flatten()
        lon = lon.flatten()
        
        dwet0 = dwetlos[:,:,kl[i]]
        dwet0 = dwet0.flatten()
        
        dwet0_cor, para, corr= remove_ramp(lat,lon,dwet0)
        print(para)
        print(corr)
        trend = func_trend_model(latvv_all,lonvv_all,para)
        
        uk = OrdinaryKriging(lon, lat, dwet0_cor, coordinates_type = 'geographic', nlags=50)
        Semivariance_trend = 2*(uk.semivariance)    
        x0 = (uk.lags)/180*np.pi*R
        y0 = Semivariance_trend
        max_length = 2/3*max(x0)
        range0 = max_length/2
        LL0 = x0[x0< max_length]
        SS0 = y0[x0< max_length]
        sill0 = max(SS0)
        sill0 = sill0.tolist()

        p0 = [sill0, range0, 0.0001]    
        vari_func = variogram_dict['spherical']        
        tt, _ = leastsq(resi_func,p0,args = (LL0,SS0))   
        corr, _ = pearsonr(SS0, vari_func(tt,LL0))
        if tt[2] < 0:
            tt[2] =0
        para = tt
        para[1] = para[1]/R/np.pi*180

        uk.variogram_model_parameters = para
        z0,s0 = uk.execute('grid', lonv, latv, n_closest_points = kriging_points_numb, backend='loop')
        z0 = z0.flatten() + trend
        dwet_intp[:,:,i] = z0.reshape((row,col))

    return dwet_intp, lonvv, latvv, hgtuseful

def interp2d_levels(lonlos,latlos,hgtlvs,delaylos,attr, maxdem, mindem, Rescale):
    # Get interp grids    
    lonStep = lonlos[0,1,0] - lonlos[0,0,0]
    latStep = latlos[1,0,0] - latlos[0,0,0]
    
    minlon,maxlon,minlat,maxlat = get_sar_area(attr)
    
    minlon = minlon - 0.5 #extend 0.5 degree
    maxlon = maxlon + 0.5
    minlat = minlat - 0.5
    maxlat = maxlat + 0.5
    
    lonStep = lonStep/Rescale
    latStep = latStep/Rescale
    
    lonv = np.arange(minlon,maxlon,lonStep)
    latv = np.arange(maxlat,minlat,latStep)
    
    lonvv,latvv = np.meshgrid(lonv,latv)
    
    lonvv_all = lonvv.flatten()
    latvv_all = latvv.flatten()
    
    # Get index of the useful hgtlvs
    mindex,maxdex = get_hgt_index(hgtlvs,mindem,maxdem)
    kl = np.arange(mindex,(maxdex+1))
    hgtuseful = hgtlvs[kl]
    row,col = lonvv.shape
    nh = len(kl)
    
    delay_intp = np.zeros((row,col,nh))
    for i in range(len(kl)):
        print_progress(i+1, nh, prefix='Interpolating delay levels:', suffix='complete', decimals=1, barLength=50, elapsed_time=None)
        
        lat = latlos[:,:,kl[i]]
        lon = lonlos[:,:,kl[i]]
        lat = lat.flatten()
        lon = lon.flatten()
        
        points = np.zeros((len(lat),2))
        points[:,0] = lon
        points[:,1] = lat
        
        delay0 = delaylos[:,:,kl[i]]
        delay0 = delay0.flatten()
        
        
        #f = interpolate.interp2d(lon, lat, delay0, kind='linear')
        #delayi = f(lonv, latv)      
        delayi = interpolate.griddata(points, delay0, (lonvv,latvv), method='linear')
        
        delay_intp[:,:,i] = delayi
        
    return delay_intp, lonvv, latvv, hgtuseful

def pyrite_griddata_los(lonlos,latlos,hgtlvs,ddrylos,dwetlos,attr, maxdem, mindem, Rescale,method,kriging_points_numb):
    
    if method =='kriging':
        dwet_intp, lonvv, latvv, hgtuse = kriging_levels(lonlos,latlos,hgtlvs,dwetlos,attr, maxdem, mindem, Rescale,kriging_points_numb)
    else:
        dwet_intp, lonvv, latvv, hgtuse = interp2d_levels(lonlos,latlos,hgtlvs,dwetlos,attr, maxdem, mindem, Rescale)

    ddry_intp, lonvv, latvv, hgtuse = interp2d_levels(lonlos,latlos,hgtlvs,ddrylos,attr, maxdem, mindem, Rescale)
    
    
    return dwet_intp,ddry_intp, lonvv, latvv, hgtuse


def kriging_levels_zenith(lonlist,latlist,hgtlvs,dwetlos,attr, maxdem, mindem, Rescale,kriging_points_numb):
    R = 6371
    # Get interp grids    
    lat = latlist.flatten()
    lon = lonlist.flatten()
    
    lonStep = lonlist[0,1] - lonlist[0,0]
    latStep = latlist[1,0] - latlist[0,0]
    
    minlon,maxlon,minlat,maxlat = get_sar_area(attr)
    
    minlon = minlon - 0.5 #extend 0.5 degree
    maxlon = maxlon + 0.5
    minlat = minlat - 0.5
    maxlat = maxlat + 0.5
    
    lonStep = lonStep/Rescale
    latStep = latStep/Rescale
    
    lonv = np.arange(minlon,maxlon,lonStep)
    latv = np.arange(maxlat,minlat,latStep)
    
    lonvv,latvv = np.meshgrid(lonv,latv)
    
    lonvv_all = lonvv.flatten()
    latvv_all = latvv.flatten()
    
    # Get index of the useful hgtlvs
    mindex,maxdex = get_hgt_index(hgtlvs,mindem,maxdem)
    kl = np.arange(mindex,(maxdex+1))
    hgtuseful = hgtlvs[kl]
    row,col = lonvv.shape
    nh = len(kl)
    
    dwet_intp = np.zeros((row,col,nh))
    
    def resi_func(m,d,y):
        variogram_function =variogram_dict['spherical'] 
        return  y - variogram_function(m,d)
    
    for i in range(len(kl)):
        print_progress(i+1, nh, prefix='Kriging wet-delay levels:', suffix='complete', decimals=1, barLength=50, elapsed_time=None)

        dwet0 = dwetlos[:,:,kl[i]]
        dwet0 = dwet0.flatten()
        
        dwet0_cor, para, corr= remove_ramp(lat,lon,dwet0)
        trend = func_trend_model(latvv_all,lonvv_all,para)
        
        uk = OrdinaryKriging(lon, lat, dwet0_cor, coordinates_type = 'geographic', nlags=50)
        Semivariance_trend = 2*(uk.semivariance)    
        x0 = (uk.lags)/180*np.pi*R
        y0 = Semivariance_trend
        max_length = 2/3*max(x0)
        range0 = max_length/2
        LL0 = x0[x0< max_length]
        SS0 = y0[x0< max_length]
        sill0 = max(SS0)
        sill0 = sill0.tolist()

        p0 = [sill0, range0, 0.0001]    
        vari_func = variogram_dict['spherical']        
        tt, _ = leastsq(resi_func,p0,args = (LL0,SS0))   
        corr, _ = pearsonr(SS0, vari_func(tt,LL0))
        if tt[2] < 0:
            tt[2] =0
        para = tt
        para[1] = para[1]/R/np.pi*180

        uk.variogram_model_parameters = para
        z0,s0 = uk.execute('grid', lonv, latv, n_closest_points = kriging_points_numb, backend='loop')
        z0 = z0.flatten() + trend
        dwet_intp[:,:,i] = z0.reshape((row,col))

    return dwet_intp, lonvv, latvv, hgtuseful

def interp2d_levels_zenith(lonlist,latlist,hgtlvs,delaylos,attr, maxdem, mindem, Rescale):
  
    # Get interp grids    
    lat = latlist.flatten()
    lon = lonlist.flatten()
    
    points = np.zeros((len(lat),2))
    points[:,0] = lon
    points[:,1] = lat
    
    lonStep = lonlist[0,1] - lonlist[0,0]
    latStep = latlist[1,0] - latlist[0,0]
    
    minlon,maxlon,minlat,maxlat = get_sar_area(attr)
    
    minlon = minlon - 0.5 #extend 0.5 degree
    maxlon = maxlon + 0.5
    minlat = minlat - 0.5
    maxlat = maxlat + 0.5
    
    lonStep = lonStep/Rescale
    latStep = latStep/Rescale
    
    lonv = np.arange(minlon,maxlon,lonStep)
    latv = np.arange(maxlat,minlat,latStep)
    
    lonvv,latvv = np.meshgrid(lonv,latv)
    
    lonvv_all = lonvv.flatten()
    latvv_all = latvv.flatten()
    
    # Get index of the useful hgtlvs
    mindex,maxdex = get_hgt_index(hgtlvs,mindem,maxdem)
    kl = np.arange(mindex,(maxdex+1))
    hgtuseful = hgtlvs[kl]
    row,col = lonvv.shape
    nh = len(kl)
    
    delay_intp = np.zeros((row,col,nh))
    for i in range(len(kl)):
        print_progress(i+1, nh, prefix='Interpolating delay levels:', suffix='complete', decimals=1, barLength=50, elapsed_time=None)
        
        delay0 = delaylos[:,:,kl[i]]
        delay0 = delay0.flatten()
        
        #f = interpolate.interp2d(lon, lat, delay0, kind='linear')
        #delayi = f(lonv, latv)
        
        delayi = interpolate.griddata(points, delay0, (lonvv,latvv), method='linear')
        delay_intp[:,:,i] = delayi
        
    return delay_intp, lonvv, latvv, hgtuseful

def pyrite_griddata_zenith(lonlist,latlist,hgtlvs,ddrylos,dwetlos,attr, maxdem, mindem, Rescale,method,kriging_points_numb):
    
    if method =='kriging':
        dwet_intp, lonvv, latvv, hgtuseful = kriging_levels_zenith(lonlist,latlist,hgtlvs,dwetlos,attr, maxdem, mindem, Rescale,kriging_points_numb)
    else:
        dwet_intp, lonvv, latvv, hgtuse = interp2d_levels_zenith(lonlist,latlist,hgtlvs,dwetlos,attr, maxdem, mindem, Rescale)

    ddry_intp, lonvv, latvv, hgtuse = interp2d_levels_zenith(lonlist,latlist,hgtlvs,ddrylos,attr, maxdem, mindem, Rescale)
    
    return dwet_intp,ddry_intp,lonvv, latvv, hgtuse
    

def losPTV2del(Presi,Tempi,Vpri,losHgt,cdict,verbose=False):
    '''Computes the delay function given Pressure, Temperature and Vapor pressure.

    Args:
        * Presi (np.array) : Pressure at height levels.
        * Tempi (np.array) : Temperature at height levels.
        * Vpri  (np.array) : Vapor pressure at height levels.
        * hgt   (np.array) : Height levels.
        * cdict (np.array) : Dictionary of constants.

    Returns:
        * DDry2 (np.array) : Dry component of atmospheric delay.
        * DWet2 (np.array) : Wet component of atmospheric delay.

    .. note::
        Computes refractive index at each altitude and integrates the delay using cumtrapz.'''

    if verbose:
        print('PROGRESS: COMPUTING DELAY FUNCTIONS')
    nhgt = Presi.shape[2]           #Number of height points
    nlat = Presi.shape[0]        #Number of stations
    nlon = Presi.shape[1]
    DryT = Presi/Tempi
    WonT = Vpri/Tempi
    WonT2 = WonT/Tempi

    k1 = cdict['k1']
    Rd = cdict['Rd']
    Rv = cdict['Rv']
    k2 = cdict['k2']
    k3 = cdict['k3']
    g = cdict['g']

    #Dry delay
    DDry2 = np.zeros((nlat,nlon,nhgt-1))
    #DDry2[:,:,:] = k1*Rd*(Presi[:,:,:] - Presi[:,:,-1][:,:,np.newaxis])*1.0e-6/g
    for i  in range(nlat):
        for j in range(nlon):
            DDry2[i,j,:] = intg.cumtrapz(DryT[i,j,:],x=losHgt[i,j,:],axis=-1)
    
    val = 2*DDry2[:,:,-1]-DDry2[:,:,-2]
    val = val[:,:,None]
    DDry2 = np.concatenate((DDry2,val),axis=-1)
    
    DDry2 = -1.0e-6*k1*DDry2
    for i in range(nlat):
        for j in range(nlon):
            DDry2[i,j,:]  = DDry2[i,j,:] - DDry2[i,j,-1]

    #Wet delay
    S1 = np.zeros((nlat,nlon,nhgt-1))
    for i  in range(nlat):
        for j in range(nlon):
            S1[i,j,:] = intg.cumtrapz(WonT[i,j,:],x=losHgt[i,j,:],axis=-1)
            
    #S1 = intg.cumtrapz(WonT,x=hgt,axis=-1)
    val = 2*S1[:,:,-1]-S1[:,:,-2]
    val = val[:,:,None]
    S1 = np.concatenate((S1,val),axis=-1)
    del WonT
    
    S2 = np.zeros((nlat,nlon,nhgt-1))
    for i  in range(nlat):
        for j in range(nlon):
            S2[i,j,:] = intg.cumtrapz(WonT2[i,j,:],x=losHgt[i,j,:],axis=-1)
    #S2 = intg.cumtrapz(WonT2,x=hgt,axis=-1)
    val = 2*S2[:,:,-1]-S2[:,:,-2]
    val = val[:,:,None]
    S2 = np.concatenate((S2,val),axis=-1)
    DWet2 = -1.0e-6*((k2-k1*Rd/Rv)*S1+k3*S2)
    
    for i in range(nlat):
        for j in range(nlon):
            DWet2[i,j,:]  = DWet2[i,j,:] - DWet2[i,j,-1]

    return DDry2,DWet2