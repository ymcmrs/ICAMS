[![Language](https://img.shields.io/badge/python-3.5%2B-blue.svg)](https://www.python.org/)
[![Citation](https://img.shields.io/badge/doi-10.1016%2Fj.jgr.solidearth.2020JB020952-blue)](https://doi.org/10.1029/2020JB020952)


# ICAMS

An open source module in python for InSAR troposphere Correction using global Atmospheric Models that considers the spatial Stochastic properties of the troposphere (ICAMS). We model (funcitonal model + stochatic model) the tropospheric parameters (P,T, and e) layer by layer along the altitude levels, and we calculate the InSAR delays along the LOS direction. ICAMS can be applied with all of the weather models (global or local). Currently, our code support to download and process [ERA5 reanalysis data](https://retostauffer.org/code/Download-ERA5/) from ECMWF, and ICAMS can be used with [MintPy](https://github.com/insarlab/MintPy) hdf5 outputs directly.

This is research code provided to you "as is" with NO WARRANTIES OF CORRECTNESS. Use at your own risk.

### 1 Download

Download the development version using git:   
   
    cd ~/python
    git clone https://github.com/ymcmrs/ICAMS
    
    
### 2 Installation

 1） To make icams importable in python, by adding the path ICAMS directory to your $PYTHONPATH
     For csh/tcsh user, add to your **_~/.cshrc_** file for example:   

    ############################  Python  ###############################
    if ( ! $?PYTHONPATH ) then
        setenv PYTHONPATH ""
    endif
    
    ##--------- Anaconda ---------------## 
    setenv PYTHON3DIR    ~/python/anaconda3
    setenv PATH          ${PATH}:${PYTHON3DIR}/bin
    
    ##--------- PyINT ------------------## 
    setenv ICAMS_HOME    ~/python/PyINT       
    setenv PYTHONPATH    ${PYTHONPATH}:${ICAMS_HOME}
    setenv PATH          ${PATH}:${ICAMS_HOME}/icams
    
 2) Install dependencies
    
    $CONDA_PREFIX/bin/pip install git+https://github.com/ymcmrs/PyKrige.git   
    $CONDA_PREFIX/bin/pip install git+https://github.com/ymcmrs/elevation.git 
    
 3） Install gdal, elevation module using pip or conda for DEM processing.
 
 4) download global geoid model (e.g., egm2008-1.pgm) and change the default geoid-data path in geoid.py

### 2 Running ICAMS

1). Zenith delay products generation (all you need to provide are region + imaging-time + date or datelist)

     tropo_icams_date.py  or  tropo_icams_date_list.py
  
     e.g. :
        tropo_icams_date.py 20201024 --region " 172.8/173.8/-42.6/-41.6 " --imaging-time 11:24 
        tropo_icams_date.py --date-list 20201024 20201023 20201025 20201026 --region " 172.8/173.8/-42.6/-41.6 " --imaging-time 11:24 --parallel 4 

2) InSAR correction. (if you are a MintPy user

     tropo_icams_sar.py  or  tropo_icams.py
     
     e.g. :
        tropo_icams_sar.py geometryRadar.h5 --date 20180101 
        tropo_icams.py timeseries.h5 geometryRadar.h5 --project zenith --method sklm
        tropo_icams.py timeseries.h5 geometryRadar.h5 --sar-par 20181025.slc.par --project los --method sklm    (if you are a GAMMA user)
        
2) icamsApp.py  (still under development)
    
        

The key steps of ICAMS can be summarized as follow: <br> 

step 1: Download weather model data [Make sure you have updated the user.cfg file].\
Step 2: Pre-processing of GAM outputs (P-layers to H-layers)\
Step 3: Estimating LOS locations at different altitude layers [You need to input slc.par file to provide orbit data] \
Step 4: Predicting trop-parameters at LOS locations (SKlm) [Make sure you have [PyKrige](https://pypi.org/project/PyKrige/) module available] \
Step 5: Calculating LOS delays \
Step 6: Interpolating LOS delays (SKlm) layer by layer \
Step 7: Interpolating SAR/InSAR tropospheric delays [Make sure you have [elevation](https://pypi.org/project/elevation/) available and download the global geoid data] 



Citation: [Cao, Y.M., Jónsson, S., Li, Z.W., 2021, Advanced InSAR Tropospheric Corrections from Global Atmospheric Models that Incorporate Spatial Stochastic Properties of the Troposphere, Journal of geophysical research: solid earth, doi: 10.1029/2020JB020952](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020JB020952)

******* ICAMS is still under development, and we will keep going to make it user friendly ******
