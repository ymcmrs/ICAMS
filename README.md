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
    
 2） Install dependencies
    
    $CONDA_PREFIX/bin/pip install git+https://github.com/ymcmrs/PyKrige.git   
    $CONDA_PREFIX/bin/pip install git+https://github.com/ymcmrs/elevation.git 
    
 3）Install gdal and download global geoid model (e.g., egm2008-1.pgm, need to change the default geoid-data path in geoid.py)
  
 4) update CDS information in user.cfg 

    [CDS]
    key = [your CDS key]

### 2 Running ICAMS

1). Zenith delay products generation (all you need to provide are region + imaging-time + date or datelist) 

     tropo_icams_date.py  or  tropo_icams_date_list.py
  
     e.g. :
        tropo_icams_date.py 20201024 --region " 172.8/173.8/-42.6/-41.6 " --imaging-time 11:24 
        tropo_icams_date.py --date-list 20201024 20201023 20201025 20201026 --region " 172.8/173.8/-42.6/-41.6 " --imaging-time 11:24 --parallel 4 

2). InSAR correction. (if you are a MintPy user) 

     tropo_icams_sar.py  or  tropo_icams.py
  
     e.g. :
        tropo_icams_sar.py geometryRadar.h5 --date 20180101 
        tropo_icams.py timeseries.h5 geometryRadar.h5 --project zenith --method sklm
        tropo_icams.py timeseries.h5 geometryRadar.h5 --sar-par 20181025.slc.par --project los --method sklm    (if you are a GAMMA user) 

3). icamsApp.py  (under development) 
        

### 3 Citation
    
 If you use our toolbox or if you find our research helpful, please cite the following paper (Thanks for your support):

Cao, Y., Jónsson, S. and Li, Z., 2021. Advanced InSAR tropospheric Corrections from global atmospheric models that incorporate spatial stochastic properties of the troposphere. Journal of Geophysical Research: Solid Earth, 126(5), p.e2020JB020952.  https://doi.org/10.1029/2020JB020952
    
 Any feedback are welcome!  (ymcmrs@gmail.com)

