[![Language](https://img.shields.io/badge/python-3.5%2B-blue.svg)](https://www.python.org/)
[![Citation](https://img.shields.io/badge/doi-10.1016%2Fj.cageo.2020JB020952-blue)](https://doi.org/10.1029/2020JB020952)


# ICAMS

An open source module in python for InSAR troposphere Correction using global Atmospheric Models that considers the spatial Stochastic properties of the troposphere (ICAMS). We model the tropospheric parameters (P,T, and e) layer by layer along the altitude levels, and we calculate the InSAR delays along the LOS direction. ICAMS can be applied with all of the weather models (global or local). Currently, our code support to download and process [ERA5 reanalysis data](https://retostauffer.org/code/Download-ERA5/) from ECMWF. 

The key steps of ICAMS can be summarized as follow: <br>  
step 1: Download weather model data [Make sure you have update the user.cfg file]. <br>  
Step 2: Pre-processing of GAM outputs (P-layers to H-layers) 
Step 3: Estimating LOS locations at different altitude layers [You need to input slc.par file to provide orbit data] <br>  
Step 4: Predicting trop-parameters at LOS locations (SKlm) [Make sure you have PyKrige module available] <br>  
Step 5: Calculating LOS delays  <br>  
Step 6: Interpolating LOS delays (SKlm) layer by layer <br>  
Step 7: Interpolating SAR/InSAR tropospheric delays [Make sure you have download the gobal geoid data] <br>  


ICAMS is still under development, and we will keep going to make it user friendly.

This is research code provided to you "as is" with NO WARRANTIES OF CORRECTNESS. Use at your own risk.

