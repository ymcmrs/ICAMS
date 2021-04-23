#! /usr/bin/env python
#################################################################
###  This program is part of PyRite  v1.0                     ### 
###  Copy Right (c): 2019, Yunmeng Cao                        ###  
###  Author: Yunmeng Cao                                      ###                                                          
###  Email : ymcmrs@gmail.com                                 ###
###  Univ. : King Abdullah University of Science & Technology ###   
#################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

__doc__ = """
PyRite
=======

Code by Yunmeng Cap and the GigPy Developers
ymcmrs@gmail.com

Summary
-------
Function definitions for elevation-correlated tropospheric models. In each function, m is a list of
defining parameters and d is an array of the elevation values at which to
calculate the elevation model.

References
----------
.. [1] Y.M., Cao, et al., 2019, under preparing.
       
Copyright (c) 2019, PyRite Developers
"""


def linear_elevation_model(m, h):
    """Linear model, m is [slope, b]"""
    k = float(m[0])
    b = float(m[1])
    return k * h + b

def onn_elevation_model(m, h):
    """Onn model, m is [C0, alpha, C_min]"""
    c,alpha,z_min = m
    return c*np.exp(-alpha*h) + h*alpha*c*np.exp(-alpha*h)+z_min

def onn_linear_elevation_model(m, h):
    """Onn-linear model, m is [C0, alpha, slope, C_min]"""
    c,alpha,k, z_min = m
    return c*np.exp(-alpha*h) + h*alpha*c*np.exp(-alpha*h) + k*h + z_min

def exp_elevation_model(m, h):
    """Exponential model, m is [slope, beta]"""
    k,beta = m
    return k*np.exp(-beta*h)


def exp_linear_elevation_model(m, h):
    """Exponential-linear model, m is [slope, beta, k]"""
    k1,beta,k2 = m
    return k1*np.exp(-beta*h) + k2*h

######### residual model
def residuals_linear(m,h,y):
    return y - linear_elevation_model(m,h)


def residuals_onn(m,h,y):
    return y - onn_elevation_model(m,h)

def residuals_onn_linear(m,h,y):
    return y - onn_linear_elevation_model(m,h)

def residuals_exp(m,h,y):
    return y - exp_elevation_model(m,h)

def residuals_exp_linear(m,h,y):
    return y - exp_linear_elevation_model(m,h)

########### initial parameters generation

def initial_linear(h,y):
    k = - (max(y[:]) - min(y[:]))/(max(h[:]) - min(h[:]))
    b = max(y[:])  
    return k,b

def initial_onn(h,y):
    c = max(y[:]) - min(y[:])
    alpha = 0.001
    z_min = min(y[:])
    return c,alpha,z_min

def initial_onn_linear(h,y):
    c = max(y[:]) - min(y[:])
    alpha = 0.001
    z_min = min(y[:])
    k = - (max(y[:]) - min(y[:]))/(max(h[:]) - min(h[:]))
    return c,alpha,k,z_min


def initial_exp(h,y):
    k = max(y[:])
    belta = 0.001
    return k, belta


def initial_exp_linear(h,y):
    k1 = max(y[:])
    belta = 0.001
    k2 = - (max(y[:]) - min(y[:]))/(max(h[:]) - min(h[:]))
    return k1, belta, k2











