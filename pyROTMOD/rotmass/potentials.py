# -*- coding: future_fstrings -*-

import numpy as np
import pyROTMOD.constants as cons
from sympy import symbols, sqrt,atan,pi,log
# Written by Aditya K.
def ISO():
    r,RHO_0,R_C = symbols('r RHO_0 R_C')
    iso = sqrt((4.*pi*cons.Gpot*RHO_0*R_C**2)* (1- (R_C/r)*atan(r/R_C)))
    return iso
# Written by Aditya K.
def NFW():
    r,C,R200= symbols('r C R200')
    nfw = (R200/0.73)*sqrt( (R200/r)*((log(1+r*(C/R200))-(r*(C/R200)/(1+r*(C/R200))))/(log(1+C) - (C/(1+C)))))
    return nfw
# Written by Aditya K.
def BURKERT():
    r,RHO_0,R_C = symbols('r RHO_0 R_C')
    Burkert = sqrt((6.4*cons.Gpot*RHO_0*((R_C**3)/r))*(log(1+(r/R_C)) - atan(r/R_C)  + 0.5*log( 1+ (r/R_C)**2) ))
    return Burkert
