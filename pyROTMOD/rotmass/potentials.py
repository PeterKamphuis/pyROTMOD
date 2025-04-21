# -*- coding: future_fstrings -*-

import numpy as np
import pyROTMOD.support.constants as cons
from sympy import symbols, sqrt,atan,pi,log
from dataclasses import dataclass,field
from typing import List





# Written by Aditya K.
def ISO():
    r,RHO0,R_C = symbols('r RHO0 R_C')
    iso = sqrt((4.*pi*cons.Gpot.value*RHO0*R_C**2)* (1- (R_C/r)*atan(r/R_C)))
    return iso

class ISO_config:
    parameters = {'RHO0': [None, None, None, True,True],\
    'R_C': [None, None, None, True,True]}
   
# Written by Aditya K.
def NFW():
    r,C,R200= symbols('r C R200')
    nfw = (R200/0.73)*sqrt( (R200/r)*((log(1+r*(C/R200))-(r*(C/R200)/(1+r*(C/R200))))/(log(1+C) - (C/(1+C)))))
    return nfw
class NFW_config:
    parameters = {'C': [None, None, None, True,True],\
    'R200': [None, None, None, True,True]}
  
def Log_NFW():
    r,lgC,lgR200= symbols('r lgC lgR200')
    nfw = ((10**lgR200)/0.73)*sqrt(((10**lgR200)/r)*((log(1+r*((10**lgC)/(10**lgR200)))
                                    -(r*((10**lgC)/(10**lgR200))/(1+r*((10**lgC)/(10**lgR200)))))
                                    /(log(1+10**lgC) - ((10**lgC)/(1+10**lgC)))))
    return nfw
class Log_NFW_config:
    parameters = {'lgC': [None, None, None, True,True],\
    'lgR200': [None, None, None, True,True]}
  
  # if V and ML are in a potential it is assumed that this is an 
  # alternative gravity where no additional RC is required beyond the baryonic RCs

# This is the formula as stated in the ROTMASS documentation of Gipsy
def MOND_CLASSIC():
    r, V,ML, a0 = symbols('r V ML a0')
    #Vt(r)=2*r*a/(mg*abs(Vg)*Vg+md*Vd*abs(Vd)+mb*Vb*abs(Vb))
    #with default: a=3734
    #V**2 = ML*V**2*sqrt(1+sqrt(1+(2*r*a/(ML*V**2))**2))/sqrt(2)  
    #Vt(r)=sqrt((mg*abs(Vg)*Vg+md*Vd*abs(Vd)+mb*Vb*abs(Vb))*sqrt(1+sqrt(1+(2*r*a/(mg*abs(Vg)*Vg+md*Vd*abs(Vd)+mb*Vb*abs(Vb)))**2))/sqrt(2))
    mond = sqrt(ML*V*abs(V)*sqrt(1+sqrt(1+(2*r*(a0*3.0856776e11)/(ML*V*abs(V)))**2))/sqrt(2)) 
    #The factor 3.08e11 is to convert from cm/s**2 to km**2/(s**2*kpc)
    return mond
#  We need this individual version to preserve the sign in the baryonic curves
def MOND_CLASSIC_INDIVIDUAL():
    r, V,ML, a0 = symbols('r V ML a0')
    mond_in = V/abs(V)*sqrt(ML*V**2*sqrt(1+sqrt(1+(2*r*(a0*3.0856776e11)/(ML*V**2))**2))/sqrt(2))
    return mond_in
class MOND_CLASSIC_config:
    parameters = {'a0': [1.2e-8, None, None, True,True],
                  'V': ['RC_input_Curve'],
                  'ML': ['Match']}

# Written by Aditya K.
def BURKERT():
    r,RHO0,R_C = symbols('r RHO0 R_C')
    Burkert = sqrt((6.4*cons.Gpot.value*RHO0*((R_C**3)/r))*(log(1+(r/R_C)) - atan(r/R_C)  + 0.5*log( 1+ (r/R_C)**2) ))
    return Burkert
class BURKERT_config:
    parameters = {'RHO0': [None, None, None, True,True],\
    'R_C': [None, None, None, True,True]}
   
