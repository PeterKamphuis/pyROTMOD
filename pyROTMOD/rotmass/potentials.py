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

#Taken from Aritra, Not pulished yet
def Fuzzy_DM():
    r,m,R_C = symbols('r m R_C')
    #m in eV  r and r_c in kpc and rho_Fuzzy_DM in M/pc**3
    #rho_Fuzzy_DM = 1.9*(m*10**23)**(-2)*R_C**(-4) /(1+0.091*(r/R_C)**2)**8
    #MR=4/3.*pi*r**3*rho_Fuzzy_DM
    #m in 10^-23 eV
    Fuzzy_DM = sqrt(cons.Gpot*4/3.*pi*r**2*(1.9*(m)**(-2)*R_C**(-4) /(1+0.091*(r/R_C)**2)**8)*10**9) 
    return Fuzzy_DM
    #(m/s)^2*kpc/Msol  *kpc^2 * M/pc**3
    #1./pc = 1./(10^-3 kpc) =10^3 1./kpc
