# -*- coding: future_fstrings -*-

def initialize():
    global H_0
    H_0 = 69.7 #km/s/Mpc
    global c
    c=299792458                  #light speed in m/s
    global pc
    pc=3.086e+18                  #parsec in cm
    global solar_mass
    solar_mass=1.98855e30          #Solar mass in kg
    global solar_luminosity
    solar_luminosity = 3.828e26    # Bolometric Solar Luminosity in W
    global HI_mass
    HI_mass=1.6737236e-27             #Mass of hydrogen in kg
    # 'BANDNAME: wavelength (m)'
    global bands #
    bands ={'WISE3.4': 3.3256e-06, 'SPITZER3.6': 3.6e-6} #m {'WISE3.4': 3.385190356820235e-06, 'SPITZER3.6': 3.6e-6}
    global zero_point_fluxes
    zero_point_fluxes = {'WISE3.4': 309.540, 'SPITZER3.6': 280.9} #Jy
    #spitzer from https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/14/#_Toc82083613/
    #WISE from http://wise2.ipac.caltech.edu/docs/release/prelim/expsup/sec4_3g.html
    global G
    G =  6.67430e-11     #m^3/(kg*s^2)
    global Gsol
    Gsol = G/(1000.**3)*solar_mass #km^3/(M_sol*s^2)
    global Grotmod
    Grotmod =0.00000431158 #km^2/s^2*kpc*/M_sol
    global Gpot
    Gpot = G*solar_mass/(pc/100.)/1000. #m^2/s^2*kpc*/M_sol
