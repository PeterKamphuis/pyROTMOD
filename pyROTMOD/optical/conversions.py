# -*- coding: future_fstrings -*-

from astropy import units as unit
import numpy as np
import pyROTMOD.support.constants as co
from pyROTMOD.support.errors import InputError,UnitError

def mag_to_lum(mag,band = 'WISE3.4',distance= 0.*unit.Mpc,debug = False):
    if band not in co.solar_magnitudes:
        raise InputError(f''' The band {band} is not yet available in pyROTMOD. 
Possible bands are {', '.join([x for x in co.solar_magnitudes])}. 
Please add the info for band {band} to pyROTMOD/constants.py.''')
    if mag.unit == unit.mag/unit.arcsec**2:
        if co.solar_magnitudes[band].unit != unit.mag:
            raise UnitError(f'The band solar magnitudes need to be in magnitudes.')
        # Surface brightness is constant with distance and hence works differently
        #from Oh 2008.
        factor = (21.56+co.solar_magnitudes[band].value)
        inL= (10**(-0.4*(mag.value-factor)))*unit.Lsun/unit.parsec**2  #L_solar/pc^2
    elif mag.unit == unit.mag:
        M= mag-2.5*np.log10((distance/(10.*unit.pc))**2)*unit.mag # Absolute magnitude
        if co.solar_magnitudes[band].unit != unit.mag:
            raise UnitError(f'The band solar magnitudes need to be in magnitudes.')
        # astropy units doesn't Really know how to convert from magnitude to Lsun.
        inL= (10**(-0.4*(M.value-co.solar_magnitudes[band].value)))*unit.Lsun # in band Luminosity in L_solar
       
    elif mag.unit == unit.Lsun/unit.parsec**2:
        inL = mag
    else:
        raise InputError(f'MAG_to_LUM: the unit {mag.unit} is not recognized for the magnitude')
    # Integrated flux is
    # and convert to L_solar
    return inL   # L_solar
mag_to_lum.__doc__ =f'''
 NAME:
    mag_to_lum

 PURPOSE:
    convert apparent magnitudes to intrinsic luminosities in L_solar

 CATEGORY:
    optical

 INPUTS:
    mag = the magnitude dictionary

 OPTIONAL INPUTS:
    band = 'WISE3.4'
        The observational band
    distance  = 0.
        Distance to the galaxy
    debug = False

 OUTPUTS:
    inL = Luminosity dictionary

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''