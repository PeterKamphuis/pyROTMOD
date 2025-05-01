# This contains the options for the vertical profiles as well as some selection 
# functions and a conversion function to transform from volume density to surface brightness

import numpy as np
import jax.numpy as jnp
import astropy.units as unit
from pyROTMOD.support.profile_classes import SBR_Profile,copy_attributes
from pyROTMOD.support.minor_functions import quantity_array
from scipy.integrate import quad

def check_height(Density_In):
    '''Check the type of the vertical distributio'''
    if not Density_In.height_type in  ['sech','exp'] and Density_In.height != 0.:
        raise InputError(f'We cannot have {Density_In.height_type} with a thick exponential disk. Use a different disk or pick exp or sech')
    sech = False
    if Density_In.height_type == 'sech':
        sech = True
    return sech

def convert_density_to_SB(density_profile,invert=False):
    '''Convert the volume density to a surface brightness'''
    '''As these conversion require integration of the height profile they are in this module '''
    #First select the height function
    vertical_function = select_vertical_function(density_profile.height_type)
    integrated_function = calculate_height_integral(vertical_function,
                density_profile.height.to(unit.pc).value)*unit.pc
    if invert:
        integrated_function = 1./integrated_function
    
    sbr_profile = SBR_Profile()

    copy_attributes(density_profile, sbr_profile)
    for attr in sbr_profile.__dict__:
        value = getattr(density_profile, attr)
        if attr == 'profile_type':
            if invert:
                new_value = 'density'
            else:
                new_value = 'sbr_dens'
        elif attr == 'error':
            new_value= []
            unit = None
            for i,x in enumerate(density_profile.values):
                if not value[i] is None:
                    new_value.append(np.sqrt(x**2*integrated_function[1]**2
                    +value[i]**2*integrated_function[0]**2))
                    unit = new_value[-1].unit
                else:
                    new_value.append(None)
            if not unit is None:
                new_value = quantity_array(new_value,unit)
            else:
                new_value = None
        elif attr == 'values':
            new_value = [x*integrated_function[0] for x in value]
            new_value = quantity_array(new_value,new_value[0].unit)
        else:
            try:
                value.unit.to(unit.Msun/unit.pc**3) 
                new_value = value*integrated_function[0]
            except unit.UnitConversionError:
                continue
        setattr(sbr_profile, attr, new_value)
    return sbr_profile

def calculate_height_integral(function,height):
    '''Calculate the height integral for the sech or exp function'''
    new_value = 2.*quad(function,0,np.inf,args=(height))
    return new_value
         


   

def exponential(r,h):
    '''Exponential function'''
    return np.exp(-1.*r/h)


def norm_exponential(radii,h):
    return exponential(radii,h)/h
def norm_sech_square(radii,h):
    '''Normalisation of the sech square function'''
    return sech_square(radii,h)/h
def norm_sech(radii,h):
    return sech(radii,h)*(2./(h*np.pi))
    #2./h/np.pi/np.cosh(radii/h)

def sech(r,h):
    '''Simple sech function'''
    #sech(x) = 1./cosh(x)
    return 1./np.float64(np.cosh(r/h))

def sech_square(r,h):
    '''Sech square function '''
    return sech(r,h)**2

def select_vertical_function(mode,normalized= False):
    if mode == 'sech-sq':
        if normalized:
            return norm_sech_square
        else:
            return sech_square
      
    elif mode == 'exp':
        if normalized:
            return norm_exponential
        else:
            return exponential
        
    elif mode == 'sech-simple':
        if normalized:
            return norm_sech
        else:
            return sech        
    else:
        raise NotImplementedError(f'We do not have not yet implemented the vertical functions for {mode}')

