# -*- coding: future_fstrings -*-
from pyROTMOD.support.minor_functions import integrate_surface_density
from pyROTMOD.optical.profiles import sersic,get_sersic_b
from astropy import units as unit
#the individul functions are quicker than the general function https://docs.scipy.org/doc/scipy/reference/special.html
from scipy.special import gamma


import numpy as np


def calculate_axis_ratio(components):
    '''The axis ratio is the ratio of the height to the scale length'''    
    if not components.axis_ratio is None:
        return
    if not components.height in [None,0.] and \
        not components.scale_length is None:
            components.axis_ratio = components.height/components.scale_length
    elif not components.height in [None,0.] and \
        not components.hernquist_scale_length is None:
            components.axis_ratio = components.height/components.hernquist_scale_length

def calculate_central_SB(components):
    '''The central SB is the SB at the center of the galaxy'''
  
    if not components.central_SB is None:
        return
    if components.profile_type == 'sbr_dens':
        if components.total_mass is None:
            calculate_total_mass(components)
        total = components.total_mass
    elif components.profile_type == 'sbr_lum':
        if components.total_luminosity is None:
            calculate_total_luminosity(components)
        total = components.total_luminosity

    if components.radii[0] == 0. and not components.values is None:  
        components.central_SB = components.values[0]
    else: 
        if components.type == 'expdisk':
            if not None in [total,components.scale_length]:
             
                # this assumes perfect ellipses for now and no deviations are allowed

                components.central_SB = total/(2.*np.pi*\
                                        components.scale_length.to(unit.pc)**2)
              
        elif components.type in ['sersic','devauc'] and not components.profile_type == 'sbr_dens': #This is based on Leffective and the sky profile so it doesn work for the density
            if components.L_effective is None and\
                not None in [total, components.R_effective,\
                            components.sersic_index,components.axis_ratio]:
                calculate_L_effective(components)
            
            components.central_SB = sersic(0.*unit.kpc,components.L_effective,\
                    components.R_effective,components.sersic_index)
                
def calculate_L_effective(components,from_central = False):
    if not components.L_effective is None:
        return
    '''The sersic profile is based on Sig_eff'''
    #kappa=2.*components.sersic_index-1./3. # From https://en.wikipedia.org/wiki/Sersic_profile
    if not components.sersic_index is None and not components.R_effective is None:
        kappa = get_sersic_b(components.sersic_index)
        if from_central and not components.profile_type == 'sbr_dens': 
             if not components.central_SB is None:
                components.L_effective = components.central_SB/np.exp(-1.*kappa*(((\
                    0.*unit.kpc)/components.R_effective)**(1./components.sersic_index)-1))
        else:
            if not components.total_luminosity is None:
                components.L_effective = components.total_luminosity/(2.*np.pi*(components.R_effective.to(unit.pc))**2*\
                            np.exp(kappa)*components.sersic_index*\
                            kappa**(-2*components.sersic_index)*\
                            components.axis_ratio*gamma(2.*components.sersic_index)) #L_solar/pc^2
   

def calculate_R_effective(components):
    '''The effective radius is the radius that contains half of the mass'''
    if not components.R_effective is None:
        return

    if not components.scale_length is None:
        components.R_effective = components.scale_length*1.678
        return
     
    if not components.hernquist_scale_length is None: 
        components.R_effective = components.hernquist_scale_length*1.8153  
        return

    if not components.radii is None and not components.values is None:
        mass,ringarea = integrate_surface_density(components.radii,\
                                                  components.values)
        
        if mass.unit == unit.Msun:
            if components.total_mass is None:
                components.total_mass = mass    
        if mass.unit == unit.Lsun:
            if components.total_luminosity is None:
                components.total_luminosity = mass
        for i in range(len(components.radii.value),1,-1):
            current_mass =  np.sum(ringarea.value[:i]*components.values.value[:i])*\
                (components.values.unit*ringarea.unit)
            if current_mass < mass/2.:
                components.R_effective =  components.radii[i]
                break
    ''' Note that in   https://articles.adsabs.harvard.edu/pdf/1990ApJ...356..359H (Hernquist 1990) Eq 39
    There is a factor 1.33 between the effective radius and the half mass radius but as we calculating the mass on a SB profile 
    and not a density profile that doesn't apply here
    '''
    if components.scale_length is None:
        components.scale_length = calculate_scale_length(components)
   
def calculate_scale_length(components):
    if not components.scale_length is None:
        return
    '''The scale length relate to the exponential function only'''
    # From https://iopscience.iop.org/article/10.1088/0004-6256/139/6/2097/pdf Peng 2010 Eq 7
    components.scale_length = components.R_effective/1.678 
    
def calculate_hernquist_scale_length(components):
    '''The scale length relates to the hernquist function only'''
    if not components.hernquist_scale_length is None:
        return
  
    # Eq 38 in https://articles.adsabs.harvard.edu/pdf/1990ApJ...356..359H (Hernquist 1990)
    components.hernquist_scale_length  = components.R_effective/1.8153 
   
def calculate_total_mass(components):
    if components.total_mass is not None:
        return
    total = None
    if not components.radii is None and not components.values is None:\
        total,ring_area = integrate_surface_density(\
                components.radii,components.values)
    elif not components.total_luminosity is None and not components.MLratio is None:
        total = components.total_luminosity*components.MLratio
    if not total is None:
        try:
            total = total.to(unit.Msun)
            components.total_mass = total 
        except:
            print('The total luminosity could not be calculated')
            components.total_mass = None

def calculate_total_luminosity(components):
    if components.total_luminosity is not None:
        return
  

    total = None
    if not components.radii is None and not components.values is None:\
        total,ring_area = integrate_surface_density(\
                components.radii,components.values)
       
    
    elif not components.total_mass is None and not components.MLratio is None:
        total = components.total_mass/components.MLratio
    elif components.type == 'expdisk' and not components.scale_length is None\
        and not components.central_SB is None:
        total = components.central_SB*(2.*np.pi*\
                                        components.scale_length**2)
    elif components.type == 'sersic' and not components.R_effective is None\
        and not components.sersic_index is None and not components.axis_ratio is None:
        #kappa=2.*components.sersic_index-1./3. # From https://en.wikipedia.org/wiki/Sersic_profile 
        kappa = get_sersic_b(components.sersic_index)
        total = components.central_SB*(2.*np.pi*(\
                    components.R_effective.to(unit.pc))**2*\
                    np.exp(kappa)*components.sersic_index*\
                    kappa**(-2*components.sersic_index)*\
                    components.axis_ratio*gamma(2.*components.sersic_index))
    if not total is None:
        try:
            total = total.to(unit.Lsun)
            components.total_luminosity = total 
        #exit()
        except:
            print('The total luminosity could not be calculated')
            components.total_luminosity = None
    
        
