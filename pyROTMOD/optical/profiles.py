# -*- coding: future_fstrings -*-

from pyROTMOD.support.errors import UnitError
from pyROTMOD.support.minor_functions import isquantity
from astropy import units as unit
from fractions import Fraction
#the individul functions are quicker than the general function https://docs.scipy.org/doc/scipy/reference/special.html
from scipy.special import k0,k1, gammaincinv
from scipy.interpolate import interp1d
from sympy import meijerg
import numpy as np
from jax import numpy as jnp
from jax import scipy as jsp
import warnings
from functools import partial
# The edge functions are  untested for now
def edge_numpyro(central,h,r):
    s = r/h
    profile = central*s*jsp.special.k1(s)
    return profile
def edge(r,central,h):
    '''This is the actual edge on sky projection of an exponential disk (vd Kruit 1981)'''
    s=r/h
    return central*s*k1(s)

def edge_luminosity(components,radii = None):
    if radii is None:
        radii = components.radii
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lum_profile = edge(radii,components.central_SB,components.scale_length)
        if np.isnan(lum_profile[0]):
            lum_profile[0] = components.central_SB
    # this assumes perfect ellipses for now and no deviations are allowed
    return lum_profile
edge_luminosity.__doc__ = f'''
 NAME:
     edge_luminosity

 PURPOSE:
    Convert the components from the galfit file into a  on sky radial profile

 CATEGORY:
    optical

 INPUTS:
    components = the components read from the galfit file.

 OPTIONAL INPUTS:
    radii = []
        the radii at which to evaluate
    zero_point_flux =0.
        The magnitude zero point flux value. Set by selecting the correct band.
    distance  = 0.
        Distance to the galaxy
    log = None
    debug = False

 OUTPUTS:
   lum_ profile  = the luminosity profile
   lum_components = a set of homogenized luminosity components for the components in galfit file.

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified
import inspect
 NOTE: This is not well tested yet !!!!!!!!!
'''

def edge_profile(components,radii = None):
   
    if radii is None:
        radii = components.radii
    #The edge luminosity in galfit is taken from vdKruit and Searle which is the
    # edge-on projection of an exponential luminosity density (See Eg 5 in vd Kruit and Searle) 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if radii.unit == components.scale_length.unit:    
            #Equation 5 in vd Kruit and Searle with z 0
            L0 = components.total_mass/(4.*np.pi*components.scale_length**2*components.height)
            ### This makes this a 3D profile with M/pc**3
            profile = exponential(radii,L0,components.scale_length)
          
        else:
            raise UnitError(f'The unit of the radii ({radii.unit}) does not match the scale length ({components.scale_length.unit})')
    #profile = [x.value for x in profile]

    return profile        
edge_profile.__doc__ = f'''
 NAME:
     edge_luminosity

 PURPOSE:
    Convert the components from the galfit file into deprojected density profiles

 CATEGORY:
    optical

 INPUTS:
    components = the components read from the galfit file.

 OPTIONAL INPUTS:
    radii = []
        the radii at which to evaluate
    zero_point_flux =0.
        The magnitude zero point flux value. Set by selecting the correct band.
    distance  = 0.
        Distance to the galaxy
    log = None
    debug = False

 OUTPUTS:
   lum_ profile  = the luminosity profile
   lum_components = a set of homogenized luminosity components for the components in galfit file.

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified
import inspect
 NOTE: This is not well tested yet !!!!!!!!!
'''

def extrapolate_zero(radii,profile):
    if 0. in radii:
        index = np.where(radii != 0.)
        extra = interp1d(radii[index].value, profile[index].value, fill_value = "extrapolate")
        index = np.where(radii == 0.)
        profile[index] = extra(radii[index].value)*profile.unit
    return profile

def extrapolate_zero_numpyro(radii,profile):
    # This is a numpyro version of the extrapolate_zero function
    profile = jnp.interp(radii, radii[1:], profile[1:], left = "extrapolate")

    return profile
def exponential_numpyro(central,h,r):

    profile = central*jnp.exp(-1.*r/h)
    return profile
def exponential(r,central,h):
    '''Exponential function'''
    return central*np.exp(-1.*r/h)

def exponential_luminosity(components,radii = None):
    #lum_components = copy.deepcopy(components)
    #Ftot = 2πrs2Σ0q
    if radii is None:
        radii = components.radii
    if radii.unit == components.scale_length.unit:    
        
        profile = exponential(radii, components.central_SB, components.scale_length) 
    else:
        raise UnitError(f'The unit of the radii ({radii.unit}) does not match the scale length ({components.scale_length.unit})')
    #profile = [x.value for x in profile]
    return profile
exponential_luminosity.__doc__ = f'''
 NAME:
    exponential_luminosity

 PURPOSE:
    Convert the components from the galfit file into luminosity profiles

 CATEGORY:
    optical

 INPUTS:
    components = the components read from the galfit file.

 OPTIONAL INPUTS:
    radii = the radii at which to evaluate the profile


 OUTPUTS:
   profile  = the luminosity profile
   lum_components = a set of homogenized luminosity components for the components in galfit file.

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''

def exponential_profile(components,radii = None):

    #lum_components = copy.deepcopy(components)
    #Ftot = 2πrs2Σ0q
    if radii is None:
        radii = components.radii
    
    if radii.unit == components.scale_length.unit:
        if components.height_type == 'inf_thin':
            profile= exponential(radii, components.central_SB, components.scale_length)  
            components.profile_type='sbr_dens' 
        else:
            #Equation 24 in Gentile and Baes
            profile = components.central_SB/(np.pi*components.scale_length)\
                *k0(radii/components.scale_length)
            components.profile_type='density' 
    else:
        raise UnitError(f'The unit of the radii ({radii.unit}) does not match the scale length ({components.scale_length.unit})')
    #profile = [x.value for x in profile]
    profile = extrapolate_zero(radii,profile)
    
    return profile
exponential_profile.__doc__ = f'''
 NAME:
    exponential_luminosity

 PURPOSE:
    Convert the components from the galfit file into luminosity profiles

 CATEGORY:
    optical

 INPUTS:
    components = the components read from the galfit file.

 OPTIONAL INPUTS:
    radii = the radii at which to evaluate the profile


 OUTPUTS:
   profile  = the luminosity profile
   lum_components = a set of homogenized luminosity components for the components in galfit file.

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''
 
# if we do not define these functions they cannot be pickled and not multiproceessed 
def hernexp_numpyro(Ltotal,hern_length,central,h,r):
    '''
    This is the sum of a Hernquist and an exponential profile
    '''
    value = hernquist_numpyro(Ltotal,hern_length,r) + exponential_numpyro(central,h,r)
   
    return value
def hernexp( r,Ltotal,hern_length,central,h):
    '''
    This is the sum of a Hernquist and an exponential profile
    '''
    value = hernquist(r,Ltotal,hern_length) + exponential(r,central,h)
   
    return value

def hernquist_numpyro(total_l,h,r):   
    '''
    This is the projected profile of a Hernquist density profile
    These are presented in Hernquist 1990 Eq 32 and what follows
    M_total/Gamma is replaced by L_total
    '''
   
    s = r/h
  
    XS_1 = 1./jnp.sqrt(1-s**2)*\
        jnp.log((1+jnp.sqrt(1-s**2))/s)
    XS_1 = jnp.where(s < 1,XS_1,0) #Casue jit need so know the size at compile time
    XS_2 = 1./jnp.sqrt(s**2-1)*1.*jnp.arccos(1./s)
    XS_2 = jnp.where(s > 1,XS_2,0) #Casue jit need so know the size at compile time
    XS = XS_1+XS_2
    profile = total_l/(2.*jnp.pi*h**2*\
        (1-s**2)**2)*((2+s**2)*XS-3)
   
    profile = extrapolate_zero_numpyro(r,profile)  
    return profile
  

def hernquist(r,total_l,h):
    '''
    This is the projected profile of a Hernquist density profile
    These are presented in Hernquist 1990 Eq 32 and what follows
    M_total/Gamma is replaced by L_total
    '''
    
    if h == 0.:
        h=1e-7

    s = r/h
    if isquantity(s):
        s = s.value
    XS_1 = 1./np.sqrt(1-s[s < 1]**2)*\
        np.log((1+np.sqrt(1-s[s<1]**2))/s[s < 1])
    XS_2 = 1./np.sqrt(s[s > 1]**2-1)*1.*np.arccos(1./s[s > 1])
    XS = np.array(list(XS_1)+list(XS_2),dtype=float)
    profile = total_l/(2.*np.pi*h**2*\
        (1-s**2)**2)*((2+s**2)*XS-3)
   
    profile = extrapolate_zero(r,profile)  
    return profile

def hernquist_luminosity(components,radii=None):
    if radii is None:
        radii = components.radii
    if radii.unit == components.hernquist_scale_length.unit:  
        profile = hernquist(radii, components.total_luminosity,components.hernquist_scale_length) 
    else:
        raise UnitError(f'The unit of the radii ({radii.unit}) does not match the scale length ({components.hernquist_scale_length.unit})')
    #profile = [x.value for x in profile]
    return profile
  
def hernquist_profile(components,radii=None):
    '''
    The hernquist density profile (Eq 2, Hernquist 1990)
    mass/(np.pi*2)*(scale_length/radii)*1./(radii+scale_length)**3
    Note that in galpy this is amp/(4*pi*a**3)*1./((r/a)(1+r/a)**2
    With amp = 2. mass
    Both have inf at r = 0. so if radii == 0 it needs to be adapted

    We want to fit the luminosity to a profile and then relate
    back Re = 1.8153 a (eq38) and  r_1/2 = 1.33 R_e
   
    This is a density profile with M/pc**3

    '''
    if radii is None:
        radii = components.radii
    if radii.unit == components.hernquist_scale_length.unit:  
        a = components.hernquist_scale_length
        profile = components.total_mass/(2.*np.pi)*a/radii*1./(radii+a)**3
    else:
        raise UnitError(f'The unit of the radii ({radii.unit}) does not match the scale length ({components.scale_length.unit})')
    profile = extrapolate_zero(radii,profile)
  
    return profile 
def sersic_numpyro(effective_luminosity,effective_radius,n,r):
    b = get_sersic_b_numpyro(n) 
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            profile = effective_luminosity*jnp.exp(-1.*b*((r/effective_radius)**(1./n)-1))
        except RuntimeWarning as e:
            if 'overflow encountered in power' in str(e):
                profile = jnp.zeros_like(r)
            else:
                raise e
    return profile
def sersic(r,effective_luminosity,effective_radius,n):
    b = get_sersic_b(n) 
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            profile = effective_luminosity*np.exp(-1.*b*((r/effective_radius)**(1./n)-1))
        except RuntimeWarning as e:
            if 'overflow encountered in power' in str(e):
                profile = np.zeros_like(r)
            else:
                raise e

        # This is the deprojected surface density profile from Baes & gentile 2010 Equation 22
        # With this it should be possible use an Einasto profile/potential


    return profile

def sersic_luminosity(components,radii=None):
    '''sersic function'''
    # as b/kappa should be numerically solved from the function Kapp = 2*gamm(2n,b) we use the astropy function
    #kappa = -1.*(2.*n-1./3.)
    #func = effective_luminosity*np.exp(kappa*((r/effective_radius)**(1./n))-1)
    if radii is None:
        radii = components.radii
    if radii.unit == components.R_effective.unit: 
        profile = sersic(radii, components.L_effective, components.R_effective,\
                         components.sersic_index ) 
    else:
        raise UnitError(f'The unit of the radii ({radii.unit}) does not match the scale length ({components.scale_length.unit})')
    #profile = [x.value for x in profile]
    return profile
     
def sersic_profile(components,radii = None):
    # This is the deprojected surface density profile from Baes & gentile 2010 Equation 22
    # With this it should be possible use an Einasto profile/potential
    if radii is None:
        radii = components.radii
    #first we need to derive the integer numbers that make up the  sersic index
 
    p, q = get_integers(components.sersic_index)
    # The a and b vectors of equation 22
 
    avect = np.array([x/q for x in range(1,q)],dtype=float)
    bvect = np.array([x/(2.*p) for x in range(1,2*p)]+\
            [x/(2.*q) for x in range(1,2*q,2)],dtype=float)
   
    # Obtain the b vector:
 
    b = get_sersic_b(components.sersic_index)
    s = radii.value/components.R_effective.value
 
    # front factor # This is lacking an 1./R_effective because it cancels with 1./s later on
    const = 2.*components.central_SB*np.sqrt(p*q)/(2*np.pi)**p
    
    # The meijer g function insympy does not accept arrays
    # We could consider mpmath but it is an additional package
    meijer_result = []
    for s_ind in s:
        meijer_input =  (b/(2*p))**(2*p) * s_ind**(2*q)
        meijer_result.append(meijerg([[],avect],[bvect,[]],meijer_input).evalf())
   
    meijer_result = np.array(meijer_result,dtype=float)
    #This is with 1/rad instead of 1/s as we dropped the R_eff from the const 
    # central_SB is M/pc**2 with /rad makes M/pc**3 ----> This is a problem for cassertano
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        density_profile =  const/radii.to(unit.pc)*meijer_result
        density_profile = extrapolate_zero(radii,density_profile)           
    return density_profile
sersic_profile.__doc__ = f'''
NAME:
    sersic_luminosity

PURPOSE:
   Convert the components from the galfit file into density profiles

CATEGORY:
   optical

INPUTS:
   components = the components read from the galfit file.

OPTIONAL INPUTS:
   radii = []
       the radii at which to evaluate
   zero_point_flux =0.
       The magnitude zero point flux value. Set by selecting the correct band.
   distance  = 0.
       Distance to the galaxy
   log = None
   debug = False

OUTPUTS:
  lum_ profile  = the luminosity profile
  lum_components = a set of homogenized luminosity components for the components in galfit file.

OPTIONAL OUTPUTS:

PROCEDURES CALLED:
   Unspecified

NOTE: This is not the correct way to get a deprojected profile should use the method from Baes & Gentile 2010
 !!!!!!!!!
'''

def get_integers(n):
    # This is a simple function to get the integers that make up the sersic index
    # to limit the array sizes we round n to 5 decimals 
    n = round(n,5)
    solution= Fraction(n).limit_denominator()
    return int(solution.numerator),int(solution.denominator)

   
def get_sersic_b_numpyro(n):
    #We cannot have an exact calculation of b as thereis no inverse gamminc function in jax
    # So we need to use the assymtotic appr wiihch iss only valid for n > 0.36 
    # Ciotti & Bertin (1999)
    b = (2*n - 1./3 + (4./405)*(n**(-1)) + (46./25515)*(n**(-2)) + (
        131./1148175)*(n**(-3)) - (2194697./30690717750)*(n**(-4)))
    return b    
def get_sersic_b(sersic_index):
        # Get the gamma function to calculate b
    b =  gammaincinv(2. * sersic_index, 0.5)    
    return b
