# -*- coding: future_fstrings -*-

import numpy as np
import warnings
import pyROTMOD.constants as co
import pyROTMOD.support as sup
from scipy.special import k1,gamma
from astropy import units as unit
import copy

class BadFileError(Exception):
    pass
class InputError(Exception):
    pass



def convert_parameters_to_luminosity(components, band='SPITZER3.6',distance= 0.,debug=False):
    lum_components =copy.deepcopy(components)
    # We plot a value every 2 pixels out to 7 x the scale length and finish with a 0.
    disk = []
    bulge = []
    for key in components:
        if key[0:4].lower() in ['expd','sers','edgedisk']:
            if components[key]['Type'] in ['expdisk']:
                tmp,tmp_components = exponential_luminosity(components[key],radii = components['radii'],
                                            band = band,distance= distance)
                lum_components[key] = tmp_components
                disk.append(tmp)
            elif components[key]['Type'] in ['edgedisk']:
                tmp,tmp_components = edge_luminosity(components[key],radii = components['radii'],band=band, t_exp = components['exposure_time'], distance= distance)
                lum_components[key] = tmp_components
                lum_components[key][1] = sup.integrate_surface_density(sup.convertskyangle(components['radii'],distance),tmp)
                lum_components[key][5] = lum_components[key][4]/lum_components[key][3]
                disk.append(tmp)

            elif components[key]['Type'] in ['sersic']:
                tmp,tmp_components = sersic_luminosity(components[key],radii = components['radii'],
                                        zero_point_flux=zero_point_flux,distance= distance,
                                        wavelength= components['wavelength'])
                lum_components[key] = tmp_components
                bulge.append(tmp)

    organized = {}
    organized['RADI'] = ['RADI','ARCSEC']+[x for x in components['radii']]
    for i,disks in enumerate(disk):
        organized[f'EXPONENTIAL_{i+1}'] = [f'EXPONENTIAL_{i+1}',f'L_SOLAR/PC^2']+[x for x in disks]
    for i,bulges in enumerate(bulge):
        organized[f'SERSIC_{i+1}'] = [f'SERSIC_{i+1}',f'L_SOLAR/PC^2']+[x for x in bulges]

    return organized,lum_components
convert_parameters_to_luminosity.__doc__ =f'''
 NAME:
    convert_parameters_to_luminosity(components,zero_point_flux=0.,distance= 0.,debug=False):

 PURPOSE:
    Convert the components from the galfit file into luminosity profiles

 CATEGORY:
    optical

 INPUTS:
    components = the components read from the galfit file.

 OPTIONAL INPUTS:
    zero_point_flux =0.
        The magnitude zero point flux value. Set by selecting the correct band.
    distance  = 0.
        Distance to the galaxy
    log = None
    debug = False

 OUTPUTS:
   organized = the luminosity profile for each component
   lum_components = a set of homogenized luminosity components for the components in galfit file.

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''
    # This is untested for now
def edge_luminosity(components,radii = [],exposure_time = 1.,band = 'WISE3.4'
                    ,distance=0 ):
    lum_components = copy.deepcopy(components)

    #mu_0 (mag/arcsec) = -2.5*log(sig_0/(t_exp*dx*dy))+mag_zpt
    #this has to be in L_solar/pc^2 but our value comes in mag/arcsec^2
    if radii.unit != unit.kpc:
        radii = sup.convertskyangle(radii.value,distance)*unit.kpc
    central_luminosity = float(mag_to_lum(components['Central Magnitude'],band = band, distance=distance))

    #Need to integrate the luminosity of the model and put it in component[1] still
    lum_components['Central Magnitude'] = central_luminosity
    lum_components['scale length'] = sup.convertskyangle(components['scale length'].value,distance)*unit.kpc
    lum_components['scale height'] = sup.convertskyangle(components['scale height'].value,distance)*unit.kpc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lum_profile = central_luminosity*radii.value/components['scale length'].value*k1(radii.value/components['scale length'].value)
        if np.isnan(lum_profile[0]):
            lum_profile[0] = central_luminosity
    # this assumes perfect ellipses for now and no deviations are allowed
    return lum_profile,lum_components
 edge_luminosity.__doc__ = f'''
 NAME:
     edge_luminosity

 PURPOSE:
    Convert the components from the galfit file into luminosity profiles

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

 NOTE: This is not well tested yet !!!!!!!!!
'''

def exponential_luminosity(components,radii = [],band = 'WISE3.4',distance= 0.,texp=1.):
    lum_components = copy.deepcopy(components)
    #Ftot = 2πrs2Σ0q

    IntLum = mag_to_lum(components['Total Magnitude'],band = band, distance=distance)
    # and transform to a face on total magnitude (where does this come from?)
    lum_components['Total Magnitude'] = IntLum/components['axis ratio']
    # convert the scale length to kpc
    lum_components['scale length'] = sup.convertskyangle(components['scale length'].value,distance)*unit.kpc
    # this assumes perfect ellipses for now and no deviations are allowed
    central_luminosity = IntLum/(2.*np.pi*lum_components['scale length'].to(unit.pc)**2*components['axis ratio']) #L_solar/pc^2

    lum_components['Central Magnitude'] = central_luminosity
    if radii.unit != unit.kpc:
        radii = sup.convertskyangle(radii.value,distance)*unit.kpc
    return central_luminosity*np.exp(-1.*(radii.value/float(components['scale length'].value))),lum_components
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
    zero_point_flux =0.
        The magnitude zero point flux value. Set by selecting the correct band.
    distance  = 0.
        Distance to the galaxy
    log = None
    debug = False

 OUTPUTS:
   profile  = the luminosity profile
   lum_components = a set of homogenized luminosity components for the components in galfit file.

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''


def get_optical_profiles(filename,distance = 0.,band = 'SPITZER3.6',exposure_time=1.,\
                            MLRatio = 0.6, log =None,debug=False, scale_height=None):
    '''Read in the optical Surface brightness profiles or the galfit file'''
    sup.print_log(f" We are reading the optical parameters from {filename}. \n",log, screen =True)
    if distance == 0.:
        raise InputError(f'We cannot convert profiles adequately without a distance')
    with open(filename) as file:
        input = file.readlines()

    firstline = input[0].split()
    correctfile  = False
    try:
        if firstline[0].strip().lower() == 'radi':
            correctfile = True
    except:
        pass
    galfit_file = False
    vel_found = False
    # If the first line and first column is not correct we assume a Galfit file
    if not correctfile:
        components = read_galfit(input,log=log,debug=debug)
        components['exposure_time'] = exposure_time*unit.second
        print(components)
        optical_profiles,lum_components = convert_parameters_to_luminosity(\
            components,band = band,distance= distance,debug=debug)
        galfit_file = True
        # Clean components to only contain the profiles in a list not a dictionary
        cleaned_components = []
        for key in lum_components:
            if key[0:4].lower() in ['expd','sers','edge']:
                cleaned_components.append(lum_components[key])

    else:
        optical_profiles = sup.read_columns(filename,debug=debug,log=log)
        found = False
        cleaned_components = []
        for types in optical_profiles:
            optical_profiles[types] = optical_profiles[types][:2]+[float(x) for x in optical_profiles[types][2:]]
            if types != 'RADI':
                if optical_profiles[types][1] in ['L_SOLAR/PC^2','M_SOLAR/PC^2']:
                    if not found:
                        found = True
                    if vel_found:
                        raise InputError('Your optical file mixes velocities with densities. We can not deal with this')
                elif optical_profiles[types][1] in ['MAG/ARCSEC^2','MAG/ARCSEC^2']:
                    if not found:
                        found = True
                    if vel_found:
                        raise InputError('Your optical file mixes velocities with densities. We can not deal with this')
                    wavelength = co.bands[band]
                    pc_conv = sup.convertskyangle(1,distance)*1000.
                    #optical_profiles[types] = [types,'L_SOLAR/PC^2']+\
                    #                          [float(mag_to_lum(x,\
                    #                                 zero_point_flux=co.zero_point_fluxes[band]\
                    #                                 ,wavelength = co.bands[band],distance= distance))\
                    #                                 /pc_conv**2 for x in optical_profiles[types][2:]]

                    optical_profiles[types] = [types,'L_SOLAR/PC^2']+\
                                              [float(mag_to_lum(x*unit.mag/unit.arcsec**2,\
                                                     band = band,distance= distance))\
                                                     /pc_conv**2 for x in optical_profiles[types][2:]]
                elif  optical_profiles[types][1] in ['KM/S','M/S']:
                    if not vel_found:
                        vel_found = True
                    if found:
                        raise InputError('Your optical file mixes velocities with densities. We can not deal with this')
                    if optical_profiles[types][1] == 'M/S':
                        optical_profiles[types][2:]=[float(y)/1000. for y in optical_profiles[types][2:]]
                        optical_profiles[types][1] = 'KM/S'
                else:
                    raise InputError(f'We do not know how to deal with the unit {optical_profiles[types][1]}. Acceptable input is M_SOLAR/PC^2, KM/S, M/S')
            # otherwise we extract the columns and such from the file
                if types[:3] == 'EXP':
                    cleaned_components.append(['expdisk',0])
                elif types[:3] == 'SER':
                    cleaned_components.append(['sersic',0])
                elif types[:3] == 'BUL':
                    cleaned_components.append(['bulge',0])
                elif types.split('_')[0] == 'DISK' and \
                     types.split('_')[1] in ['S','STELLAR']:
                    cleaned_components.append(['inifite_disk',0])
                else:
                    raise InputError(f'We do not know how to deal with the column {types}. Acceptable input is SERSIC, EXPONENTIAL, BULGE')

    for types in optical_profiles:
        optical_profiles[types]=optical_profiles[types][:2]+[float(x) for x in optical_profiles[types][2:]]
        if types != 'RADI':
            if optical_profiles[types][1] == 'L_SOLAR/PC^2':
                optical_profiles[types][1] = 'M_SOLAR/PC^2'
                optical_profiles[types][2:] = [float(y)*MLRatio for y in optical_profiles[types][2:]]
    if galfit_file:
        for i in range(len(cleaned_components)):
            cleaned_components[i][1] = cleaned_components[i][1]*MLRatio
            cleaned_components[i][2] = cleaned_components[i][2]*MLRatio

    return optical_profiles,cleaned_components,galfit_file,vel_found

get_optical_profiles.__doc__ =f'''
 NAME:
    get_optical_profiles

 PURPOSE:
    Read in the optical file provided and convert the input to homogenuous profiles to be used by rotmod.

 CATEGORY:
    optical

 INPUTS:
    filename = name of the file to be read.

 OPTIONAL INPUTS:
    distance = 0.
        Distance to the galaxy for converting flux parameters required for a galfit file or when input is in magnitude/arcsec^2

    exposure_time = 1.
        certain galfit components require a exposure time from the header of the image.

    band = 'SPITZER3.6'
        band used for the observation
    MLRatio = 0.6
        mass to light ratio used for light profile. Note that this is used to multiply the luminosity profile.
        Hence when doing the mass decomposition this factor is incorporated. Set to 1. if you want to get  the MD and MB from the decomposition.

    log = None
    debug = False
 OUTPUTS:

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''

def mag_to_lum(mag,band = 'WISE3.4',distance= 0.,debug = False):
    if mag.unit == unit.mag/unit.arcsec**2:
        # Surface brightness is constant with distance and hence works differently
        #from Oh 2008.
        factor = (21.56+co.solar_magnitudes[band])*unit.mag
        inL= (10**(-0.4*(mag-factor)))*unit.Lsun/unit.parsec**2  #L_solar/pc^2
    elif mag.unit == unit.mag:
        M= mag.value-2.5*np.log10((distance*1e6/10.)**2) # Absolute magnitude
        inL= (10**(-0.4*(M-co.solar_magnitudes[band])))*unit.Lsun # in band Luminosity in L_solar
    else:
        raise InputError('MAG_to_LUM: this unit is not recognized for the magnitude')
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

def read_galfit(lines,log=None,debug=False):
    recognized_components = ['expdisk','sersic','edgedisk']
    counter = [0 for x in recognized_components]
    mag_zero = []
    plate_scale = []
    read_component = False
    components = {}
    max_radius = 0.
    for line in lines:
        tmp = [x.strip().lower() for x in line.split()]

        if len(tmp) > 0:
            if tmp[0] == 'j)':
                mag_zero = [float(tmp[1])]
            if tmp[0] == 'k)':
                plate_scale = [float(tmp[1]), float(tmp[2])] # [arcsec per pixel]
            if tmp[0] == 'z)':
                read_component = False
            if len(tmp) > 1:
                if tmp[1] == 'component':
                    read_component = True
            if tmp[0] == '0)' and read_component:
                current_component = tmp[1]
                if current_component not in recognized_components:
                    sup.print_log(f'''pyROTMOD does not know how to process {current_component} not reading it
''',log)
                    read_component = False
                else:
                    counter[recognized_components.index(current_component)] += 1
                    components[f'{current_component}_{counter[recognized_components.index(current_component)]}'] = {'Type':current_component,\
                                                                                                                    'Central Magnitude': None ,
                                                                                                                    'Total Magnitude': None ,
                                                                                                                    'R effective': None ,
                                                                                                                    'scale height':None,
                                                                                                                    'scale lenght':None,
                                                                                                                    'sersic index':None,
                                                                                                                    'central position':None,
                                                                                                                    'axis ratio':None,
                                                                                                                    'PA':None,}

            if tmp[0] == '3)' and read_component:
                if  current_component in ['edgedisk']:
                    components[f'{current_component}_{counter[recognized_components.index(current_component)]}']['Central Magnitude'] = float(tmp[1])*unit.mag/unit.arcsec**2
                else:
                    components[f'{current_component}_{counter[recognized_components.index(current_component)]}']['Total Magnitude'] = float(tmp[1])*unit.mag
            if tmp[0] == '4)' and read_component:
                if current_component in ['sersic']:
                    components[f'{current_component}_{counter[recognized_components.index(current_component)]}']['R effective'] = float(tmp[1])*np.mean(plate_scale)*unit.arcsec
                    if max_radius < 5* float(tmp[1]): max_radius = 5 * float(tmp[1])
                if current_component in ['edgedisk']:
                    components[f'{current_component}_{counter[recognized_components.index(current_component)]}']['scale height'] = float(tmp[1])*np.mean(plate_scale)*unit.arcsec
                if current_component in ['expdisk']:
                    components[f'{current_component}_{counter[recognized_components.index(current_component)]}']['scale length'] = float(tmp[1])*np.mean(plate_scale)*unit.arcsec

                    if max_radius < 10 * float(tmp[1]): max_radius = 10 * float(tmp[1])


            if tmp[0] == '5)' and read_component and current_component in ['sersic','edgedisk']:
                if current_component == 'sersic':
                    components[f'{current_component}_{counter[recognized_components.index(current_component)]}']['sersic index'] = float(tmp[1])
                elif current_component in ['edgedisk']:
                    if max_radius < 10 * float(tmp[1]): max_radius = 10 * float(tmp[1])
                    components[f'{current_component}_{counter[recognized_components.index(current_component)]}']['scale length'] = float(tmp[1])*np.mean(plate_scale)*unit.arcsec

            if tmp[0] == '9)' and read_component and current_component in ['expdisk','sersic']:
                components[f'{current_component}_{counter[recognized_components.index(current_component)]}']['axis ratio'] = float(tmp[1])
            if tmp[0] == '10)' and read_component:
                components[f'{current_component}_{counter[recognized_components.index(current_component)]}']['PA'] = float(tmp[1])*unit.degree

    if len(plate_scale) == 0 or len(mag_zero) == 0:
        raise BadFileError(f'Your file  is not recognized by pyROTMOD')
    components['radii'] = np.linspace(0,max_radius,int(max_radius/2.))*np.mean(plate_scale)*unit.arcsec # in arcsec
    components['plate_scale'] = plate_scale*unit.arcsec #[arcsec per pixel]
    components['magnitude_zero'] = mag_zero*unit.mag
    # Components are returned as [type,integrated magnitude, central magnitude/arcsec^2 ,scale parameter in arcsec,sercic index or scaleheight in arcsec, axis ratio]
    return components

read_galfit.__doc__ =f'''
 NAME:
    read_galfit

 PURPOSE:
    Read in the galfit file and extract the parameters for each component in there

 CATEGORY:
    optical

 INPUTS:
    lines = the string instance of the opened file

 OPTIONAL INPUTS:
    log = None
    debug = False

 OUTPUTS:
   components = set of parameters with units for each component as well as some global components



 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''


def sersic_luminosity(components,radii = [],zero_point_flux=0.,distance= 0.,wavelength = 0.):
    # This is not a deprojected surface density profile we should use the formula from Baes & gentile 2010
    lum_components = copy.deepcopy(components)
    IntLum = mag_to_lum(components[1],zero_point_flux=zero_point_flux,
                        wavelength=wavelength, distance=distance)
    kappa=2.*components[4]-1./3. # From https://en.wikipedia.org/wiki/Sersic_profile
    h_kpc = sup.convertskyangle(components[3],distance)
    central_luminosity = IntLum/(2.*np.pi*(h_kpc*1000.)**2*np.exp(kappa)*components[4]*\
                                kappa**(-2*components[4])*components[5]*gamma(2.*components[4])) #L_solar/pc^2
    lum_components[1] = IntLum/components[5]
    lum_components[2] = central_luminosity
    lum_components[3] = sup.convertskyangle(components[3],distance)
    return central_luminosity*np.exp(-1.*kappa*((radii/components[3])**(1./components[4])-1)),lum_components
sersic_luminosity.__doc__ = f'''
NAME:
    sersic_luminosity

PURPOSE:
   Convert the components from the galfit file into luminosity profiles

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
