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

class CalculateError(Exception):
    pass


def convert_parameters_to_luminosity(components, band='SPITZER3.6',distance= 0.,
                                    log=None, debug=False):

    lum_components =copy.deepcopy(components)
    # We plot a value every 2 pixels out to 7 x the scale length and finish with a 0.
    disk = []
    bulge = []
    sersic_disk = []
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
                lum_components[key]['Central SB'] = sup.integrate_surface_density(sup.convertskyangle(components['radii'],distance,quantity=True),tmp)
                lum_components[key]['axis ratio'] = lum_components[key]['scale height']/lum_components[key]['scale length']
                disk.append(tmp)

            elif components[key]['Type'] in ['sersic']:
                tmp,tmp_components = sersic_luminosity(components[key],radii = components['radii'],
                                            band = band,distance= distance)
                if (0.75 < tmp_components['sersic index']< 1.25):
                    sup.print_log(f''' Your sersic index = {tmp_components['sersic index']} which is so close to 1 that we will treat this as an exponential disk.
''',log)
                    lum_components[key] = tmp_components
                    disk.append(tmp)
                elif 3.75 < tmp_components['sersic index'] < 4.25:
                    sup.print_log(f''' Your sersic index = {tmp_components['sersic index']} which is so close to 4 that we will treat this as a Bulge.
''',log)
                    lum_components[key] = tmp_components
                    bulge.append(tmp)
                else:
                    sup.print_log(f''' Your sersic index = {tmp_components['sersic index']} which cannot be estimated by a density profile.
''',log)
                    lum_components[key] = tmp_components
                    sersic_disk.append(tmp)


    organized = {}
    organized['RADI'] = ['RADI','ARCSEC']+[x.value for x in components['radii']]
    for i,disks in enumerate(disk):
        organized[f'EXPONENTIAL_{i+1}'] = [f'EXPONENTIAL_{i+1}',f'L_SOLAR/PC^2']\
                    +[x.value if x.unit == unit.Lsun/unit.pc**2 else float('NaN') for x in disks]
        if float('NaN') in organized[f'EXPONENTIAL_{i+1}']:
            print(f'We got {organized[f"EXPONENTIAL_{i+1}"]}')
            raise CalculateError(f'Something went wrong in EXPONENTIAL_{i+1}')
    for i,bulges in enumerate(bulge):
        organized[f'BULGE_{i+1}'] = [f'BULGE_{i+1}',f'L_SOLAR/PC^2']\
                    +[x.value if x.unit == unit.Lsun/unit.pc**2 else float('NaN') for x in bulges]
        if float('NaN') in organized[f'BULGE_{i+1}']:
            print(f'We got {organized[f"BULGE_{i+1}"]}')
            raise CalculateError(f'Something went wrong in BULGE_{i+1}')
    for i,sersic_disks in enumerate(sersic_disk):
        organized[f'SERSIC_{i+1}'] = [f'SERSIC_{i+1}',f'L_SOLAR/PC^2']\
                    +[x.value if x.unit == unit.Lsun/unit.pc**2 else float('NaN') for x in sersic_disks]
        if float('NaN') in organized[f'SERSIC_{i+1}']:
            print(f'We got {organized[f"SERSIC_{i+1}"]}')
            raise CalculateError(f'Something went wrong in SERSIC_{i+1}')

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
        radii = sup.convertskyangle(radii,distance,quantity=True)
    central_luminosity = mag_to_lum(components['Central SB'],band = band, distance=distance)

    #Need to integrate the luminosity of the model and put it in component[1] still
    lum_components['Central SB'] = central_luminosity
    if lum_components['scale length'].unit != unit.kpc:
        lum_components['scale length'] = sup.convertskyangle(components['scale length'],distance,quantity=True)
    if lum_components['scale height'].unit != unit.kpc:
        lum_components['scale height'] = sup.convertskyangle(components['scale height'],distance,quantity=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lum_profile = central_luminosity*radii/components['scale length']*k1(radii.value/components['scale length'].value)
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

def exponential_luminosity(components,radii = [],band = 'WISE3.4',distance= 0.):
    lum_components = copy.deepcopy(components)
    #Ftot = 2πrs2Σ0q

    IntLum = mag_to_lum(components['Total SB'],band = band, distance=distance)
    # and transform to a face on total magnitude (where does this come from?)
    lum_components['Total SB'] = IntLum/components['axis ratio']
    # convert the scale length to kpc
    lum_components['scale length'] = sup.convertskyangle(components['scale length'],distance, quantity=True)
    # this assumes perfect ellipses for now and no deviations are allowed
    central_luminosity = IntLum/(2.*np.pi*lum_components['scale length'].to(unit.pc)**2*components['axis ratio']) #L_solar/pc^2

    lum_components['Central SB'] = central_luminosity
    if radii.unit != unit.kpc:
        radii = sup.convertskyangle(radii,distance,quantity=True)
    profile= central_luminosity*np.exp(-1.*(radii/lum_components['scale length']))
    #profile = [x.value for x in profile]
    return profile,lum_components
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


def get_optical_profiles(filename,distance = 0.,band = 'SPITZER3.6',exposure_time=1.,\
                            MLRatio = 0.6, log =None,debug=False, scale_height=None):
    '''Read in the optical Surface brightness profiles or the galfit file'''
    # as we do a lot of conversions in the optical module we make distance a quantity with unit Mpc
    distance = distance * unit.Mpc
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
        optical_profiles,lum_components = convert_parameters_to_luminosity(\
            components,band = band,distance= distance,debug=debug,log=log)
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
                    pc_conv = sup.convertskyangle(1,distance.value)*1000.
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
                cleaned_components.append({'Type': None,
                                    'Central SB': None ,
                                    'Total SB': None ,
                                    'R effective': None ,
                                    'scale height':None,
                                    'scale length':None,
                                    'sersic index':None,
                                    'central position':None,
                                    'axis ratio':None,
                                    'PA':None})

                if types[:3] == 'EXP':
                    cleaned_components[-1]['Type']= 'expdisk'
                elif types[:3] == 'SER':
                    cleaned_components[-1]['Type']= 'sersic'

                elif types[:3] == 'BUL':
                    cleaned_components[-1]['Type']= 'bulge'
                elif types.split('_')[0] == 'DISK' and \
                     types.split('_')[1] in ['S','STELLAR']:
                    cleaned_components[-1]['Type']= 'infinite_disk'
                else:
                    raise InputError(f'We do not know how to deal with the column {types}. Acceptable input is SERSIC, EXPONENTIAL, BULGE')

    for types in optical_profiles:
        if types != 'RADI':
            if optical_profiles[types][1] == 'L_SOLAR/PC^2':
                optical_profiles[types][1] = 'M_SOLAR/PC^2'
                optical_profiles[types][2:] = [float(y)*MLRatio for y in optical_profiles[types][2:]]
    if galfit_file:
        for i in range(len(cleaned_components)):
            if cleaned_components[i]['Central SB']:
                cleaned_components[i]['Central SB'] = cleaned_components[i]['Central SB']*MLRatio
            if cleaned_components[i]['Total SB']:
                cleaned_components[i]['Total SB'] = cleaned_components[i]['Total SB']*MLRatio

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
        factor = (21.56+co.solar_magnitudes[band])
        inL= (10**(-0.4*(mag.value-factor)))*unit.Lsun/unit.parsec**2  #L_solar/pc^2
    elif mag.unit == unit.mag:
        M= mag-2.5*np.log10((distance/(10.*unit.pc))**2)*unit.mag # Absolute magnitude
        inL= (10**(-0.4*(M.value-co.solar_magnitudes[band])))*unit.Lsun # in band Luminosity in L_solar
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
                                                                                                                    'Central SB': None ,
                                                                                                                    'Total SB': None ,
                                                                                                                    'R effective': None ,
                                                                                                                    'scale height':None,
                                                                                                                    'scale length':None,
                                                                                                                    'sersic index':None,
                                                                                                                    'central position':None,
                                                                                                                    'axis ratio':None,
                                                                                                                    'PA':None,}
                    if current_component in ['expdisk','sersic']:
                        components[f'{current_component}_{counter[recognized_components.index(current_component)]}']['scale height'] = 0. * unit.kpc


            if tmp[0] == '3)' and read_component:
                if  current_component in ['edgedisk']:
                    components[f'{current_component}_{counter[recognized_components.index(current_component)]}']['Central SB'] = float(tmp[1])*unit.mag/unit.arcsec**2
                else:
                    components[f'{current_component}_{counter[recognized_components.index(current_component)]}']['Total SB'] = float(tmp[1])*unit.mag
            if tmp[0] == '4)' and read_component:
                if current_component in ['sersic']:
                    components[f'{current_component}_{counter[recognized_components.index(current_component)]}']['R effective'] = float(tmp[1])*np.mean(plate_scale)*unit.arcsec
                    if max_radius < 5* float(tmp[1]): max_radius = 5 * float(tmp[1])

                if current_component in ['edgedisk']:
                    components[f'{current_component}_{counter[recognized_components.index(current_component)]}']['scale height'] = float(tmp[1])*np.mean(plate_scale)*unit.arcsec
                if current_component in ['expdisk']:
                    components[f'{current_component}_{counter[recognized_components.index(current_component)]}']['scale length'] = float(tmp[1])*np.mean(plate_scale)*unit.arcsec
                    components[f'{current_component}_{counter[recognized_components.index(current_component)]}']['scale height'] = 0.*unit.arcsec

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


def sersic_luminosity(components,radii = [],band = 'WISE3.4',distance= 0.):
    # This is not a deprojected surface density profile we should use the formula from Baes & gentile 2010
    # Once we implement that it should be possible use an Einasto profile/potential
    lum_components = copy.deepcopy(components)
    IntLum = mag_to_lum(components['Total SB'],band=band, distance=distance)
    kappa=2.*components['sersic index']-1./3. # From https://en.wikipedia.org/wiki/Sersic_profile
    if components['R effective'].unit != unit.kpc:
        R_kpc = sup.convertskyangle(components['R effective'],distance,quantity =True)

    if radii.unit != unit.kpc:
        radii = sup.convertskyangle(radii,distance,quantity=True)
    effective_luminosity = IntLum/(2.*np.pi*(R_kpc.to(unit.pc))**2*np.exp(kappa)*components['sersic index']*\
                                kappa**(-2*components['sersic index'])*components['axis ratio']*gamma(2.*components['sersic index'])) #L_solar/pc^2
    lum_components['Total SB'] = IntLum/components['axis ratio']
    lum_components['R effective'] = R_kpc
    lum_profile = effective_luminosity*np.exp(-1.*kappa*((radii/lum_components['R effective'])**(1./components['sersic index'])-1))
    lum_components['Central SB'] = effective_luminosity*np.exp(-1.*kappa*(((0.*unit.kpc)/lum_components['R effective'])**(1./components['sersic index'])-1))
    return lum_profile,lum_components
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
