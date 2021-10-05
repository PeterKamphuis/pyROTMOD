# -*- coding: future_fstrings -*-

import numpy as np
import warnings
import pyROTMOD.constants as c
import pyROTMOD.support as sup
from scipy.special import k1,gamma
import copy

class BadFileError(Exception):
    pass
class InputError(Exception):
    pass

def get_optical_profiles(filename,distance = 0.,\
                            band = 'SPITZER3.6',exp_time = -1.,luminosity = False,
                            MLRatio = 0.6, log =None,debug=False ):
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
        components['wavelength'] = c.bands[band]
        components['exp_time'] = exp_time
        optical_profiles,lum_components = convert_parameters_to_luminosity(\
            components,zero_point_flux=c.zero_point_fluxes[band],distance= distance,debug=debug)
        galfit_file = True
        # Clean components to only contain the profiles in a list not a dictionary
        cleaned_components = []
        for key in lum_components:
            if key[0:4].lower() in ['expd','sers','edge']:
                cleaned_components.append(lum_components[key])

    else:
        optical_profiles = sup.read_columns(filename,debug=debug)
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
                    wavelength = c.bands[band]
                    pc_conv = sup.convertskyangle(1,distance)*1000.
                    optical_profiles[types] = [types,'L_SOLAR/PC^2']+\
                                              [float(mag_to_lum(x,\
                                                     zero_point_flux=c.zero_point_fluxes[band]\
                                                     ,wavelength = c.bands[band],distance= distance))\
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
                else:
                    raise InputError(f'We do not know how to deal with the column {types}. Acceptable input SERSIC, EXPONENTIAL, BULGE')






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
    #if len(firstline) = 0 :
    #    if firstline[0].lower() !=

    return optical_profiles,cleaned_components,galfit_file,vel_found



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
                    components[f'{current_component}_{counter[recognized_components.index(current_component)]}'] = [current_component,0.,0.,0.,0.,0.]
            if tmp[0] == '3)' and read_component:
                if  current_component in ['edgedisk']:
                    components[f'{current_component}_{counter[recognized_components.index(current_component)]}'][2] = float(tmp[1])
                else:
                    components[f'{current_component}_{counter[recognized_components.index(current_component)]}'][1] = float(tmp[1])
            if tmp[0] == '4)' and read_component:
                if current_component in ['expdisk','sersic']:
                    components[f'{current_component}_{counter[recognized_components.index(current_component)]}'][3] = float(tmp[1])*np.mean(plate_scale)
                if current_component in ['edgedisk']:
                    components[f'{current_component}_{counter[recognized_components.index(current_component)]}'][4] = float(tmp[1])*np.mean(plate_scale)
                if current_component in ['expdisk']:
                    if max_radius < 10 * float(tmp[1]): max_radius = 10 * float(tmp[1])
                elif current_component in ['sersic']:
                    if max_radius < 5* float(tmp[1]): max_radius = 5 * float(tmp[1])
            if tmp[0] == '5)' and read_component and current_component in ['sersic','edgedisk']:
                if current_component == 'sersic':
                    components[f'{current_component}_{counter[recognized_components.index(current_component)]}'][4] = float(tmp[1])
                elif current_component in ['edgedisk']:
                    if max_radius < 10 * float(tmp[1]): max_radius = 10 * float(tmp[1])
                    components[f'{current_component}_{counter[recognized_components.index(current_component)]}'][3] = float(tmp[1])*np.mean(plate_scale)
                else:
                    components[f'{current_component}_{counter[recognized_components.index(current_component)]}'][4] = float(tmp[1])*np.mean(plate_scale)
            if tmp[0] == '9)' and read_component and current_component in ['expdisk','sersic']:
                components[f'{current_component}_{counter[recognized_components.index(current_component)]}'][5] = float(tmp[1])
    if len(plate_scale) == 0 or len(mag_zero) == 0:
        raise BadFileError(f'Your file  is not recognized by pyROTMOD')
    components['radii'] = np.linspace(0,max_radius,int(max_radius/2.))*np.mean(plate_scale) # in arcsec
    components['plate_scale'] = plate_scale #[arcsec per pixel]
    components['magnitude_zero'] = mag_zero
    # Components are returned as [type,integrated magnitude, central magnitude/arcsec^2 ,scale parameter in arcsec,sercic index or scaleheight in arcsec, axis ratio]
    return components



def convert_parameters_to_luminosity(components,zero_point_flux=0.,distance= 0.,debug=False):
    lum_components =copy.deepcopy(components)
    # We plot a value every 2 pixels out to 7 x the scale length and finish with a 0.
    disk = []
    bulge = []
    for key in components:
        if key[0:4].lower() in ['expd','sers','edge']:
            if components[key][0] in ['expdisk']:
                tmp,tmp_components = exponential_luminosity(components[key],radii = components['radii'],
                                            zero_point_flux=zero_point_flux,distance= distance,
                                            wavelength= components['wavelength'])
                lum_components[key] = tmp_components
                disk.append(tmp)
            elif components[key][0] in ['edgedisk']:
                if components['exp_time'] == -1:
                    raise InputError('For the edgedisk parameter in GalFit you need to provide an exposure time with --et')
                tmp,tmp_components = edge_luminosity(components[key],radii = components['radii'],
                                            values = components,zero_point_flux=zero_point_flux, distance= distance)
                lum_components[key] = tmp_components
                lum_components[key][1] = sup.integrate_surface_density(sup.convertskyangle(components['radii'],distance),tmp)
                lum_components[key][5] = lum_components[key][4]/lum_components[key][3]
                disk.append(tmp)

            elif components[key][0] in ['sersic']:
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


def mag_to_lum(mag, zero_point_flux=0.,wavelength =0.,distance= 0.):
    # Integrated flux is
    IntFlux = zero_point_flux*10**(-1*mag/2.5)
    # Convert it to a luminosity
    #The distance is in Mpc and the parsec in m hence here a factor 1e12*1e32 = 1e44 is missing.
    # However converting from Jy to Watt takes out a factor of 1e-26 so we multiply by a factor of 1e18
    IntLum  =  IntFlux*np.pi*4.*distance**2*3.085677581**2*1e18 #
    # We need to multiply this with the band frequency
    # 3.6 micron for Spitzer 3.4 for WISE

    frequency = c.c/wavelength

    # and convert to L_solar
    return IntLum*frequency/c.solar_luminosity      # L_solar

def exponential_luminosity(components,radii = [],zero_point_flux=0.,distance= 0.,wavelength =0.):
    lum_components = copy.deepcopy(components)
    #Ftot = 2πrs2Σ0q
    IntLum = mag_to_lum(components[1],zero_point_flux=zero_point_flux,
                        wavelength=wavelength, distance=distance)
    lum_components[1] = IntLum/components[5]
    # convert the scale length to kpc
    h_kpc = sup.convertskyangle(components[3],distance)
    lum_components[3] = h_kpc
    # this assumes perfect ellipses for now and no deviations are allowed
    central_luminosity = IntLum/(2.*np.pi*(h_kpc*1000.)**2*components[5]) #L_solar/pc^2
    lum_components[2] = central_luminosity
    return central_luminosity*np.exp(-1.*(radii/float(components[3]))),lum_components


    # This is untested for now
def edge_luminosity(components,radii = [],zero_point_flux=0.,
                    values= {'magnitude_zero': 0,'wavelength':0.,'plate_scale':[0.,0.], 'exp_time':1.}
                    ,distance=0 ):
    lum_components = copy.deepcopy(components)

    #mu_0 (mag/arcsec) = -2.5*log(sig_0/(t_exp*dx*dy))+mag_zpt
    #this has to be in L_solar/pc^2 but our value comes in mag/arcsec^2

    central_luminosity = float(mag_to_lum(components[2],zero_point_flux=zero_point_flux,\
                        wavelength=float(values['wavelength']), distance=distance))/\
                        (float(sup.convertskyangle(1.,distance))*1000.)**2.
    
    #Need to integrate the luminosity of the model and put it in component[1] still
    lum_components[2] = central_luminosity
    lum_components[3] = sup.convertskyangle(components[3],distance)
    lum_components[4] = sup.convertskyangle(components[4],distance)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lum_profile = central_luminosity*radii/components[3]*k1(radii/components[3])
        if np.isnan(lum_profile[0]):
            lum_profile[0] = central_luminosity
    # this assumes perfect ellipses for now and no deviations are allowed
    return lum_profile,lum_components

def sersic_luminosity(components,radii = [],zero_point_flux=0.,distance= 0.,wavelength = 0.):
    # This is not a deprojected surface density profile we should use the formual from Baes & gentile 2010
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
