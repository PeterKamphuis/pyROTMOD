# -*- coding: future_fstrings -*-

import numpy as np
import pyROTMOD.constants as c
import pyROTMOD.support as sup
from scipy.special import k1,gamma

class BadFileError(Exception):
    pass
class InputError(Exception):
    pass

def get_optical_profiles(filename,zero_point_flux = 0.,distance = 0.,band = 'SPITZER3.6',exp_time = -1.,luminosity = False, MLRatio = 0.6 ):

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
    # If the first line and first column is not correct we assume a Galfit file
    if not correctfile:
        components = read_galfit(input)
        components['wavelength'] = c.bands[band]
        components['exp_time'] = exp_time
        optical_profiles = convert_parameters_to_luminosity(components,zero_point_flux=zero_point_flux,distance= distance)
        luminosity = True
        # Celan components to only contain the profiles in a list not a dictionary
        cleaned_components = []
        for key in components:
            if key[0:4].lower() in ['expd','sers','edge']:
                cleaned_components.append(components[key])

    else:
        # otherwise we extract the columns and such from the file
        input
        print("Not operational yet")


    if luminosity:
        for x in range(1,len(optical_profiles)):
            optical_profiles[x][1] = 'M_SOLAR/PC^2'
            optical_profiles[x][2:] = [float(y)*MLRatio for y in optical_profiles[x][2:]]
    #if len(firstline) = 0 :
    #    if firstline[0].lower() !=

    return optical_profiles,cleaned_components



def read_galfit(lines):
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
                plate_scale = [float(tmp[1]), float(tmp[2])]
            if tmp[0] == 'z)':
                read_component = False
            if len(tmp) > 1:
                if tmp[1] == 'component':
                    read_component = True
            if tmp[0] == '0)' and read_component:
                current_component = tmp[1]
                if current_component not in recognized_components:
                    print(f'''pyROTMOD does not know how to process {current_component} not reading it
''')
                    read_component = False
                else:
                    counter[recognized_components.index(current_component)] += 1
                    components[f'{current_component}_{counter[recognized_components.index(current_component)]}'] = [current_component,0.,0.,0.,0.]
            if tmp[0] == '3)' and read_component:
                components[f'{current_component}_{counter[recognized_components.index(current_component)]}'][1] = float(tmp[1])
            if tmp[0] == '4)' and read_component:
                components[f'{current_component}_{counter[recognized_components.index(current_component)]}'][2] = float(tmp[1])*np.mean(plate_scale)
                if current_component in ['expdisk','edgedisk']:
                    if max_radius < 10 * float(tmp[1]): max_radius = 10 * float(tmp[1])
                elif current_component in ['sersic']:
                    if max_radius < 5* float(tmp[1]): max_radius = 5 * float(tmp[1])
            if tmp[0] == '5)' and read_component and current_component in ['sersic','edgedisk']:
                components[f'{current_component}_{counter[recognized_components.index(current_component)]}'][3] = float(tmp[1])
            if tmp[0] == '9)' and read_component and current_component in ['expdisk','sersic']:
                components[f'{current_component}_{counter[recognized_components.index(current_component)]}'][4] = float(tmp[1])
    if len(plate_scale) == 0 or len(mag_zero) == 0:
        print(plate_scale,mag_zero)
        raise BadFileError(f'Your file  is not recognized by pyROTMOD')
    components['radii'] = np.linspace(0,max_radius,int(max_radius/2.))*np.mean(plate_scale) # in arcsec
    components['plate_scale'] = plate_scale
    components['magnitude_zero'] = mag_zero
    return components



def convert_parameters_to_luminosity(components,zero_point_flux=0.,distance= 0.):

    # We plot a value every 2 pixels out to 7 x the scale length and finish with a 0.
    disk = []
    bulge = []
    for key in components:
        if key[0:4].lower() in ['expd','sers','edge']:
            if components[key][0] in ['expdisk']:
                tmp = exponential_luminosity(components[key],radii = components['radii'],
                                            zero_point_flux=zero_point_flux,distance= distance,
                                            wavelength= components['wavelength'])
                disk.append(tmp)
            elif components[key][0] in ['edgedisk']:
                if components['exp_time'] == -1:
                    raise InputError('For the edgedisk parameter in GalFit you need to provide an exposure time with --et')
                tmp = edge_luminosity(components[key],radii = components['radii'],
                                            values = components, distance= distance)
                disk.append(tmp)

            elif components[key][0] in ['sersic']:
                tmp = sersic_luminosity(components[key],radii = components['radii'],
                                        zero_point_flux=zero_point_flux,distance= distance,
                                        wavelength= components['wavelength'])
                bulge.append(tmp)
    organized = [['RADI','ARCSEC']]
    counter = 1
    for x in range(len(disk)):
        organized.append([])
        organized[counter].append(f'EXPONENTIAL_{x+1}')
        organized[counter].append(f'L_SOLAR/PC^2')
        for i in range(len(components['radii'])):
            if x == 0:
                organized[0].append(components['radii'][i])
            organized[counter].append(disk[x][i])
        counter += 1
    for x in range(len(bulge)):
        organized.append([])
        organized[counter].append(f'SERSIC_{x+1}')
        organized[counter].append(f'L_SOLAR/PC^2')
        for i in range(len(components['radii'])):
            organized[counter].append(bulge[x][i])
        counter += 1
    return organized


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
    IntLum = mag_to_lum(components[1],zero_point_flux=zero_point_flux,
                        wavelength=wavelength, distance=distance)
    # convert the scale length to kpc
    h_kpc = sup.convertskyangle(components[2],distance)
    # this assumes perfect ellipses for now and no deviations are allowed
    central_luminosity = IntLum/(2.*np.pi*(h_kpc*1000.)**2*components[4]) #L_solar/pc^2
    return central_luminosity*np.exp(-1*radii/components[2])


    # This is untested for now
def edge_luminosity(components,radii = [],zero_point_flux=0.,
                    values= {'magnitude_zero': 0,'wavelength':0.,'plate_scale':[0.,0.], 'exp_time':0.}
                    ,distance=0 ):
    central_brightness = 10**((component[1]-values['magnitude_zero'])/-2.5)*values['exp_time']*values['plate_scale'][0]* values['plate_scale'][1]
    central_luminosity= mag_to_lum(central_brightness,zero_point_flux=zero_point_flux,
                        wavelength=values['wavelength'], distance=distance)
    # this assumes perfect ellipses for now and no deviations are allowed
    return central_luminosity*radii/components[2]*k1(radii/components[2])

def sersic_luminosity(components,radii = [],zero_point_flux=0.,distance= 0.,wavelength = 0.):
    IntLum = mag_to_lum(components[1],zero_point_flux=zero_point_flux,
                        wavelength=wavelength, distance=distance)
    kappa=2.*components[3]-1./3. # From https://en.wikipedia.org/wiki/Sersic_profile
    h_kpc = sup.convertskyangle(components[2],distance)
    central_luminosity = IntLum/(2.*np.pi*(h_kpc*1000.)**2*np.exp(kappa)*components[3]*\
                                kappa**(-2*components[3])*components[4]*gamma(2.*components[3])) #L_solar/pc^2
    return central_luminosity*np.exp(-1.*kappa*((radii/components[2])**(1./components[3])-1))
