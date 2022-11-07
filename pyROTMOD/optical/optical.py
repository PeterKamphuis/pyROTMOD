# -*- coding: future_fstrings -*-

import numpy as np
import warnings
import pyROTMOD.constants as co
import pyROTMOD.support as sup
from scipy.special import k1,gamma
from scipy.optimize import curve_fit
from astropy import units as unit
import copy
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt

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
    sersic = []
    for key in components:
        if key[0:4].lower() in ['expd','sers','edge']:
            if components[key]['Type'] in ['expdisk']:
                tmp,tmp_components = exponential_luminosity(components[key],radii = components['radii'],
                                            band = band,distance= distance)
                lum_components[key] = tmp_components
                disk.append(tmp)
            elif components[key]['Type'] in ['edgedisk']:
                tmp,tmp_components = edge_luminosity(components[key],radii = components['radii'],band=band, distance= distance)
                lum_components[key] = tmp_components
                lum_components[key]['axis ratio'] = lum_components[key]['scale height']/lum_components[key]['scale length']
                disk.append(tmp)

            elif components[key]['Type'] in ['sersic']:
                tmp,tmp_components = sersic_luminosity(components[key],radii = components['radii'],
                                            band = band,distance= distance)
                lum_components[key] = tmp_components
                sersic.append(tmp)


    organized = {}
    organized['RADI'] = ['RADI','ARCSEC']+[x.value for x in components['radii']]
    for i,disks in enumerate(disk):
        organized[f'EXPONENTIAL_{i+1}'] = [f'EXPONENTIAL_{i+1}',f'L_SOLAR/PC^2']\
                    +[x.value if x.unit == unit.Lsun/unit.pc**2 else float('NaN') for x in disks]
        if float('NaN') in organized[f'EXPONENTIAL_{i+1}']:
            print(f'We got {organized[f"EXPONENTIAL_{i+1}"]}')
            raise CalculateError(f'Something went wrong in EXPONENTIAL_{i+1}')
    for i,bulges in enumerate(bulge):
        organized[f'HERNQUIST_{i+1}'] = [f'HERNQUIST_{i+1}',f'L_SOLAR/PC^2']\
                    +[x.value if x.unit == unit.Lsun/unit.pc**2 else float('NaN') for x in bulges]
        if float('NaN') in organized[f'HERNQUIST_{i+1}']:
            print(f'We got {organized[f"HERNQUIST_{i+1}"]}')
            raise CalculateError(f'Something went wrong in HERNQUIST_{i+1}')
    for i,sersic_disks in enumerate(sersic):
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
        lum_profile = central_luminosity*np.array([float(x/lum_components['scale length']) for x in radii],dtype=float)\
            *k1(np.array([float(x/lum_components['scale length']) for x in radii],dtype=float))
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


def exponential(radii,central,h):
    '''Exponential function'''
    return central*np.exp(-1.*radii/h)

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
def fit_profile(radii,density,components,function='EXPONENTIAL_1',output_dir = './',debug =False, log = None):

    radii = np.array(radii,dtype=float)
    density = np.array(density,dtype=float)

    if components['Total SB'] == None:
        components['Total SB'] = sup.integrate_surface_density(radii,density)*unit.Msun
    print(f'What is goinf on')
    print(components['Total SB'],density,radii)
    if components['R effective'] == None:
        # if this is specified as an exponential disk without the parameters defined we fit a expoenential
        ring_area = sup.integrate_surface_density(radii,density,calc_ring_area=True)
        print(ring_area)
        components['R effective'] = radii[-1]*unit.kpc
        for i in range(1,len(radii)):
            current_mass = np.sum(ring_area[:i]*density[:i])*unit.Msun
            print(current_mass)
            if current_mass.value > components['Total SB'].value/2.:
                components['R effective'] = radii[i]*unit.kpc
                break
    type_split=function.split('_')
    type = type_split[0]
    try:
        count = type_split[1]
    except:
        count='1'
    fit_function_dictionary = {'EXPONENTIAL':
                {'initial':[density[0],radii[density < density[0]/np.e ][0]],
                'out':['Central SB','scale length'],
                'function': exponential,
                'Type':'expdisk,
                'max_red_sqr': 1000,
                'name':'Exponential',
                'fail':'random_disk'},
                'HERNQUIST':
                {'initial':[components['Total SB'].value,float(components['R effective'].value/1.8153)][0]],
                'out':['Total SB','scale length'],
                'Type':'hernquist',
                'max_red_sqr': 3000,
                'name':'Hernquist',
                'fail':'failed'},
                'SERSIC':
                {'initial':[density[radii < components['R effective'].value][0],\
                            components['R effective'].value, 2.],
                'out':[None,'R effective','sersic index'],
                'Type':'sersic',
                'max_red_sqr': 1000,
                'name':'Sersic',
                'fail':'random_disk'},
                'EXP+HERN':
                {'initial':[components['Total SB'].value/3.,float(components['R effective'].value/(2.*1.8153)),\
                                density[0]/2.,radii[density < density[0]/np.e][0]],
                'out':[['Total SB','scale length'],['Central SB','scale length']],
                'function': fit_function = lambda r,mass,hern_length,central,scale_length:\
                        hernquist_profile(r,mass,hern_length) + exponential(r,central,scale_length),
                'separate_functions': [hernquist_profile,exponential],
                'Type':'split',
                'max_red_sqr': 3000,
                'name':'Exp_Hern',
                'fail':'failed'},
                }



    if type == 'DENSITY':
        evaluate = ['EXPONENTIAL','EXP+HERN']
    elif type == 'BULGE'
        evaluate = ['HERNQUIST']
    elif type == 'DISK'
        evaluate = ['EXPONENTIAL']
    else:
        evaluate = [type]

    fitted_dict = {}

    try:
    for ev in evaluate
        try:
            tmp_fit_parameters, tmp_red_chisq,tmp_profile = single_fit_profile(\
                    fit_function_dictionary[ev]['function'],\
                    radii[density > 0.],density[density > 0.],\
                    fit_function_dictionary[ev]['initial'],\
                    debug=debug,log=log,name=fit_function_dictionary[ev]['name'],\
                    ,output_dir=output_dir,\
                    count= count)
            if tmp_red_chisq > fit_function_dictionary[ev]['max_red_sqr']:
                tmp_red_chisq = float('NaN')
            if fit_function_dictionary[ev]['Type'] == 'split':
                tmp_profile = []
                prev= 0
                for i in len(fit_function_dictionary[ev]['separate_functions']):
                    len_parameter= len(fit_function_dictionary[ev]['out'][i])
                    tmp_profile.append(fit_function_dictionary[ev]['seperate_functions'][i]\
                                        (radii,tmp_fit_parameters[prev:prev+len_parameter])))


            fitted_dict[ev]= {'parameters': tmp_fit_parameters,
                'red_chi_sq':tmp_red_chisq,
                'profile': tmp_profile,
                'result': fit_function_dictionary[ev]['Type'],
                'component_parameter': fit_function_dictionary[ev]['out']}
        except:
            fitted_dict[ev]= {'result': fit_function_dictionary[ev]['fail'],\
                              'red_chi_sq': float('NaN')}

    red_chi = [x['red_chi_sq']  for x in fitted_dict]
    if np.min(red_chi) != float('NaN'):
        for ev in fitted_dict:
            if fitted_dict[ev]['red_chi_sq'] == np.min(red_chi):
                break
        fitted_dict =fitted_dict[ev]
    else:
        for ev in fitted_dict:
            if fitted_dict[ev]['result'] == 'random_disk':
                profile = density
                components['Type'] = 'random_disk'
                return 'ok',profile,components

    if fitted_dict['result'] != 'split':
        profile = [[],[]]
        for i in [0,1]:
            profile[i] =

        profile = [hernquist_profile(radii,tot_parameters[0],tot_parameters[1]),
                    exponential(radii,tot_parameters[2],tot_parameters[3]) ]
        components_com = [copy.deepcopy(components),copy.deepcopy(components)]
        components_com[0]['Total SB'] = tot_parameters[0]*unit.Msun
        components_com[1]['Total SB'] = sup.integrate_surface_density(profile[1],density)*unit.Msun
        components_com[0]['Central SB'] =  profile[1][0]*  unit.Msun /unit.pc**2
        components_com[1]['Central SB'] = tot_parameters[2]*unit.Msun/unit.pc**2
        components_com[1]['scale length'] = tot_parameters[3]*  unit.kpc
        components_com[0]['scale length'] = tot_parameters[1]*  unit.kpc
        components_com[0]['Type'] = 'hernquist'
        components_com[1]['Type'] = 'expdisk'
        components = copy.deepcopy(components_com)
        result = 'process'

    else:
        for i,parameter in enumerate(fitted_dict['component_parameter']):
            component[parameter]=fitted_dict['parameters'][i]
        components['Type'] = fitted_dict['result']
        result = 'ok'





            if red_chisq < 1000.:
                components['scale length'] = fit_parameters[1]*unit.kpc
                components['Central SB'] =  fit_parameters[0]*unit.Msun/unit.pc**2
                components['Type'] = 'expdisk'
            else:
                sup.print_log(f'''FIT_EXPONENTIAL:: You claim the profile is an {type}, but the fit is a really bad.''',log)
                profile = density
                components['Type'] = 'random_disk'
                sup.print_log(f'''FIT_PROFILE: We will fit a random density disk. \n''', log)
            result = 'ok'
        elif type[0:3] =='HER':
            initial =  [components['Total SB'].value,float(components['R effective'].value/1.8153)]

            fit_parameters, red_chisq,profile = single_fit_profile(hernquist_profile,\
                    radii[density > 0.],density[density > 0.],initial,\
                    debug=debug,log=log,name='Hernquist',output_dir=output_dir,\
                    count= count)

            if red_chisq < 1500.:
                components['Total SB'] = tot_parameters[0]*unit.Msun
                components['scale length'] = tot_parameters[1]*unit.kpc
                components['Central SB'] =  profile[0]*unit.Msun/unit.pc**2
                components['Type'] = 'hernquist'
                result = 'ok'
            else:
                sup.print_log(f'''FIT_EXPONENTIAL:: You claim the profile is an {type}, but the fit is a really bad.''',log)
                profile = density
                components['Type'] = 'Failed'
                sup.print_log(f'''FIT_PROFILE: We will remove this profile from the fit. \n''', log)
                result = 'Failed'
            #fit_function = lambda r,scale_length:\
            #        hernquist_profile(r,components['Total SB'].value,scale_length)

        elif type[0:3] =='SER':
            initial = [density[radii < components['R effective'].value][0],\
                        components['R effective'].value, 2.]
            name='Sersic'
            initial =  [components['Total SB'].value,float(components['R effective'].value/1.8153)]

            fit_parameters, red_chisq,profile = single_fit_profile(sersic,\
                    radii[density > 0.],density[density > 0.],initial,\
                    debug=debug,log=log,name='Sersic',output_dir=output_dir,\
                    count= count)

            if red_chisq < 1000.:
                components['R effective'] = tot_parameters[1]*unit.kpc
                components['sersic index'] =  tot_parameters[2]
                components['Type'] = 'sersic'
                result = 'ok'
            else:
                sup.print_log(f'''FIT_EXPONENTIAL:: You claim the profile is an {type}, but the fit is a really bad.''',log)
                profile = density
                components['Type'] = 'Failed'
                sup.print_log(f'''FIT_PROFILE: We will remove this profile from the fit. \n''', log)
                result = 'Failed'
        elif type[0:3] == 'DEN':
            fit_parameters,red_chisq,profile,type = multiple_fit_profile(\
                    radii[density > 0.],density[density > 0.],\
                    debug=debug,log=log,output_dir=output_dir,\
                    count= count)
            fit_function = lambda r,mass,hern_length,central,scale_length:\
                    hernquist_profile(r,mass,hern_length) + exponential(r,central,scale_length)
            guess_scale = radii[density < density[0]/np.e][0]
            initial= [components['Total SB'].value/3.,float(components['R effective'].value/(2.*1.8153)),\
                        density[0]/2.,guess_scale]
            name= 'Combined'

    # if this is specified as an exponential disk without the parameters defined we fit a expoenential
    except:

          components['Type'] = 'random_disk'
          sup.print_log(f'''FIT_PROFILE: You claim the density profile is {function}, but we fail to fit the function.
''', log)
          if type[0:3] in ['SER','BUL','HER']:
              components['Type'] = 'Failed'
              sup.print_log(f'''FIT_PROFILE: We will remove this profile from the fit. \n''', log)
          else:
              components['Type'] = 'random_disk'
              sup.print_log(f'''FIT_PROFILE: We will fit a random density disk. \n''', log)
    try:


        result = 'ok'
        if red_chisq > 1000.:
            sup.print_log(f'''FIT_EXPONENTIAL:: You claim the profile is an {name}, but the fit is a really bad.''',log)
            profile = density
            exit()
            if type[0:3] in ['SER','BUL','HER']:
                components['Type'] = 'Failed'
                sup.print_log(f'''FIT_PROFILE: We will remove this profile from the fit. \n''', log)
            else:
                components['Type'] = 'random_disk'
                sup.print_log(f'''FIT_PROFILE: We will fit a random density disk. \n''', log)
        else:
            if type[0:3] =='EXP':
                components['scale length'] = tot_parameters[1]*unit.kpc
                components['Central SB'] =  tot_parameters[0]*unit.Msun/unit.pc**2
                components['Type'] = 'expdisk'
            elif type[0:3] =='HER':
                components['Total SB'] = tot_parameters[0]*unit.Msun
                components['scale length'] = tot_parameters[1]*unit.kpc
                components['Central SB'] =  profile[0]*unit.Msun/unit.pc**2
                components['Type'] = 'hernquist'

            elif type[0:3] =='SER':
                components['R effective'] = tot_parameters[1]*unit.kpc
                components['sersic index'] =  tot_parameters[2]
                components['Type'] = 'sersic'
            elif type[0:3] == 'DEN':
                profile = [hernquist_profile(radii,tot_parameters[0],tot_parameters[1]),
                            exponential(radii,tot_parameters[2],tot_parameters[3]) ]
                components_com = [copy.deepcopy(components),copy.deepcopy(components)]
                components_com[0]['Total SB'] = tot_parameters[0]*unit.Msun
                components_com[1]['Total SB'] = sup.integrate_surface_density(profile[1],density)*unit.Msun
                components_com[0]['Central SB'] =  profile[1][0]*  unit.Msun /unit.pc**2
                components_com[1]['Central SB'] = tot_parameters[2]*unit.Msun/unit.pc**2
                components_com[1]['scale length'] = tot_parameters[3]*  unit.kpc
                components_com[0]['scale length'] = tot_parameters[1]*  unit.kpc
                components_com[0]['Type'] = 'hernquist'
                components_com[1]['Type'] = 'expdisk'
                components = copy.deepcopy(components_com)
                result = 'process'

    except:

    #    sup.print_log(f'''FIT_PROFILE: You claim the density profile is {function}, but we fail to fit the function.
#''', log)
    #    if type[0:3] in ['SER','BUL','HER']:
    #        components['Type'] = 'Failed'
#            sup.print_log(f'''FIT_PROFILE: We will remove this profile from the fit. \n''', log)
#        else:
#            components['Type'] = 'random_disk'
#            sup.print_log(f'''FIT_PROFILE: We will fit a random density disk. \n''', log)
#        result = 'ok'
#        profile=density

    return result,profile,components
fit_profile.__doc__ =f'''
 NAME:
    fit_exponential(radii,density,components,output_dir = './',debug =False, log = None)


 PURPOSE:
    fit an exponential to the provided density distribution

 CATEGORY:
    optical

 INPUTS:
    radii = radi at which the profile is evaluated
    density = density profile to fit
    components = the components fitted to the profile


 OPTIONAL INPUTS:
    output_dir = './'

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
def  multiple_fit_profile(radii[density > 0.],density[density > 0.],\
        debug=debug,log=log,output_dir=output_dir,\
        count= count):

    functions_to_fit= {'exponential': []
    fit_function = lambda r,mass,hern_length,central,scale_length:\
            hernquist_profile(r,mass,hern_length) + exponential(r,central,scale_length)
    guess_scale = radii[density < density[0]/np.e][0]
    initial= [components['Total SB'].value/3.,float(components['R effective'].value/(2.*1.8153)),\
                density[0]/2.,guess_scale]
    name= 'Combined'
def single_fit_profile(fit_function,radii,density,initial,debug=False,log=None,\
                        name='Generic',output_dir='./',count= 0):

    tot_parameters, tot_covariance = curve_fit(fit_function, radii,\
                    density,sigma= 0.1*density,\
                    p0=initial,maxfev=5000)


    # let's see if our fit has a reasonable reduced chi square
    profile = fit_function(radii,*tot_parameters)
    components['Total SB'] = sup.integrate_surface_density(radii,density)*unit.Msun
    red_chi = np.sum((density[density > 0.]-profile[density > 0.])**2/(0.1*density[density > 0.]))
    red_chisq = red_chi/(len(density[density > 0.])-len(tot_parameters))
    plot_exponentials(radii,density,profile,name=name,
                    output_dir=output_dir,red_chi= red_chisq,count=count)
    sup.print_log(f'''FIT_PROFILE: We fit the {name} with a reduced Chi^2 = {red_chisq} with a 10% error.''', log)
    return tot_parameters,red_chisq

single_fit_profile.__doc__ =f'''
 NAME:
    fit_exponential(fit_function,radii,density,initial,\
    output_dir = './',debug =False, log = None, count = 0)


 PURPOSE:
    fit an exponential to the provided density distribution

 CATEGORY:
    optical

 INPUTS:
    fit_function = function to fit
    radii = radi at which the profile is evaluated
    density = density profile to fit
    inital = initial_guess
 OPTIONAL INPUTS:
    output_dir = './'
    log = None
    debug = False
    count = 0
        iterative number to seperate different instances of the same function

 OUTPUTS:
    tot_paramets = fitted_parametrs
    red_chisq = reduced chisquare of the fit
 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''

def get_optical_profiles(filename,distance = 0.,band = 'SPITZER3.6',exposure_time=1.,\
                            MLRatio = 0.6, log =None,debug=False, scale_height=None,
                            output_dir='./'):
    '''Read in the optical Surface brightness profiles or the galfit file'''
    # as we do a lot of conversions in the optical module we make distance a quantity with unit Mpc
    distance = distance * unit.Mpc
    MLRatio = MLRatio*unit.Msun/unit.Lsun
    sup.print_log(f"GET_OPTICAL_PROFILES: We are reading the optical parameters from {filename}. \n",log, screen =True)
    if distance == 0.:
        raise InputError(f'We cannot convert profiles adequately without a distance.')
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
        cleaned_components = []
        for types in optical_profiles:
            optical_profiles[types] = optical_profiles[types][:2]+[float(x) for x in optical_profiles[types][2:]]
            if types != 'RADI':
                if optical_profiles[types][1] in ['MAG/ARCSEC^2','MAG/ARCSEC^2']:
                    wavelength = co.bands[band]
                    pc_conv = sup.convertskyangle(1,distance.value)*1000.
                    optical_profiles[types] = [types,'L_SOLAR/PC^2']+\
                                              [float(mag_to_lum(x*unit.mag/unit.arcsec**2,\
                                                     band = band,distance= distance))\
                                                     /pc_conv**2 for x in optical_profiles[types][2:]]
                elif  optical_profiles[types][1] in ['M/S']:
                    optical_profiles[types][2:]=[float(y)/1000. for y in optical_profiles[types][2:]]
                    optical_profiles[types][1] = 'KM/S'
                elif optical_profiles[types][1] in ['KM/S','L_SOLAR/PC^2','M_SOLAR/PC^2']:
                    sup.print_log(f"GET_OPTICAL_PROFILES: The read {optical_profiles[types][0]} profile can be used directly",log)
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

    for types in optical_profiles:
        if types != 'RADI':
            if optical_profiles[types][1] == 'L_SOLAR/PC^2':
                optical_profiles[types][1] = 'M_SOLAR/PC^2'
                optical_profiles[types][2:] = [float(y)*MLRatio.value for y in optical_profiles[types][2:]]
    if galfit_file:
        for i in range(len(cleaned_components)):
            if cleaned_components[i]['Central SB']:
                cleaned_components[i]['Central SB'] = cleaned_components[i]['Central SB']*MLRatio
            if cleaned_components[i]['Total SB']:
                cleaned_components[i]['Total SB'] = cleaned_components[i]['Total SB']*MLRatio
    else:
        #if we want to read from file we want to fit

        optical_profiles,cleaned_components = process_read_profile(optical_profiles,cleaned_components,\
            output_dir = output_dir, debug=debug, log=log)

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
def hernquist_profile(radii, mass, scale_length):
    'The hernquist density profile (Eq 2, Hernquist 1990)'
    'mass/(np.pi*2)*(scale_length/radii)*1./(radii+scale_length)**3'
    'Note that in galpy this is amp/(4*pi*a**3)*1./((r/a)(1+r/a)**2'
    'With amp = 2. mass'
    'Both have inf at r = 0. so if radii == 0 it needs to be adapted'
    central = False
    if radii[0] == 0:
        radii[0] = radii[1]/3.
        central = True
    profile = mass/(np.pi*2)*(scale_length/radii)*1./(radii+scale_length)**3
    if central:
        #Do a linear extrapolation from the inner two points
        try:
            profile[0] = profile[0]+(profile[0]-profile[1])*((radii[0])/(radii[1]-radii[0]))
            radii[0] = 0.
        except:
            profile[0] = float('inf')
            radii[0] = 0.
    return profile



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

def plot_exponentials(in_radii,density, exponential,name='generic',\
                    output_dir='./',red_chi = None, count = '1'):
    '''This function makes a simple plot of the optical profiles'''
    figure = plt.figure(figsize=(10,7.5) , dpi=300, facecolor='w', edgecolor='k')
    ax = figure.add_subplot(1,1,1)
    ax.plot(in_radii,density, label='Input Profile')
    ax.plot(in_radii,exponential, label='Fitted Profile')


    #plt.xlim(0,6)
    ax.set_ylabel('Density (M$_\odot$/pc$^2$)')
    ax.set_xlabel('Radius (kpc)')
    if red_chi:
        ax.text(0.5,1.05,f'''Red. $\chi^{{2}}$ = {red_chi:.4f}''',rotation=0, va='bottom',ha='center', color='black',\
            bbox=dict(facecolor='white',edgecolor='white',pad=0.5,alpha=0.),\
            zorder=7, backgroundcolor= 'white',fontdict=dict(weight='bold',size=16),transform=ax.transAxes)
    ax.set_yscale('log')
    plt.legend()
    if int(count) > 1:
        name = f'{name}_{count}'
    plt.savefig(f'{output_dir}/{name}_Profiles.png')
    plt.close()

plot_exponentials.__doc__ =f'''
 NAME:
     plot_exponentials(in_radii,density, exponential,name='generic',\
                        output_dir='./',red_chi = None,bulge =False, count = 1):
 PURPOSE:
    plot the fitted function over the input profile

 CATEGORY:
    optical

 INPUTS:
    in_radii = radii
    density = original profile
    exponential = fitted profile

 OPTIONAL INPUTS:
    name = 'generic'
        Name of the fitted function to be used in the file name

    output_dir = './'
        Destination directory for filename

    red_chi = None
        Reduced Chi square

    count = '1'
        number of function fitting used in file name

 OUTPUTS:
    png plot
 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''

def process_read_profile(optical_profiles,cleaned_components,\
                    output_dir = './',debug =False, log = None):
    optical_profiles_out = {}
    component_out  = []
    exponen_count = 0
    hern_count = 0
    for type in optical_profiles:
        prof= type.split('_')
        if prof[0] == 'EXPONENTIAL':
            if int(prof[1]) > int(exponen_count):
                exponen_count = int(prof[1])
        if prof[0] == 'HERNQUIST':
            if int(prof[1]) > int(hern_count):
                hern_count = int(prof[1])


    for i,type in enumerate(optical_profiles):
        if type == 'RADI':
            optical_profiles_out[type] = optical_profiles[type]
        elif optical_profiles[type][1] in ['KM/S']:
            if type[:3] in ['EXP','DEN','DIS','SER']:
                cleaned_components[i-1]['Type'] = 'random_disk'
            else:
                cleaned_components[i-1]['Type'] = 'random_bulge'
            optical_profiles_out[type] = optical_profiles[type]
            component_out.append(cleaned_components[i-1])
        elif optical_profiles[type][1] in ['M_SOLAR/PC^2'] and \
            type[:3] in ['EXP','HER','SER','DEN'] :
            result,profiles,components = fit_profile(optical_profiles['RADI'][2:],optical_profiles[type][2:],\
                    cleaned_components[i-1],function=type, output_dir = output_dir\
                    ,debug = debug, log = log)
            if result == 'process':
                exponen_count += 1
                optical_profiles_out[f'EXPONENTIAL_{exponen_count}'] = \
                    [f'EXPONENTIAL_{exponen_count}','M_SOLAR/PC^2'] + list(profiles[1])
                component_out.append(components[1])
                hern_count += 1
                optical_profiles_out[f'HERNQUIST_{hern_count}'] = \
                    [f'HERNQUIST_{hern_count}','M_SOLAR/PC^2'] + list(profiles[0])
                component_out.append(components[0])
            else:
                optical_profiles_out[type] = optical_profiles[type]
                component_out.append(cleaned_components[i-1])
        elif type[:3] == 'DIS':
            cleaned_components[i-1]['Type'] = 'random_disk'
            optical_profiles_out[type] = optical_profiles[type]
            component_out.append(cleaned_components[i-1])
        else:
            print_log(f'''PROCESS_READ_PROFILES: We do not know how convert the profile {optical_profiles[type][0]} with the units {optical_profiles[type][1]} to velocities
''',log)
            raise InputError(f'''PROCESS_READ_PROFILES: We do not know how convert the profile {optical_profiles[type][0]} with the units {optical_profiles[type][1]} to velocities''')
    return optical_profiles_out,component_out
process_read_profile.__doc__ =f'''
 NAME:
    process_read_profile(optical_profiles,cleaned_components,fit_random = False,\
                        output_dir = './',debug =False, log = None):
 PURPOSE:
    Fit the components of the read profile with their specified function
    in case the densities are read from file and not a galfit file

 CATEGORY:
    optical

 INPUTS:
    optical_profiles = the read input profiles dictionary
    cleaned_components = list of components to match the profiles
                        (this runs i-1 compared to the profiles as the radii doesn't have components )



 OPTIONAL INPUTS:
    fit_random = False
        if input is a radom density disk (DISK_) do we want to fit it?
    output_dir = './'
        standard for directory name where to put comparison plots
    log = None
    debug = False

 OUTPUTS:
    updated components list

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE: For now we can not model the sersic profile into a density distribution so
       sersic profiles are converted to hernquist or exponential profiles in rotmod
        but only if 0.75 < n < 1.25 (to exponential) or 3.75 < n < 4.25 (hernquist)
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

def sersic(r,effective_luminosity,effective_radius,n):
    '''sersic function'''
    kappa = -1.*(2.*n-1./3.)
    func = effective_luminosity*np.exp(kappa*((r/effective_radius)**(1./n))-1)
    return func

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
    effective_luminosity = IntLum/(2.*np.pi*(R_kpc.to(unit.pc))**2*\
                            np.exp(kappa)*components['sersic index']*\
                            kappa**(-2*components['sersic index'])*\
                            components['axis ratio']*gamma(2.*components['sersic index'])) #L_solar/pc^2

    #lum_components['Total SB'] = IntLum/components['axis ratio']
    lum_components['Total SB'] = IntLum
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
