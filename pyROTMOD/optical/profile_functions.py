# -*- coding: future_fstrings -*-
from pyROTMOD.fitters.fitters import initial_guess,lmfit_run
from pyROTMOD.support.errors import InputError,UnitError
from pyROTMOD.support.minor_functions import integrate_surface_density,\
    strip_unit,get_uncounted
from pyROTMOD.support.log_functions import print_log
from pyROTMOD.optical.profiles import exponential,hernquist,sersic,hernexp
from pyROTMOD.optical.calculate_profile_components import calculate_R_effective,\
    calculate_total_SB
from astropy import units as unit
#the individul functions are quicker than the general function https://docs.scipy.org/doc/scipy/reference/special.html


import numpy as np
import warnings
import inspect
import traceback


def determine_density_profile(type, n):
    prof_type = None  
    if type in ['expdisk','edgedisk', 'random_disk']:
        prof_type = 'exponential'
    elif type in ['devauc','hernquist']:
        prof_type= 'hernquist'
    elif type in ['sersic']:
        if 0.9 < n < 1.1:
            prof_type = 'exponential'
        elif  3.9 < n < 4.1:
            prof_type= 'hernquist'
        else:
            prof_type = 'sersic'
    return prof_type

def determine_profiles_to_fit(type):
    if type.upper() in ['DENSITY','RANDOM_DISK','RANDOM_BULGE']:
        evaluate = ['EXPONENTIAL','EXP+HERN','SERSIC']
    elif type.upper() in ['BULGE', 'HERNQUIST']:
        evaluate = ['HERNQUIST']
    elif type.upper() in ['DISK', 'EXPONENTIAL']:
        evaluate = ['EXPONENTIAL','SERSIC']
    else:
        evaluate = [type]
    return evaluate
 
def fit_profile(profile_to_fit,cfg=None):
    '''We only implemented the functions for the luminosity profiles 
    So this always needs to be done on a sky distribution aka Luminosity_Profile
    '''
   

    if profile_to_fit.values.unit not in [unit.Lsun/unit.pc**2,unit.Msun/unit.pc**2]:
        raise UnitError(f'''The profile {profile_to_fit.name} is not a ky profile that we can fit 
The unit {profile_to_fit.values.unit} will not lead to the right result.                         
''')
    #main_unit = density.unit*unit.pc**2
    #if components is None:
    #    raise InputError(f'We need a place to store the results please set components =') 

  
    if  profile_to_fit.R_effective == None:
        calculate_R_effective(profile_to_fit)
    
    if  profile_to_fit.total_SB == None:
        calculate_total_SB(profile_to_fit)
    
    type,count = get_uncounted(profile_to_fit.name)
   
    radii = strip_unit(profile_to_fit.radii,variable_type='radii')
    density = strip_unit(profile_to_fit.values,variable_type=profile_to_fit.profile_type)
   
    fit_function_dictionary = {'EXPONENTIAL':
                {'initial':[density[0],radii[density < density[0]/np.e ][0]],
                'out':['Central_SB','scale_length'],
                'function': exponential,
                'out_units':[profile_to_fit.values.unit,unit.kpc],
                'Type':'expdisk',
                'max_red_sqr': 1000,
                'name':'Exponential',
                'fail':'random_disk'},
                'HERNQUIST':
                {'initial':[profile_to_fit.total_SB.value/2.,float(profile_to_fit.R_effective.value/1.8153)],
                'out':['Total_SB','scale_length'],
                'out_units':[profile_to_fit.total_SB.unit,unit.kpc],
                'function': hernquist,
                'Type':'hernquist',
                'max_red_sqr': 3000,
                'name':'Hernquist',
                'fail':'failed'},
                'SERSIC':
                {'initial':[density[radii < profile_to_fit.R_effective.value][0],\
                            profile_to_fit.R_effective.value, 2.],
                'out':['L_effective','R_effective','sersic_index'],
                'out_units':[profile_to_fit.total_SB.unit/unit.pc**2,unit.kpc,1],
                'function': sersic,
                'Type':'sersic',
                'max_red_sqr': 1000,
                'name':'Sersic',
                'fail':'random_disk'},
                'EXP+HERN':
                {'initial':[profile_to_fit.total_SB.value/10.,float(profile_to_fit.R_effective.value/(10.*1.8153)),\
                                density[0]/2.,radii[density < density[0]/np.e][0]],
                'out':['Total_SB','scale_length_hernquist','Central_SB','scale_length'],
                'out_units':[profile_to_fit.total_SB.unit,unit.kpc,\
                                profile_to_fit.values.unit,unit.kpc],
                #'function': lambda r,Ltotal,hern_length,central,h:\
                #        hernquist(r,Ltotal,hern_length) + exponential(r,central,h),
                'function': hernexp,
                'separate_functions': [hernquist,exponential],
                'Type':'hernq+expdisk',
                'max_red_sqr': 3000,
                'name':'Exp_Hern',
                'fail':'failed'},
                }


    evaluate = determine_profiles_to_fit(type)
    fitted_dict = {}
    for ev in evaluate:
        try:
            tmp_fit_parameters, tmp_red_chisq,tmp_profile = single_fit_profile(\
                    profile_to_fit,fit_function_dictionary[ev]['function'],\
                    fit_function_dictionary[ev]['initial'],\
                    cfg=cfg,name=fit_function_dictionary[ev]['name'],\
                    count= count)

            if ev == 'EXPONENTIAL' and len(evaluate) == 1 and tmp_red_chisq > 50:
                evaluate.append('EXP+HERN')

            if tmp_red_chisq > fit_function_dictionary[ev]['max_red_sqr']:

                print_log(f'''The fit to {ev} has a red Chi^2 {tmp_red_chisq}.
As this is higher than {fit_function_dictionary[ev]['max_red_sqr']} we declare a mis fit''',\
                    cfg,case=['main'])
                tmp_red_chisq = float('NaN')

            if len(fit_function_dictionary[ev]['Type']) == 2:
                if tmp_fit_parameters[1] > tmp_fit_parameters[3]:
                    #If the hern_scale length is longer then the exponential something is wrong
                    tmp_red_chisq = float('NaN')
                else:
                    tmp_profile = []
                    prev= 0
                    for i in range(len(fit_function_dictionary[ev]['separate_functions'])):
                        len_parameter= len(fit_function_dictionary[ev]['out'][i])
                        tmp_profile.append(fit_function_dictionary[ev]['separate_functions'][i]\
                                            (radii,*tmp_fit_parameters[prev:prev+len_parameter]))
                        prev += len_parameter


            fitted_dict[ev]= {'parameters': tmp_fit_parameters,
                'red_chi_sq':tmp_red_chisq,
                'function': fit_function_dictionary[ev]['function'],
                'profile': tmp_profile,
                'result': fit_function_dictionary[ev]['Type'],
                'component_parameter': fit_function_dictionary[ev]['out'],
                'component_units':fit_function_dictionary[ev]['out_units']}
        except Exception as exiting:
            print_log(f'FIT_PROFILE: We failed to fit {ev}:',cfg,case=['main'])
            print_log(traceback.print_exception(type(exiting),exiting,exiting.__traceback__),\
                cfg,case=['main'])
            #exit()
            fitted_dict[ev]= {'result': fit_function_dictionary[ev]['fail'],\
                              'red_chi_sq': float('NaN')}


    red_chi = [fitted_dict[x]['red_chi_sq']  for x in fitted_dict]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        min_red_chi = np.nanmin(red_chi)
    #red_chi = [float('NaN'),float('NaN
    # ')]
    if not np.isnan(min_red_chi):
        print_log('FIT_PROFILE: We have found at least one decent fit.',cfg,case=['main'])
        for ev in fitted_dict:
            if fitted_dict[ev]['red_chi_sq'] == min_red_chi:
                # if the sersic index is 0.9 < 1.1 we prefer an exponential
                if ev.lower() == 'sersic':
                    if 0.9 < fitted_dict[ev]['parameters'][2] < 1.1: 
                        ev = 'EXPONENTIAL'
                    elif 3.9 < fitted_dict[ev]['parameters'][2] < 4.1: 
                        ev = 'HERNQUIST'
                break

        fitted_dict =fitted_dict[ev]

    else:
        for ev in fitted_dict:
            if fitted_dict[ev]['result'] == 'random_disk' or type == 'DENSITY':
                profile_to_fit.type = 'random_disk'
                return 
        print_log(f'You claim your profile is {type} but we fail to fit a proper function an it is not a disk.',\
            cfg,case=['main','screen'])
        raise InputError(f'We can not process profile {function}')

    profile = fitted_dict['profile']*unit.Lsun/unit.pc**2
   
    for i,parameter in enumerate(fitted_dict['component_parameter']):
        setattr(profile_to_fit,parameter,fitted_dict['parameters'][i]*\
            fitted_dict['component_units'][i])
   
    if not cfg.RC_Construction.keep_random_profiles:
        profile_to_fit.type = fitted_dict['result']
        profile_to_fit.values = profile 
       
    return
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

def single_fit_profile(profile_to_fit,fit_function,initial,cfg=None,\
                        name='Generic',count= 0):
 

    inp_fit_function = {'function':fit_function, 'variables':[]}
    parameter_settings = {}
    for i,parameter in enumerate(inspect.signature(fit_function).parameters):
        if parameter != 'r':
            parameter_settings[parameter] = [initial[i-1],0.,None,True,True]
            if parameter == 'mass':
                parameter_settings[parameter][2] =\
                    integrate_surface_density(profile_to_fit.radii,profile_to_fit.values)
            if parameter == 'central':
                parameter_settings[parameter][2] = 2*profile_to_fit.values[0].value
            inp_fit_function['variables'].append(parameter)
    setattr(profile_to_fit,'numpy_curve',inp_fit_function)
    setattr(profile_to_fit,'fitting_variables',parameter_settings)

    initial_parameters,original_settings = initial_guess(profile_to_fit,\
        cfg=cfg,negative=False,minimizer = 'leastsq')
    print_log(f'Starting mcmc for  {name}',cfg,case=['main'])
   
    if profile_to_fit.errors is None:
        profile_to_fit.errors = 0.1*profile_to_fit.values
   
    optical_fits,emcee_results = lmfit_run(cfg,profile_to_fit)
   
    tot_parameters = [optical_fits[x][0] for x in optical_fits]

    # let's see if our fit has a reasonable reduced chi square
    profile = fit_function(profile_to_fit.radii.value,*tot_parameters)
    red_chi = np.sum((profile_to_fit.values[profile_to_fit.values.value > 0.].value\
        -profile[profile_to_fit.values.value > 0.])**2/(profile_to_fit.errors[profile_to_fit.values.value > 0.].value))
    red_chisq = red_chi/(len(profile_to_fit.values[profile_to_fit.values.value > 0.])\
                         -len(tot_parameters))
    #plot_profile(radii,density,profile,name=name,
    #                output_dir=output_dir,red_chi= red_chisq,count=count)
    print_log(f'''FIT_PROFILE: We fit the {name} with a reduced Chi^2 = {red_chisq}. \n'''\
        , cfg,case=['main'])
    return tot_parameters,red_chisq,profile
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

def calc_truncation_function(radii, truncation_radius,softening_length):
    '''Calculate the truncation function following GALFIT Peng 2010
    (see Equation (B2) in Appendix B)'''
    if truncation_radius.unit !=  softening_length.unit:
        raise RuntimeError(f'For the truncation radius and the softening length should be the same. ({truncation_radius.unit,softening_length.unit})')
    B = 2.65-4.98*(truncation_radius.value/(softening_length.value))
    Pn = 0.5*(np.tanh((2.-B)*radii.value/truncation_radius.value +B)+1.)
    return Pn

def truncate_density_profile(profile):
    if not profile.truncation_radius is None:
        if profile.truncation_radius < profile.radii[-1]:
            Pn= calc_truncation_function(profile.radii,\
                profile.truncation_radius,profile.softening_length)
         
            profile.values =  profile.values*(1.-Pn)
            