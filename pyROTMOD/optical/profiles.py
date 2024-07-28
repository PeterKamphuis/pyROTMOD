# -*- coding: future_fstrings -*-
from pyROTMOD.rotmass.rotmass import initial_guess,mcmc_run
from pyROTMOD.support.errors import InputError
from pyROTMOD.optical.conversions import mag_to_lum
from pyROTMOD.support.minor_functions import convertskyangle,integrate_surface_density,\
    print_log, get_effective_radius
from astropy import units as unit
from scipy.special import k1,gamma
import numpy as np
import warnings
import copy
import inspect

def determine_profiles_to_fit(type):
    if type == 'DENSITY':
        evaluate = ['EXPONENTIAL','EXP+HERN']
    elif type == 'BULGE':
        evaluate = ['HERNQUIST']
    elif type == 'DISK':
        evaluate = ['EXPONENTIAL']
    else:
        evaluate = [type]
    return evaluate
# This is untested for now
def edge_luminosity(components,radii = None,exposure_time = 1.,band = 'WISE3.4'\
                    ,distance=0.*unit.Mpc ):
   
    if radii is None:
        radii = components.radii
    #mu_0 (mag/arcsec) = -2.5*log(sig_0/(t_exp*dx*dy))+mag_zpt
    #this has to be in L_solar/pc^2 but our value comes in mag/arcsec^2
    if radii.unit != unit.kpc:
        radii = convertskyangle(radii,distance,quantity=True)
    central_luminosity = mag_to_lum(components.central_SB,band = band, distance=distance)

    #Need to integrate the luminosity of the model and put it in component[1] still
    components.central_SB = central_luminosity
    if components.scale_length.unit != unit.kpc:
        components.scale_length = convertskyangle(components.scale_length,\
                                                  distance,quantity=True)
    if components.scale_height.unit != unit.kpc:
        components.scale_height = convertskyangle(components.scale_height,\
                                                  distance,quantity=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lum_profile = central_luminosity*np.array([float(x/components.scale_length) for x in radii],dtype=float)\
            *k1(np.array([float(x/components.scale_length) for x in radii],dtype=float))
        if np.isnan(lum_profile[0]):
            lum_profile[0] = central_luminosity

    # this assumes perfect ellipses for now and no deviations are allowed
    return lum_profile
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
import inspect
 NOTE: This is not well tested yet !!!!!!!!!
'''

def exponential(r,central,h):
    '''Exponential function'''
    return central*np.exp(-1.*r/h)


def exponential_luminosity(components,radii = None,band = 'WISE3.4',distance= 0.*unit.Mpc):
    #lum_components = copy.deepcopy(components)
    #Ftot = 2πrs2Σ0q
    if radii is None:
        radii = components.radii
    IntLum = mag_to_lum(components.total_SB,band = band, distance=distance)
   
    # and transform to a face on total magnitude (where does this come from?)
    components.total_SB = IntLum/components.axis_ratio
    # convert the scale length to kpc
    components.scale_length = convertskyangle(components.scale_length,distance, quantity=True)
    # this assumes perfect ellipses for now and no deviations are allowed
    central_luminosity = IntLum/(2.*np.pi*components.scale_length.to(unit.pc)**2*components.axis_ratio) #L_solar/pc^2

    components.central_SB = central_luminosity
    if radii.unit != unit.kpc:
        radii = convertskyangle(radii,distance,quantity=True)
    profile = central_luminosity*np.exp(-1.*(radii/components.scale_length))
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


def fit_profile(radii,density,function='EXPONENTIAL_1',\
                components = None,debug =False, log = None):

    if components is None:
        raise InputError(f'We need a place to store the results please set components =') 

    if components.total_SB == None or components.R_effective == None:
        totalsb,reff = get_effective_radius(radii,density, debug=debug,log=log)
        if  components.total_SB == None:
            components.total_SB = totalsb#*unit.Msun
        if components.R_effective == None:
            components.R_effective = reff
       
    type_split=function.split('_')
    type = type_split[0]
    try:
        count = type_split[1]
    except:
        count='1'
    fit_function_dictionary = {'EXPONENTIAL':
                {'initial':[density[0],radii[density < density[0]/np.e ][0]],
                'out':['Central SB','scale length'],
                'out_units':[unit.Msun/unit.pc**2,unit.kpc],
                'function': exponential,
                'Type':'expdisk',
                'max_red_sqr': 1000,
                'name':'Exponential',
                'fail':'random_disk'},
                'HERNQUIST':
                {'initial':[components.total_SB.value/2.,float(components.R_effective.value/1.8153)],
                'out':['Total SB','scale length'],
                'out_units':[unit.Msun,unit.kpc],
                'function': hernquist_profile,
                'Type':'hernquist',
                'max_red_sqr': 3000,
                'name':'Hernquist',
                'fail':'failed'},
                'SERSIC':
                {'initial':[density[radii < components.R_effective.value][0],\
                            components.R_effective.value, 2.],
                'out':[None,'R effective','sersic index'],
                'out_units':[1,unit.kpc,1],
                'function': sersic,
                'Type':'sersic',
                'max_red_sqr': 1000,
                'name':'Sersic',
                'fail':'random_disk'},
                'EXP+HERN':
                {'initial':[components.total_SB.value/10.,float(components.R_effective.value/(10.*1.8153)),\
                                density[0]/2.,radii[density < density[0]/np.e][0]],
                'out':[['Total SB','scale length'],['Central SB','scale length']],
                'out_units':[[unit.Msun,unit.kpc],[unit.Msun/unit.pc**2,unit.kpc]],
                'function': lambda r,mass,hern_length,central,h:\
                        hernquist_profile(r,mass,hern_length) + exponential(r,central,h),
                'separate_functions': [hernquist_profile,exponential],
                'Type':['hernquist','expdisk'],
                'max_red_sqr': 3000,
                'name':'Exp_Hern',
                'fail':'failed'},
                }


    evaluate = determine_profiles_to_fit(type)
    fitted_dict = {}
    for ev in evaluate:
        try:
            tmp_fit_parameters, tmp_red_chisq,tmp_profile,total_sb = single_fit_profile(\
                    fit_function_dictionary[ev]['function'],\
                    radii,density,\
                    fit_function_dictionary[ev]['initial'],\
                    debug=debug,log=log,name=fit_function_dictionary[ev]['name'],\
                    count= count)

            if ev == 'EXPONENTIAL' and len(evaluate) == 1 and tmp_red_chisq > 50:
                evaluate.append('EXP+HERN')

            if tmp_red_chisq > fit_function_dictionary[ev]['max_red_sqr']:

                print_log(f'''The fit to {ev} has a red Chi^2 {tmp_red_chisq}.
As this is higher than {fit_function_dictionary[ev]['max_red_sqr']} we declare a mis fit''',log)
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
                'profile': tmp_profile,
                'result': fit_function_dictionary[ev]['Type'],
                'component_parameter': fit_function_dictionary[ev]['out'],
                'component_units':fit_function_dictionary[ev]['out_units']}
        except Exception as exiting:
            print_log(f'FIT_PROFILE: We failed to fit {ev}:',log)
            #sup.print_log(traceback.print_exception(type(exiting),exiting,exiting.__traceback__),\
            #              log, screen=False)
            #exit()
            fitted_dict[ev]= {'result': fit_function_dictionary[ev]['fail'],\
                              'red_chi_sq': float('NaN')}


    red_chi = [fitted_dict[x]['red_chi_sq']  for x in fitted_dict]
    #red_chi = [float('NaN'),float('NaN')]
    if not np.isnan(np.nanmin(red_chi)):
        print_log('FIT_PROFILE: We have found at least one decent fit.',log)
        for ev in fitted_dict:
            if fitted_dict[ev]['red_chi_sq'] == np.nanmin(red_chi):
                break

        fitted_dict =fitted_dict[ev]

    else:
        for ev in fitted_dict:
            if fitted_dict[ev]['result'] == 'random_disk' or type == 'DENSITY':
                profile = density
                components.type = 'random_disk'
                return True,components,profile
        print_log(f'You claim your profile is {type} but we fail to fit a proper function an it is not a disk.',log)
        raise InputError(f'We can not process profile {function}')

    profile = fitted_dict['profile']
    if len(fitted_dict['result']) == 2:
        components_com = [copy.deepcopy(components),copy.deepcopy(components)]
        offset =0
        for x in [0,1]:
            for i,parameter in enumerate(fitted_dict['component_parameter'][x]):
                setattr(components_com[x],parameter,fitted_dict['parameters'][i+offset]*fitted_dict['component_units'][x][i])
            offset = len(fitted_dict['component_parameter'][x])
            components_com[x].type = fitted_dict['result'][x]
        components = copy.deepcopy(components_com)
        result = 'process'
    else:
        for i,parameter in enumerate(fitted_dict['component_parameter']):
            setattr(components,parameter,fitted_dict['parameters'][i]*fitted_dict['component_units'][i])
        components.type = fitted_dict['result']
        result = True
    return result,components,profile
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

def hernquist_profile(r, mass, h):
    '''
    The hernquist density profile (Eq 2, Hernquist 1990)
    mass/(np.pi*2)*(scale_length/radii)*1./(radii+scale_length)**3
    Note that in galpy this is amp/(4*pi*a**3)*1./((r/a)(1+r/a)**2
    With amp = 2. mass
    Both have inf at r = 0. so if radii == 0 it needs to be adapted
    This is a 3D profile, we need a 2 profile
    We want to fit the Vaucouleur to the profile and the relate back Re = 1.8153 a (eq38) and get I_0 from the mass
    The projected profiles are in eq 32
    here if h is in kpc then the output is in M_solar/kpc**2
    '''
    s = r/h
    hpc=1000.*h
    #XS_1 = 1./np.sqrt(1-s[s < 1]**2)*1./np.sech(s[s > 1])
    #XS_2 = 1./np.sqrt(s[s > 1]**2-1)*1./np.sec(s[s > 1])
    XS_1 = 1./np.sqrt(1-s[s < 1]**2)*np.log((1+np.sqrt(1-s[s<1]**2))/s[s < 1])
    XS_2 = 1./np.sqrt(s[s > 1]**2-1)*1./np.cos(1./s[s > 1])
    XS = np.array(list(XS_1)+list(XS_2),dtype=float)
    profile = mass/(2.*np.pi*hpc**2*(1-s**2)**2)*((2+s**2)*XS-3)

    return profile

'''
#Calculate the effective radius
def obtain_R_eff(radii,density):
    # if this is specified as an exponential disk without the parameters defined we fit a expoenential
    ring_area = integrate_surface_density(radii,density,calc_ring_area=True)
    R_eff = radii[-1] #*unit.kpc
    for i in range(1,len(radii)):
        current_mass = np.sum(ring_area[:i]*density[:i]) #*unit.Msun
        if current_mass.value > components['Total SB'].value/2.:
            R_eff = radii[i] #*unit.kpc
            break
    return R_eff
'''

def sersic(r,effective_luminosity,effective_radius,n):
    '''sersic function'''
    kappa = -1.*(2.*n-1./3.)
    func = effective_luminosity*np.exp(kappa*((r/effective_radius)**(1./n))-1)
    return func

def sersic_luminosity(components,radii = None,band = 'WISE3.4',distance= 0.*unit.Mpc):
    # This is not a deprojected surface density profile we should use the formula from Baes & gentile 2010
    # Once we implement that it should be possible use an Einasto profile/potential
    if radii is None:
        radii = components.radii
      
    IntLum = mag_to_lum(components.total_SB,band=band, distance=distance)
    kappa=2.*components.sersic_index-1./3. # From https://en.wikipedia.org/wiki/Sersic_profile

    if components.R_effective.unit != unit.kpc:
        R_kpc = convertskyangle(components.R_effective,distance,quantity =True)

    if radii.unit != unit.kpc:
        radii = convertskyangle(radii,distance,quantity=True)
    effective_luminosity = IntLum/(2.*np.pi*(R_kpc.to(unit.pc))**2*\
                            np.exp(kappa)*components.sersic_index*\
                            kappa**(-2*components.sersic_index)*\
                            components.axis_ratio*gamma(2.*components.sersic_index)) #L_solar/pc^2

    #lum_components['Total SB'] = IntLum/components['axis ratio']
    components.total_SB = IntLum
    components.R_effective = R_kpc
    lum_profile = effective_luminosity*np.exp(-1.*kappa*((radii/components.R_effective)**(1./components.sersic_index)-1))
    components.central_SB = effective_luminosity*np.exp(-1.*kappa*(((0.*unit.kpc)/components.R_effective)**(1./components.sersic_index)-1))

    return lum_profile
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

def single_fit_profile(fit_function,radii,density,initial,debug=False,log=None,\
                        name='Generic',count= 0):

    '''
    model = lmfit.Model(fit_function)
    for i,parameter in enumerate(inspect.signature(fit_function).parameters):
        if parameter != 'radii':
            model.set_param_hint(parameter,value=initial[i-1],\
                min=0,\
                max=np.inf,\
                vary=True
                )
    parameters= model.make_params()
    print(parameters)
    exit()
    tot_parameters, tot_covariance = curve_fit(fit_function, radii[density > 0.],\
                    density[density > 0.],sigma= 0.05*density[density > 0.],\
                    p0=initial,bounds =[0.,np.inf],maxfev=5000)
    '''

    inp_fit_function = {'function':fit_function, 'variables':[]}
    parameter_settings = {}
    for i,parameter in enumerate(inspect.signature(fit_function).parameters):
        if parameter != 'r':
            parameter_settings[parameter] = [initial[i-1],0.,None,True,True]
            if parameter == 'mass':
                parameter_settings[parameter][2] = integrate_surface_density(radii,density)
            if parameter == 'central':
                parameter_settings[parameter][2] = 2*density[0]

            inp_fit_function['variables'].append(parameter)
    input_density = [density[density > 0.], 0.1*density[density > 0.]]
    initial_parameters = initial_guess(inp_fit_function,radii[density > 0.],input_density,parameter_settings,\
                    debug=debug,function_name=name,log=log,negative=False, minimizer = 'leastsq')
    print_log(f'Starting mcmc for  {name}',log,screen=True)
    optical_fits,emcee_results = mcmc_run(inp_fit_function,radii[density > 0.],input_density,\
                            initial_parameters,parameter_settings,\
                            debug=debug,log=log,\
                            negative=False,\
                            steps=1000,
                            results_name= f'Optical_{name}')
    tot_parameters = [optical_fits[x][0] for x in optical_fits]

    # let's see if our fit has a reasonable reduced chi square
    profile = fit_function(radii,*tot_parameters)
    red_chi = np.sum((density[density > 0.]-profile[density > 0.])**2/(0.1*density[density > 0.]))
    red_chisq = red_chi/(len(density[density > 0.])-len(tot_parameters))
    #plot_profile(radii,density,profile,name=name,
    #                output_dir=output_dir,red_chi= red_chisq,count=count)
    print_log(f'''FIT_PROFILE: We fit the {name} with a reduced Chi^2 = {red_chisq} with a 5% error. \n''', log)
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
