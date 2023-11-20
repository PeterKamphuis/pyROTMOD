# -*- coding: future_fstrings -*-

class InitialGuessWarning(Exception):
    pass
class InputError(Exception):
    pass
import numpy as np
import warnings
import pyROTMOD.constants as cons
import copy
from types import FunctionType
import lmfit
import corner
import pyROTMOD.rotmass.potentials as V
import pyROTMOD.constants as cons
from pyROTMOD.support import print_log,get_uncounted
from sympy import symbols, sqrt,atan,pi,log,Abs,lambdify
from scipy.optimize import differential_evolution,curve_fit,OptimizeWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt

class RunTimeError(Exception):
    pass
def build_curve(function_variables,disk_var,dm_halo,Baryonic_RCs,debug=False,log=None):
    # First get the DM function in sympy format
    DM_curve_sym = getattr(V, dm_halo)()
    dm_variables= [x for x in list(DM_curve_sym.free_symbols) if str(x) not in ['r'] ]
    dm_variables.insert(0,symbols('r'))
    #Then start added the RCs in symbols
    curve_sym = DM_curve_sym**2
    fixed_variables = []
    replace_dict = {}
    symbols_to_replace = []
    values_to_be_replaced = []
    for component in Baryonic_RCs:

        fitM, fitV = symbols(f"{disk_var[component][0]} {disk_var[component][1]}")
        curve_sym = curve_sym +fitM*fitV*Abs(fitV)
        symbols_to_replace.append(fitV)
        # keep a record of the values we need to enter
        replace_dict[disk_var[component][1]]=Baryonic_RCs[component]['RC']
        values_to_be_replaced.append(Baryonic_RCs[component]['RC'])
 
    curve_sym = sqrt(curve_sym)
    #Now let's transform these to actual functions
    #for dm this is straight forward
    DM_curve_np = lambdify(dm_variables,DM_curve_sym,"numpy")
    print_log(f'''BUILD_CURVE: We are fitting this DM halo :
{'':8s}{DM_curve_np.__doc__}
''',log,debug=debug,screen =True)

    #For the actual fit curve we need to replace the  V components with their actual values
    #make sure that r is always the first input on the function and we will replace the RCs

    curve_symbols_out = [x for x in list(curve_sym.free_symbols) if str(x) not in ['r']+[str(y) for y in symbols_to_replace] ]
    curve_symbols_out.insert(0,symbols('r'))
    curve_symbols_in = symbols_to_replace+curve_symbols_out
    initial_formula = lambdify(curve_symbols_in,curve_sym,"numpy")

    print_log(f'''BUILD_CURVE:: We are fitting this complete formula:
{'':8s}{initial_formula.__doc__}
''',log,debug=debug,screen =True)
    # since lmfit is a piece of shit we have to constract or final formula through exec
    clean_code = create_formula_code(initial_formula,replace_dict, function_name='curve_np',debug=debug,log=log)
    exec(clean_code,globals())
   
    # This piece of code can be used to check the exec made fit function
    curve_lamb = lambda *curve_symbols_out: initial_formula(*values_to_be_replaced, *curve_symbols_out)
    DM_final = {'function': DM_curve_np , 'variables': [str(x) for x in dm_variables] }
    Curve_final = {'function': curve_np , 'variables': [str(x) for x in curve_symbols_out]}
    check_final = {'function': curve_lamb , 'variables': [str(x) for x in curve_symbols_out]}
    return DM_final,Curve_final,check_final

build_curve.__doc__ =f'''
 NAME:
    build_curve

 PURPOSE:
    build the combined curve that we want to fit, load the DM function

 CATEGORY:
    rotmass

 INPUTS:

 OPTIONAL INPUTS:

 OUTPUTS:

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''

def calculate_confidence_area(radii,full_curve,final_variable_fits):
    all_possible_curve = []
    all_sets =[]
    ranges = []
    variables = [x for x in full_curve['variables'] if x != 'r']
    coll_variables = []
    for variable in variables:
        fits_variable = None
        for x in final_variable_fits:
            if final_variable_fits[x]['Variables'][0] == variable:
                fits_variable = x
        if fits_variable != None:  
            coll_variables.append(final_variable_fits[fits_variable]['Variables'][0])
            if final_variable_fits[fits_variable]['Settings'][3] and \
                final_variable_fits[fits_variable]['Settings'][1] != None and \
                    final_variable_fits[fits_variable]['Settings'][2] != None:
                ranges.append([final_variable_fits[fits_variable]['Settings'][1],final_variable_fits[fits_variable]['Settings'][2]])
            else:
                ranges.append([final_variable_fits[fits_variable]['Settings'][0]])

    if not np.array_equal(variables,coll_variables):
        print(f'''We have messed up the collection of variables for the curve {full_curve['function'].__name__}
requested variables = {variables}
collected variables = {coll_variables}''' )
        raise RunTimeError(f'Ordering Error in variables collection')
    if  None in ranges:
         return [np.zeros(len(radii)),np.zeros(len(radii))]
    all_sets = np.array(np.meshgrid(*ranges)).T.reshape(-1,len(ranges))
    
    for set in all_sets: 
        curve = full_curve['function'](radii,*set)
        all_possible_curve.append(curve)
    
    all_possible_curve=np.array(all_possible_curve,dtype=float)
   
    minim = np.argmin(all_possible_curve,axis=0)
    maxim =  np.argmax(all_possible_curve,axis=0)
    confidence_area = [[all_possible_curve[loc,i] for i,loc in enumerate(minim) ],\
                       [all_possible_curve[loc,i] for i,loc in enumerate(maxim)]]
   
    return confidence_area

calculate_confidence_area.__doc__ =f'''
 NAME:
    calculate_confidence_area

 PURPOSE:
    calculate the minimum and maximum possible for every point based on the 1 sigma errors in fit

 CATEGORY:
    rotmass

 INPUTS:

 OPTIONAL INPUTS:

 OUTPUTS:

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''

def calculate_curves(radii ,total_RC,Baryonic_RC,full_curve,DM_curve,function_variable_settings,disk_var):
    combined_RC = {'RADI': radii, 'V_total': [total_RC[0],total_RC[0]-total_RC[1],total_RC[0]+total_RC[1]]}
   
    for key in Baryonic_RC:

        combined_RC[disk_var[key][1]]=[np.sqrt(function_variable_settings[key]['Settings'][0])*Baryonic_RC[key]['RC'],[],[]]
        if function_variable_settings[key]['Settings'][3]:
            
            if np.sum(Baryonic_RC[key]['Errors']) != 0.:
                low_curve = Baryonic_RC[key]['RC']-Baryonic_RC[key]['Errors']
                high_curve =  Baryonic_RC[key]['RC']+Baryonic_RC[key]['Errors']
            else:
                low_curve = Baryonic_RC[key]['RC']
                high_curve =  Baryonic_RC[key]['RC'] 
            below_low = []
            below_high = []
            if function_variable_settings[key]['Settings'][1]:
                low_curve = np.sqrt(abs(function_variable_settings[key]['Settings'][1]))*low_curve 
                #below_low = np.where(low_curve < 0.)[0]
              
            if function_variable_settings[key]['Settings'][2]:
                high_curve = np.sqrt(abs(function_variable_settings[key]['Settings'][2]))*high_curve
                #below_high = np.where(high_curve < 0.)[0]
           
            if np.array_equal(low_curve,Baryonic_RC[key]['RC']) and np.array_equal(high_curve,Baryonic_RC[key]['RC']):
                combined_RC[disk_var[key][1]][1] = np.zeros(len(radii))
                combined_RC[disk_var[key][1]][2] = np.zeros(len(radii))
            elif not np.array_equal(low_curve,Baryonic_RC[key]['RC']) and not np.array_equal(high_curve,Baryonic_RC[key]['RC']):
                combined_RC[disk_var[key][1]][1] = low_curve
                combined_RC[disk_var[key][1]][2] = high_curve
            elif np.array_equal(low_curve,Baryonic_RC[key]['RC']):
                combined_RC[disk_var[key][1]][1] = 2.*combined_RC[disk_var[key][1]][0]-high_curve
                combined_RC[disk_var[key][1]][2] = high_curve
            else:
                combined_RC[disk_var[key][1]][2] = 2.*combined_RC[disk_var[key][1]][0]-low_curve
                combined_RC[disk_var[key][1]][1] = low_curve 
          
        else:
            combined_RC[disk_var[key][1]][1] = np.zeros(len(radii))
            combined_RC[disk_var[key][1]][2] = np.zeros(len(radii))
   
    dm_variables = [function_variable_settings[x]['Settings'][0] for x in DM_curve['variables'] if x != 'r']
   
    if any(x == None for x in dm_variables):
        pass
    else:
        #This is where we should calculate an error on the DM curve
        combined_RC['V_dm'] = [DM_curve['function'](radii,*dm_variables),[],[]]
        confidence_curve = calculate_confidence_area(radii,DM_curve,function_variable_settings)
        combined_RC['V_dm'][1] = confidence_curve[0]
        combined_RC['V_dm'][2] = confidence_curve[1]

   
    full_variables = []
    for x in full_curve['variables']:
        if x != 'r':
            for key in function_variable_settings:
                if x == function_variable_settings[key]['Variables'][0]:
                    full_variables.append( function_variable_settings[key]['Settings'][0])
  
    if any(x == None for x in full_variables):
        pass
    else:
        combined_RC['V_fit'] = [full_curve['function'](radii,*full_variables),[],[]]
        confidence_curve = calculate_confidence_area(radii,full_curve,function_variable_settings)
        combined_RC['V_fit'][1] = confidence_curve[0]
        combined_RC['V_fit'][2] = confidence_curve[1]
   
    return combined_RC

calculate_curves.__doc__ =f'''
 NAME:
    calculate_curves

 PURPOSE:
    calculate a dictionary that contains all  the curves together with their applied variables

 CATEGORY:
    rotmass

 INPUTS:

 OPTIONAL INPUTS:

 OUTPUTS:

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''
def calculate_red_chisq(curves,parameters):
    free_parameters=0
    for var in parameters:
        if parameters[var]['Settings'][3]:
            free_parameters  += 1.
    errors = np.array([np.mean([x-y,z-x]) for x,y,z in zip(*curves['V_total'])],dtype=float)
    Chi_2 = np.sum((curves['V_total'][0]-curves['V_fit'][0])**2/errors**2)
    red_chisq = Chi_2/(len(curves['V_total'][0])-free_parameters)
    return red_chisq
calculate_red_chisq.__doc__ =f'''
 NAME:
    calculate_curves

 PURPOSE:
    calculate a the reduced chi square of the fit
 CATEGORY:
    rotmass

 INPUTS:
    curves= the fitted curves +observed curve
    parameters = the parameters in the total curve
 OPTIONAL INPUTS:

 OUTPUTS:

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''

def create_disk_var(collected_RCs):
    disk_var = {}
    counters = {'GAS': 1, 'DISK': 1, 'BULGE': 1} 
    for RC in collected_RCs:
        key = RC[0]
        if key != 'RADI':
            bare,no = get_uncounted(key)
            if bare in ['EXPONENTIAL','SERSIC']:
                disk_var[key] = [f'Gamma_disk_{counters["DISK"]}',f'V_disk_{counters["DISK"]}']
                counters['DISK'] += 1
            elif bare in ['DISK_GAS']:
                disk_var[key] = [f'Gamma_gas_{counters["GAS"]}',f'V_gas_{counters["GAS"]}']
                counters['GAS'] += 1
            elif bare in ['HERNQUIST','BULGE']:
                disk_var[key] = [f'Gamma_gas_{counters["BULGE"]}',f'V_bulge_{counters["BULGE"]}']
                counters['BULGE'] += 1
    return disk_var

def create_formula_code(initial_formula,replace_dict,function_name='python_formula' ,log=None,debug =False):
    lines=initial_formula.__doc__.split('\n')

    dictionary_trans = {'sqrt':'np.sqrt', 'arctan': 'np.arctan', 'pi': 'np.pi','log': 'np.log', 'abs': 'np.abs'}
    found = False
    code =''
    for line in lines:
        inline = line.split()
        if len(inline) > 0:
            if line.split()[0].lower() == 'source':
                found = True
                continue
            if found:
                code += line+'\n'
                if line.split()[0].lower() == 'return':
                    break
    clean_code = ''
    for i,line in enumerate(code.split('\n')):
        if i == 0:
            line = line.replace('_lambdifygenerated',function_name)
            for key in replace_dict:
                line = line.replace(key+',','')
        if i == 1:
            for key in dictionary_trans:
                line = line.replace(key,dictionary_trans[key])
            for key in replace_dict:
                line = line.replace(key,'np.array(['+', '.join([str(i) for i in replace_dict[key]])+'],dtype=float)')
        clean_code += line+'\n'
    if debug:
        print_log(f''' This the code for the formula that is finally fitted.
{clean_code}
''',log,debug=True)
    return clean_code
create_formula_code.__doc__ =f'''
 NAME:
    create_formula_code

 PURPOSE:
    create the code that can be ran through exec to get the function to be fitted.

 CATEGORY:
    rotmass

 INPUTS:

 OPTIONAL INPUTS:

 OUTPUTS:

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''

def get_baryonic_RC(radii,total_RC,derived_RCs,
                    V_total_error=[0.,0.,0.],baryonic_error= {'Empty':True},
                    log = None, debug=False):
    if 'Empty' not in baryonic_error:
        baryonic_error['Empty']=False
    #We can strip the total rc and the radius from their indicators
    new_radii= np.array(radii[2:])
    new_RC = np.array(total_RC[2:])
    new_RC_error = np.array(V_total_error[2:])
    #Radius of 0. messses thing up so lets interpolate to the second point
    if new_radii[0] == 0.:
        new_radii[0]=new_radii[1]/2.
        new_RC[0]=(new_RC[0]+new_RC[1])/2.
        if np.sum(new_RC_error) != 0.:
            new_RC_error[0] = (new_RC_error[0]+new_RC_error[1])/2.


    RC ={}
    for x in range(len(derived_RCs)):
        component = ['Empty','Empty']
        if derived_RCs[x][0] == 'RADI':
            rad_in = np.array(derived_RCs[x][2:])
            component = ['Empty','Empty']
        elif derived_RCs[x][0][:6] in ['EXPONE','DISK_S','SERSIC']:
            component = [derived_RCs[x][0],'MD']
        elif derived_RCs[x][0][:3] in ['BUL','HER']:
            component = [derived_RCs[x][0],'MB']
        elif derived_RCs[x][0][:6] == 'DISK_G':
            component =  [derived_RCs[x][0],'MG']
        else:
            if derived_RCs[x][0][:3] not in ['EXP','BUL','SER','DIS','HER']:
                print_log(f"We do not recognize this type ({derived_RCs[x][0][:3]}) of RC and don't know what to do with it",log)
                exit()
   
        if component[0] != 'Empty':
            if component[0] not in RC:
                RC[component[0]] = {'Type': component[1], 'RC': np.array(derived_RCs[x][2:])}
                if x in baryonic_error:
                    RC[component[0]]['Errors'] = np.array(baryonic_error[x][2:])
                else:
                    RC[component[0]]['Errors'] = np.zeros(len( RC[component[0]]['RC']))
            else:
                if x in baryonic_error:
                    RC[component[0]]['Errors'] = np.array([np.sqrt(x*x_err/np.sqrt(x**2+y**2)+y*y_err/np.sqrt(x**2+y**2))\
                                     for x,y,x_err,y_err in \
                                        zip(RC[component[0]]['RC'],derived_RCs[x][2:],\
                                     RC[component[0]]['Errors'],baryonic_error[x][2:])])

                RC[component[0]]['RC'] = np.array([np.sqrt(x**2+y**2) for x,y in zip(RC[component[0]]['RC'],derived_RCs[x][2:])])


    # if our requested radii do not correspond to the wanted radii we interpolat
    for key in RC:
        if len(RC[key]['RC']) > 0.:
            if np.sum([float(x)-float(y) for x,y in zip(new_radii,rad_in)]) != 0.:
                RC[key]['RC'] = np.array(np.interp(new_radii,rad_in,RC[key]['RC']),dtype=float)
                RC[key]['Errors'] = np.array(np.interp(new_radii,rad_in,RC[key]['Errors']),dtype=float)
        else:
            raise InputError(f'The parameter {key} is requested to be added to the formula but the RC is missing')
    if np.sum(new_RC_error) != 0.:
        new_RC = [new_RC,new_RC_error]
    else:
        new_RC = [new_RC,np.zeros(len(new_RC))]
    return new_radii, new_RC, RC

get_baryonic_RC.__doc__ =f'''
 NAME:
    get_baryonic_RC

 PURPOSE:
    Define the baryonic RCs that are defined for the fit. and transform radii and total_RC to float numpy arrays
    This combines the different components of the galfit or tirific fit to a single RC for Bulge, Disk and Gas disk

 CATEGORY:
    rotmass

 INPUTS:

 OPTIONAL INPUTS:

 OUTPUTS:

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''


def initial_guess(fit_function,radii,total_RC,function_variable_settings,\
                  debug =False,log=None,negative = False,\
                  minimizer = 'differential_evolution'):

    from numpy import nan
  
    #succes_outer = False
    guess_variables = copy.deepcopy(function_variable_settings)
    #First set the model
    #model = lmfit.Model(sum_of_square)

    model = lmfit.Model(fit_function['function'])
    #no_input = False
   
    for variable in function_variable_settings:
        parameter,no = get_uncounted(guess_variables[variable]['Variables'][0])
        if (function_variable_settings[variable]['Settings'][1] is None):
            if parameter in ['Gamma_disk','Gamma_gas','Gamma_bulge']:
                guess_variables[variable]['Settings'][1] = function_variable_settings[variable]['Settings'][0]/10.
            else:
                guess_variables[variable]['Settings'][1] = 0.1

        if (function_variable_settings[variable]['Settings'][2] is None):
            if parameter in ['Gamma_disk','Gamma_gas','Gamma_bulge']:
                guess_variables[variable]['Settings'][2] = function_variable_settings[variable]['Settings'][0]*20.
            else:
                guess_variables[variable]['Settings'][2] = 1000.
        if (function_variable_settings[variable]['Settings'][0] is None):
            #no_input = True
            guess_variables[variable]['Settings'][0] = float(np.random.rand()*(guess_variables[variable]['Settings'][2]-guess_variables[variable]['Settings'][1])+guess_variables[variable]['Settings'][1])
        if guess_variables[variable]['Settings'][3]:
            print_log(f'''Setting {variable} with value {guess_variables[variable]['Settings'][0]} and fitting = {guess_variables[variable]['Settings'][3]}.
limits are between {guess_variables[variable]['Settings'][1]} - {guess_variables[variable]['Settings'][2]}
''',log)
        model.set_param_hint(guess_variables[variable]['Variables'][0],value=guess_variables[variable]['Settings'][0],\
                    min=guess_variables[variable]['Settings'][1],\
                    max=guess_variables[variable]['Settings'][2],\
                    vary=guess_variables[variable]['Settings'][3]
                    )
  
    parameters= model.make_params()
    no_errors = True
    counter =0
    while no_errors:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")

            #initial_fit = model.fit(data=total_RC[0], params=parameters, r=radii,method='differential_evolution'\
            #                         , nan_policy='omit',scale_covar=False)
          
            initial_fit = model.fit(data=total_RC[0], params=parameters, r=radii,method= minimizer\
                                     ,nan_policy='omit',scale_covar=False)
            if not initial_fit.errorbars or not initial_fit.success:
                print(f"\r The initial guess did not produce errors, retrying with new guesses: {counter/float(500.)*100.:.1f} % of maximum attempts.", end =" ",flush = True)
                for variable in guess_variables:
                    if not function_variable_settings[variable]['Settings'][0]:
                        guess_variables[variable]['Settings'][0] = float(np.random.rand()*\
                                (guess_variables[variable]['Settings'][2]-guess_variables[variable]['Settings'][1])\
                                +guess_variables[variable]['Settings'][1])
                        parameters[ guess_variables[variable]['Variables'][0] ].value =guess_variables[variable]['Settings'][0]
                counter+=1
                if counter > 501.:
                    raise InitialGuessWarning(f'We could not find errors and initial guesses for the function. Try smaller boundaries or set your initiial values')
            else:
                print_log(f'\n',log)
                print_log(f'The initial guess is a succes. \n',log)
                for variable in guess_variables:
                    fit_variable = guess_variables[variable]['Variables'][0]
                    buffer = np.max([abs(initial_fit.params[fit_variable].value*0.25)\
                                     ,10.*initial_fit.params[fit_variable].stderr])
                    guess_variables[variable]['Settings'][1] = float(initial_fit.params[fit_variable].value-buffer \
                                            if (function_variable_settings[variable]['Settings'][1] is None) \
                                            else initial_fit.params[fit_variable].min)
                    if not negative:
                        if guess_variables[variable]['Settings'][1] < 0.:
                            guess_variables[variable]['Settings'][1] =1e-7
                    guess_variables[variable]['Settings'][2] = float(initial_fit.params[fit_variable].value+buffer \
                                             if (function_variable_settings[variable]['Settings'][2] is None)\
                                             else initial_fit.params[fit_variable].max)

                    guess_variables[variable]['Settings'][0] = float(initial_fit.params[fit_variable].value)
                no_errors = False
    print_log(f'''INITIAL_GUESS: These are the statistics and values found through {minimizer} fitting of the residual.
{initial_fit.fit_report()}
''',log)
    return guess_variables
initial_guess.__doc__ =f'''
 NAME:
    initial_guess

 PURPOSE:
    Make sure that we have decent values and boundaries for all values, also the ones that were left unset

 CATEGORY:
    rotmass

 INPUTS:
    rotmass_settings = the original settings from the yaml including all the defaults.
 OPTIONAL INPUTS:

 OUTPUTS:

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''

def mcmc_run(fit_function,radii,total_RC,function_variable_settings,original_variable_settings,\
                  out_dir = None,log_directory=None,debug=False,negative=False,log=None,steps=2000.,\
                  results_name = 'MCMC',fixed_boundaries = False):
    steps=int(steps*(len(fit_function['variables'])-1))
    burning=int(steps/4.)
    #First set the model
    model = lmfit.Model(fit_function['function'])
    #then set the hints

    for variable in function_variable_settings:
            parameter = function_variable_settings[variable]["Variables"][0]
            if function_variable_settings[variable]["Settings"][1] == function_variable_settings[variable]["Settings"][2]:
                function_variable_settings[variable]["Settings"][1] = function_variable_settings[variable]["Settings"][1]*0.9
                function_variable_settings[variable]["Settings"][2] = function_variable_settings[variable]["Settings"][2]*1.1
            if function_variable_settings[variable]["Settings"][3]:
                print_log(f'''Setting {parameter} with value {function_variable_settings[variable]["Settings"][0]}.
With the boundaries between {function_variable_settings[variable]["Settings"][1]} - {function_variable_settings[variable]["Settings"][2]}
''',log)
            else:
                print_log(f'''Keeping {parameter} fixed at {function_variable_settings[variable]["Settings"][0]}.
''',log)
            model.set_param_hint(parameter,value=function_variable_settings[variable]["Settings"][0],\
                        min=function_variable_settings[variable]["Settings"][1],\
                        max=function_variable_settings[variable]["Settings"][2],
                        vary=function_variable_settings[variable]["Settings"][3])
    parameters = model.make_params()
    emcee_kws = dict(steps=steps, burn=burning, thin=10, is_weighted=True)
    no_succes =True
    count = 0
    while no_succes:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
            warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
            warnings.filterwarnings("ignore", message="invalid value encountered in log")
            result_emcee = model.fit(data=total_RC[0], r=radii, params=parameters, method='emcee',nan_policy='omit',
                             fit_kws=emcee_kws,weights=1./total_RC[1])
            no_succes=False

            triggered =False
            # we have to check that our results are only limited by the boundaries in the user has set them
            for variable in original_variable_settings:
                parameter = function_variable_settings[variable]["Variables"][0]
                if function_variable_settings[variable]["Settings"][3]:
                    prev_bound = [parameters[parameter].min,parameters[parameter].max]
                    mini = result_emcee.params[parameter].value-3.*result_emcee.params[parameter].stderr
                    if not negative and mini < 0.:
                        mini = 1e-7
                    if mini < result_emcee.params[parameter].min:
                        triggered =True
                        parameters[parameter].min = result_emcee.params[parameter].value - 10.*result_emcee.params[parameter].stderr
                        if not negative and parameters[parameter].min < 0.:
                                parameters[parameter].min = 1e-7
                        if original_variable_settings[variable]["Settings"][1]:
                            if original_variable_settings[variable]["Settings"][1] > parameters[parameter].min:
                                parameters[parameter].min=original_variable_settings[variable]["Settings"][1]

                    triggered = False
                    if result_emcee.params[parameter].value+\
                        3.*result_emcee.params[parameter].stderr >\
                        result_emcee.params[parameter].max:
                        triggered =True
                        parameters[parameter].max = result_emcee.params[parameter].value + 10.*result_emcee.params[parameter].stderr
                        if original_variable_settings[variable]["Settings"][2]:
                            if original_variable_settings[variable]["Settings"][2] < parameters[parameter].max:
                                parameters[parameter].max=original_variable_settings[variable]["Settings"][2]

                    if np.array_equal(prev_bound, [parameters[parameter].min,parameters[parameter].max]) or fixed_boundaries :
                        if triggered:
                            if fixed_boundaries:
                                print_log(f''' The boundaries for {parameter} were too small, consider changing them
''',log)
                            else:
                                print_log(f''' The boundaries for {parameter} were too small, we sampled an incorrect area
Changing the parameter and its boundaries and trying again.
''',log)
                                parameters[parameter].value = result_emcee.params[parameter].value
                                print_log(f''' Setting {parameter} = {parameters[parameter].value} between {parameters[parameter].min}-{parameters[parameter].max}
''',log)
                                no_succes =True   
                        else:
                            print_log(f'''{parameter} is fitted wel in the boundaries {parameters[parameter].min} - {parameters[parameter].max}.
''',log)
                    else:
                        count += 1
                        if count >= 10 :
                            print_log(f''' Your boundaries are not converging. Condidering fitting less variables or manually fix the boundaries
''',log)
                        else:
                            print_log(f''' The boundaries for {parameter} were too small, we sampled an incorrect area
Changing the parameter and its boundaries and trying again.
''',log)
                            parameters[parameter].value = result_emcee.params[parameter].value
                            print_log(f''' Setting {parameter} = {parameters[parameter].value} between {parameters[parameter].min}-{parameters[parameter].max}
''',log)
                            no_succes =True


    print_log(result_emcee.fit_report(),log,screen=False)
    print_log('\n',log)

    if out_dir:

        label_dictionary = {'Gamma_disk':'$\mathrm{M/L_{disk}}$',
                         'Gamma_bulge':'$\mathrm{M/L_{bulge}}$',
                         'Gamma_gas':'$\mathrm{M/L_{gas}}$',
                         'RHO_0': '$\mathrm{\\rho_{c}\\times 10^{-3}(M_{\\odot}/pc^{3})}$',
                         'R_C': '$ \mathrm{R_{c}(kpc)}$',
                         'C':'$\mathrm{c}$',
                         'R200':'$ \mathrm{R_{200}(kpc)}$',
                         'm': '$\mathrm{Axion Mass\\times 10^{-23}(eV)}$',
                         'central': '$\mathrm{Central SBR} (M_{\\odot}/pc^{2})$',
                         'h': '$\mathrm{Scale length} (kpc)$',
                         'mass': '$\mathrm{Total Mass} (M_{\\odot})$',
                         'hern_length': '$\mathrm{Hernquist length} (kpc)$',
                         'effective_luminosity': '$\mathrm{L_{e}} (M_{\\odot})$' ,
                         'effective_radius': '$\mathrm{R_{e}} (kpc)$' ,
                         'n': 'Sersic Index'
                         }
        lab = []
        for x in function_variable_settings:
            parameter = function_variable_settings[x]["Variables"][0]
            strip_parameter,no = get_uncounted(parameter)
            lab.append(label_dictionary[strip_parameter])
       
        fig = corner.corner(result_emcee.flatchain, quantiles=[0.16, 0.5, 0.84],show_titles=True,
                        title_kwargs={"fontsize": 15},labels=lab)
        fig.savefig(f"{out_dir}{results_name}_COV_Fits.pdf",dpi=300)
        plt.close()
    print_log(f''' MCMC_RUN: We find the following parameters for this fit. \n''',log)

    for variable in function_variable_settings:
        parameter = function_variable_settings[variable]["Variables"][0]
        if function_variable_settings[variable]["Settings"][3]:
            function_variable_settings[variable]["Settings"][1] = float(result_emcee.params[parameter].value-\
                                    result_emcee.params[parameter].stderr)
            function_variable_settings[variable]["Settings"][2] = float(result_emcee.params[parameter].value+\
                                    result_emcee.params[parameter].stderr)

            function_variable_settings[variable]["Settings"][0] = float(result_emcee.params[parameter].value)
            print_log(f'''{parameter} = {result_emcee.params[parameter].value} +/- {result_emcee.params[parameter].stderr} within the boundary {result_emcee.params[parameter].min}-{result_emcee.params[parameter].max}
''',log)




    return function_variable_settings,result_emcee

mcmc_run.__doc__ =f'''
 NAME:
    mcmc_run

 PURPOSE:
    run emcee under the lmfit package to fine tune the initial guesses.

 CATEGORY:
    rotmass

 INPUTS:
    rotmass_settings = the original settings from the yaml including all the defaults.
 OPTIONAL INPUTS:

 OUTPUTS:

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''

def write_output_file(final_variable_fits,result_emcee,output_dir='./',\
                results_file = 'Final_Results.txt', red_chisq= None):
    #variables_fitted = [x for x in final_variable_fits]
    variable_line = f'{"Variable":>15s} {"Value":>15s} {"Error":>15s} {"Lower Bound":>15s} {"Upper Bound":>15s} \n'
    with open(f'{output_dir}{results_file}.txt','w') as file:
        file.write('# These are the results from pyROTMOD. \n')
        file.write('# An error designated as Fixed indicates a parameter that is not fitted. \n')
        file.write(f'# The Reduced Xi^2 for this fit is {red_chisq}.\n')
        file.write(f'# The Bayesian Information Criterion for this fit is {result_emcee.bic}.\n')
        file.write(variable_line)
        for variable in final_variable_fits:
            parameter = final_variable_fits[variable]["Variables"][0]
            if final_variable_fits[variable]["Settings"][4]:
                variable_line = f'{parameter:>15s} {result_emcee.params[parameter].value:>15.4f}'
                min = result_emcee.params[parameter].min
                max = result_emcee.params[parameter].max
                if not final_variable_fits[variable]["Settings"][3]:
                    variable_line += f' {"Fixed":>15s} {min:>15.4f} {max:>15.4f} \n'
                else:
                    err = result_emcee.params[parameter].stderr
                    variable_line += f' {err:>15.7f} {min:>15.4f} {max:>15.4f} \n'
                file.write(variable_line)


def plot_curves(name,curves,variables=None, halo = 'NFW',interactive = False, font = 'Times New Roman'):

    style_library = {'V_total': {'color': 'k','lines':'-','name': 'V$_{Obs}$','marker':'o', 'alpha': 0.5 ,'order' :7},
                     'V_fit': {'color': 'r','lines':'-','name': 'V$_{Total}$','marker':'o', 'alpha': 1,'order' :6},
                     'V_dm': {'color': 'b','lines':'dotted','name': f'V$_{{{halo}}}$','marker':None, 'alpha': 1,'order' :5},
                     'V_gas': {'color': 'k','lines':'dotted','name': f'V$_{{Gas}}$','marker':None, 'alpha': 1,'order' :4},
                     'V_disk': {'color': 'k','lines':'--','name': f'V$_{{Disk}}$','marker':None, 'alpha': 1,'order' :3},
                     'V_bulge': {'color': 'k','lines':'-.','name': f'V$_{{Bulge}}$','marker':None, 'alpha': 1,'order' :2},

                    }
    styles = ['-','-.',':','--','-','-.',':','--']
    labelfont = {'family': font,
             'weight': 'bold',
             'size': 14}
    plt.rc('font',**labelfont)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rc('axes', linewidth=2)
    if 'V_fit' in curves:
        figure,  (ax1, ax2) = plt.subplots(nrows=2, ncols=1,figsize=(10,10), gridspec_kw={'height_ratios': [3, 1]})
    else:
        figure = plt.figure(figsize=(10,7.5) , dpi=300, facecolor='w', edgecolor='k')
        ax1 = figure.add_subplot(1,1,1)

    for rcs_full in curves:
        rcs,no = get_uncounted(rcs_full)
        if rcs != 'RADI':
            lab = style_library[rcs]['name']
            if no != None:
                lab = f'{lab}_{no}'
          
            ax1.plot(curves['RADI'],curves[rcs_full][0],label=lab\
                     ,lw=5,linestyle = style_library[rcs]['lines'],\
                     markerfacecolor='white', markeredgewidth=4, \
                     marker=style_library[rcs]['marker'],
                     alpha = style_library[rcs]['alpha'],
                     zorder=style_library[rcs]['order'],ms=15, \
                     color= style_library[rcs]['color'])
            if np.sum(curves[rcs_full][1]) != 0.:
                if style_library[rcs]['alpha'] < 1.:
                    use_alpha = style_library[rcs]['alpha'] -0.3
                else:
                    use_alpha = style_library[rcs]['alpha'] -0.5
                ax1.fill_between(curves['RADI'], curves[rcs_full][1],\
                                curves[rcs_full][2] ,
                                color= style_library[rcs]['color'],
                                alpha = use_alpha,
                                edgecolor='none',zorder=1)

    ax1.legend()
    ax1.set_xlabel('R(kpc)', fontdict=dict(weight='bold',size=16))
    ax1.set_ylabel('V(km/s)', fontdict=dict(weight='bold',size=16))
    if 'V_fit' in curves:
        ax2.plot(curves['RADI'],curves['V_total'][0]-curves['V_fit'][0],marker='o', \
                ms=15,color='r',linestyle =None,zorder=1,lw=5)
        ax2.plot(curves['RADI'], np.zeros(len(curves['RADI'])),marker=None,lw=5, \
                ms=10,color='k',linestyle ='-',zorder=0)
        ax2.set_xlabel('Radius(kpc)', fontdict=dict(weight='bold',size=16))
        ax2.set_ylabel('Residual(km/s)', fontdict=dict(weight='bold',size=16))
        ymin,ymax=ax2.get_ylim()
        buffer = (ymax-ymin)/20.
        ax2.set_ylim(ymin-buffer,ymax+buffer)
        if variables:
            red_Chi_2 = calculate_red_chisq(curves,variables)

            ax2.text(1.0,1.0,f'''Red. $\chi^{{2}}$ = {red_Chi_2:.4f}''',rotation=0, va='top',ha='right', color='black',\
              bbox=dict(facecolor='white',edgecolor='white',pad=0.5,alpha=0.),\
              zorder=7, backgroundcolor= 'white',fontdict=dict(weight='bold',size=16),transform = ax2.transAxes)

    if interactive:
        raise InputError(f"An interactive mode is not available yet, feel free to right a Gui. Here you can plot the curves")
    else:
        plt.savefig(name,dpi=300)
    #return red_Chi_2



plot_curves.__doc__ =f'''
 NAME:
    plot_curves

 PURPOSE:
    Plot the current curves, best run right after pl
 CATEGORY:
    rotmass

 INPUTS:

 OPTIONAL INPUTS:

 OUTPUTS:

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''

def rotmass_main(radii, derived_RCs, total_RC,total_RC_err,no_negative =True,out_dir = None,\
                interactive = False,rotmass_settings = None,log_directory=None,\
                rotmass_parameter_settings = None,\
                results_file = 'Final_Results',log=None,debug = False, font = 'Times New Roman'):

    #Dictionary for translate RC and mass parameter for the baryonic disks
    disk_var = create_disk_var(derived_RCs) 
    
    # First combine all parameters that need to be included in the total fit in a single dictionary
    function_variable_settings = set_fitting_parameters(rotmass_settings,rotmass_parameter_settings,disk_var)
   
    # Then we will group the baryonic  rotation curves that are define,the error of the total_RC is added.
    # When introducing errors on the RCs of the curves we need to do that here.
    radii ,total_RC,Baryonic_RC= get_baryonic_RC(radii,total_RC,derived_RCs,\
                                            V_total_error=total_RC_err,log=log)
   

    if debug:
        print_log(f'We will use the following initial RCs for the mass decomposition. \n',log)
        print_log(f'''Radi = {radii}
total RC = {total_RC[0]}, error = {total_RC[1]}. \n''',log)
        for key in Baryonic_RC:
            print_log(f'{key} = {Baryonic_RC[key]} \n',log)
    # Then we check for which parameters we actaully have an RC !!!!!This should always be the same
    for component in disk_var:
        if component in function_variable_settings and component not in Baryonic_RC:
            raise RuntimeError(f'We have more baryonic compnents than RCs or vice versa')
            #function_variable_settings.pop(component)
    if debug:
        print_log(f'We will use the following initial fitting settings: \n',log)
        for key in function_variable_settings:
            print_log(f'{key} = {function_variable_settings[key]} \n',log)
    
    # Construct the function to be fitted, note that the actual fit_curve is
    DM_curve,full_curve,check_curve = build_curve(function_variable_settings,disk_var,\
                                  rotmass_settings['HALO'],Baryonic_RC,\
                                  debug=debug,log=log)
   
    # make a dictionary with the current RCs
    current_curves = calculate_curves(radii ,total_RC,Baryonic_RC,full_curve,\
                                        DM_curve,function_variable_settings,disk_var)
    if interactive:
        #We want to bring up a GUI to allow for the fitting
        print_log(f'''Unfortunately the interactive function of fitting is not yet implemented. Feel free to write a GUI.
For now you can set, fix and apply boundaries in the yml input file.
for the {rotmass_settings} DM halo the parameters are {','.join([key for key in function_variable_settings if key not in disk_var])}
''',log,screen=True)
        exit()
        #Start GUI with default settings or input yml settings. these are already in function_parameters
    else:

        plot_curves(f'{log_directory}/{results_file}_Input_Curves.pdf', current_curves,\
                    variables=function_variable_settings, halo=rotmass_settings['HALO'],\
                    font= font)

        #Then download settings




    # calculate the initial guesses
    initial_variable_settings = initial_guess(full_curve,radii,total_RC,function_variable_settings,\
                            debug=debug,log=log,negative=rotmass_settings.negative_values)

    current_curves = calculate_curves(radii ,total_RC,Baryonic_RC,full_curve,\
                                        DM_curve,initial_variable_settings,disk_var)
   
    plot_curves(f'{log_directory}/{results_file}_Initial_Guess_Curves.pdf', current_curves,variables= initial_variable_settings,halo=rotmass_settings['HALO'])

    final_variable_fits,emcee_results = mcmc_run(full_curve,radii,total_RC,\
                            initial_variable_settings,function_variable_settings,\
                            out_dir = out_dir, debug=debug,log=log,\
                            negative=rotmass_settings.negative_values,\
                            steps=rotmass_settings.mcmc_steps,
                            results_name= results_file)

    current_curves = calculate_curves(radii ,total_RC,Baryonic_RC,full_curve,\
                                        DM_curve,final_variable_fits,disk_var)
   
    print_log('Plotting and writing',log)
    plot_curves(f'{out_dir}/{results_file}_Final_Curves.pdf', current_curves ,variables=final_variable_fits,halo=rotmass_settings['HALO'])


    red_chisq = calculate_red_chisq(current_curves,final_variable_fits)

    write_output_file(final_variable_fits,emcee_results,output_dir=out_dir,\
                results_file = results_file, red_chisq = red_chisq)


rotmass_main.__doc__ =f'''
 NAME:
    rotmass_main

 PURPOSE:
    The main fitting module for the RCs

 CATEGORY:
    rotmass

 INPUTS:

 OPTIONAL INPUTS:

 OUTPUTS:

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''


def set_fitting_parameters(rotmass_settings,parameters, disk_var):
    # Get the variables of the DM function
    dm_parameters = [str(x) for x in getattr(V, rotmass_settings['HALO'])().free_symbols if str(x) != 'r']
    #define a dictionary
    fitting_parameter = {}
    # then set the DM parameters
   
    for key in dm_parameters:
        fitting_parameter[key] = {'Variables': [key,key], 'Settings': parameters[key]}
        if not bool(fitting_parameter[key]['Settings'][4]):
            raise  InputError(f'''You have requested the {rotmass_settings['HALO']} DM halo but want to exclude {key} from the fitting.
You cannot change the DM formula, if this is your aim please add a potential in the rotmass potential file.
If you merely want to fix the variable, set an initial guess and fix it in the input (e.g. rotmass.{key} = [100, null,null, False,True]). ''')
   
    for key in disk_var:
        if bool(parameters[key][4]):
            fitting_parameter[key] = {'Variables': disk_var[key], 'Settings': parameters[key]}
    return fitting_parameter
set_fitting_parameters.__doc__ =f'''
 NAME:
    set_fitting_parameters

 PURPOSE:
    Create a dictionary where all parameters of the final function are set with their boundaries and initial guesses.

 CATEGORY:
    rotmass

 INPUTS:
    rotmass_settings = the original settings from the yaml including all the defaults.
 OPTIONAL INPUTS:

 OUTPUTS:

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''






#
