# -*- coding: future_fstrings -*-


import numpy as np
import warnings
from pyROTMOD.support.errors import InitialGuessWarning
import copy
import lmfit
import corner
from pyROTMOD.support.minor_functions import get_uncounted,\
    get_correct_label,get_exponent
from pyROTMOD.support.log_functions import print_log
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt




def set_initial_guesses(input_settings,cfg = None):
    variables = copy.deepcopy(input_settings)
   
    for variable in variables:
        print(variable)
        #parameter,no = get_uncounted(guess_variables[variable]['Variables'][0])
        if variables[variable][1] is None:
            if variables[variable][0] is not None:
                variables[variable][1] = variables[variable][0]/10.
            else:
                variables[variable][1] = 0.1
        if variables[variable][2] is None:
            if variables[variable][0] is not None:
                variables[variable][2] = variables[variable][0]*20.
            else:
                variables[variable][2] = 1000.
        if variables[variable][0] is None:
                #no_input = True
            variables[variable][0] = float(np.random.rand()*\
                (variables[variable][2]-variables[variable][1])\
                    +variables[variable][1])
        if variables[variable][3]:
            print_log(f'''Setting {variable} with value {variables[variable][0]} and fitting = {variables[variable][3]}.
limits are between {variables[variable][1]} - {variables[variable][2]}
''',cfg, case=['debug_add'])
    return variables

def initial_guess(total_RC, cfg=None, negative = False,\
                  minimizer = 'differential_evolution'):
    #First initiate the model with the numpy function we want to fit
    model = lmfit.Model(total_RC.numpy_curve['function'])
    #no_input = False
   
    guess_variables = set_initial_guesses(total_RC.fitting_variables,cfg=cfg)

    for variable in total_RC.numpy_curve['variables']:
        if variable == 'r':
            #We don't need guesses for r
            continue
        model.set_param_hint(variable,value=guess_variables[variable][0],\
            min=guess_variables[variable][1],\
            max=guess_variables[variable][2],\
            vary=guess_variables[variable][3]
                        )
           
    parameters= model.make_params()
    no_errors = True
    counter =0
    while no_errors:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
            warnings.filterwarnings("ignore", message="invalid value encountered in divide")

            initial_fit = model.fit(data=total_RC.values.value, \
                params=parameters, r=total_RC.radii.value, method= minimizer\
                ,nan_policy='omit',scale_covar=False)
            if not initial_fit.errorbars or not initial_fit.success:
                print(f"\r The initial guess did not produce errors, retrying with new guesses: {counter/float(500.)*100.:.1f} % of maximum attempts.",\
                    end =" ",flush = True) 
                for variable in guess_variables:
                    if total_RC.fitting_variables[variable][0] is None:
                        guess_variables[variable][0] = float(np.random.rand()*\
                                (guess_variables[variable][2]-guess_variables[variable][1])\
                                +guess_variables[variable][1])
                        
                counter+=1
                if counter > 501.:
                    raise InitialGuessWarning(f'We could not find errors and initial guesses for the function. Try smaller boundaries or set your initial values')
            else:
                print_log(f'The initial guess is a succes. \n',cfg, case=['debug_add']) 
                for variable in guess_variables:
                    buffer = np.max([abs(initial_fit.params[variable].value*0.25)\
                                     ,10.*initial_fit.params[variable].stderr])
                    #We modify the limit if it was originally unset else we keep it as was
                    guess_variables[variable][1] = float(initial_fit.params[variable].value-buffer \
                                            if (total_RC.fitting_variables[variable][1] is None) \
                                            else initial_fit.params[variable].min)
                    if not negative:
                        if guess_variables[variable][1] < 0.:
                            guess_variables[variable][1] = 0.
                    guess_variables[variable][2] = float(initial_fit.params[variable].value+buffer \
                                             if (total_RC.fitting_variables[variable][2] is None)\
                                             else initial_fit.params[variable].max)

                    guess_variables[variable][0] = float(initial_fit.params[variable].value)
                no_errors = False
    print_log(f'''INITIAL_GUESS: These are the statistics and values found through {minimizer} fitting of the residual.
{initial_fit.fit_report()}
''',cfg,case=['main'])
    return guess_variables,copy.deepcopy(total_RC.fitting_variables)
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
def calculate_steps_burning(steps,function_variable_settings):
     # the steps should be number free variable times the require steps
    free_variables  = 0.
    for key in function_variable_settings:
        if function_variable_settings[key][3]:
            free_variables +=1 
    
    steps=int(steps*free_variables)
    return steps, int(steps/4.)

def mcmc_run(total_RC,original_variable_settings,out_dir = None,\
        negative=False,cfg=None,steps=2000.,\
        results_name = 'MCMC'):
    function_variable_settings = copy.deepcopy(total_RC.fitting_variables)
    steps,burning = calculate_steps_burning(steps,function_variable_settings)
    #First set the model
    model = lmfit.Model(total_RC.numpy_curve['function'])
    #then set the hints
    

    fixed_boundaries = {}
    added = []
    for variable in function_variable_settings:
            parameter = variable
            if parameter not in added:
                if original_variable_settings[variable][1] is not None or \
                    original_variable_settings[variable][2] is not None:
                        fixed_boundaries[parameter] = True
                else:
                    fixed_boundaries[parameter] = False
                if function_variable_settings[variable][1] == function_variable_settings[variable][2]:
                    function_variable_settings[variable][1] = function_variable_settings[variable][1]*0.9
                    function_variable_settings[variable][2] = function_variable_settings[variable][2]*1.1
                if function_variable_settings[variable][3]:
                    print_log(f'''Setting {parameter} with value {function_variable_settings[variable][0]}.
        With the boundaries between {function_variable_settings[variable][1]} - {function_variable_settings[variable][2]}
        ''',cfg, case=['main'])
                else:
                    print_log(f'''Keeping {parameter} fixed at {function_variable_settings[variable][0]}.
        ''',cfg, case=['main'])
                model.set_param_hint(parameter,value=function_variable_settings[variable][0],\
                            min=function_variable_settings[variable][1],\
                            max=function_variable_settings[variable][2],
                            vary=function_variable_settings[variable][3])
                added.append(parameter)
    parameters = model.make_params()
    emcee_kws = dict(steps=steps, burn=burning, thin=10, is_weighted=True)
    no_succes =True
    count = 0
    while no_succes:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
            warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
            warnings.filterwarnings("ignore", message="invalid value encountered in log")
            warnings.filterwarnings("ignore", message="invalid value encountered in divide")
            result_emcee = model.fit(data=total_RC.values.value, r=total_RC.radii.value, params=parameters, method='emcee',nan_policy='omit',
                             fit_kws=emcee_kws,weights=1./total_RC.errors.value)
            no_succes=False

            triggered =False
            # we have to check that our results are only limited by the boundaries in the user has set them
            added = []
            for variable in original_variable_settings:
                parameter = variable
                if parameter not in added:
                    if function_variable_settings[variable][3]:
                        prev_bound = [parameters[parameter].min,parameters[parameter].max]
                        change = np.nanmax([10.*result_emcee.params[parameter].stderr,\
                                                0.25*result_emcee.params[parameter].value])    

                        mini = result_emcee.params[parameter].value-3.*result_emcee.params[parameter].stderr
                        if not negative and mini < 0.:
                            mini = 0.
                        if mini < result_emcee.params[parameter].min:
                            triggered =True
                            
                            parameters[parameter].min = result_emcee.params[parameter].value -\
                                change  
                            if not negative and parameters[parameter].min < 0.:
                                    parameters[parameter].min = 0.
                            if original_variable_settings[variable][1]:
                                if original_variable_settings[variable][1] > parameters[parameter].min:
                                    parameters[parameter].min=original_variable_settings[variable][1]

                        triggered = False
                        if result_emcee.params[parameter].value+\
                            3.*result_emcee.params[parameter].stderr >\
                            result_emcee.params[parameter].max:
                            triggered =True
                            parameters[parameter].max = result_emcee.params[parameter].value\
                                +change
                            if original_variable_settings[variable][2]:
                                if original_variable_settings[variable][2] < parameters[parameter].max:
                                    parameters[parameter].max=original_variable_settings[variable][2]

                        if np.array_equal(prev_bound, [parameters[parameter].min,parameters[parameter].max]) or fixed_boundaries[parameter] :
                            if triggered:
                                if fixed_boundaries[parameter]:
                                    print_log(f''' The boundaries for {parameter} were too small, consider changing them
    ''',cfg,case=['main'])
                                else:
                                    print_log(f''' The boundaries for {parameter} were too small, we sampled an incorrect area
    Changing the parameter and its boundaries and trying again.
    ''',cfg,case=['main'])
                                    parameters[parameter].value = result_emcee.params[parameter].value
                                    print_log(f''' Setting {parameter} = {parameters[parameter].value} between {parameters[parameter].min}-{parameters[parameter].max}
    ''',    cfg,case=['main'])
                                    no_succes =True   
                            else:
                                print_log(f'''{parameter} is fitted wel in the boundaries {parameters[parameter].min} - {parameters[parameter].max}.
    ''',    cfg,case=['main'])
                        else:
                            count += 1
                            if count >= 10 :
                                print_log(f''' Your boundaries are not converging. Condidering fitting less variables or manually fix the boundaries
    ''',cfg,case=['main'])
                            else:
                                print_log(f''' The boundaries for {parameter} were too small, we sampled an incorrect area
    Changing the parameter and its boundaries and trying again.
    ''',cfg,case=['main'])
                                parameters[parameter].value = result_emcee.params[parameter].value
                                print_log(f''' Setting {parameter} = {parameters[parameter].value} between {parameters[parameter].min}-{parameters[parameter].max}
    ''',    cfg,case=['main'])
                                no_succes =True
                    added.append(parameter)

    print_log(result_emcee.fit_report(),cfg,case=['main'])
    print_log('\n',cfg,case=['main'])

    if out_dir:
        lab = []
        added  = []
        for parameter_mc in result_emcee.params:
            if result_emcee.params[parameter_mc].vary:
                for x in function_variable_settings:
                    parameter = x
                    if parameter_mc == parameter and parameter not in added:
                        strip_parameter,no = get_uncounted(parameter) 
                        edv,correction = get_exponent(np.mean(result_emcee.flatchain[parameter_mc]),threshold=3.)
                        result_emcee.flatchain[parameter_mc] = result_emcee.flatchain[parameter_mc]*correction
                        lab.append(get_correct_label(strip_parameter,no,exponent= edv))
                        added.append(parameter)  
                              
      
        fig = corner.corner(result_emcee.flatchain, quantiles=[0.16, 0.5, 0.84],show_titles=True,
                        title_kwargs={"fontsize": 15},labels=lab)
        fig.savefig(f"{out_dir}{results_name}_COV_Fits.pdf",dpi=300)
        plt.close()
    print_log(f''' MCMC_RUN: We find the following parameters for this fit. \n''',cfg,case=['main'])
    added= []
    for variable in function_variable_settings:
        parameter = variable
        if parameter not in added:
            if function_variable_settings[variable][3]:
                function_variable_settings[variable][1] = float(result_emcee.params[parameter].value-\
                                        result_emcee.params[parameter].stderr)
                function_variable_settings[variable][2] = float(result_emcee.params[parameter].value+\
                                        result_emcee.params[parameter].stderr)

                function_variable_settings[variable][0] = float(result_emcee.params[parameter].value)
                print_log(f'''{parameter} = {result_emcee.params[parameter].value} +/- {result_emcee.params[parameter].stderr} within the boundary {result_emcee.params[parameter].min}-{result_emcee.params[parameter].max}
''',cfg,case=['main'])
                added.append(parameter)                




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






