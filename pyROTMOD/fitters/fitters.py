# -*- coding: future_fstrings -*-


import numpy as np
import warnings
from pyROTMOD.support.errors import InitialGuessWarning,FailedFitError
import copy
import lmfit
import inspect
import corner
import arviz
import xarray
import sys
from pyROTMOD.support.minor_functions import get_uncounted,\
    get_correct_label,get_exponent,get_output_name
from pyROTMOD.support.log_functions import print_log
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt    
from jax import numpy as jnp
from jax import random
from tinygp import GaussianProcess as tgpGaussianProcess
from tinygp import kernels as tgpkernels
import numpyro
from functools import partial

import pickle
from pyROTMOD.optical.profiles import hernexp





def initial_guess(cfg, total_RC,r_last = False):
    
    negative= cfg.fitting_general.negative_values
    minimizer = cfg.fitting_general.initial_minimizer
    #First initiate the model with the numpy function we want to fit
    ivars = 'r'
    paras = []
    for variable in total_RC.numpy_curve['variables']:
        if (variable not in inspect.signature(total_RC.numpy_curve['function']).parameters
            or variable == 'r'):
            #We don't need guesses for r
            continue
        paras.append(variable) 
    model = lmfit.Model(total_RC.numpy_curve['function'],independent_vars = ivars,
                        param_names= paras)
   
    #no_input = False
   
    guess_variables = copy.deepcopy(total_RC.fitting_variables)
    for variable in guess_variables:
        guess_variables[variable].fill_empty()

     #Test that the models works
  
    for variable in total_RC.numpy_curve['variables']:
        if variable == 'r' or variable not in inspect.signature(total_RC.numpy_curve['function']).parameters:
            #We don't need guesses for r
            continue
        print_log(f'''INITIAL_GUESS: Setting the parameter {variable} with the following values:
    Value: {guess_variables[variable].value}
    Min: {guess_variables[variable].min}
    Max: {guess_variables[variable].max}
    Vary: {guess_variables[variable].variable}
    ''',cfg,case=['debug_add'])
        model.set_param_hint(variable,value=guess_variables[variable].value,\
            min=guess_variables[variable].min,\
            max=guess_variables[variable].max,\
            vary=guess_variables[variable].variable)
                        
           
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
                    if total_RC.fitting_variables[variable].value is None:
                        guess_variables[variable].value = float(np.random.rand()*\
                                (guess_variables[variable].max-guess_variables[variable].min)\
                                +guess_variables[variable].min)
                        
                counter+=1
                if counter > 501.:
                    raise InitialGuessWarning(f'We could not find errors and initial guesses for the function. Try smaller boundaries or set your initial values')
            else:
                print_log(f'The initial guess is a succes. \n',cfg, case=['debug_add']) 
                for variable in guess_variables:
                    # as the GP for the numpyro backend is not in the function we need to check the signature
                    if variable in inspect.signature(total_RC.numpy_curve['function']).parameters:
     
                        buffer = np.max([abs(initial_fit.params[variable].value*0.25)\
                                        ,10.*initial_fit.params[variable].stderr])
                        if cfg.fitting_general.use_gp:
                            buffer = buffer*3.
                        #We modify the limit if it was originally unset else we keep it as was
                        guess_variables[variable].min = float(initial_fit.params[variable].value-buffer \
                                                if (total_RC.fitting_variables[variable].min is None) \
                                                else initial_fit.params[variable].min)
                        if not negative:
                            if guess_variables[variable].min < 0.:
                                guess_variables[variable].min = 0.
                        guess_variables[variable].max = float(initial_fit.params[variable].value+buffer \
                                                if (total_RC.fitting_variables[variable].max is None)\
                                                else initial_fit.params[variable].max)
                        guess_variables[variable].std = float(initial_fit.params[variable].stderr)
                        guess_variables[variable].value = float(initial_fit.params[variable].value)
                no_errors = False
    print_log(f'''INITIAL_GUESS: These are the statistics and values found through {minimizer} fitting of the residual.
{initial_fit.fit_report()}
''',cfg,case=['main'])
    
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
def calculate_steps_burning(cfg,function_variable_settings):
     # the steps should be number free variable times the require steps
    free_variables  = 0.
    for key in function_variable_settings:
        if function_variable_settings[key].variable:
            free_variables +=1 
    
    steps=int(cfg.fitting_general.mcmc_steps*free_variables)
    burns = int(cfg.fitting_general.burn*free_variables)
    return steps, burns

def build_GP(total_RC, fitting_variables, cfg=None, no_log=False, no_gp=False):
  
    yerr = jnp.array(total_RC.errors.value)
    x = jnp.array(total_RC.radii.value)
    params = []
    for par in inspect.signature(total_RC.numpy_curve['function']).parameters:
        if str(par) != 'r':
            params.append(fitting_variables[par])
    if not no_log:
        print_log(f'''The function total_numpy_curve takes the parameters:     
{[par for par in inspect.signature(total_RC.numpy_curve['function']).parameters ]}
This should correspond to,
{params}''',cfg,case=['debug_add'])
    #print(params)
    function_fill = partial(total_RC.numpy_curve['function'], *params) 
  
    if no_gp:
        return function_fill, yerr   
    else:
        kernel = (
            10**fitting_variables['lgamplitude']
            * tgpkernels.ExpSquared(
                fitting_variables['length_scale'],
                distance=tgpkernels.distance.L1Distance()
            )
        )
        return tgpGaussianProcess(kernel, x, diag=yerr**2,mean=function_fill)
        
   

def tiny_gp_model(total_RC, fitting_variables, cfg=None):
    parameters ={}
    for parameter in fitting_variables:
        if fitting_variables[parameter].variable:
            parameters[parameter] = numpyro.sample(
                parameter,
                    numpyro.distributions.Uniform(fitting_variables[parameter].min,
                                                  fitting_variables[parameter].max)
                    )
        else:
            parameters[parameter] = fitting_variables[parameter].value
   
    y = jnp.array(total_RC.values.value)
    x = jnp.linspace(total_RC.radii.value.min(), total_RC.radii.value.max(), 1000)
    gp = build_GP(total_RC, parameters, cfg=cfg, no_log=True)
    #, no_log=True)
    numpyro.sample("y", gp.numpyro_dist(), obs=y)   
    # calculate the predicted V_rot (i.e. the mean function) of the model
    mu = gp.mean_function(x)
    numpyro.deterministic("mu", mu)

def simple_model(total_RC, fitting_variables, cfg=None):
    parameters ={}
    for parameter in fitting_variables:
        if fitting_variables[parameter].variable:
            parameters[parameter] = numpyro.sample(
                parameter,
                    numpyro.distributions.Uniform(fitting_variables[parameter].min,
                                                  fitting_variables[parameter].max)
                    )
        else:
            parameters[parameter] = fitting_variables[parameter].value
   
    y =jnp.array(total_RC.values.value)
    x = jnp.array(total_RC.radii.value)
    x_res = jnp.linspace(total_RC.radii.value.min(), total_RC.radii.value.max(), 1000) 
    function_with_parr,yerr = build_GP(total_RC, parameters, cfg=cfg,no_log=True,no_gp=True)
    numpyro.sample("y", numpyro.distributions.Normal(function_with_parr(x), yerr), obs=y)
        
    # calculate properties of the model
    numpyro.deterministic("mu", function_with_parr(x_res)) 
    
       



def numpyro_run(cfg,total_RC,out_dir = None,optical_profile = False):
    fit_tracking = {'convergence': False,
                    'iterations': -1,
                    'initial_guess': False,
                    'total_rhat': 0.,
                    'prev_rhat': -1,
           
                    'initial_guess': False}
    
                    
                    
    

    negative = cfg.fitting_general.negative_values
   
    numpyro.set_host_device_count(cfg.input.ncpu)
   
    if cfg.fitting_general.numpyro_chains is None:
        chains = cfg.input.ncpu
    else:
        chains = cfg.fitting_general.numpyro_chains
    results_name = get_output_name(cfg,profile_name = total_RC.name,
            function_name= total_RC.numpy_curve['function'].__name__)
  
    #numpyro.set_host_device_count(1)
    rng_key = random.PRNGKey(67)  # Replace 0 with a seed value if needed
    guess_variables = copy.deepcopy(total_RC.fitting_variables)
  
    for variable in guess_variables:
        guess_variables[variable].fill_empty()
    steps,burning = calculate_steps_burning(cfg,guess_variables)
   
   
   
    setbounds = {}
    for variable in guess_variables:   
        setbounds[variable] = [float(guess_variables[variable].min),float(guess_variables[variable].max)]
   
    if cfg.fitting_general.use_gp and not optical_profile:
        mod= tiny_gp_model
    else:
        mod = simple_model   
   
    sampler = numpyro.infer.MCMC(
                numpyro.infer.NUTS(
                    mod,
                    dense_mass=True,
                    step_size = 0.1,
                    target_accept_prob=0.9,
                    find_heuristic_step_size=True,
                    regularize_mass_matrix=False,
                ),
                num_warmup=burning,
                num_samples=steps,
                num_chains= chains, # The more chains the better
                progress_bar=True,
            )
             # Split the PRNG key for the sampler
    rng_key, subkey = random.split(rng_key)
    labels_map = {}
    parameter_names = []
    for parameter in guess_variables:
        if guess_variables[parameter].variable:
            parameter_names.append(parameter)
            strip_parameter,no = get_uncounted(parameter) 
            #edv,correction = get_exponent(np.mean(result_emcee.flatchain[parameter_mc]),threshold=3.)
            labels_map[parameter] = get_correct_label(strip_parameter,no)
    azLabeller = arviz.labels.MapLabeller(var_name_map=labels_map)    
    
    while (not fit_tracking['convergence'] and 
        fit_tracking['iterations'] <= cfg.fitting_general.max_iterations):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
            warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
            warnings.filterwarnings("ignore", message="invalid value encountered in log")
            warnings.filterwarnings("ignore", message="invalid value encountered in divide")
            warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
            warnings.filterwarnings("ignore", message="overflow encountered in power") 
            warnings.filterwarnings("ignore", message="overflow encountered in reduce")
            warnings.filterwarnings("ignore", message="overflow encountered in square")
            warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
            fit_tracking['iterations'] +=1
            sampler.run(subkey,total_RC, guess_variables)            
            sampler.print_summary()
            data = arviz.from_numpyro(sampler,log_likelihood=True)  

            fit_tracking,guess_variables,setbounds = process_numpyro_results(
                    cfg,guess_variables,data,fit_tracking,setbounds,total_RC)
      
          
    if fit_tracking['iterations'] >= cfg.fitting_general.max_iterations:
        print_log(f''' Your boundaries are not converging. Consider fitting less variables or manually fix the boundaries
We stopped the iterations  process and will use the last fit as the best guess output.
''',cfg,case=['main'])
        
     
   
    if out_dir:
        if not cfg.output.chain_data is None:
            with open(f"{out_dir}{results_name}_chain_data.pickle", "wb") as f:
                pickle.dump(data, f)
        
            
        arviz.plot_trace(data, var_names= parameter_names, figsize=(12,9), 
                 labeller = azLabeller, legend =True, compact =False)
        plt.tight_layout()
       
        plt.savefig(f"{out_dir}{results_name}_Numpyro_trace.pdf",dpi=300)
        plt.close()
      
        lab = []
        ranges = []
        for parameter_mc in parameter_names:    
            strip_parameter,no = get_uncounted(parameter_mc) 
            edv,correction = get_exponent(fit_summary[parameter_mc].values
                [available_metrics.index('mean')],threshold=3.)
            #inf_data[parameter_mc] = inf_data[parameter_mc]*correction
            lab.append(get_correct_label(strip_parameter,no,exponent= edv))
            ranges.append((setbounds[parameter_mc][0]
                           ,setbounds[parameter_mc][1]))
          
       
       
        fig = corner.corner(data, bins=40, ranges =ranges, labeller = azLabeller, 
            show_titles=True,title_kwargs={"fontsize": 15},quantiles=[0.16, 0.5, 0.84]
            ,var_names=parameter_names,divergence =True)
        #,labels=lab)
        plt.savefig(f"{out_dir}{results_name}_Numpyro_COV_Fits.pdf",dpi=150)
        plt.close()
    print_log(f''' Numpyro_RUN: We find the following parameters for this fit. \n''',cfg,case=['main'])
    for variable in guess_variables:
        if guess_variables[variable].variable:
            guess_variables[variable].min = float(fit_summary[variable].values
                [available_metrics.index('mean')]-fit_summary[variable].values
                [available_metrics.index('sd')])
            guess_variables[variable].max = float(fit_summary[variable].values
                [available_metrics.index('mean')]+fit_summary[variable].values
                [available_metrics.index('sd')])
            guess_variables[variable].value = float(fit_summary[variable].values
                [available_metrics.index('mean')])
            guess_variables[variable].stddev = float(fit_summary[variable].values
                [available_metrics.index('sd')])
            print_log(f'''{variable} = {guess_variables[variable].value} +/- {guess_variables[variable].stddev} within the boundary {guess_variables[variable].min}-{guess_variables[variable].max}
''',cfg,case=['main'])
    with warnings.catch_warnings():
        warnings.filterwarnings("error")        
        try:
            BIC = arviz.loo(data)
            print_log(f'''The LOO value is {BIC}''',cfg,case=['main'])  
        except UserWarning as e:
            if str(e) == 'Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.':
                warnings.filterwarnings("ignore")  
                BIC = arviz.loo(data)
                print_log(f'''The LOO value ({BIC}) is not reliable''',cfg,case=['main'])
                succes = False
            else:
                raise UserWarning(e)
        except RuntimeWarning as e:
            warnings.filterwarnings("ignore")  
            BIC = arviz.loo(data)
            print_log(f'''The LOO value is {BIC}''',cfg,case=['main'])
            pass
    return guess_variables,BIC,fit_tracking

numpyro_run.__doc__ =f'''
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

def set_statistics(cfg,fit_variables,fit_summary):
    available_metrics = list(fit_summary.metric.values)
    for variable in fit_variables:
        if variable in fit_summary:
            metric_values = fit_summary[variable].values
            collected_stats = {}
            for metric in available_metrics:
                collected_stats[metric] = float(metric_values[available_metrics.index(metric)])
            fit_variables[variable].fit_stats = collected_stats
        else:
            print_log(f'''The parameter {variable} is not in the summary. 
''',cfg,case=['debug_add'])     


def process_parameter_stats(cfg,parameter_names,fit_tracking,fit_variables,total_RC):
    for var_name in parameter_names:
        parameter_stats = fit_variables[var_name].fit_stats
        print_log(f'''The stats for {var_name} are {parameter_stats}.
''',cfg,case=['debug_add'])
        if parameter_stats['r_hat'] > 1.5:
            fit_tracking['rhat_count'] += 1
        fit_tracking['total_rhat'] += parameter_stats['r_hat']
        if abs(parameter_stats['mean']-parameter_stats['median'])/parameter_stats['sd'] > 1.0:        
            fit_tracking['med_mean'] += 1.
def process_accumulated_stats(cfg,fit_tracking,parameter_count):
    fit_tracking['statistics_quality'] = {'average_rhat': True,
                                          'med_mean': True,
                                          'total_rhat': True, 
                                          'grow_rhat': True}
                                          
    if fit_tracking['rhat_count']/parameter_count > 0.5:
        print_log(f'''More than 50% of the parameters have a rhat > 2.  This is not good. 
''',cfg,case=['main'])
        fit_tracking['statistics_quality']['average_hat'] = False
    if fit_tracking['med_mean'] >= 1.:
        print_log(f'''There is parameter where the mean and median differ by more than the std.  This is not good.
''',cfg,case=['main'])
        fit_tracking['statistics_quality']['med_mean'] = False
    
    if fit_tracking['total_rhat']/parameter_count > 3:
        print_log(f'''The average rhat is > 3. This is not good.
''',cfg,case=['main'])
        fit_tracking['statistics_quality']['total_rhat'] = False
    if fit_tracking['total_rhat'] > 2.*fit_tracking['prev_hat']:
        print_log(f'''The total rhat is increasing. This is not good.
''',cfg,case=['main'])
        fit_tracking['statistics_quality']['grow_rhat'] = False     

def process_numpyro_results(cfg,fit_variables, mcmc_result,fit_tracking,setbounds):
    func_dict = stat_func_dict()
    parameter_names = [name for name in fit_variables if fit_variables[name].variable]    
    fit_summary = arviz.summary(mcmc_result, var_names=parameter_names,
        fmt='xarray',round_to= None,stat_funcs=func_dict)
    set_statistics(cfg,fit_variables,fit_summary)
    fit_tracking['prev_hat'] = fit_tracking['total_rhat']
    for reset in ['rhat_count','total_rhat','med_mean','statistics_quality']:
        fit_tracking[reset] = 0
   
    process_parameter_stats(cfg,parameter_names,fit_tracking,fit_variables)                                           
    process_accumulated_stats(cfg,fit_tracking,len(parameter_names))    
    if any([not fit_tracking['statistics_quality'][x] for x in fit_tracking['statistics_quality']]):
        if not fit_tracking['initial_guess']:
            fit_tracking['prev_hat'] = -1   
            print_log(f'''We will run a differential evolution to estimate the parameters.
''',cfg,case=['main'])
            guess_variables = initial_guess(cfg,total_RC)
            fit_tracking['initial_guess'] = True
            for var_name in guess_variables:
                guess_variables[var_name].min = (guess_variables[var_name].value
                    - 3.*guess_variables[var_name].stddev)  
                if not negative and guess_variables[var_name].min < 0.:
                    guess_variables[var_name].min = 0.
                guess_variables[var_name].max = (guess_variables[var_name].value
                    + 3.*guess_variables[var_name].stddev)
                
        else:
            if fit_tracking['statistics_quality']['med_mean']:
                err_message = create_error_message(fit_tracking)
                raise FailedFitError(f'''{err_message}''')
            else:
                use_median =True    
    fit_tracking['convergence'],setbounds = check_boundaries(cfg,guess_variables,
                fit_summary,negative=cfg.fitting_general.negative_values,
                median=use_median,arviz_output=True,prev_bound = setbounds)
    

def lmfit_run(cfg,total_RC,original_settings, out_dir = None,optical_profile = False):
    fit_tracking = {'convergence': False,
                    'iterations': -1,
                    'initial_guess': False}

    function_variable_settings = copy.deepcopy(total_RC.fitting_variables)
    negative = cfg.fitting_general.negative_values
    results_name = get_output_name(cfg,profile_name = total_RC.name)
   
    steps,burning = calculate_steps_burning(cfg,function_variable_settings)
    #First set the model
    model = lmfit.Model(total_RC.numpy_curve['function'])
    #then set the hints
    if cfg.fitting_general.use_gp and not optical_profile:
        results_name = 'GP_' + results_name

    
    added = []
    setbounds = {}
    for variable in function_variable_settings:
            function_variable_settings[variable].fill_empty()
            if variable not in added:
                setbounds[variable] = [float(original_settings[variable].min),
                    float(original_settings[variable].max)]
                model.set_param_hint(variable,value=function_variable_settings[variable].value,\
                            min=function_variable_settings[variable].min,\
                            max=function_variable_settings[variable].max,
                            vary=function_variable_settings[variable].variable)
                added.append(variable)
   
    parameters = model.make_params()
    if total_RC.numpy_curve['function'].__name__ == 'total_numpy_curve':
        workers =1
    else:
        workers = cfg.input.ncpu

    emcee_kws = dict(steps=steps, burn=burning, thin=10, is_weighted=True,\
        workers=workers)
    #,\
    #    workers = cfg.input.ncpu)
    
  
    while (not fit_tracking['convergence'] and 
        fit_tracking['iterations'] <= cfg.fitting_general.max_iterations):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
            warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
            warnings.filterwarnings("ignore", message="invalid value encountered in log")
            warnings.filterwarnings("ignore", message="invalid value encountered in divide")
            warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
            warnings.filterwarnings("ignore", message="overflow encountered in power")
            fit_tracking['iterations'] +=1
            result_emcee = model.fit(data=total_RC.values.value, \
                r=total_RC.radii.value, params=parameters, method='emcee'\
                ,nan_policy='omit',fit_kws=emcee_kws,weights=1./total_RC.errors.value)
            
            fit_tracking['convergence'],setbounds = check_boundaries(cfg, function_variable_settings,
                result_emcee, negative=negative, prev_bound = setbounds)
           
            for variable in function_variable_settings:
                if function_variable_settings[variable].variable:
                    parameters[variable].min = function_variable_settings[variable].min
                    parameters[variable].max = function_variable_settings[variable].max 
                    parameters[variable].value = function_variable_settings[variable].value 
                print_log(f''' {variable} = {parameters[variable].value}  within the boundary {parameters[variable].min}-{parameters[variable].max})
''',cfg,case=['debug_add'])
    print_log(result_emcee.fit_report(),cfg,case=['main'])
    print_log('\n',cfg,case=['main'])
 
    if out_dir:
        if not cfg.output.chain_data is None:
            with open(f"{out_dir}{results_name}_chain_data.pickle", "wb") as f:
                pickle.dump(result_emcee.flatchain, f)
        lab = []
        for parameter_mc in result_emcee.params:
            if result_emcee.params[parameter_mc].vary:
                strip_parameter,no = get_uncounted(parameter_mc) 
                edv,correction = get_exponent(np.mean(result_emcee.flatchain[parameter_mc]),threshold=3.)
                result_emcee.flatchain[parameter_mc] = result_emcee.flatchain[parameter_mc]*correction
                lab.append(get_correct_label(strip_parameter,no,exponent= edv))

        #xdata= xarray.Dataset.from_dataframe(result_emcee.flatchain)
        #ardata = arviz.InferenceData(xdata) 
        fig = corner.corner(result_emcee.flatchain, quantiles=[0.16, 0.5, 0.84],show_titles=True,
                        title_kwargs={"fontsize": 15},labels=lab)
        fig.savefig(f"{out_dir}{results_name}_COV_Fits.pdf",dpi=300)
        plt.close()
    print_log(f''' MCMC_RUN: We find the following parameters for this fit. \n''',cfg,case=['main'])
 
    for variable in function_variable_settings:
        if function_variable_settings[variable].variable:
            function_variable_settings[variable].min = float(result_emcee.params[variable].value-\
                                    result_emcee.params[variable].stderr)
            function_variable_settings[variable].max = float(result_emcee.params[variable].value+\
                                    result_emcee.params[variable].stderr)

            function_variable_settings[variable].value = float(result_emcee.params[variable].value)
            print_log(f'''{variable} = {result_emcee.params[variable].value} +/- {result_emcee.params[variable].stderr} within the boundary {result_emcee.params[variable].min}-{result_emcee.params[variable].max}
''',cfg,case=['main'])
                    
    BIC = result_emcee.bic



    return function_variable_settings,BIC,fit_tracking

lmfit_run.__doc__ =f'''
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

def create_error_message(fit_tracking):
    err_message = 'The statistics are off and getting an initial gues does not help. \n'
    if not fit_tracking['statistics_quality']['total_rhat']:
        err_message += f'The total rhat ({fit_tracking["total_rhat"]}) is on average > 3. \n'
    if not fit_tracking['statistics_quality']['average_hat']:
        err_message += 'More than half of the parameters have a rhat > 1.5 \n'
    if not fit_tracking['statistics_quality']['grow_rhat']:
        err_message += f'The total rhat ({fit_tracking["total_rhat"]}) is increasing ({fit_tracking["prev_hat"]}). \n'
    return err_message

def update_parameter_values(output,var_name,parameter,arviz_output=False,median=False):
    if arviz_output:
        stats = parameter.fit_stats
       
        if median:
            parameter.value = float(stats['median'])
            parameter.stddev = float(stats['mad'])
        else:
            parameter.value = float(stats['mean'])
            parameter.stddev = float(stats['sd'])
        if not parameter.fixed_boundaries:
            parameter.min = float(stats['low_3%'])
            parameter.max = float(stats['high_97%'])
    else:
        if not parameter.fixed_boundaries:
            parameter.min = output.params[var_name].min
            parameter.max = output.params[var_name].max
        parameter.stddev = output.params[var_name].stderr
        parameter.value = float(output.params[var_name].value)

def check_boundaries(cfg,function_variable_settings,output,count=0.,arviz_output=False,
        prev_bound= None,negative=False,median=False):
    succes=True
    req_fraction =0.25 #Arrays should be within 25% of each other
    if prev_bound is None:
        prev_bound = {}
        for parameter in function_variable_settings:
            prev_bound[parameter] = [function_variable_settings[parameter].min, 
                                     function_variable_settings[parameter].max]
    bounds_out = {}        
    # we have to check that our results are only limited by the boundaries in the user has set them
    for var_name in function_variable_settings:
        if function_variable_settings[var_name].variable:
            current_parameter =  copy.deepcopy(function_variable_settings[var_name])
            update_parameter_values(output,var_name, current_parameter
                        ,arviz_output=arviz_output,median=median)
            new_bounds = [float(current_parameter.min),
                          float(current_parameter.max)]
            
            change = abs(3.*current_parameter.stddev)
            min_bounds = [current_parameter.value-change,
                          current_parameter.value+change]
            if not negative and min_bounds[0] < 0. and var_name[0:2] != 'lg':
                    min_bounds[0] = 0.
          
            change = abs(5.*current_parameter.stddev)
            if change < abs(0.2*current_parameter.stddev):
                change = abs(0.2*current_parameter.stddev) 
                             
            lower_bound = current_parameter.value - change
            upper_bound = current_parameter.value + change
            if current_parameter.fixed_boundaries:
                print_log(f'''''The boundaries for {var_name} are fixed check that they are reasonable.                            
''',cfg,case=['main'])
                if prev_bound[var_name][0] < lower_bound:
                    print_log(f'''The lower bound ({prev_bound[var_name][0]}) for {var_name} deviates more than 5. * std (std = {current_parameter.stddev}).
consider changing it''',cfg,case=['main'])
                if prev_bound[var_name][1] > upper_bound:
                    print_log(f'''The upper bound ({prev_bound[var_name][1]}) for {var_name} deviates more than 5. * std (std = {current_parameter.stddev}).
consider changing it''',cfg,case=['main'])
                
                bounds_out[var_name] = prev_bound[var_name]
                    
                #function_variable_settings[var_name].min = prev_bound[var_name][0]
                #function_variable_settings[var_name].max = prev_bound[var_name][1]     
                #function_variable_settings[var_name].value = current_parameter.value
                #function_variable_settings[var_name].stddev = current_parameter.stddev
                continue

            #lower_bound = current_parameter.value - 5.*current_parameter.stddev
            #upper_bound = current_parameter.value + 5.*current_parameter.stddev         
         
            print_log(f'''{var_name} = {current_parameter.value} +/- {current_parameter.stddev} 
change = {change} lowerbound = {lower_bound}    upperbound = {upper_bound} bounds = {new_bounds} 
minbounds = {min_bounds} prev_bound = {prev_bound[var_name]}
''',cfg,case=['debug_add'])
            

             # We expect that the boundaries are symmetrical around the value
            min_distance = abs(current_parameter.value - new_bounds[0])
            max_distance = abs(current_parameter.value - new_bounds[1])

            if change > min_distance:
                new_bounds[0] = float(current_parameter.value - np.max([change,max_distance]))
                if not negative and new_bounds[0] < 0. and var_name[0:2] != 'lg':
                    new_bounds[0] = 0.
           
            if change > max_distance:
                new_bounds[1] = float(current_parameter.value + np.max([change,min_distance]))
      
            if new_bounds[0] > min_bounds[0]:
                new_bounds[0] = min_bounds[0]
            if new_bounds[1] < min_bounds[1]:
                new_bounds[1] = min_bounds[1]
            if count > 0.:
                if new_bounds[0] > prev_bound[var_name][0]:
                    new_bounds[0] = prev_bound[var_name][0]
                if new_bounds[1] < prev_bound[var_name][1]:
                    new_bounds[1] = prev_bound[var_name][1]    
            if np.allclose(np.array(prev_bound[var_name])/current_parameter.value, 
                    np.array(new_bounds)/current_parameter.value,rtol=req_fraction):
                print_log(f'''{var_name} is fitted wel in the boundaries {new_bounds[0]} - {new_bounds[1]}. (Old is {prev_bound[var_name][0]} - {prev_bound[var_name][1]} )
Compared array {np.array(prev_bound[var_name])/current_parameter.value} to {np.array(new_bounds)/current_parameter.value} with a tolerance of {req_fraction}
''',    cfg,case=['main'])
                function_variable_settings[var_name].min = prev_bound[var_name][0]
                function_variable_settings[var_name].max = prev_bound[var_name][1]    
            else:
                print_log(f''' The boundaries for {var_name} are deviating more that {int(req_fraction*100.)}% from those set by 5*std (std = {current_parameter.stddev}) change.
Setting {var_name} = {current_parameter.value} between {new_bounds[0]}-{new_bounds[1]} (old ={prev_bound[var_name][0]}-{prev_bound[var_name][1]})
Compared array {np.array(prev_bound[var_name])/current_parameter.value} to {np.array(new_bounds)/current_parameter.value} with a tolerance of {req_fraction}
''',cfg,case=['main'])
                succes = False
               
                function_variable_settings[var_name].min = new_bounds[0]
                function_variable_settings[var_name].max = new_bounds[1]     
            function_variable_settings[var_name].value = current_parameter.value
            function_variable_settings[var_name].stddev = current_parameter.stddev
            bounds_out[var_name] = new_bounds
            del current_parameter
        

    #no_success=True
    return succes,bounds_out           
            
def stat_func_dict():
    '''This function returns a dictionary with the functions that are used to calculate the statistics
    for the arviz summary. The default is mean and sd.'''
    fdict = {'mean':np.mean,
             'sd':np.std,
             'median':np.median,
             'mad': lambda x: np.median(np.abs(x - np.median(x))),
             'low_3%': lambda x: np.percentile(x, 3),
             'high_97%': lambda x: np.percentile(x, 97),
             'r_hat': arviz.rhat
             }
    return fdict