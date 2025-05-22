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
    get_correct_label,get_exponent,get_output_name,setup_fig
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





def initial_guess(cfg, total_RC,chain_data= False):
    
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
        if variable[0:2] == 'lg':
            #the diffirential does not like larget ranges of boundaries
            
            diff = guess_variables[variable].boundaries[1]-\
                guess_variables[variable].boundaries[0]
            if diff > 8.:
                change = (diff-8.)/2.
                guess_variables[variable].boundaries[1] -= change
                guess_variables[variable].boundaries[0] += change
            '''
            for i in [0,1]:
                if abs(guess_variables[variable].boundaries[i]) > 10.:
                    guess_variables[variable].boundaries[i] =\
                          4.*guess_variables[variable].boundaries[i]\
                            /abs(guess_variables[variable].boundaries[i])+5*i
            '''
        print_log(f'''INITIAL_GUESS: Setting the parameter {variable} with the following values:
    Value: {guess_variables[variable].value}
    Min, Max: {','.join([f'{x}' for x in guess_variables[variable].boundaries])}
    Vary: {guess_variables[variable].variable}
    ''',cfg,case=['debug_add'])
       
        model.set_param_hint(variable,value=guess_variables[variable].value,\
            min=guess_variables[variable].boundaries[0],\
            max=guess_variables[variable].boundaries[1],\
            vary=guess_variables[variable].variable)
                        
           
    parameters= model.make_params()
    no_errors = True
    counter =0
    print_log(f'''We are trying to fit {total_RC.numpy_curve["function"].__name__} to the data.
''',cfg,case=['main','screen'])
    while no_errors:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            '''
            warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
            warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
            warnings.filterwarnings("ignore", message="invalid value encountered in log")
            warnings.filterwarnings("ignore", message="invalid value encountered in divide")
            warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
            warnings.filterwarnings("ignore", message="overflow encountered in power") 
            warnings.filterwarnings("ignore", message="overflow encountered in reduce")
            warnings.filterwarnings("ignore", message="overflow encountered in square")
            warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
            warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
            warnings.filterwarnings("ignore", message="invalid value encountered in divide")
            warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
            warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")
            '''

            initial_fit = model.fit(data=total_RC.values.value, \
                params=parameters, r=total_RC.radii.value, method= minimizer\
                ,nan_policy='omit',scale_covar=False)
            if not initial_fit.errorbars or not initial_fit.success:
                print(f"\r The initial guess did not produce errors, retrying with new guesses: {counter/float(500.)*100.:.1f} % of maximum attempts.",\
                    end =" ",flush = True) 
                
                for variable in guess_variables:
                    if total_RC.fitting_variables[variable].value is None:
                        guess_variables[variable].value = float(np.random.rand()*
                                (guess_variables[variable].boundaries[1]
                                -guess_variables[variable].boundaries[0])
                                +guess_variables[variable].boundaries[0])
                        
                counter+=1
                if counter >= 501.:
                    raise InitialGuessWarning(f'''We could not find errors and initial guesses for the function. 
Try smaller boundaries or set your initial values or use a different minimizer''')
            else:
                print(f"\n")
                print_log(f'The initial guess is a succes. \n',cfg, case=['debug_add']) 
                for variable in guess_variables:
                    # as the GP for the numpyro backend is not in the function we need to check the signature
                    if variable in inspect.signature(total_RC.numpy_curve['function']).parameters:
     
                        buffer = np.max([abs(initial_fit.params[variable].value*0.25)\
                                        ,10.*initial_fit.params[variable].stderr])
                        if cfg.fitting_general.use_gp:
                            buffer = buffer*2.
                        #We modify the limit if it was originally unset else we keep it as was
                        for i in [0,1]:
                            guess_variables[variable].previous_boundaries[i] =\
                                copy.deepcopy(guess_variables[variable].boundaries[i]) 
                        if not all(guess_variables[variable].fixed_boundaries):
                            guess_variables[variable].previous_boundaries =\
                                copy.deepcopy(guess_variables[variable].boundaries)
                        if not guess_variables[variable].fixed_boundaries[0]:
                            guess_variables[variable].boundaries[0] = float(
                                initial_fit.params[variable].value-buffer)                
                            if not negative and not guess_variables[variable].log:
                                if guess_variables[variable].boundaries[0] < 0.:
                                    guess_variables[variable].boundaries[0] = 0.
                        if not guess_variables[variable].fixed_boundaries[1]:
                            guess_variables[variable].boundaries[1] = float(
                                initial_fit.params[variable].value+buffer) 
                        '''    
                        for i in [0,1]:
                            if guess_variables[variable].original_boundaries[i] is None:
                                guess_variables[variable].original_boundaries[i] =\
                                    copy.deepcopy(guess_variables[variable].boundaries[i]) 
                        '''
                        guess_variables[variable].previous_value =  copy.deepcopy(guess_variables[variable].value)            
                        guess_variables[variable].std = float(initial_fit.params[variable].stderr)
                        guess_variables[variable].value = float(initial_fit.params[variable].value)
                        guess_variables[variable].fit_direction = ['initial_guess','initial_guess']
                no_errors = False
    print_log(f'''INITIAL_GUESS: These are the statistics and values found through {minimizer} fitting of the residual.
{initial_fit.fit_report()}
''',cfg,case=['main'])
    if chain_data:
        return guess_variables,initial_fit.flatchain
    else:
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
        return tgpGaussianProcess(kernel, x, diag=yerr**2,mean=function_fill),yerr
        
   
def set_parameters_for_model(fitting_variables):
    parameters ={}
    for parameter in fitting_variables:
        if fitting_variables[parameter].variable:
            parameters[parameter] = numpyro.sample(
                parameter,
                numpyro.distributions.Uniform(
                    fitting_variables[parameter].boundaries[0],
                    fitting_variables[parameter].boundaries[1])
                )
        else:
            parameters[parameter] = fitting_variables[parameter].value
    return parameters

def set_model(total_RC, fitting_variables, cfg=None, simple=True):
    parameters = set_parameters_for_model(fitting_variables)
    y = jnp.array(total_RC.values.value)
    x = jnp.linspace(total_RC.radii.value.min(), total_RC.radii.value.max(), 1000)
    gp_function,yerr = build_GP(total_RC, parameters, cfg=cfg,
        no_log=True,no_gp=simple)
    if simple:
        x_res = jnp.array(total_RC.radii.value) 
        numpyro.sample("y", numpyro.distributions.Normal(gp_function(x_res), yerr), obs=y)
        # calculate properties of the model
        numpyro.deterministic("mu", gp_function(x)) 

    else:
        numpyro.sample("y", gp_function.numpyro_dist(), obs=y)   
        # calculate the predicted V_rot (i.e. the mean function) of the model
        mu = gp_function.mean_function(x)
        numpyro.deterministic("mu", mu)

     


def set_tracking():
    fit_tracking = {'value_convergence': False, # Wther all parameter values are stable from one fit to the next
                    'boundary_convergence': False, #Wehther the boundaries are stable from one to the next.
                    'reliable': True,
                    'iterations': -1,
                    'total_rhat': 0.,
                    'non_gaussian': 0.,
                    'prev_rhat': -1,
                    'failure_count': 0,
                    'statistics_quality': reset_statistics_quality() 
        }
    return fit_tracking

def reset_statistics_quality():
    return {'low_ind_rhat': True, #Traces that the individual rhat is < 1.5 for more than half of the parameter
            'similar_stat': True, #Ensures that the mean and median are similar
            'low_average_rhat': True, #Traces the avreage rhat is < 3
            'stable_rhat': True} #Traces that the rhat is not doubling

def numpyro_run(cfg,total_RC,out_dir = None,optical_profile = False):
    fit_tracking = set_tracking()
    numpyro.set_host_device_count(cfg.input.ncpu)
    if cfg.fitting_general.numpyro_chains is None:
        chains = cfg.input.ncpu
    else:
        chains = cfg.fitting_general.numpyro_chains
    results_name = get_output_name(cfg,profile_name = total_RC.name,
            function_name= total_RC.numpy_curve['function'].__name__)

    rng_key = random.PRNGKey(67)  # Replace 0 with a seed value if needed
    guess_variables = copy.deepcopy(total_RC.fitting_variables)
    steps,burning = calculate_steps_burning(cfg,guess_variables)
   
    #setbounds = {}
    #for variable in guess_variables:   
    #    setbounds[variable] = [float(guess_variables[variable].min),float(guess_variables[variable].max)]
    if cfg.fitting_general.use_gp and not optical_profile:
        mod= partial(set_model,simple=False)
    else:
        mod = partial(set_model,simple=True)

    if cfg.fitting_general.log_parameters:
        step_size = 0.1
    else:
        step_size = 0.1
    sampler = numpyro.infer.MCMC(
                numpyro.infer.NUTS(
                    mod,
                    dense_mass=True,
                    step_size = step_size,
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
    
    while ((not fit_tracking['value_convergence'] or
        not fit_tracking['boundary_convergence']) and 
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
            print_log(f'''--------Numpyro_RUN: We are running the {fit_tracking['iterations']} iteration of the fitting process.--------
''',cfg,case=['main'])
            sampler.run(subkey,total_RC, guess_variables)            
            sampler.print_summary()

            data = arviz.from_numpyro(sampler,log_likelihood=True,num_chains =int(chains))  
            fit_tracking,guess_variables = process_results(
                    cfg,guess_variables,data,fit_tracking)
           
          
    if fit_tracking['iterations'] >= cfg.fitting_general.max_iterations:
        print_log(f''' Your boundaries are not converging. Consider fitting less variables or manually fix the boundaries
We stopped the iterations  process and will use the last fit as the best guess output.
''',cfg,case=['main'])
        
     
   
    if out_dir:
        #fit_summary = arviz.summary(data, var_names=parameter_names,
        #fmt='xarray',round_to= None,stat_funcs=stat_func_dict())
        if not cfg.output.chain_data is None:
            with open(f"{out_dir}{results_name}_chain_data.pickle", "wb") as f:
                pickle.dump(data, f)
        
            
        arviz.plot_trace(data, var_names= parameter_names, figsize=(12,9), 
                 labeller = azLabeller, legend =True, compact =False)
        plt.tight_layout()
       
        plt.savefig(f"{out_dir}{results_name}_Numpyro_trace.pdf",dpi=300)
        plt.close()
        create_corner_plot(cfg,data,guess_variables,parameter_names,
            results_plot_name=f"{out_dir}{results_name}_Numpyro_COV_Fits.pdf")
       
    print_log(f''' Numpyro_RUN: We find the following parameters for this fit. \n''',cfg,case=['main'])
    for variable in guess_variables:
        if guess_variables[variable].variable:
            print_log(f'''{variable} = {guess_variables[variable].value} +/- {guess_variables[variable].stddev} within the boundary {'-'.join([f'{x}' for x in guess_variables[variable].previous_boundaries])}
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
                fit_tracking['reliable'] = False
            else:
                raise UserWarning(e)
        except RuntimeWarning as e:
            warnings.filterwarnings("ignore")  
            BIC = arviz.loo(data)
            print_log(f'''The LOO value is {BIC}''',cfg,case=['main'])
            pass
    tmp= f'{BIC}'
    BIC = ':\n# '
    for char in tmp:
        new_character = f"|{char}|"
        if "\n" in new_character:
            BIC += "\n# "
        else:
            BIC += char
           
    BIC.replace(r"\n",r"\n# ")    
    return guess_variables,BIC,fit_tracking,data

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

def create_corner_plot(cfg,data,guess_variables,parameter_names,
        results_plot_name = 'COV_plot.pdf'):

        ndim = len(parameter_names)
     #lab = []
        ranges = []
        levels = []
        for parameter_mc in parameter_names:    
            #strip_parameter,no = get_uncounted(parameter_mc) 
            #edv,correction = get_exponent(guess_variables[parameter_mc].value
            #    ,threshold=3.)
            #inf_data[parameter_mc] = inf_data[parameter_mc]*correction
            #lab.append(get_correct_label(strip_parameter,no,exponent= edv))
            ranges.append((guess_variables[parameter_mc].previous_boundaries[0],
                           guess_variables[parameter_mc].previous_boundaries[1]))
            levels.append(guess_variables[parameter_mc].stddev)
       
        fig = setup_fig(figsize=(3*ndim,3*ndim),size_factor=1.5)
     
        fig = corner.corner(data, bins=40, ranges =ranges,
            levels= [0.393,0.864],fig=fig ,       
            show_titles=False,title_kwargs={"fontsize": 15},
            var_names=parameter_names,divergence =True)
      
        axes = np.array(fig.axes).reshape((ndim, ndim))
        for i in range(ndim):
            edv,correction = get_exponent(guess_variables[parameter_names[i]].value,threshold=3.)
            parameter_label = get_correct_label(*get_uncounted(parameter_names[i]))
            for y in range(i+1):
                edvy,correctiony = get_exponent(guess_variables[parameter_names[y]].value,threshold=3.)
                parameter_label_y = get_correct_label(*get_uncounted(parameter_names[y]))
                ax_hist = axes[i, y]
                if y == i:
                 
                    ax_hist.axvline(guess_variables[parameter_names[i]].value, color='b', linestyle='-')
                    ax_hist.axvline(guess_variables[parameter_names[i]].value+\
                                    guess_variables[parameter_names[i]].stddev, color='cyan', linestyle='--')
                    ax_hist.axvline(guess_variables[parameter_names[i]].value-\
                                    guess_variables[parameter_names[i]].stddev, color='cyan', linestyle='--')
                    title_string = f'{parameter_label} = {guess_variables[parameter_names[i]].value/correction:.2f} $\\pm$ {guess_variables[parameter_names[i]].stddev/correction:.2f}'
                    if edv != 0.:
                        title_string += f'$\\times 10^{{{edv}}} '
                    ax_hist.set_title(title_string,fontsize=15.,loc='left',
                        verticalalignment='top')
                else:
                   
                    ax_hist.axvline(guess_variables[parameter_names[y]].value,
                        color="b")
                    ax_hist.axhline(guess_variables[parameter_names[i]].value,
                        color="b")
                    ax_hist.plot(guess_variables[parameter_names[y]].value,
                        guess_variables[parameter_names[i]].value,'sb')
                if i == ndim-1:
                    ax_hist.set_xlabel(parameter_label_y,fontsize=15.,labelpad=0.5)
                if y == 0 and i > 0:
                    ax_hist.set_ylabel(parameter_label,fontsize=15.,labelpad=0.5)
                #ax_hist.set_xlabel(f'y = {y} i= {i}',fontsize=15.)
                #ax_hist.set_ylabel(f'y = {y} i= {i}',fontsize=15.)

        fig.tight_layout()
        plt.savefig(results_plot_name,dpi=150)
        plt.close()
       

def set_parameter_statistics(cfg,fit_variables,fit_summary):
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


def process_parameter_stats(cfg,parameter_names,fit_tracking,fit_variables):
   
    for var_name in parameter_names:
        parameter_stats = fit_variables[var_name].fit_stats

        if parameter_stats['r_hat'] > 1.5:
            fit_tracking['count_rhat'] += 1
        fit_tracking['total_rhat'] += parameter_stats['r_hat']



       
        stat_err = np.mean([parameter_stats['sd'],parameter_stats['mad'],parameter_stats['max_error']])
        if abs(parameter_stats['mean']-parameter_stats['max'])/stat_err > 0.5: 
            if parameter_stats['max_error'] > parameter_stats['sd']:
                fit_tracking['non_gaussian'] += 1       
            fit_variables[var_name].stat_use = 'Max'
        else:
            fit_variables[var_name].stat_use = 'Mean'
          

def process_accumulated_stats(cfg,fit_tracking,parameter_count):
    fit_tracking['previous_stats'] = copy.deepcopy(fit_tracking['statistics_quality'])
    fit_tracking['statistics_quality'] = reset_statistics_quality()
                                          
    if fit_tracking['count_rhat']/parameter_count > 0.5:
        print_log(f'''More than 50% of the parameters have a rhat > 1.5  This is not good. 
''',cfg,case=['main'])
        fit_tracking['statistics_quality']['low_ind_rhat'] = False
    if fit_tracking['non_gaussian']/parameter_count >= 0.5:
        print_log(f'''More than half of the parameters differ by more than the std in mena max.  This indicates non-gausianity of the distribution .
''',cfg,case=['main'])
        fit_tracking['statistics_quality']['similar_stat'] = False
    if fit_tracking['total_rhat']/parameter_count > 3:
        print_log(f'''The average rhat is > 3. This is not good.
''',cfg,case=['main'])
        fit_tracking['statistics_quality']['low_average_rhat'] = False
    if fit_tracking['total_rhat'] > 1.2*fit_tracking['prev_rhat'] and\
        fit_tracking['prev_rhat'] > 0.:
        print_log(f'''The total rhat is increasing. This is not good.
''',cfg,case=['main'])
        fit_tracking['statistics_quality']['stable_rhat'] = False     

def process_results(cfg,fit_variables, mcmc_result,fit_tracking,lmfit=False):
    func_dict = statistic_func_dict()
    parameter_names = [name for name in fit_variables if fit_variables[name].variable]   
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        #, message="Shape validation failed")
        if lmfit:
            fit_summary = lmfit_summary(mcmc_result, var_names=parameter_names,
                stat_funcs=func_dict)    
        else:
            fit_summary = arviz.summary(mcmc_result, var_names=parameter_names,
                fmt='xarray',round_to= None,stat_funcs=func_dict)
    print(fit_summary)
    set_parameter_statistics(cfg,fit_variables,fit_summary)
    fit_tracking['prev_rhat'] = copy.deepcopy(fit_tracking['total_rhat'])
    for reset in ['count_rhat','total_rhat','non_gaussian']:
        fit_tracking[reset] = 0
 
    process_parameter_stats(cfg,parameter_names,fit_tracking,fit_variables)                                           
    process_accumulated_stats(cfg,fit_tracking,len(parameter_names))   
   
    if any([not fit_tracking['statistics_quality'][x] for x in fit_tracking['statistics_quality']]):
        fit_tracking['failure_count'] +=1
        err_message = create_error_message(fit_tracking)
        print_log(f'''{err_message}''',cfg,case=['main'])
        if (not fit_tracking['statistics_quality']['stable_rhat'] and fit_tracking['failure_count'] > 2.)\
            or fit_tracking['failure_count'] > 4.:
            raise FailedFitError(f'''{err_message}''')
       
    check_boundaries(cfg,fit_variables,fit_summary, fit_tracking)
    
    
    return  fit_tracking,fit_variables  

def lmfit_summary(mcmc_result, var_names=None,stat_funcs=None):
   
    from typing import List
    """
    Create a summary of the MCMC results.
    Parameters
    ----------
    mcmc_result : lmfit.ModelResult
        The result of the MCMC fitting.
    var_names : list of str, optional
        The names of the variables to include in the summary. If None, all
        variables are included.
    fmt : str, optional         
        The format of the summary. Can be 'xarray' or 'pandas'.
    round_to : int, optional
        The number of decimal places to round the summary to. If None, no
        rounding is applied.
    stat_funcs : dict, optional  
        A dictionary of functions to calculate statistics. The keys are the
        names of the statistics and the values are the functions. If None,
        the default statistics are used.
    **kwargs : keyword arguments
        Additional keyword arguments to pass to the summary function.
    Returns
    -------
    summary : xarray.DataArray or pandas.DataFrame
        The summary of the MCMC results.
    """
    if var_names is None:
        var_names = mcmc_result.var_names
    chain = mcmc_result.flatchain
    
    tmp = {}
    metric_names: List[str] = []
    for i,names in enumerate(var_names):
        #print(chain[names])
        varmetr = []
        for func_name,func in stat_funcs.items():
            if func_name == 'r_hat':
              
                meanaccpt = np.mean(mcmc_result.acceptance_fraction)
                
                if meanaccpt > 0.5:
                    rhat  = 1. 
                else:
                    rhat = 1./abs(0.5 - meanaccpt)
                varmetr.append(rhat)
            else:
                varmetr.append(func(chain[names]))
        
            if i == 0:                
                metric_names.append(func_name)
        print(f'for {names} we find {",".join([f"{x}={y}" for x,y in zip(metric_names,varmetr)])} ')
        tmp[names] = xarray.DataArray(np.array(varmetr),dims=['metric'],\
            coords=[metric_names])
    metrics = xarray.Dataset(tmp)
    #print(metrics)
    return metrics
  


def lmfit_run(cfg,total_RC, out_dir = None,optical_profile = False):
    fit_tracking = set_tracking()
    function_variable_settings = copy.deepcopy(total_RC.fitting_variables)
    
    results_name = get_output_name(cfg,profile_name = total_RC.name,
        function_name= total_RC.numpy_curve['function'].__name__)
   
    steps,burning = calculate_steps_burning(cfg,function_variable_settings)
    #First set the model
    model = lmfit.Model(total_RC.numpy_curve['function'])
    #then set the hints
    if cfg.fitting_general.use_gp and not optical_profile:
        results_name = 'GP_' + results_name

    
    added = []
   
    for variable in function_variable_settings:
            function_variable_settings[variable].fill_empty()
            if variable not in added:
                model.set_param_hint(variable,value=function_variable_settings[variable].value,\
                            min=function_variable_settings[variable].boundaries[0],\
                            max=function_variable_settings[variable].boundaries[1],
                            vary=function_variable_settings[variable].variable)
                added.append(variable)
   
    parameters = model.make_params()
    #Why only 1 worker? Because lmfit doesn't spawn the multiprocesses correct on MAC
    
    if total_RC.numpy_curve['function'].__name__ == 'total_numpy_curve' and\
        sys.platform == 'darwin':
        workers =1
    else:
        workers = cfg.input.ncpu

    emcee_kws = dict(steps=steps, burn=burning, thin=10, is_weighted=True,\
        workers=workers)
 
    while ((not fit_tracking['value_convergence'] or 
        not fit_tracking['boundary_convergence']) and
        fit_tracking['iterations'] <= cfg.fitting_general.max_iterations):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            '''
            , message="invalid value encountered in double_scalars")
            warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
            warnings.filterwarnings("ignore", message="invalid value encountered in log")
            warnings.filterwarnings("ignore", message="invalid value encountered in divide")
            warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
            warnings.filterwarnings("ignore", message="overflow encountered in power")
            '''
            fit_tracking['iterations'] +=1
            print_log(  f'''We are running the {fit_tracking['iterations']} iteration of the fitting process.
''',cfg,case=['main','screen'])
            result_emcee = model.fit(data=total_RC.values.value, \
                r=total_RC.radii.value, params=parameters, method='emcee'\
                ,nan_policy='omit',fit_kws=emcee_kws,weights=1./total_RC.errors.value)
            
            fit_tracking,function_variable_settings = process_results(
                    cfg,function_variable_settings,result_emcee,fit_tracking,
                    lmfit = True)
           
            print_log(f'''The fit has the following statistics:
{result_emcee.fit_report()}
The fit has the following evaluation:
{fit_tracking}
''',cfg,case=['main','screen'])
           
           
            for variable in function_variable_settings:
                if function_variable_settings[variable].variable:
                    parameters[variable].min = function_variable_settings[variable].boundaries[0]
                    parameters[variable].max = function_variable_settings[variable].boundaries[1]
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
        ranges= []
        for parameter_mc in result_emcee.params:
            if result_emcee.params[parameter_mc].vary:
                print(f'for {parameter_mc} we find:')
                strip_parameter,no = get_uncounted(parameter_mc) 
                edv,correction = get_exponent(np.mean(result_emcee.flatchain[parameter_mc]),threshold=3.)
                result_emcee.flatchain[parameter_mc] = result_emcee.flatchain[parameter_mc]*correction
                lab.append(get_correct_label(strip_parameter,no,exponent= edv))
                ranges.append((function_variable_settings[parameter_mc].previous_boundaries[0],
                           function_variable_settings[parameter_mc].previous_boundaries[1]))
            
        #xdata= xarray.Dataset.from_dataframe(result_emcee.flatchain)
        #ardata = arviz.InferenceData(xdata) 
         
       
        fig = corner.corner(result_emcee.flatchain, bins=40, ranges =ranges, labels=lab, 
            show_titles=True,title_kwargs={"fontsize": 15},quantiles=[0.393, 0.864]
            ,divergence =True)
        #fig = corner.corner(result_emcee.flatchain, quantiles=[0.16, 0.5, 0.84],show_titles=True,
        #                title_kwargs={"fontsize": 15},labels=lab)
        fig.savefig(f"{out_dir}{results_name}_COV_Fits.pdf",dpi=300)
        plt.close()
    print_log(f''' MCMC_RUN: We find the following parameters for this fit. \n''',cfg,case=['main'])
    '''
    for variable in function_variable_settings:
        if function_variable_settings[variable].variable:
            function_variable_settings[variable].boundaries[0] = float(result_emcee.params[variable].value-\
                                    result_emcee.params[variable].stderr)
            function_variable_settings[variable].boundaries[1] = float(result_emcee.params[variable].value+\
                                    result_emcee.params[variable].stderr)

            function_variable_settings[variable].value = float(result_emcee.params[variable].value)
            print_log(f''{variable} = {result_emcee.params[variable].value} +/- {result_emcee.params[variable].stderr} within the boundary {result_emcee.params[variable].min}-{result_emcee.params[variable].max}
'',cfg,case=['main'])
    '''                
    BIC = result_emcee.bic



    return function_variable_settings,BIC,fit_tracking,result_emcee.flatchain

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
    err_message = f'''The fit did not converge. The following problems were detected: \n'''
    if not fit_tracking['statistics_quality']['similar_stat']:
        err_message += f'# The mean and max differ by more than the std for more than half of the parameters. \n'
    if fit_tracking['failure_count'] > 4:
        err_message += f'# The fit has failed more than 4 times. \n'
    if not fit_tracking['statistics_quality']['low_average_rhat']:
        err_message += f'# The total rhat ({fit_tracking["total_rhat"]}) is on average > 3. \n'
    if not fit_tracking['statistics_quality']['low_ind_rhat']:
        err_message += '# More than half of the parameters have a rhat > 1.5 \n'
    if not fit_tracking['statistics_quality']['stable_rhat']:
        err_message += f'# The total rhat ({fit_tracking["total_rhat"]}) is increasing ({fit_tracking["prev_rhat"]}). \n'
    return err_message

def avoid_negative(cfg,parameter,value):
    result = False
    if (not cfg.fitting_general.negative_values
        and not parameter.log 
        and value < 0.):
            result = True
    return result

def update_parameter_values(cfg,var_name,parameter):
        
    parameter.previous_value = copy.deepcopy(parameter.value)
    parameter.previous_boundaries = copy.deepcopy(parameter.boundaries)
   
    stats = parameter.fit_stats
    if parameter.stat_use == 'Max':
        print_log(f'''We will use the maximum for {var_name}''',cfg,case=['main'])
        parameter.value = float(stats['max'])
        parameter.stddev = float(stats['max_error'])
    elif parameter.stat_use == 'Mean':
        parameter.value = float(stats['mean'])
        parameter.stddev = float(stats['sd'])
    elif parameter.stat_use == 'Median':
        parameter.value = float(stats['median'])
        parameter.stddev = float(stats['mad'])
    else:
        raise ValueError(f'''The statistic {parameter.stat_use} is not valid.
''')    
    boun_stats = [float(stats['low_3%']),
                float(stats['high_97%'])]
    if avoid_negative(cfg,parameter,boun_stats[0]):
        boun_stats[0] = 0.
    for i in range(2):
        if not parameter.fixed_boundaries[i]:
            parameter.boundaries[i] = boun_stats[i]

def check_boundaries(cfg,function_variable_settings,output,fit_tracking):
    # we have to check that our results are only limited by the boundaries in the user has set them
    for var_name in function_variable_settings:
        if function_variable_settings[var_name].variable:
            parameter_message = f'------We are checking the boundaries for {var_name}------\n'
            current_parameter =  copy.deepcopy(function_variable_settings[var_name])
          
            update_parameter_values(cfg,var_name, current_parameter)
           
            #the boundaries should always be wider than the 3 * standard deviation
            change = abs(5.*current_parameter.stddev)
            min_bounds = [current_parameter.value-change,
                          current_parameter.value+change]
           
            if avoid_negative(cfg,current_parameter,min_bounds[0]):
                min_bounds[0] = 0.
          
            # if the boundaries are fixed we should not change them
            # We do issue a warning if they appear out                   
            if all(current_parameter.fixed_boundaries):
                parameter_message +=\
                    f'''The boundaries for {var_name} (min = {current_parameter.previous_boundaries[0]}, max = {current_parameter.previous_boundaries[1]}) are fixed.
!!!!!!!!! Check that they are reasonable. !!!!!!!!!\n'''                           
                if (current_parameter.previous_boundaries[0] <
                    min_bounds[0]):
                    parameter_message +=\
                        f'''For {var_name} the set lower bound ({current_parameter.previous_boundaries[0]}) is lower than the optimal lower bound ({min_bounds[0]})
Consider modifying the lower bound.\n'''
                if current_parameter.previous_boundaries[1] > min_bounds[1]:
                   parameter_message +=\
                        f'''For {var_name} the set upper bound ({current_parameter.previous_boundaries[1]}) is higher than the optimal upper bound ({min_bounds[1]})
Consider modifying the lower bound.\n'''
                parameter_message +=\
                    f'''The error on {var_name} is {current_parameter.stddev}. \n'''
                
               
                print_log(parameter_message,cfg,case=['main'])    
                function_variable_settings[var_name] = copy.deepcopy(current_parameter)
              
                continue

            print_log(f'''{var_name} = {current_parameter.value} +/- {current_parameter.stddev} 
change = {change}  bounds = {current_parameter.boundaries} original_boundaries = {current_parameter.original_boundaries}
minbounds = {min_bounds} prev_bound = {current_parameter.previous_boundaries}
''',cfg,case=['debug_add'])
            
 
    
            for i in range(2):
                '''
                if change > boundary_distance[i]:
                    parameter_message +=\
                        f''The boundaries for {var_name} are not symmetrical around the value. \n''
                    current_parameter.boundaries[i] = float(
                        current_parameter.value + (i*2-1) * 
                        np.max([change,boundary_distance[abs(i-1)]]))
                    if (i == 0 and #for the lower boundary
                        not negative and #if we do not want negative values
                        var_name[0:2] != 'lg' and # and this is not log
                        current_parameter.boundaries[i] < 0.): # and the parameter is less than 0
                        current_parameter.boundaries[i] = 0. #set it to 0
                '''
                # if the boundaries are less wide than 5 sig       
                if (-2.*i+1.)*current_parameter.boundaries[i]+(i*2.-1)*min_bounds[i] > 0.:
                    current_parameter.boundaries[i] = min_bounds[i]
                # after the first succesful fit we do not allow the 
                # boundaries to shrink towards the value 
                if fit_tracking['iterations']-fit_tracking['failure_count'] > 0.:
                    if ((-2.*i+1)*current_parameter.boundaries[i]+
                        (i*2.-1)*current_parameter.previous_boundaries[i] > 0.):
                        current_parameter.boundaries[i] = copy.deepcopy(current_parameter.previous_boundaries[i])
                # We never allow the boundaries to be grow beyond the original boundaries
                if not current_parameter.original_boundaries[i] is None:
                    if ((-2.*i+1)*current_parameter.boundaries[i]+
                        (i*2.-1)*current_parameter.original_boundaries[i] < 0.):
                      current_parameter.boundaries[i] = copy.deepcopy(current_parameter.original_boundaries[i])


            boundary_distance = [abs(current_parameter.value - x) for x in current_parameter.previous_boundaries]
          
            if abs(boundary_distance[0]-boundary_distance[1]) > 0.05*np.mean(boundary_distance):
                parameter_message +=\
                        f'''The boundaries for {var_name} are not symmetrical around the value. \n'''
             
                if boundary_distance[0] > boundary_distance[1]:
                  
                    current_parameter.boundaries[1] = current_parameter.value + boundary_distance[0]
                elif boundary_distance[1] > boundary_distance[0]:
                  
                    if not avoid_negative(cfg,current_parameter,current_parameter.value - boundary_distance[1]):
                        current_parameter.boundaries[0] = current_parameter.value - boundary_distance[1]
                    else:
                        current_parameter.boundaries[0] = 0. 
           
            #If we have very small errors we should still allow for some variation            
            tolerans = np.max([1.5*current_parameter.stddev])
          
            if np.allclose(np.array(current_parameter.previous_boundaries),np.array(current_parameter.boundaries)
                ,atol=tolerans):
                parameter_message += f'''{var_name} is fitted wel in the boundaries {'-'.join([f'{x}' for x in current_parameter.boundaries])}. 
(Old is {'-'.join([f'{x}' for x in current_parameter.previous_boundaries])})'''

                current_parameter.boundaries = copy.deepcopy(
                    current_parameter.previous_boundaries) 
                current_parameter.fit_direction[1] = 'stable'
               
               
            else:
                parameter_message += f''' The boundaries for {var_name} are deviating more than {2.*tolerans} from the previous boundaries.
Setting {var_name} = {current_parameter.value} between {'-'.join([f'{x}' for x in current_parameter.boundaries])} (old = {'-'.join([f'{x}' for x in current_parameter.previous_boundaries])})
'''
                current_parameter.fit_direction[1] = 'diverging'
            
            parameter_message += f'''We compared the array {np.array(current_parameter.previous_boundaries)} to {np.array(current_parameter.boundaries)} with a tolerance of {2.*tolerans}
'''
            if np.allclose(np.array(current_parameter.previous_value)
                ,np.array(current_parameter.value),
                atol=0.5*tolerans):
              
                parameter_message += f'''for {var_name} the value does not change much.'''
                current_parameter.fit_direction[0] = 'stable'
            else:
             
                parameter_message += f'''for {var_name} the value changes more than {tolerans}. '''
                current_parameter.fit_direction[0] = 'diverging'
            parameter_message += f'''old = {current_parameter.previous_value} new = {current_parameter.value}.\n'''
            function_variable_settings[var_name] = copy.deepcopy(current_parameter)    
            del current_parameter
            print_log(parameter_message,cfg,case=['main'])

    fit_tracking['boundary_convergence'] = True
    fit_tracking['value_convergence'] =True       
    for parameter in function_variable_settings:
        if function_variable_settings[parameter].variable:
            if function_variable_settings[parameter].fit_direction[0] == 'diverging':
                fit_tracking['value_convergence'] = False
                print_log(f'''The value for {parameter} is diverging. \n''',cfg,case=['debug_add'])
            if function_variable_settings[parameter].fit_direction[1] == 'diverging':
                fit_tracking['boundary_convergence'] = False
                print_log(f'''The boundaries for {parameter} is diverging. \n''',cfg,case=['debug_add'])
   

        
            
def statistic_func_dict():
    '''This function returns a dictionary with the functions that are used to calculate the statistics
    for the arviz summary. The default is mean and sd.'''
    fdict = {'mean':np.mean,
             'sd':np.std,
             'median':np.median,
             'mad': lambda x: np.median(np.abs(x - np.median(x))),
             'low_3%': lambda x: np.percentile(x, 3),
             'high_97%': lambda x: np.percentile(x, 97),
             'max': calc_max_occurence,
             'max_error': lambda x: calc_max_occurence(x,error=True),
             'r_hat': arviz.rhat
             }
    return fdict

def calc_max_occurence(x,error =False):
    #Create the histogram
    hist, bin_edges = np.histogram(x, bins=40)
    # get the index of the maximum value in the histogram
    max_index = np.where(np.nanmax(hist) == hist)[0][0]
    # get the value of the maximum occurence
    max_value = (bin_edges[max_index]+bin_edges[max_index+1])/2. 
    if error:
        # get the index of the maximum value in the histogram
        indices = np.where(np.nanmax(hist)/2. < hist)[0]
        minv = (bin_edges[indices[0]]+bin_edges[indices[0]+1])/2.
        maxv = (bin_edges[indices[-1]]+bin_edges[indices[-1]+1])/2.
        error = ((maxv-max_value)+(max_value-minv))/2. 
        return error
    return max_value