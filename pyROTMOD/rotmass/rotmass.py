# -*- coding: future_fstrings -*-


import numpy as np
import warnings
from pyROTMOD.support.errors import InputError,RunTimeError,FailedFitError
import copy
import pyROTMOD.rotmass.potentials as potentials
import pyROTMOD.support.constants as cons
from pyROTMOD.support.parameter_classes import Parameter,set_parameter_from_cfg
from pyROTMOD.support.profile_classes import Rotation_Curve
from pyROTMOD.support.minor_functions import get_uncounted,add_font,get_output_name
from pyROTMOD.support.log_functions import print_log
from pyROTMOD.fitters.fitters import initial_guess,lmfit_run, numpyro_run
from sympy import symbols, sqrt,lambdify

from jax import numpy as jnp
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

def build_curve(all_RCs, total_RC, cfg=None):
    # First set individual sympy symbols  and the curve for each RC

    ML, V = symbols('ML V')
    replace_dict = {'symbols': []}
    total_sympy_curve = None
    for name in all_RCs:
        RC_symbols = [x for x in list(all_RCs[name].curve.free_symbols) if str(x) != 'r']
        print_log(f'''##########################{name}##################
{'':8s}{[all_RCs[name].fitting_variables[x].print() for x in all_RCs[name].fitting_variables]}
{'':8s}{all_RCs[name].curve}
{'':8s}############################################''',cfg,case=['debug_add'])
        for symbol in RC_symbols:
            if symbol == V:
                V_replace = symbols(f'V_{all_RCs[name].name}')
                for attr in ['curve', 'individual_curve']:
                    setattr(all_RCs[name],attr,getattr(all_RCs[name],attr).subs({V: V_replace}))

                all_RCs[name].match_radii(total_RC)
                #Here we need to set it all to the radii of the total_RC else we can not match  
                if all_RCs[name].include: 
                    if cfg.fitting_general.backend.lower() == 'lmfit':
                        replace_dict[f'V_{all_RCs[name].name}']= {'values':
                            all_RCs[name].matched_values.value, 'radii':
                            all_RCs[name].matched_radii.value}
                    elif cfg.fitting_general.backend.lower() == 'numpyro':
                        replace_dict[f'V_{all_RCs[name].name}']= {'values':
                            all_RCs[name].values.value, 'radii':
                            all_RCs[name].radii.value}

                    
                    replace_dict['symbols'].append(V_replace)
            if symbol == ML:
                for variable in all_RCs[name].fitting_variables:
                    if variable.split('_')[0].lower() in ['gamma','ml']:    
                        ML_replace = symbols(variable)
                for attr in ['curve', 'individual_curve']:
                    setattr(all_RCs[name],attr,getattr(all_RCs[name],attr).subs({ML: ML_replace}))
      
        #RC_symbols = [x for x in list(all_RCs[name].individual_curve.free_symbols) if str(x) != 'r']
        RC_symbols = [x for x in list(all_RCs[name].individual_curve.free_symbols)]
        all_RCs[name].numpy_curve ={'function': lambdify(RC_symbols,all_RCs[name].individual_curve,"numpy"),
                                    'variables': [str(x) for x in RC_symbols]}
        if  all_RCs[name].include:
            if  total_sympy_curve is None:
                total_sympy_curve =  all_RCs[name].curve**2
            else:
                total_sympy_curve += all_RCs[name].curve**2
    
   
    total_sympy_curve = sqrt(total_sympy_curve)

    #For the actual fit curve we need to replace the  V components with their actual values
    #make sure that r is always the first input on the function and we will replace the RCs

    curve_symbols_out = [x for x in list(total_sympy_curve.free_symbols) if str(x) not in ['r']+[str(y) for y in replace_dict['symbols']] ]
    if cfg.fitting_general.backend.lower() == 'numpyro':
        #For the partial filling it is useful to have r at the end
        curve_symbols_out.append(symbols('r'))
    else:
        curve_symbols_out.insert(0,symbols('r'))
    curve_symbols_in = replace_dict['symbols']+curve_symbols_out
   
    initial_formula = lambdify(curve_symbols_in, total_sympy_curve ,"numpy")

    print_log(f'''BUILD_CURVE:: We are fitting this complete formula:
{'':8s}{initial_formula.__doc__}
''',cfg,case=['main','screen'])
    # since lmfit is a piece of shit we have to construct our final formula through exec
   
    clean_code = create_formula_code(initial_formula,replace_dict,total_RC,\
        function_name='total_numpy_curve',cfg=cfg)
   
    exec(clean_code,globals())
    
    total_RC.numpy_curve =  {'function': total_numpy_curve , 'variables': [str(x) for x in curve_symbols_out]}    
        

    
    total_RC.curve = total_sympy_curve
    # This piece of code can be used to check the exec made fit function
    #curve_lamb = lambda *curve_symbols_out: initial_formula(*values_to_be_replaced, *curve_symbols_out)
    #DM_final = {'function': DM_curve_np , 'variables': [str(x) for x in dm_variables] }
    #Curve_final = {'function': curve_np , 'variables': [str(x) for x in curve_symbols_out]}
    #baryonic_curve_final = {'function':baryonic_curve_np, 'variables': [str(x) for x in baryonic_variables]}
    # The line below can used to check whether the full curev has been build properly
    #check_final = {'function': curve_lamb , 'variables': [str(x) for x in curve_symbols_out]}
    #return DM_final,Curve_final,baryonic_curve_final

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



   

def calculate_red_chisq(RC,cfg=None):
    free_parameters= 0.
    for var in RC.fitting_variables:
        if RC.fitting_variables[var].variable:
            free_parameters  += 1.
    if RC.errors is None:
        raise InputError(f'In the RC {RC.name} we cannot calculate a chi^2 as we have no errors. ')
    Chi_2 = np.nansum((RC.values-RC.calculated_values)**2/RC.errors**2)
    count = 0.
    for x in RC.calculated_values:
        if not np.isnan(x):
            count += 1
    red_chisq = Chi_2/(count-free_parameters)
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


def inject_GP(total_RC,header = False):
    if header:
        code= f'''from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel\n'''
   
    else:
        code = f'''{'':6s}# Define the Gaussian Process kernel
{'':6s}x = r.reshape(-1, 1) 
{'':6s}kernel = ConstantKernel(10**lgamplitude) * RBF(length_scale=length_scale)
{'':6s}# Initialize the Gaussian Process Regressor
{'':6s}yerr=np.array([{', '.join([str(i.value) for i in total_RC.errors])}],dtype=float)
{'':6s}gp = GaussianProcessRegressor(kernel=kernel, alpha=yerr**2, n_restarts_optimizer=3, normalize_y=True)
{'':6s}# Evaluate the model using the current parameters
{'':6s}# Fit the GP to the residuals (data - model)
{'':6s}if any(np.isnan(vmodelled)):
{'':12s}y_pred = np.array([np.nan for x in vmodelled])
{'':6s}else: 
{'':12s}gp.fit(x, vmodelled)
{'':12s}# Predict the residuals
{'':12s}y_pred = gp.predict(x, return_std=False)
{'':6s}return y_pred
'''
    return code
def inject_interp_array(array,rad):
    array1=f'jnp.array([{", ".join([str(i) for i in array])}])'
    array2=f'jnp.array([{", ".join([str(i) for i in rad])}])'
    total = f'jnp.interp(r,{array2},{array1})'
    return total

def inject_simple_array(array,rad):
    return f'np.array([{", ".join([str(i) for i in array])}])'

def create_formula_code(initial_formula,replace_dict,total_RC,\
            function_name='python_formula' ,cfg=None):
    lines=initial_formula.__doc__.split('\n')
    if cfg.fitting_general.backend.lower() == 'numpyro':
        dictionary_trans = {'sqrt':'jnp.sqrt', 'arctan': 'jnp.arctan', \
                        'pi': 'jnp.pi','log': 'jnp.log', 'abs': 'jnp.abs'}
        array_name = inject_interp_array
        #'jnp.array'
    else:
        
        dictionary_trans = {'sqrt':'np.sqrt', 'arctan': 'np.arctan', \
                        'pi': 'np.pi','log': 'np.log', 'abs': 'np.abs'}
        array_name = inject_simple_array
        #simp'np.array'
    #rad = total_RC.radii.value
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
    if cfg.fitting_general.use_gp and cfg.fitting_general.backend.lower() == 'lmfit':
            clean_code += inject_GP(total_RC,header=True)
    
    for i,line in enumerate(code.split('\n')):
        if i == 0:
            #This is the header line of the code
            line = line.replace('_lambdifygenerated',function_name)
            for key in replace_dict:
                if key != 'symbols':
                    line = line.replace(key+',','')
            if cfg.fitting_general.use_gp and cfg.fitting_general.backend.lower() == 'lmfit':
                line = line.replace('):',', lgamplitude, length_scale):')
            line += '\n'
        if i == 1:
            for key in dictionary_trans:
                line = line.replace(key,dictionary_trans[key])
            for key in replace_dict:
                if key != 'symbols':
                    print_log(f'REPLACE: {key} with {replace_dict[key]}',cfg,
                        case=['debug_add'])
                    line = line.replace(key,array_name(replace_dict[key]['values']
                                            ,replace_dict[key]['radii']))
                  
            line = f'''{'':6s}{line.replace('return','vmodelled = ').strip()}\n'''
            if cfg.fitting_general.use_gp and cfg.fitting_general.backend == 'lmfit':
                line += inject_GP(total_RC)
            else:
                line += f'''{'':6s}return vmodelled \n'''
        clean_code += line

    print_log(f''' This the code for the formula that is finally fitted.
{clean_code}
''',cfg,case=['debug_add','screen'])
   
    return clean_code
create_formula_code.__doc__ =f'''
 NAME:
    create_formula_code

 PURPOSE:
    create the code that can be ran through exec to get the function to be fitted.
    The input formula is already lambidified so this is merely a matter of  replacing the variables with the correct 
    values for each radius.

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

def write_output_file(cfg,final_variable_fits,BIC,output_dir='./',\
                results_file = None, red_chisq= None):
    if results_file is None:
        results_file = f'{get_output_name(cfg)}_results.txt'
    #variables_fitted = [x for x in final_variable_fits]
    variable_line = f'{"Variable":>15s} {"Value":>15s} {"Error":>15s} {"Lower Bound":>15s} {"Upper Bound":>15s} \n'
    with open(f'{output_dir}{results_file}.txt','w') as file:
        file.write('# These are the results from pyROTMOD. \n')
        file.write('# An error designated as Fixed indicates a parameter that is not fitted. \n')
        file.write(f'# The Reduced Xi^2 for this fit is {red_chisq}.\n')
        file.write(f'# The Bayesian Information Criterion for this fit is {BIC}.\n')
        file.write(variable_line)
        added = []
        for variable in final_variable_fits:
            if variable not in added:
                if final_variable_fits[variable].include:
                    if 0.001 < final_variable_fits[variable].value < 10000.:
                        form = ">15.4f"
                    else:
                        form = ">15.4e"
                    variable_line = f'{variable:>15s} {final_variable_fits[variable].value:{form}}'
                    min =final_variable_fits[variable].min
                    max = final_variable_fits[variable].max
                    if not final_variable_fits[variable].variable:
                        variable_line += f' {"Fixed":>15s} {min:{form}} {max:{form}} \n'
                    else:
                        err = final_variable_fits[variable].stddev
                        variable_line += f' {err:{form}} {min:{form}} {max:{form}} \n'
                    file.write(variable_line)
                    added.append(variable)

def set_RC_style(RC,input=False):
    style_dictionary = {'label':  'V$_{Obs}$',\
                        'lw': 5, \
                        'linestyle':'-',\
                        'markerfacecolor': colors.to_rgba('k',alpha=0.25),\
                        'markeredgecolor': colors.to_rgba('k',alpha=0.5),\
                        'markeredgewidth':4, \
                        'marker': 'o',\
                        #'alpha': 0.5,\
                        'zorder': 7,\
                        'ms': 15, \
                        'color': colors.to_rgba('k',alpha=0.5)}
    if not input:
        #style_dictionary['alpha'] = 1.

        if RC.component.lower() == 'all':
            style_dictionary['color'] = colors.to_rgba('r',alpha=1.)
            style_dictionary['label'] =  r'V$_{Total}$'
            style_dictionary['zorder'] = 6
            style_dictionary['markerfacecolor'] = colors.to_rgba(style_dictionary['color'],alpha=0.5)
            style_dictionary['markeredgecolor'] = colors.to_rgba(style_dictionary['color'],alpha=0.75)

        elif RC.component.lower() == 'dm':
            style_dictionary['linestyle'] = '-.'
            style_dictionary['color'] = colors.to_rgba('b',alpha=1.)
            style_dictionary['label'] =  f'V$_{{{RC.halo}}}$'
            style_dictionary['zorder'] = 6
        elif RC.component.lower() == 'gas':
            rcs,no = get_uncounted(RC.name)
            style_dictionary['linestyle'] = ':'
            style_dictionary['color'] = colors.to_rgba('g',alpha=1.)
            style_dictionary['label'] = f'V$_{{Gas\\_Disk_{no}}}$'
            style_dictionary['zorder'] = 4
        elif RC.component.lower() == 'stars':
            rcs,no = get_uncounted(RC.name)
            style_dictionary['linestyle'] = '--'
            if rcs in ['EXPONENTIAL','DISK']:
                style_dictionary['color'] = colors.to_rgba('cyan',alpha=1.)
                style_dictionary['label'] =  f'V$_{{Stellar\\_Disk_{no}}}$'
                style_dictionary['zorder'] = 3
            elif rcs in ['HERNQUIST','BULGE','SERSIC_BULGE']:
                style_dictionary['color'] = colors.to_rgba('purple',alpha=1.)
                style_dictionary['label'] = f'V$_{{Stellar\\_Bulge_{no}}}$'
                style_dictionary['zorder'] = 2
            elif rcs in ['SERSIC','SERSIC_DISK']:
                style_dictionary['color'] = colors.to_rgba('blue',alpha=1.)
                style_dictionary['label'] =  f'V$_{{Stellar\\_Sersic_{no}}}$'
                style_dictionary['zorder'] = 3
            else:
                style_dictionary['color'] = colors.to_rgba('orange',alpha=1.)
                style_dictionary['label'] =  f'V$_{{Stellar\\_Random_{no}}}$'
                style_dictionary['zorder'] = 3

        else:
            raise InputError(f'The component {RC.component} in {RC.name} is not a component familiar to us')
        if not RC.component.lower() == 'all':    
            style_dictionary['markerfacecolor'] = colors.to_rgba(style_dictionary['color'],alpha=0.25)
            style_dictionary['markeredgecolor'] = colors.to_rgba(style_dictionary['color'],alpha=0.5)
            style_dictionary['ms'] = 8
    return style_dictionary
        


def plot_individual_RC(RC,ax1,input=False):
    if input:
        plot_values = RC.values
    else:
        #make sure we are using the latest settings
        RC.calculate_RC()
        plot_values = RC.calculated_values
    if plot_values is None:
        #If we have no values to plot we skip this curve
        return ax1
    style_library = set_RC_style(RC,input=input)
    ax1.plot(RC.radii,plot_values,**style_library)
    
    plot_err = None
    if input:
        if not RC.errors is None:
            plot_err = [RC.values.value-RC.errors.value,\
                        RC.values.value+RC.errors.value]
    else:
        if not RC.calculated_errors is None:
            plot_err =  RC.calculated_errors.value 

    if not plot_err is None:
        if style_library['markeredgecolor'][3] < 1.:
            use_alpha =  style_library['markeredgecolor'][3] -0.3
        else:
            use_alpha = style_library['markeredgecolor'][3] -0.5
        if use_alpha < 0.1:
            use_alpha = 0.1
        ax1.fill_between(RC.radii.value,plot_err[0] ,\
                        plot_err[1] ,
                        color= style_library['color'],
                        alpha = use_alpha,
                        edgecolor='none',zorder=1)

    ax1.legend()
    ax1.set_xlabel(f'R (kpc)', fontdict=dict(weight='bold',size=16))
    ax1.set_ylabel(f'V (km/s)', fontdict=dict(weight='bold',size=16))
    return ax1

def calculate_residual(RC,ax2):
    if RC.values is None or RC.calculated_values is None:
        ax2.remove()
        return ax2
    ax2.plot(RC.radii,RC.values-RC.calculated_values,marker='o', \
            ms=15,color='r',linestyle =None,zorder=2,lw=5)
    ax2.plot(RC.radii, np.zeros(len(RC.radii)),marker=None,lw=5, \
            ms=10,color='k',linestyle ='-',zorder=1)
    
    ax2.set_xlabel('Radius (kpc)', fontdict=dict(weight='bold',size=16))
    ax2.set_ylabel('Residual (km/s)', fontdict=dict(weight='bold',size=16))
    ymin,ymax=ax2.get_ylim()
    buffer = (ymax-ymin)/20.
    ax2.set_ylim(ymin-buffer,ymax+buffer)
    if not RC.errors is None:
        ax2.fill_between(RC.radii.value,-1*RC.errors.value ,\
                        RC.errors.value  ,
                        color= 'k',
                        alpha = 0.2,
                        edgecolor='none',zorder=0)
        red_Chi_2 = calculate_red_chisq(RC)

        ax2.text(1.0,1.0,f'''Red. $\\chi^{{2}}$ = {red_Chi_2:.4f}''',rotation=0, va='top',ha='right', color='black',\
            bbox=dict(facecolor='white',edgecolor='white',pad=0.5,alpha=0.),\
            zorder=7, backgroundcolor= 'white',fontdict=dict(weight='bold',size=16),transform = ax2.transAxes)
    return ax2

def plot_curves(filename,RCs,total_RC,interactive = False, font = 'Times New Roman'):

    
    labelfont = {'family': font,
             'weight': 'bold',
             'size': 14}
    plt.rc('font',**labelfont)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rc('axes', linewidth=2)
    figure,  (ax1, ax2) = plt.subplots(nrows=2, ncols=1,figsize=(10,10), gridspec_kw={'height_ratios': [3, 1]})
    for name in RCs:
        if RCs[name].include:
            ax1 = plot_individual_RC(RCs[name],ax1)
    ax1 = plot_individual_RC(total_RC,ax1)
    ax1 = plot_individual_RC(total_RC,ax1, input=True)
    ax2 = calculate_residual(total_RC,ax2)
    if interactive:
        raise InputError(f"An interactive mode is not available yet, feel free to right a Gui. Here you can plot the curves")
    else:
        plt.savefig(filename,dpi=300, bbox_inches='tight')
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
def update_RCs(update,RCs,total_RC):
    
    for variable in update:
        for name in RCs:
            if variable in [key for key in RCs[name].fitting_variables]:
                RCs[name].fitting_variables[variable] = update[variable]
                    
        total_RC.fitting_variables[variable] = update[variable]  
      


def rotmass_main(cfg,baryonic_RCs, total_RC,interactive = False):
    out_dir = f'{cfg.output.output_dir}{cfg.fitting_general.HALO}/'
    original_fitting_settings = copy.deepcopy(total_RC.fitting_variables)
    results_file = get_output_name(cfg,profile_name = total_RC.name)
    font = add_font(cfg.input.font)        
    # First combine all RCs that need to be included in the total fit in a single dictionary
    # With their parameters and individual RC curves set 
    all_RCs = set_fitting_parameters(cfg,baryonic_RCs,total_RC)
  
    for names in all_RCs:
        print_log(f'''For {names} we find the following parameters and fit variables:
{all_RCs[names].print()}
''',cfg,case=['debug_add'])
        

    # Construct the function to be fitted, note that the actual fit_curve is
    build_curve(all_RCs,total_RC,cfg=cfg)                      
    
       
    if interactive:
        #We want to bring up a GUI to allow for the fitting
        print_log(f'''Unfortunately the interactive function of fitting is not yet implemented. Feel free to write a GUI.
For now you can set, fix and apply boundaries in the yml input file.
for your current settings the variables are {','.join(total_RC.numpy_curve['variables'])}
''',cfg,case=['main'])
        exit()
        #Start GUI with default settings or input yml settings. these are already in function_parameters
    else:

        plot_curves(f'{out_dir}/{results_file}_Input_Curves.pdf', all_RCs,\
            total_RC,font= font)
   
    # calculate the initial guesses

    if cfg.fitting_general.use_gp :
        #total_RC.fitting_variables['amplitude'] = [1.,0.1,10.,True,True]
        #total_RC.fitting_variables['length_scale'] = [1.,0.1,10.,True,True]
        total_RC.fitting_variables['lgamplitude'] = Parameter(name='lgamplitude'
            ,value= 1.2, min= -3. ,max = 3., variable = True, include = True, 
            fixed_boundaries= False)
        total_RC.fitting_variables['length_scale'] = Parameter(name='length_scale'
            ,value= 1., min= 0.5 ,max = 5., variable = True, include = True ,
            fixed_boundaries= False)
        
        total_RC.numpy_curve['variables'] = total_RC.numpy_curve['variables'] + ['lgamplitude','length_scale']
    skip_output = False
    if cfg.fitting_general.backend.lower() == 'numpyro':
        try:
            variable_fits,BIC,succes = numpyro_run(cfg,total_RC, out_dir = out_dir)
        except FailedFitError as e:
            print_log(f'''The parameters for the function could not be fitted.
Skipping the output Fase''', cfg, case=['main'])
            succes = False
            skip_output =True

    else:
        initial_guesses = initial_guess(cfg, total_RC)
        original_settings = copy.deepcopy(total_RC.fitting_variables)
        for variable in original_settings:
            original_settings[variable].fill_empty()
        update_RCs(initial_guesses,all_RCs,total_RC) 
        
        plot_curves(f'{out_dir}/{results_file}_Initial_Guess_Curves.pdf',\
            all_RCs,total_RC,font=font)
       
        variable_fits,BIC,succes = lmfit_run(cfg, total_RC, original_settings,out_dir = out_dir)
    if not skip_output:
        update_RCs(variable_fits,all_RCs,total_RC) 
        
        print_log('Plotting and writing',cfg,case=['main'])
        plot_curves(f'{out_dir}/{results_file}_Final_Curves.pdf', \
            all_RCs,total_RC,font=font)      
        
        red_chisq = calculate_red_chisq(total_RC)

        write_output_file(cfg,variable_fits,BIC,output_dir=out_dir, 
            red_chisq = red_chisq, results_file=f'{results_file}_results.txt')
    if not succes:
        raise RunTimeError(f'''The fitting procedure was not successful. See log for details
''')

rotmass_main.__doc__ =f'''
 NAME:
    rotmass_main

 PURPOSE:
    The main fitting module for the RCs

 CATEGORY:
    rotmass

 INPUTS:
    baryonic_RCs - Dictionary of baryonic rotation curves.
    total_RC - The total rotation curve object.
    no_negative - Whether to disallow negative values in the fit.
    out_dir - Directory to save output files.
    interactive - Whether to enable interactive mode (not implemented).
    rotmass_settings - Settings for the rotation curve fitting.
    cfg - Configuration object for logging.
    rotmass_parameter_settings - Parameter settings for the fit.
    results_file - Name of the results file.
    font - Font to use for plots.
    use_gp - Whether to use Gaussian Process fitting for data correlations.
    gp_kernel - Kernel type for Gaussian Process fitting (default: "RBF").

 OUTPUTS:
    None

 OPTIONAL OUTPUTS:
    Saves plots and results to the specified output directory.

 PROCEDURES CALLED:
    build_curve, initial_guess, mcmc_run, gp_fitter, plot_curves, write_output_file

 NOTE:
    Ensure all required modules and dependencies are installed.
'''


def add_fitting_dict(name, parameters, component_type = 'stars', fitting_dictionary = {}):

    variable = None
    #V_disk and V_bulge, V_sersic are place holders for the values to be inserted in the final formulas
    base,number = get_uncounted(name)
    component_type = component_type.lower() 

    if base in ['EXPONENTIAL','DISK','DISK_GAS']:
        variable  = f'Gamma_disk_{component_type}_{number}'
    elif base in ['HERNQUIST','BULGE'] and component_type == 'stars':
        variable = f'Gamma_bulge_{component_type}_{number}'
    elif base in ['SERSIC'] and component_type == 'stars':
        variable = f'Gamma_sersic_{component_type}_{number}'

    if variable is None:
        variable = f'Gamma_random_{component_type}_{number}'
    
    # We need to add the parameter to the fitting dictionary    
    fitting_dictionary[variable] = set_parameter_from_cfg(variable,
        parameters) 
    # We need to add the parameter to the fitting dictionary    
  
def set_fitting_parameters(cfg, baryonic_RCs,total_RC):
    # Get the variables of the DM function
    dm_parameters = []
    no_dm = False
    baryonic = []
    total_RC.fitting_variables = {}
    for x in getattr(potentials, cfg.fitting_general['HALO'])().free_symbols: 
        if str(x) == 'r':
            pass
        elif str(x) in ['ML','V']:
            baryonic.append(str(x))
        else:
            dm_parameters.append(str(x))
    if len(baryonic) == 2:
        no_dm = True 
    

    all_RCs = copy.deepcopy(baryonic_RCs)
    # Let's set the initial parameters for all the baryonic curve
    for name in all_RCs:
        
        fitting_dictionary = {} 
        all_RCs[name].check_component()
        add_fitting_dict(all_RCs[name].name,cfg.fitting_parameters[all_RCs[name].name],\
                         component_type=all_RCs[name].component,\
                        fitting_dictionary=fitting_dictionary)
        #Check whether we want to include this RC tot the total
        if not cfg.fitting_parameters[all_RCs[name].name][4]:
            all_RCs[name].include=False
     
        if no_dm:
            all_RCs[name].halo = cfg.fitting_general['HALO']
            all_RCs[name].curve = getattr(potentials, cfg.fitting_general['HALO'])()       
            all_RCs[name].individual_curve = getattr(potentials, f"{cfg.fitting_general['HALO']}_INDIVIDUAL")()
            for variable in dm_parameters:
                # Using a dictionary make the parameter always to be added
                fitting_dictionary[variable] = set_parameter_from_cfg(variable,
                    cfg.fitting_parameters[variable]) 
        else:
            ML, V = symbols(f"ML V")
            all_RCs[name].halo = 'NEWTONIAN'
            all_RCs[name].curve = sqrt(ML*V*abs(V))
            all_RCs[name].individual_curve = V/abs(V)*sqrt(ML*V**2)
        

  

        all_RCs[name].fitting_variables= fitting_dictionary 
        
        
        all_RCs[name].check_unified(cfg.fitting_general.single_stellar_ML,\
                                    cfg.fitting_general.single_gas_ML)
        if all_RCs[name].include:
            total_RC.fitting_variables.update(all_RCs[name].fitting_variables)
      
    if not no_dm:
        #We need add the DM RC and the parameters
        all_RCs[cfg.fitting_general['HALO']] = Rotation_Curve(component='DM',\
            name=cfg.fitting_general['HALO'])
        fitting_dictionary = {} 
        for variable in dm_parameters:
            # Using a dictionary make the parameter always to be added
            fitting_dictionary[variable] =set_parameter_from_cfg(variable,
                    cfg.fitting_parameters[variable]) 
            if not bool(cfg.fitting_parameters[variable][4]):
                raise  InputError(f'''You have requested the {cfg.fitting_generals['HALO']} DM halo but want to exclude {variable} from the fitting.
You cannot change the DM formula, if this is your aim please add a potential in the rotmass potential file.
If you merely want to fix the variable, set an initial guess and fix it in the input (e.g. rotmass.{variable} = [100, null,null, False,True]). ''')
        all_RCs[cfg.fitting_general['HALO']].fitting_variables = fitting_dictionary
        all_RCs[cfg.fitting_general['HALO']].radii = total_RC.radii
        all_RCs[cfg.fitting_general['HALO']].values = np.zeros(len(total_RC.radii))*\
                                                    total_RC.values.unit
        
        all_RCs[cfg.fitting_general['HALO']].halo = cfg.fitting_general['HALO']
        all_RCs[cfg.fitting_general['HALO']].curve = getattr(potentials, cfg.fitting_general['HALO'])()
        #Thewre are no negatives for the DM supossedly so the individual curve is the same
        all_RCs[cfg.fitting_general['HALO']].individual_curve = getattr(potentials, cfg.fitting_general['HALO'])()
        total_RC.fitting_variables.update(all_RCs[cfg.fitting_general['HALO']].fitting_variables)
   
    return all_RCs
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
