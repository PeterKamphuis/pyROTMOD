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
import traceback
import corner
import pyROTMOD.rotmass.potentials as V
import pyROTMOD.constants as cons
from pyROTMOD.support import print_log
from sympy import symbols, sqrt,atan,pi,log,Abs,lambdify
from scipy.optimize import differential_evolution,curve_fit,OptimizeWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt

def rotmass_main(radii, derived_RCs, total_RC,total_RC_err,no_negative =True,out_dir = None,\
                interactive = False,rotmass_settings = None,log_directory=None,log=None,debug = False):
    #Dictionary for translate RC and mass parameter for the baryonic disks
    disk_var = {'MD': 'Vdisk','MG': 'Vgas','MB': 'Vbulge' }

    # First combine all parameters that to be included in the total fit in a single dictionary
    function_variable_settings = set_fitting_parameters(rotmass_settings,disk_var)

    # Then we will group the baryonic  rotation curves that are define,the error of the total_RC is added.
    # When introducing errors on the RCs of the curves we need to do that here.
    radii ,total_RC,Baryonic_RC= get_baryonic_RC(radii,total_RC,derived_RCs,\
                                            function_variable_settings,V_total_error=total_RC_err)

    if debug:
        print_log(f'We will use the following initial RCs for the mass decomposition',log)
        print_log(f'''Radi = {radii}
total RC = {total_RC[0]}, error = {total_RC[1]}''',log)
        for key in Baryonic_RC:
            print_log(f'{key} = {Baryonic_RC[key]}',log)
    # Then we check for which parameters we actaully have an RC
    for component in disk_var:
        if component in function_variable_settings and component not in Baryonic_RC:
            function_variable_settings.pop(component)
    if debug:
        print_log(f'We will use the following initial fitting settings',log)
        for key in function_variable_settings:
            print_log(f'{key} = {function_variable_settings[key]}',log)

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
for the {rotmass_settings} DM halo the parameters are {','.join([key for key in function_parameters if key not in disk_var])}
''',log,screen=True)
        exit()
        #Start GUI with default settings or input yml settings. these are already in function_parameters
    else:
        plot_curves(f'{log_directory}/Input_Curves.pdf', current_curves)
        pass

        #Then download settings




    # calculate the initial guesses
    initial_variable_settings = initial_guess(full_curve,radii,total_RC,function_variable_settings,\
                            debug=debug,log=log)

    current_curves = calculate_curves(radii ,total_RC,Baryonic_RC,full_curve,\
                                        DM_curve,initial_variable_settings,disk_var)
    plot_curves(f'{log_directory}/Initial_Guess_Curves.pdf', current_curves)

    final_variable_fits = mcmc_run(full_curve,radii,total_RC,initial_variable_settings,\
                            out_dir = out_dir, debug=debug,log=log)
'''
    boundaries= {}
    for i,parameter in enumerate(free_sym):
        if str(parameter) != 'r':
            if rotmass_settings[str(parameter)][4] and rotmass_settings[str(parameter)][3]:
                if rotmass_settings[str(parameter)][1]:
                    min_in=rotmass_settings[str(parameter)][1]
                else:
                    min_in = initial[i-1]-3.*errors[i-1]
                if rotmass_settings[str(parameter)][2]:
                    max_in=rotmass_settings[str(parameter)][2]
                else:
                    max_in = initial[i-1]+3.*errors[i-1]
            if not rotmass_settings['negative_ML']:
                if min_in < 0.:
                    min = 0.
            boundaries[str(parameter)] = [min_in,max_in]

    results = mcmc_run(python_formula,radii[2:],total_RC[2:],total_RC_err[2:],free_sym,initial\
            ,errors,steps,burning,rotmass_settings,boundaries = boundaries,log_directory=log_directory,no_negative = no_negative,debug=debug,log=log)
'''
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
        fitM, fitV = symbols(f"{component} {disk_var[component]}")
        curve_sym = curve_sym +fitM*fitV*Abs(fitV)
        symbols_to_replace.append(fitV)
        # keep a record of the values we need to enter
        replace_dict[disk_var[component]]=Baryonic_RCs[component][0]
        values_to_be_replaced.append(Baryonic_RCs[component][0])
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

def calculate_curves(radii ,total_RC,Baryonic_RC,full_curve,DM_curve,function_variable_settings,disk_var):
    combined_RC = {'RADI': radii, 'V_Total': total_RC}
    for key in Baryonic_RC:
        combined_RC[disk_var[key]]=[function_variable_settings[key][0]*Baryonic_RC[key][0],Baryonic_RC[key][1]]

    dm_variables = [function_variable_settings[x][0] for x in DM_curve['variables'] if x != 'r']
    if any(x == None for x in dm_variables):
        pass
    else:
        #This is where we should calculate an error on the DM curve
        combined_RC['V_DM'] = [DM_curve['function'](radii,*dm_variables),np.zeros(len(radii))]
    full_variables = [function_variable_settings[x][0] for x in full_curve['variables'] if x != 'r']
    if any(x == None for x in full_variables):
        pass
    else:
        combined_RC['V_fit'] = [full_curve['function'](radii,*full_variables),np.zeros(len(radii))]
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


def fix_fixed_variables(fit_curve,fix_variables,rotmass_settings):
    variables = list(fit_curve.free_symbols)
    apply_curve = fit_curve
    for key in variables:
        if f"{key}" in fix_variables:
            apply_curve = apply_curve.subs(key,rotmass_settings[f"{key}"][0])
        if f"{key}" == 'G':
            apply_curve = apply_curve.subs(key,cons.Gpot)
    return apply_curve
fix_fixed_variables.__doc__ =f'''
 NAME:
    fix_fixed_variables

 PURPOSE:
    Replace variables that should not vary

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
def get_baryonic_RC(radii,total_RC,derived_RCs,variables,
                    V_total_error=[0.,0.,0.],baryonic_error= {'Empty':True},debug=False):
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

        if derived_RCs[x][0] == 'RADI':
            rad_in = np.array(derived_RCs[x][2:])
            component = 'Empty'
        elif derived_RCs[x][0][:3] == 'EXP' and 'MD' in variables:
            component = 'MD'
        elif (derived_RCs[x][0][:3] == 'SER' or derived_RCs[x][0][:3] == 'BUL') and 'MB' in variables:
            component = 'MB'
        elif derived_RCs[x][0][:6] == 'DISK_G' and 'MG' in variables:
            component = 'MG'
        else:
            if derived_RCs[x][0][:3] not in ['EXP','BUL','SER'] and derived_RCs[x][0][:6] != 'DISK_G':
                print("We do not recognize this type of RC and don't know what to do with it")
                exit()
        if component != 'Empty':
            if component not in RC:
                RC[component] = np.array(derived_RCs[x][2:])
                if x in baryonic_error:
                    RC[component] = [RC[component],np.array(baryonic_error[x][2:])]
                else:
                    RC[component] = [RC[component],np.zeros(len(RC[component]))]
            else:
                if x in baryonic_error:
                    RC[component][1] = np.array([np.sqrt(x*x_err/np.sqrt(x**2+y**2)+y*y_err/np.sqrt(x**2+y**2))\
                                     for x,y,x_err,y_err in zip(RC[component][0],derived_RCs[x][2:])],\
                                     RC[component][1],baryonic_error[x][2:])

                RC[component][0] = np.array([np.sqrt(x**2+y**2) for x,y in zip(RC[component][0],derived_RCs[x][2:])])



    # if our requested radii do not correspond to the wanted radii we interpolat
    for key in RC:
        if len(RC[key][0]) > 0.:
            if np.sum([float(x)-float(y) for x,y in zip(new_radii,rad_in)]) != 0.:
                RC[key][0] = np.array(np.interp(new_radii,rad_in,RC[key][0]),dtype=float)
                RC[key][1] = np.array(np.interp(new_radii,rad_in,RC[key][1]),dtype=float)
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
                  function_name='curve_np',debug =False,log=None):


    code = f'sum_of_square =lambda {",".join([x for x in fit_function["variables"] ])}: np.sum(((np.array([{", ".join([str(i) for i in total_RC[0]])}],dtype=float)-{function_name}({",".join([x for x in fit_function["variables"] ])}))/np.array([{", ".join([str(i) for i in total_RC[1]])}],dtype=float))**2)'
    exec(code,globals())

    succes_outer = False
    guess_variables = copy.deepcopy(function_variable_settings)
    #First set the model
    model = lmfit.Model(sum_of_square)
#    model = lmfit.Model(fit_function['function'])
    no_input = False
    for variable in guess_variables:
        if not function_variable_settings[variable][1]:
            if variable in ['MD','MB','MG']:
                guess_variables[variable][1] = function_variable_settings[variable][0]/10.
            else:
                guess_variables[variable][1] = 0.1

        if not function_variable_settings[variable][2]:
            if variable in ['MD','MB','MG']:
                guess_variables[variable][2] = function_variable_settings[variable][0]*20.
            else:
                guess_variables[variable][2] = 1000.
        if not function_variable_settings[variable][0]:
            no_input = True
            guess_variables[variable][0] = float(np.random.rand()*(guess_variables[variable][2]-guess_variables[variable][1])+guess_variables[variable][1])
        print_log(f'''Setting {variable} with value {guess_variables[variable][0]} and fitting = {guess_variables[variable][3]}.
limits are between {guess_variables[variable][1]} - {guess_variables[variable][2]}
''',log)
        model.set_param_hint(variable,value=guess_variables[variable][0],\
                    min=guess_variables[variable][1],\
                    max=guess_variables[variable][2],\
                    vary=guess_variables[variable][3]
                    )

    parameters= model.make_params()

    #if we are completely guessing we first get the global minimum
    #if no_input:
    #    with warnings.catch_warnings():
    #        warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
    #        initial_fit = model.fit(data=total_RC[0], params=parameters, r=radii,method='ampgo'\
    #                             , nan_policy='omit',scale_covar=False)
    #    for variable in parameters:
    #        if not function_variable_settings[variable][0]:
    #            parameters[variable].value=initial_fit.params[variable].value

    no_errors = True
    counter =0
    while no_errors:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
            initial_fit = model.fit(data=total_RC[0], params=parameters, r=radii,method='differential_evolution'\
                                     , nan_policy='omit',scale_covar=False)

            if not initial_fit.errorbars or not initial_fit.success:
                print(f"\r The initial guess did not produce errors, retrying with new guesses: {counter/float(500.)*100.:.1f} % of maximum attempts.", end =" ",flush = True)
                for variable in parameters:
                    if not function_variable_settings[variable][0]:
                        guess_variables[variable][0] = float(np.random.rand()*(guess_variables[variable][2]-guess_variables[variable][1])+guess_variables[variable][1])
                        parameters[variable].value =guess_variables[variable][0]
                counter+=1
                if counter > 501.:
                    raise InitialGuessWarning(f'We could not find errors and initial guesses for the function. Try smaller boundaries or set your initiial values')
            else:
                print(f'\n')
                print_log(f'The initial guess is a succes',log)
                for variable in guess_variables:
                    guess_variables[variable][1] = float(initial_fit.params[variable].value-10*\
                                            initial_fit.params[variable].stderr if not function_variable_settings[variable][1] \
                                            else initial_fit.params[variable].min)
                    guess_variables[variable][2] = float(initial_fit.params[variable].value+10*\
                                            initial_fit.params[variable].stderr if not function_variable_settings[variable][2]\
                                             else initial_fit.params[variable].max)

                    guess_variables[variable][0] = float(initial_fit.params[variable].value)
                no_errors = False
    print_log(f'''INITIAL_GUESS: These are the statistics and values found through differential evolution fitting of the residual.
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

def plot_curves(name,curves,interactive = False):
    styles = ['-','-.',':','--','-','-.',':','--']
    figure, ax = plt.subplots(nrows=1, ncols=1,figsize=(15,10))
    for i,rcs in enumerate(curves):
        if rcs != 'RADI':
            x=i
            while x >= len(styles):
                x -= len(styles)
            if np.sum(curves[rcs][1]) != 0.:
                ax.errorbar(curves['RADI'], curves[rcs][0], yerr=curves[rcs][1], ms=10,lw=5,alpha=1,
                 capsize=10,marker='o', barsabove=True,label=f"{str(rcs)}",linestyle = styles[x])
            else:
                ax.plot(curves['RADI'],curves[rcs][0],label=str(rcs),lw=5,linestyle = styles[x],marker='o', ms=10)
    plt.legend()
    if interactive:
        raise InputError(f"An interactive mode is not available yet, feel free to right a Gui. Here you can plot the curves")
    else:
        ax.set_xlabel('R(kpc)')
        ax.set_ylabel('V(km/s)')
        plt.savefig(name,dpi=300)


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

def set_fitting_parameters(rotmass_settings, disk_var):
    # Get the variables of the DM function
    dm_parameters = [str(x) for x in getattr(V, rotmass_settings['HALO'])().free_symbols if str(x) != 'r']
    #define a dictionary
    fitting_parameter = {}
    # then set the DM parameters
    for key in dm_parameters:
        fitting_parameter[key] = rotmass_settings[key]
        if not fitting_parameter[key][4]:
            raise  InputError(f'''You have requested the {rotmass_settings['HALO']} DM halo but want to exclude {key} from the fitting.
You cannot change the DM formula, if this is your aim please add a potential in the rotmass potential file.
If you merely want to fix the variable, set an initial guess and fix it in the input (e.g. rotmass.{key} = [100, null,null, False,True]). ''')

    for key in disk_var:
        if rotmass_settings[key][4]:
            fitting_parameter[key] = rotmass_settings[key]
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
def mcmc_run(fit_function,radii,total_RC,function_variable_settings,\
                  out_dir = None,log_directory=None,debug =False,log=None):
    steps=750*(len(fit_function['variables'])-1)
    burning=15
    #First set the model
    model = lmfit.Model(fit_function['function'])
    #then set the hints
    for variable in function_variable_settings:
            if function_variable_settings[variable][1] == function_variable_settings[variable][2]:
                function_variable_settings[variable][1] = function_variable_settings[variable][1]*0.9
                function_variable_settings[variable][2] = function_variable_settings[variable][2]*1.1

            print(f'''Setting {variable} with value {function_variable_settings[variable][0]}.
With the boundaries between {function_variable_settings[variable][1]} - {function_variable_settings[variable][2]}''')
            model.set_param_hint(variable,value=function_variable_settings[variable][0],\
                        min=function_variable_settings[variable][1],\
                        max=function_variable_settings[variable][2],
                        vary=function_variable_settings[variable][3])
    parameters = model.make_params()
    #with warnings.catch_warnings():
    #    warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
    #    result = model.fit(data=RC, params=p, r=radius,method=minimizer_use, nan_policy='omit',weights=1/RC_err)
    emcee_kws = dict(steps=steps, burn=burning, thin=10, is_weighted=True)
    #emcee_params = result.params.copy()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
        warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
        result_emcee = model.fit(data=total_RC[0], r=radii, params=parameters, method='emcee',nan_policy='omit',
                         fit_kws=emcee_kws,weights=1./total_RC[0])
    print_log(result_emcee.fit_report(),log,screen=False)


    if out_dir:
        lab = [x for x in result_emcee.params if function_variable_settings[x][3]]
        fig = corner.corner(result_emcee.flatchain, quantiles=[0.16, 0.5, 0.84],show_titles=True,
                        title_kwargs={"fontsize": 15},labels=lab)
        fig.savefig(f"{out_dir}MCMC_COV_Fits.pdf",dpi=300)
    print_log(f''' MCMC_RUN: We find the following parameters for this fit.''',log)
    for variable in function_variable_settings:
        if function_variable_settings[variable][3]:
            function_variable_settings[variable][1] = float(result_emcee.params[variable].value-\
                                    result_emcee.params[variable].stderr)
            function_variable_settings[variable][2] = float(result_emcee.params[variable].value+\
                                    result_emcee.params[variable].stderr)

            function_variable_settings[variable][0] = float(result_emcee.params[variable].value)
            print_log(f'''{variable} = {result_emcee.params[variable].value} +/- {result_emcee.params[variable].stderr} within the boundary {result_emcee.params[variable].min}-{result_emcee.params[variable].max}''',log)



    return function_variable_settings







#
