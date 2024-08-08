# -*- coding: future_fstrings -*-


import numpy as np
import warnings
from pyROTMOD.support.errors import InputError,RunTimeError
import copy
import pyROTMOD.rotmass.potentials as potentials
import pyROTMOD.support.constants as cons
from pyROTMOD.support.classes import Rotation_Curve
from pyROTMOD.support.minor_functions import print_log,get_uncounted
from pyROTMOD.fitters.fitters import initial_guess,mcmc_run
from sympy import symbols, sqrt,lambdify

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors


def build_curve(all_RCs,total_RC,debug=False,log=None):
    # First set individual sympy symbols  and the curve for each RC

    ML, V = symbols('ML V')
    replace_dict = {'symbols': []}
    total_sympy_curve = None
    for name in all_RCs:
        RC_symbols = [x for x in list(all_RCs[name].curve.free_symbols) if str(x) != 'r']
        
        for symbol in RC_symbols:
            if symbol == V:
                V_replace = symbols(f'V_{all_RCs[name].name}')
                for attr in ['curve', 'individual_curve']:
                    setattr(all_RCs[name],attr,getattr(all_RCs[name],attr).subs({V: V_replace}))

                all_RCs[name].match_radii(total_RC)
                #Here we need to set it all to the radii of the total_RC else we can not match    
                replace_dict[f'V_{all_RCs[name].name}'] =   all_RCs[name].matched_values.value
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
        if  total_sympy_curve is None:
            total_sympy_curve =  all_RCs[name].curve**2
        else:
            total_sympy_curve += all_RCs[name].curve**2
 
   
    total_sympy_curve = sqrt(total_sympy_curve)

    #For the actual fit curve we need to replace the  V components with their actual values
    #make sure that r is always the first input on the function and we will replace the RCs

    curve_symbols_out = [x for x in list(total_sympy_curve.free_symbols) if str(x) not in ['r']+[str(y) for y in replace_dict['symbols']] ]
    curve_symbols_out.insert(0,symbols('r'))
    curve_symbols_in = replace_dict['symbols']+curve_symbols_out
   
    initial_formula = lambdify(curve_symbols_in, total_sympy_curve ,"numpy")

    print_log(f'''BUILD_CURVE:: We are fitting this complete formula:
{'':8s}{initial_formula.__doc__}
''',log,debug=debug,screen =True)
    # since lmfit is a piece of shit we have to constract or final formula through exec
    
    clean_code = create_formula_code(initial_formula,replace_dict, function_name='total_numpy_curve',debug=debug,log=log)
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



   

def calculate_red_chisq(RC):
    free_parameters= 0.
    for var in RC.fitting_variables:
        if RC.fitting_variables[var][3]:
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



'''
def create_disk_var(collected_RCs,single_stellar_ML=True,single_gas_ML=True):
    disk_var = {}
    counters = {'GAS': 1, 'DISK': 1, 'BULGE': 1} 
    for RC in collected_RCs:
        key = collected_RCs[RC].name
        if key != 'RADI':
            bare,no = get_uncounted(key)
            if collected_RCs[RC].component in ['stars']:
                if bare in ['EXPONENTIAL','SERSIC']:
                    disk_var[key] = [f'Gamma_disk_{counters["DISK"]}',f'V_disk_{counters["DISK"]}']
                    counters['DISK'] += 1
                elif bare in ['HERNQUIST','BULGE']:
                    disk_var[key] = [f'Gamma_bulge_{counters["BULGE"]}',f'V_bulge_{counters["BULGE"]}']
                    counters['BULGE'] += 1
                if single_stellar_ML:
                    disk_var[key][0]='ML_stars'
            
            elif bare in ['DISK_GAS']:
                disk_var[key] = [f'Gamma_gas_{counters["GAS"]}',f'V_gas_{counters["GAS"]}']
                counters['GAS'] += 1
            elif bare in ['HERNQUIST','BULGE']:
                disk_var[key] = [f'Gamma_bulge_{counters["BULGE"]}',f'V_bulge_{counters["BULGE"]}']
                counters['BULGE'] += 1
            if bare in ['EXPONENTIAL','SERSIC','HERNQUIST','BULGE']:
                if single_stellar_ML:
                    disk_var[key][0]='ML_optical'
            if bare in ['DISK_GAS']:
                if single_gas_ML: 
                    disk_var[key][0] = 'ML_gas'
    return disk_var
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
                if key != 'symbols':
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
'''
def get_baryonic_RC(radii,total_RC,derived_RCs,
                    V_total_error=[0.,0.,0.],baryonic_error= {'Empty':True},
                    log = None, debug=False,settings = None):
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
        if not settings is None and derived_RCs[x][0] != 'RADI':
            #if parameter 5 is set to false we do not want to include the rc
            if not settings[derived_RCs[x][0]][4]:
                continue
            
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
'''
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
        added = []
        for variable in final_variable_fits:
            if variable not in added:
                if final_variable_fits[variable][4]:
                    if 0.001 < result_emcee.params[variable].value < 10000.:
                        form = ">15.4f"
                    else:
                        form = ">15.4e"
                    variable_line = f'{variable:>15s} {result_emcee.params[variable].value:{form}}'
                    min = result_emcee.params[variable].min
                    max = result_emcee.params[variable].max
                    if not final_variable_fits[variable][3]:
                        variable_line += f' {"Fixed":>15s} {min:{form}} {max:{form}} \n'
                    else:
                        err = result_emcee.params[variable].stderr
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

        if RC.component == 'All':
            style_dictionary['color'] = colors.to_rgba('r',alpha=1.)
            style_dictionary['label'] =  r'V$_{Total}$'
            style_dictionary['zorder'] = 6
            style_dictionary['markerfacecolor'] = colors.to_rgba(style_dictionary['color'],alpha=0.5)
            style_dictionary['markeredgecolor'] = colors.to_rgba(style_dictionary['color'],alpha=0.75)

        elif RC.component == 'DM':
            style_dictionary['linestyle'] = '-.'
            style_dictionary['color'] = colors.to_rgba('b',alpha=1.)
            style_dictionary['label'] =  f'V$_{{{RC.halo}}}$'
            style_dictionary['zorder'] = 6
        elif RC.component == 'gas':
            rcs,no = get_uncounted(RC.name)
            style_dictionary['linestyle'] = ':'
            style_dictionary['color'] = colors.to_rgba('g',alpha=1.)
            style_dictionary['label'] = f'V$_{{Gas_{no}}}$'
            style_dictionary['zorder'] = 4
        elif RC.component == 'stars':
            rcs,no = get_uncounted(RC.name)
            style_dictionary['linestyle'] = '--'
            if rcs in ['SERSIC','EXPONENTIAL','DISK','SERSIC_DISK']:
                style_dictionary['color'] = colors.to_rgba('cyan',alpha=1.)
                style_dictionary['linestyle'] = '--'
                style_dictionary['label'] =  f'V$_{{Disk_{no}}}$'
                style_dictionary['zorder'] = 3
            if rcs in ['HERNQUIST','BULGE','SERSIC_BULGE']:
                style_dictionary['color'] = colors.to_rgba('purple',alpha=1.)
                style_dictionary['label'] = f'V$_{{Bulge_{no}}}$'
                style_dictionary['zorder'] = 2
        else:
            raise InputError(f'The component {RC.component} in {RC.name} is not a component familiar to us')
        if not RC.component == 'All':    
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




def rotmass_main(baryonic_RCs, total_RC,no_negative =True,out_dir = None,\
                interactive = False,rotmass_settings = None,log_directory=None,\
                rotmass_parameter_settings = None,\
                results_file = 'Final_Results',log=None,debug = False, font = 'Times New Roman'):

    # First combine all RCs that need to be included in the total fit in a single dictionary
    # With their parameters and individual RC curves set 
    all_RCs = set_fitting_parameters(rotmass_settings,rotmass_parameter_settings,\
                                    baryonic_RCs,total_RC)
    screen = False
    if debug:
        screen =True
    for names in all_RCs:
        print_log(f'For {names} we find the following parameters and fit variables:',log, screen =screen)
        for attr in vars(all_RCs[names]):
            print_log(f'{"":8s} {attr} = {getattr(all_RCs[names],attr)}',log, screen=screen)

    # Construct the function to be fitted, note that the actual fit_curve is
    build_curve(all_RCs,total_RC,debug=debug,log=log)
  
       
                        
    if interactive:
        #We want to bring up a GUI to allow for the fitting
        print_log(f'''Unfortunately the interactive function of fitting is not yet implemented. Feel free to write a GUI.
For now you can set, fix and apply boundaries in the yml input file.
for your current settings the variables are {','.join(total_RC.numpy_curve['variables'])}
''',log,screen=True)
        exit()
        #Start GUI with default settings or input yml settings. these are already in function_parameters
    else:

        plot_curves(f'{out_dir}/{results_file}_Input_Curves.pdf', all_RCs,\
            total_RC,font= font)

    
    # calculate the initial guesses
    initial_guesses, original_settings = initial_guess(total_RC,debug=debug,log=log,\
            negative=rotmass_settings.negative_values,\
            minimizer = rotmass_settings.initial_minimizer)
    update_RCs(initial_guesses,all_RCs,total_RC) 
    
    plot_curves(f'{out_dir}/{results_file}_Initial_Guess_Curves.pdf',\
        all_RCs,total_RC,font=font)
  
    
    variable_fits,emcee_results = mcmc_run(total_RC,original_settings,\
                            out_dir = out_dir, debug=debug,log=log,\
                            negative=rotmass_settings.negative_values,\
                            steps=rotmass_settings.mcmc_steps,
                            results_name= results_file)
    update_RCs(variable_fits,all_RCs,total_RC) 
    
    print_log('Plotting and writing',log)
    plot_curves(f'{out_dir}/{results_file}_Final_Curves.pdf', \
        all_RCs,total_RC,font=font)      
    
    red_chisq = calculate_red_chisq(total_RC)

    write_output_file(variable_fits,emcee_results,output_dir=out_dir,\
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


def add_fitting_dict(name, parameters, component_type = 'stars', fitting_dictionary = {}):
    variable = None
    #V_disk and V_Bulge are place holders for the values to be inserted in the final formulas
    base,number = get_uncounted(name)
    if base in ['EXPONENTIAL','DISK','DISK_GAS']:
        if component_type == 'stars':
            variable  = f'Gamma_disk_{number}'
        elif component_type == 'gas':
            variable  = f'Gamma_gas_{number}'
    elif base in ['HERNQUIST','BULGE'] and component_type == 'stars':
        variable = f'Gamma_bulge_{number}'

    if variable is None:
        variable = name
    fitting_dictionary[variable] = parameters
   
def set_fitting_parameters(rotmass_settings,fit_settings, baryonic_RCs,total_RC):
    # Get the variables of the DM function
    dm_parameters = []
    no_dm = False
    baryonic = []
    total_RC.fitting_variables = {}
    for x in getattr(potentials, rotmass_settings['HALO'])().free_symbols: 
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
        add_fitting_dict(all_RCs[name].name,fit_settings[all_RCs[name].name],\
                         component_type=all_RCs[name].component,\
                        fitting_dictionary=fitting_dictionary)
        if no_dm:
            all_RCs[name].halo = rotmass_settings['HALO']
            all_RCs[name].curve = getattr(potentials, rotmass_settings['HALO'])()       
            all_RCs[name].individual_curve = getattr(potentials, f"{rotmass_settings['HALO']}_INDIVIDUAL")()
            for variable in dm_parameters:
                # Using a dictionary make the parameter always to be added
                fitting_dictionary[variable] = fit_settings[variable]
        else:
            ML, V = symbols(f"ML V")
            all_RCs[name].halo = 'NEWTONIAN'
            all_RCs[name].curve = sqrt(ML*V*abs(V))
            all_RCs[name].individual_curve = V/abs(V)*sqrt(ML*V**2)
        

  

        all_RCs[name].fitting_variables= fitting_dictionary 
        all_RCs[name].check_unified(rotmass_settings.single_stellar_ML,\
                                    rotmass_settings.single_gas_ML)
      
        total_RC.fitting_variables.update(all_RCs[name].fitting_variables)
      
    if not no_dm:
        #We need add the DM RC and the parameters
        all_RCs[rotmass_settings['HALO']] = Rotation_Curve(component='DM',\
            name=rotmass_settings['HALO'])
        fitting_dictionary = {} 
        for variable in dm_parameters:
            # Using a dictionary make the parameter always to be added
            fitting_dictionary[variable] = fit_settings[variable]
            if not bool(fit_settings[variable][4]):
                raise  InputError(f'''You have requested the {rotmass_settings['HALO']} DM halo but want to exclude {variable} from the fitting.
You cannot change the DM formula, if this is your aim please add a potential in the rotmass potential file.
If you merely want to fix the variable, set an initial guess and fix it in the input (e.g. rotmass.{variable} = [100, null,null, False,True]). ''')
        all_RCs[rotmass_settings['HALO']].fitting_variables = fitting_dictionary
        all_RCs[rotmass_settings['HALO']].radii = total_RC.radii
        all_RCs[rotmass_settings['HALO']].values = np.zeros(len(total_RC.radii))*\
                                                    total_RC.values.unit
        
        all_RCs[rotmass_settings['HALO']].halo = rotmass_settings['HALO']
        all_RCs[rotmass_settings['HALO']].curve = getattr(potentials, rotmass_settings['HALO'])()
        #Thewre are no negatives for the DM supossedly so the individual curve is the same
        all_RCs[rotmass_settings['HALO']].individual_curve = getattr(potentials, rotmass_settings['HALO'])()
        total_RC.fitting_variables.update(all_RCs[rotmass_settings['HALO']].fitting_variables)

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
