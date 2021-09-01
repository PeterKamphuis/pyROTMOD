# -*- coding: future_fstrings -*-

class InitialGuessWarning(Exception):
    pass

import numpy as np
import warnings
import copy
import traceback
import pyROTMOD.rotmass.potentials as V
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

def the_action_is_go(radii, derived_RCs, total_RC,total_RC_err,debug=False,interactive = False,rotmass_settings = None,log_directory=None,log=None):

    disk_type =['MB','MD','MG']
    Baryonic_RC = get_three_RC(radii,derived_RCs,types = disk_type)
    #Radius of 0. messses thing up so lets interpolate to the second point
    if radii[2] == 0.:
        radii[2]=radii[3]/2.
        total_RC[2]=(total_RC[2]+total_RC[3])/2.
        for rc in disk_type:
            if np.sum(Baryonic_RC[rc]) != 0.:
                Baryonic_RC[rc][0] = (Baryonic_RC[rc][0]+Baryonic_RC[rc][1])/2.
    Disk_Var = {'MD': 'Vdisk','MG': 'Vgas','MB': 'Vbulge' }

    for component in disk_type:
        if len(Baryonic_RC[component]) < 1:
            getattr(rotmass_settings,component)[2] = False
            #rotmass_settings[component][2] = False
    if interactive:
        #We want to bring up a GUI to allow for the fitting
        print_log(f'''The interactive function of fitting is not yet implemented.
''',log,screen=True)
        #Start GUI

        #Then download settings
    else:

        fit_curve,fix_variables,DM_RC = build_curve(rotmass_settings,types = disk_type)
        apply_curve = fix_fixed_variables(fit_curve,fix_variables,rotmass_settings)
        #make sure that r is always the first input on the function
        tmp_free_sym = [x for x in list(apply_curve.free_symbols) if str(x) not in ['r','Vgas','Vbulge','Vdisk'] ]
        tmp_free_sym.insert(0,symbols('r'))
        free_sym = copy.deepcopy(tmp_free_sym)
        to_be_replaced = []
        for input_RC in disk_type:
            if symbols(Disk_Var[input_RC]) in  apply_curve.free_symbols:
                tmp_free_sym.insert(0,symbols(Disk_Var[input_RC]))
                to_be_replaced.insert(0,Baryonic_RC[input_RC])

        initial_formula = lambdify(tmp_free_sym,apply_curve,"numpy")

        print_log(f'''THE_ACTION_IS_GO: We are fitting this formula:
{'':8s}{initial_formula.__doc__}
''',log,debug=debug,screen =True)

        python_formula = lambda *free_sym: initial_formula(*to_be_replaced, *free_sym)

    #print(python_formula.__doc__)
    #import inspect
    #lines = inspect.getsource(python_formula)
    #print(lines)
    #transfer the RCs without unit, must be KM/S by this point

    initial = initial_guess(python_formula,radii[2:],total_RC[2:],total_RC_err[2:],\
                            free_sym,Baryonic_RC,rotmass_settings,disk_type,DM_RC,\
                            log_directory=log_directory,debug=debug,log=log)

    mcmc_run(python_formula,initial,debug=debug,log=log)

def mcmc_run(fit_function,initial_guesses,debug=False,log=None):
    

'''
    if interactive:
        #We want to bring up a GUI to allow for the fitting
        print("Not functional yet")
    else:
        #First we get the DM function we want
        #DM_RC = get_DM_RC(rotmass_settings['HALO'])
        #Then we build the combined rotation curve
        fit_curve,fix_variables = build_curve(rotmass_settings,Baryonic_RC,types = type)
        print(fit_curve,fit_variable,fix_variables)
        print(fit_curve)
        #for key in dir(ML_ratios):
        #    print(f" for the {key} we start with M/L {getattr(ML_ratios,key)}")

        fit_curve = build_curve(rotmass_settings,DM_RC,Bulge_RC,Disk_RC,Gas_RC)
    return np.sqrt(NFW_h * NFW_h + Mg*V_gas*abs(V_gas)  + Md *V_disc*abs(V_disc) + Mb*V_bulge*abs(V_bulge) )
          return np.sqrt(NFW_h* NFW_h + Mg*V_gas*abs(V_gas)  + ML *V_disc*abs(V_disc) + Mb*V_bulge*abs(V_bulge) )

        print(DM_RC)
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
def build_curve(rotmass_settings,types = ['MG']):
    DM_curve = getattr(V, rotmass_settings['HALO'])()
    curve = DM_curve**2
    fixed_variables = []
    Disk_Var = {'MD': 'Vdisk','MG': 'Vgas','MB': 'Vbulge' }
    for component in types:
        if rotmass_settings[component][2]:
            fitM, fitV = symbols(f"{component} {Disk_Var[component]}")
            if not rotmass_settings.negative_ML:
                curve = curve +Abs(fitM)*fitV*Abs(fitV)
            else:
                curve = curve +fitM*fitV*Abs(fitV)
            #curve = curve +fitM*RC[component]*Abs(RC[component])
            if not rotmass_settings[component][1]:
                fixed_variables.append(component)
    #print(curve)
    curve = sqrt(curve)
    #curve.sub({MG:1.4})
    #print(curve)
    #G=symbols('G')
    #DM_curve=DM_curve.subs(G,cons.Gpot)
    return curve,fixed_variables,DM_curve

def get_three_RC(radii,derived_RCs,types =['MG']):
    radii=radii[2:]

    RC ={}
    for component in types:
        RC[component]=[]

    for x in range(len(derived_RCs)):
        if derived_RCs[x][0] == 'RADII':
            rad_in = derived_RCs[x][2:]
        elif derived_RCs[x][0][:3] == 'EXP':
            component = 'MD'
        elif derived_RCs[x][0][:3] == 'SER':
            component = 'MB'
        elif derived_RCs[x][0][:6] == 'DISK_G':
            component = 'MG'
        else:
            print("We do not recognize this type of RC and don't know what to do with it")
            exit()
        if len(RC[component]) < 1:
            RC[component] = derived_RCs[x][2:]
        else:
            RC[component] = [np.sqrt(x**2+y**2) for x,y in zip(RC[component],derived_RCs[x][2:])]

    # if our requested radii do not correspond to the wanted radii we interpolat


    for key in RC:
        if np.sum([float(x)-float(y) for x,y in zip(radii,rad_in)]) != 0.:
            RC[key] = np.array(np.interp(np.array(radii),np.array(rad_in),np.array(RC[key])),dtype=float)
        else:
            RC[key] = np.array(RC[key],dtype=float)

    return RC


#Written by Aditya K.
def initial_guess(fit_function,xData,yData,err,free_sym,Baryonic_RC,rotmass_settings,disk_types,DM_RC,log_directory=None,debug =False,log=None):
    #parameterTuple = [str(x) for x in free_sym if str(x) != 'r']
    #print(*parameterTuple)
    #print(xData,yData,err)
    xData = np.array(xData)
    yData=  np.array(yData)
    err =  np.array(err)
    sum_of_square =lambda parameterTuple: np.sum(((yData-fit_function(xData, *parameterTuple))/err)**2)
    succes_outer = False
    upper_bounds =1000.
    parameterBounds = [[0.1, upper_bounds] for x in free_sym if str(x) != 'r']

    for types in disk_types:
        if symbols(types) in free_sym:
            parameterBounds[free_sym.index(symbols(types))-1] = [rotmass_settings[types][0]/10.,rotmass_settings[types][0]*20.]
    while not succes_outer:
        if debug:
            print_log(f'''INITIAL_GUESS: We are doing a curve fit and differential evolution with these boundaries:
{'':8s} {','.join([f'{x}={y}' for x,y in zip(free_sym[1:],parameterBounds)] )}
''',log,debug=True)
        initial_guess =  differential_evolution(sum_of_square, parameterBounds)
        with warnings.catch_warnings():
            warnings.filterwarnings("error",category = OptimizeWarning)
            maxfev = 1000
            succes = False
            while not succes:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="invalid value encountered in log")
                        warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
                        fitted_parameters, pcov = curve_fit(fit_function, xData, yData,initial_guess.x,maxfev=maxfev )
                    succes_outer= True
                    succes = True
                except OptimizeWarning:
                    maxfev += 1000
                except RuntimeError:
                    maxfev += 1000
                if maxfev > 1e5:
                    upper_bounds -= 100.
                    if upper_bounds > 200.:
                        succes = True
                        for i,set_bounds in enumerate(parameterBounds):
                            if set_bounds[1] == upper_bounds+100.:
                                parameterBounds[i]=[0.1,upper_bounds]
                    else:
                        print_log(f'''INITIAL_GUESS: We cannot fit the initial guesses and thus we will probably run into trouble
''',log,debug=debug,screen=True)
                        raise InitialGuessWarning("We failed to obtain the initial guesses")
    for types in disk_types:
        if symbols(types) in free_sym:
            rotmass_settings[types][0]=float(fitted_parameters[free_sym.index(symbols(types))-1])

    fitted_errors = [np.sqrt(pcov[j,j]) for j in range(fitted_parameters.size)]
    model_predictions = fit_function(xData, *fitted_parameters)
    prediction_errors = (model_predictions - yData)/err
    squared_prediction_errors = np.square(prediction_errors)
    mean_sp_errors = np.mean(squared_prediction_errors)
    mean_prediction_errors = np.sqrt(mean_sp_errors)
    Rsquared = 1.0 - (np.var(prediction_errors) / np.var(yData))

    chi_square = np.sum(((yData - model_predictions)/err)**2)
    reduced_chi_square = chi_square/(len(xData)-len(fitted_parameters))
    print_log(f'''INITIAL_GUESS: These are the statistics and values found through differential evolution fitting using a {rotmass_settings['HALO']} DM halo.
{'':8s}{','.join([f'{str(para)} = {val:.3f}' for para,val in zip(free_sym[1:],initial_guess.x) ])}
{'':8s}These are the statistics and values found through linear regression fitting using a {rotmass_settings['HALO']} DM halo.
{'':8s}{','.join([f'{str(para)} = {val:.3f} +/- {val_err:.3f}' for para,val,val_err in zip(free_sym[1:],fitted_parameters,fitted_errors) ])}
{'':8s}Mean Squared Error= {mean_sp_errors}
{'':8s}Root Mean Squared Error= {mean_prediction_errors}
{'':8s}Reduced Chi-Square= {reduced_chi_square}
''',log,screen=True)

    fitted_DM_curve = copy.deepcopy(DM_RC)
    for i,key in enumerate(free_sym):
        if str(key) not in ['r','MG','MD','MG']:
            fitted_DM_curve=fitted_DM_curve.subs(key,fitted_parameters[i-1])
        else:
            pass

    python_DM_curve =  lambdify(free_sym[0],fitted_DM_curve,"numpy")

    initial_DM_curve = python_DM_curve(xData)

    collected_curves = {'V_Total':[yData,err],
                        'V_DM':[initial_DM_curve,[]],
                        'V_Model_Total':[fit_function(xData,*fitted_parameters),[]]

     }

    for types in disk_types:
        if rotmass_settings[types][2]:
            collected_curves[f'V_{types}']=[Baryonic_RC[types]*np.sqrt(np.abs(rotmass_settings[types][0])),[]]

    if debug:
        fitted_DM_curve = copy.deepcopy(DM_RC)
        for i,key in enumerate(free_sym):
            if str(key) not in ['r','MG','MD','MB']:
                fitted_DM_curve=fitted_DM_curve.subs(key,initial_guess.x[i-1])
            else:
                pass

        python_DM_curve =  lambdify(free_sym[0],fitted_DM_curve,"numpy")
        initial_DM_curve = python_DM_curve(xData)
        collected_curves['V_DM_diff']= [initial_DM_curve,[]]
        diff_MD = rotmass_settings['MD'][0]
        for types in disk_types:
            if rotmass_settings[types][2]:
                collected_curves[f'V_{types}_diff']=[Baryonic_RC[types]*np.sqrt(np.abs(initial_guess.x[free_sym.index(symbols(types))-1])),[]]
        collected_curves['V_diff_Total']=[fit_function(xData,*initial_guess.x),[]]
    new_para =[]
    for key in free_sym:
        if str(key) != 'r':
            new_para.append(fitted_parameters[free_sym.index(key)-1])

    new_para = np.array(new_para)

    plot_curves(f'{log_directory}/Initial_Guesses.pdf', xData,collected_curves)
    return fitted_parameters




def plot_curves(name,radius,curves):
    styles = ['-','-.',':','--','-','-.',':','--']
    figure, ax = plt.subplots(nrows=1, ncols=1,figsize=(15,10))
    for i,rcs in enumerate(curves):
        if np.sum(curves[rcs][1]) != 0.:
            ax.errorbar(radius, curves[rcs][0], yerr=curves[rcs][1], ms=10, color='black',lw=5,alpha=1, capsize=10,marker='o', barsabove=True,label=f"{str(rcs)}_err")
        x=i
        while x >= len(styles):
            x -= len(styles)
        ax.plot(radius,curves[rcs][0],label=str(rcs),lw=5,linestyle = styles[x])
    plt.legend()
    ax.set_xlabel('R(kpc)')
    ax.set_ylabel('V(km/s)')
    plt.savefig(name,dpi=300)
#
