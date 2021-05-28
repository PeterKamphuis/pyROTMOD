# -*- coding: future_fstrings -*-



import numpy as np
import pyROTMOD.rotmass.potentials as V
from sympy import symbols, sqrt,atan,pi,log,Abs,lambdify

def the_action_is_go(radii, derived_RCs, total_RC,total_RC_err,debug=False,interactive = False,config = None):
    if not config:
            #let's make a dictionary with default settings
            # for masses of disk [Initial M/L, Fixed, Include]
            config = { 'MG': [1.4, True,True],
                       'MD': [1., True,True],
                       'MB': [1., False,False],
                       'HALO': 'NFW'
            }


    type =['MB','MD','MG']
    Baryonic_RC = get_three_RC(radii,derived_RCs,types = type)

    for component in type:
        if len(Baryonic_RC[component]) < 1:
            config[component][2] = False

    fit_curve,fix_variables = build_curve(config,Baryonic_RC,types = type)

    apply_curve = fix_fixed_variables(fit_curve,fix_variables,config)
    print(apply_curve)
    python_formula = lambdify(list(apply_curve.free_symbols),apply_curve,"numpy")
    print(python_formula)

'''
    if interactive:
        #We want to bring up a GUI to allow for the fitting
        print("Not functional yet")
    else:
        #First we get the DM function we want
        #DM_RC = get_DM_RC(config['HALO'])
        #Then we build the combined rotation curve
        fit_curve,fix_variables = build_curve(config,Baryonic_RC,types = type)
        print(fit_curve,fit_variable,fix_variables)
        print(fit_curve)
        #for key in dir(ML_ratios):
        #    print(f" for the {key} we start with M/L {getattr(ML_ratios,key)}")

        fit_curve = build_curve(config,DM_RC,Bulge_RC,Disk_RC,Gas_RC)
    return np.sqrt(NFW_h * NFW_h + Mg*V_gas*abs(V_gas)  + Md *V_disc*abs(V_disc) + Mb*V_bulge*abs(V_bulge) )
          return np.sqrt(NFW_h* NFW_h + Mg*V_gas*abs(V_gas)  + ML *V_disc*abs(V_disc) + Mb*V_bulge*abs(V_bulge) )

        print(DM_RC)
'''

def fix_fixed_variables(fit_curve,fix_variables,config):
    variables = list(fit_curve.free_symbols)
    apply_curve = fit_curve
    for key in variables:
        if f"{key}" in fix_variables:
            apply_curve = apply_curve.subs(key,config[f"{key}"][0])

    return apply_curve
def build_curve(config,RC,types = ['MG']):

    curve = getattr(V, config['HALO'])()**2
    fixed_variables = []
    Disk_Var = {'MD': 'Vdisk','MG': 'Vgas','MB': 'Vbulge' }
    for component in types:
        if config[component][2]:
            fitM, fitV = symbols(f"{component} {Disk_Var[component]}")
            curve = curve +fitM*fitV*Abs(fitV)
            if config[component][1]:
                fixed_variables.append(component)
    curve = sqrt(curve)
    #curve.sub({MG:1.4})
    #print(curve)
    return curve,fixed_variables

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






#
