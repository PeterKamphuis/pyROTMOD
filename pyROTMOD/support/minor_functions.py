# -*- coding: future_fstrings -*-
# This file should only import Errors from pyROTMOD but nothing else 
# such that it can be imported every where without singular imports
# any function that imports anywhere else from pyROTMOD should be in major_functions


import copy
import numpy as np
import os
import warnings
import sys
import pyROTMOD

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as mpl_fm

from astropy import units as u
from omegaconf import OmegaConf
from datetime import datetime
from pyROTMOD.support.errors import UnitError,InputError,SupportRunError,\
    RunTimeError
from pyROTMOD.support.log_functions import print_log
def add_font(file):
    try:
        mpl_fm.fontManager.addfont(file)
        font_name = mpl_fm.FontProperties(fname=file).get_name()
    except FileNotFoundError:
        font_name = 'DejaVu Sans'
    return font_name



#Check wether a variable is a unit quantity and if not multiply with the supplied unit
# unlees is none
def check_quantity(value):
    if not isinstance(value,u.quantity.Quantity):
        if isiterable(value):
        #if it is an iterable we make sure it it is an numpy array
            if not isinstance(value,np.ndarray) and not value is None:
                value = quantity_array(value)
        if not value is None and not isinstance(value,u.quantity.Quantity):
            raise UnitError(f'This value {value} is unitless it shouldnt be')
    return value


def isquantity(value):
    verdict= True
    if not isinstance(value,u.quantity.Quantity):
        if isiterable(value):
        #if it is an iterable we make sure it it is an numpy array
            if not isinstance(value,np.ndarray) and not value is None:
                value = quantity_array(value)
            else:
                verdict = False
        else:
            verdict = False
       
    return verdict


def create_directory(directory,base_directory,debug=False):
    split_directory = [x for x in directory.split('/') if x]
    split_directory_clean = [x for x in directory.split('/') if x]
    split_base = [x for x in base_directory.split('/') if x]
    #First remove the base from the directory but only if the first directories are the same
    if split_directory[0] == split_base[0]:
        for dirs,dirs2 in zip(split_base,split_directory):
            if dirs == dirs2:
                split_directory_clean.remove(dirs2)
            else:
                if dirs != split_base[-1]:
                    raise InputError(f"You are not arranging the directory input properly ({directory},{base_directory}).")
    for new_dir in split_directory_clean:
        if not os.path.isdir(f"{base_directory}/{new_dir}"):
            os.mkdir(f"{base_directory}/{new_dir}")
        base_directory = f"{base_directory}/{new_dir}"
create_directory.__doc__ =f'''
 NAME:
    create_directory

 PURPOSE:
    create a directory recursively if it does not exists and strip leading directories when the same fro the base directory and directory to create

 CATEGORY:
    support

 INPUTS:
    directory = string with directory to be created
    base_directory = string with directory that exists and from where to start the check from

 OPTIONAL INPUTS:
    debug = False

 OUTPUTS:

 OPTIONAL OUTPUTS:
    The requested directory is created but only if it does not yet exist

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''






# function for converting kpc to arcsec and vice versa

def convertskyangle(angle, distance=1., unit='arcsec', distance_unit='Mpc', \
                    physical=False,debug = False, quantity= False,cfg=None):

    if quantity:
        try:
            angle = angle.to(u.kpc)
            unit = 'kpc'
            if not physical:
                raise InputError(f'CONVERTSKYANGLE: {angle} is a distance but you claim it is a sky angle.\n')
        except u.UnitConversionError:
            if physical:
                raise InputError(f'CONVERTSKYANGLE: {angle} is sky angle but you claim it is a distance.\n')
            angle = angle.to(u.arcsec)
            unit='arcsec'
        angle = angle.value
        distance = distance.to(u.Mpc)
        distance = distance.value
        distance_unit= 'Mpc'

    if debug:
            print_log(f'''CONVERTSKYANGLE: Starting conversion from the following input.
    {'':8s}Angle = {angle}
    {'':8s}Distance = {distance}
''',cfg,case=['debug_add'])
       
    try:
        _ = (e for e in angle)
    except TypeError:
       
        angle = [angle]

        # if physical is true default unit is kpc
    angle = np.array(angle)
    if physical and unit == 'arcsec':
        unit = 'kpc'
    if distance_unit.lower() == 'mpc':
        distance = distance * 10 ** 3
    elif distance_unit.lower() == 'kpc':
        distance = distance
    elif distance_unit.lower() == 'pc':
        distance = distance / (10 ** 3)
    else:
        print('CONVERTSKYANGLE: ' + distance_unit + ' is an unknown unit to convertskyangle.\n')
        print('CONVERTSKYANGLE: please use Mpc, kpc or pc.\n')
        raise SupportRunError('CONVERTSKYANGLE: ' + distance_unit + ' is an unknown unit to convertskyangle.')
    if not physical:
        if unit.lower() == 'arcsec':
            radians = (angle / 3600.) * ((2. * np.pi) / 360.)
        elif unit.lower() == 'arcmin':
            radians = (angle / 60.) * ((2. * np.pi) / 360.)
        elif unit.lower() == 'degree':
            radians = angle * ((2. * np.pi) / 360.)
        else:
            print('CONVERTSKYANGLE: ' + unit + ' is an unknown unit to convertskyangle.\n')
            print('CONVERTSKYANGLE: please use arcsec, arcmin or degree.\n')
            raise SupportRunError('CONVERTSKYANGLE: ' + unit + ' is an unknown unit to convertskyangle.')


        kpc = 2. * (distance * np.tan(radians / 2.))
        if quantity:
            kpc = kpc*u.kpc
    else:
        if unit.lower() == 'kpc':
            kpc = angle
        elif unit.lower() == 'mpc':
            kpc = angle / (10 ** 3)
        elif unit.lower() == 'pc':
            kpc = angle * (10 ** 3)
        else:
            print('CONVERTSKYANGLE: ' + unit + ' is an unknown unit to convertskyangle.\n')
            print('CONVERTSKYANGLE: please use kpc, Mpc or pc.\n')
            raise SupportRunError('CONVERTSKYANGLE: ' + unit + ' is an unknown unit to convertskyangle.')

        radians = 2. * np.arctan(kpc / (2. * distance))
        kpc = (radians * (360. / (2. * np.pi))) * 3600.
        if quantity:
            kpc = kpc*u.arcsec
    if len(kpc) == 1:
        if not quantity:
            kpc = float(kpc[0])
        else:
            kpc = float(kpc[0].value)*kpc.unit

    return kpc

def check_input(cfg, fitting=False):
    'Check various input values to avoid problems later on.'
    #check the slashes and get the ininitial dir for the output dir
    if cfg.output.output_dir[-1] != '/':
        cfg.output.output_dir = f"{cfg.output.output_dir}/"
    if cfg.output.output_dir[0] == '/':
        first_dir= cfg.output.output_dir.split('/')[1]
    else:
        first_dir= cfg.output.output_dir.split('/')[0]
    #check the slashes and get the ininitial dir for the logging dir
    if cfg.output.log_directory[-1] != '/':
        cfg.output.log_directory = f"{cfg.output.log_directory}/"
    if cfg.output.log_directory[0] == '/':
        cfg.output.log_directory = cfg.output.log_directory[1:]
    first_log= cfg.output.log_directory.split('/')[0]

    # If the first directories in the log dir and the output dir are not the same
    # we assume the log dir should start at the output dir
    if first_dir != first_log:
        cfg.output.log_directory = f'{cfg.output.output_dir}{cfg.output.log_directory}'
    else:
        cfg.output.log_directory = f'/{cfg.output.log_directory}'
    #Create the directories if non-existent   
    if not fitting:
        if not os.path.isdir(cfg.output.output_dir):
            os.mkdir(cfg.output.output_dir)
        create_directory(cfg.output.log_directory,cfg.output.output_dir)
        
        log = f"{cfg.output.log_directory}{cfg.output.log}"
        #If it exists move the previous Log
        if os.path.exists(log):
            os.rename(log,f"{cfg.output.log_directory}/{cfg.output.out_base}_Previous_Log.txt")

        #Start a new log
        print_log(f'''This file is a log of the modelling process run at {datetime.now()}.
This is version {pyROTMOD.__version__} of the program.
''',cfg,case=['main'])

        # check the input files
        if not cfg.RC_Construction.gas_file and not cfg.RC_Construction.optical_file \
            and cfg.RC_Construction.enable:
            print(f'''You did not set the gas file input or the optical file input.
    We cannot model the profiles without these inputs.''')
            if not cfg.RC_Construction.gas_file:
                cfg.RC_Construction.gas_file = input('''Please add the gas file or tirific output to be evaluated: ''')
            if not cfg.RC_Construction.optical_file:
                cfg.RC_Construction.optical_file = input('''Please add the optical file or galfit output to be evaluated: ''')
        if not cfg.RC_Construction.gas_file is None:
            print_log(f'''We are using the input from {cfg.RC_Construction.gas_file} for the gaseous component.
''',cfg,case=['debug_add'])
            if cfg.RC_Construction.gas_file.split('.')[1].lower() == 'def' and \
                cfg.RC_Construction.gas_scaleheight[1] is None\
                and cfg.RC_Construction.enable:
                cfg.RC_Construction.gas_scaleheight = [0.,None,'ARCSEC','tir']
            
        if not cfg.RC_Construction.optical_file is None:
            print_log(f'''We are using the input from {cfg.RC_Construction.optical_file} for the optical component.
''',cfg,case=['debug_add'])
    
        if cfg.input.distance is None:
            raise InputError(f'We cannot model profiles adequately without a proper distance')
        
        print_log(f'''We are using the following distance = {cfg.input.distance}.
''',cfg,case=['main'])
    # return the cfg and log name

    #write the input to the log dir.
   
    name = f'{cfg.output.log_directory}/{cfg.output.out_base}_input'
    if fitting:
        name += '_RC_fitting'
    else:
        name += '_RC_construction'
    name +='.yml'   
    with open(name,'w') as input_write:
        input_write.write(OmegaConf.to_yaml(cfg))
       

    return cfg
check_input.__doc__ =f'''
 NAME:
    check_input(cfg)
 PURPOSE:
    Handle a set check parameters on the input omega conf

 CATEGORY:
    support_functions

 INPUTS:
    cfg = input omega conf object

 OPTIONAL INPUTS:
    debug = False

 OUTPUTS:
    checked and modified object and the name of the log file

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:


 NOTE:

'''

def get_correct_label(par,no,exponent = 0.):
    '''Get the correct label for the parameter'''
    split_parameter = par.split('_')
    morph_type ='unknown'
    component_type = 'unknown'
    log = False
    if len(split_parameter) > 1:
        if split_parameter[0] in ['effective','R','length','hern']:
            pass
        else:
            par = split_parameter[0]
            morph_type = split_parameter[1]
            if len(split_parameter) > 2:
                component_type = split_parameter[2]
    if par[0:2] == 'lg':
        par = par[2:]
        log = True
    #Need to use raw strings here to avoid problems with \
    label_dictionary = {'Gamma':[f'$\\mathrm{{M/L_{{{morph_type}}}}}$ {no}', ''],
                        'ML': [f'$\\mathrm{{M/L_{{{morph_type}}}}}$',''],
                        'RHO': [r'$\mathrm{\rho_{c}}$',r'$\mathrm{(M_{\odot} \,\, pc^{-3})}$'],
                        'RHO0': [r'$\mathrm{\rho_{c}}$',r'$\mathrm{(M_{\odot} \,\, pc^{-3})}$'],
                        'R_C': [r'$ \mathrm{R_{c}}$','(kpc)'],
                        'C':[r'C',''],
                        'R200':[r'$ \mathrm{R_{200}}$','(kpc)'],
                        'm': [r'Axion Mass','(eV)'],
                        'central': [r'Central SBR',r'$\mathrm{(M_{\odot}\,\,pc^{-2})}$'],
                        'h': [r'Scale Length','(kpc)'],
                        'Ltotal':   [r'Total Luminosity',r'$\mathrm{(L_{\odot})}$'],    
                        'mass': [r'Total Mass', r'$\mathrm{(M_{\odot})}$'],
                        'hern_length': ['Hernquist scale length','(kpc)}$'],
                        'effective_luminosity': [r'$\mathrm{L_{e}}$',r'$\mathrm{(M_{\odot})}$'] ,
                        'effective_radius': [r'$\mathrm{R_{e}}$','(kpc)'] ,
                        'n': [r'Sersic Index',''],
                        'a0': [r'$\mathrm{a_{0}}$',r'$\mathrm{(cm\,\,s^{-2})}$'],
                        'amplitude': [r'GP Amplitude',''],
                        'length_scale': [r'GP Length Scale','(RC step)'],
                        }
    if par in label_dictionary:
        string = label_dictionary[par][0] 
        if abs(exponent) >= 1.:
            string += f'$\\times10^{{{exponent}}}$' 
        if log:
            string = f'log10({string})'
            if label_dictionary[par][1] != '':    
                string += f'log10({label_dictionary[par][1]})'
        else:
            string += f'{label_dictionary[par][1]}'

    else:
        print(f''' The parameter {par} has been stripped
Unfortunately we can not find it in the label dictionary.''')
        string = f'{"_".join(split_parameter)}'
       
    return string   

def get_output_name(cfg,profile_name =None,function_name = None):
    name = f'{cfg.output.out_base}_{cfg.fitting_general.backend}'
    if cfg.fitting_general.use_gp:
        name += '_GP'
    if profile_name is not None:
        name += f'_{profile_name}' 
    if function_name is not None:
        name += f'_{function_name}'   
    return name
'''Stripe any possible _counters from the key'''
def get_uncounted(key):
    number = None
    try:
        gh = int(key[-1])
        splitted = key.split('_')
        if len(splitted) == 1:
            component = key
        else:
            component = '_'.join([x for x in splitted[:-1]])
            try:
                int(splitted[-1])
                number = splitted[-1]
            except ValueError:
                component = key
    except ValueError:
        component = key
    if number is None:
        number = '1'    
       
    return component,number


def quantity_array(list,unit):
    #Because astropy is coded by nincompoops Units to not convert into numpy arrays well.
    #It seems impossible to convert a list of Quantities into a quantity  with a list or np array
    #This means we have to pull some ticks when using numpy functions because they don't accept lists of Quantities
    # Major design flaw in astropy unit and one think these nincompoops could incorporate a function like this 
    #Convert a list of quantities into quantity with a numpy array
    return np.array([x.to(unit).value for x in list],dtype=float)*unit 

def integrate_surface_density(radii,density, log=None):
   
    ringarea= [0*u.kpc**2 if radii[0] == 0 else np.pi*((radii[0]+radii[1])/2.)**2]
     #Make sure the ring radii are in pc**2 to match the densities
    ringarea = quantity_array(ringarea +\
        [np.pi*(((y+z)/2.)**2-((y+x)/2.)**2) for x,y,z in zip(radii,radii[1:],radii[2:])]\
        +[np.pi*((radii[-1]+0.5*(radii[-1]-radii[-2]))**2-((radii[-1]+radii[-2])/2.)**2)]\
        ,u.pc**2)
    #Make sure the ring radii are in pc**2 to match the densities
   
    mass_array = [x*y for x,y in zip(ringarea,density)]
    #Convert to a proper np array quantity and sum
    mass =np.sum(quantity_array(mass_array,mass_array[0].unit))
    return mass,ringarea


def isiterable(variable):
    '''Check whether variable is iterable'''
    #First check it is not a string as those are iterable
    if isinstance(variable,str):
        return False
    try:
        iter(variable)
    except TypeError:
        return False

    return True
isiterable.__doc__ =f'''
 NAME:
    isiterable

 PURPOSE:
    Check whether variable is iterable

 CATEGORY:
    support_functions

 INPUTS:
    variable = variable to check

 OPTIONAL INPUTS:

 OUTPUTS:
    True if iterable False if not

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''


def plot_individual_profile(profile,min,max,ax,log = None,cfg=None):
    stellar_profile=False
    if not profile.values.unit in [u.Lsun/u.pc**2,u.Msun/u.pc**3,u.Msun/u.pc**2] :
        print_log(f'''The units of {profile.name} are not L_SOLAR/PC^2, M_SOLAR/PC^2 OR M_SOLAR/PC^3 .
Unit = {profile.values.unit}                  
Not plotting this profile.
''',cfg, case=['main'] )
        return max,stellar_profile
    if profile.radii.unit != u.kpc:
        print_log(f'''The units of {profile.name} are not KPC.
Not plotting this profile.
''',cfg, case=['main'])
        return max,stellar_profile
    
    lineout = ax.plot(profile.radii.value,profile.values.value, \
            label = profile.name)
    if np.nanmax(profile.values.value) > max:
        max =  np.nanmax(profile.values.value)
    if np.nanmin(profile.values[profile.values.value > 0.].value) < min:
        min =  np.nanmin(profile.values[profile.values.value > 0.].value)
    if profile.original_values is not None:
        if profile.original_values.unit == profile.values.unit:
           
            #plt.gca().twiny()
            secline = ax.plot(profile.radii.value,profile.original_values.value, \
                    label = f'Original {profile.name}')
            if np.nanmax(profile.original_values.value) > max:
                max =  np.nanmax(profile.original_values.value)
            lineout = lineout +secline
   
      
        if 'rejected' in profile.__dict__:
            for rej_ev in profile.rejected:
                secline = ax.plot(profile.radii.value, profile.rejected[rej_ev]['profile'], 
                            label = f'Rejected { profile.rejected[rej_ev]["name"]}',
                            linestyle='--',zorder=-2,alpha=0.5)
                lineout = lineout + secline   
    if not profile.component is None:
        if profile.component.lower() == 'stars':
            stellar_profile = True
    return min,max,stellar_profile,ax,lineout

def calculate_total_profile(total,profile):
    if len(total['Profile']) ==  0.:
            total['Profile'] = profile.values
            total['Radii'] = profile.radii
    else:
        # We checked the units when plotting the profile
        if np.array_equal(total['Radii'],profile.radii):
            total['Profile'] = np.array([x+y for x,y in \
                    zip(total['Profile'].value,profile.values.value)],dtype=float)*total['Profile'].unit
        else:
            #Else we interpolate to the lower resolution
            if total['Radii'][1] < profile.radii[1]:
                add_profile  = np.interp(profile.radii.values,\
                     total['Radii'].value,total['Profile'].value)
                total['Profile'] = np.array([x+y for x,y in \
                        zip(add_profile,profile['Profile'].value)],type=float)*total['Profile'].unit
                total['Radii']  = profile.radii
            else:
                add_profile  = np.interp(total['Radii'].value, \
                    profile.radii.value,profile['Profile'].value)
                   
                total['Profile'] = np.array([x+y for x,y in \
                        zip(add_profile,total['Profile'].value)],type=float)*total['Profile'].unit
    return total

def get_accepted_unit(search_dictionary,attr, acceptable_units = \
                      [u.Lsun/u.pc**2,u.Msun/u.pc**2,u.Msun/u.pc**3],
                      cfg=None,first_value=None):
    funit = None
    iters = iter(search_dictionary)
    while funit is None:
        try:
            check = next(iters)
        except StopIteration:
            break
        values  = getattr(search_dictionary[check],attr)
          
        if isquantity(values):
            funit = values.unit
        else:
            continue
        print_log(f'''The units of {search_dictionary[check].name} for {attr} are {funit}.
''',cfg,case=['debug_add','screen'])
        if first_value is not None:
            if funit == first_value:
                funit = None          
        if not funit in acceptable_units:
            funit = None
    return funit

def get_exponent(level,threshold = 2.):
   
    if np.isnan(level) or level == 0.:
        logexp= 0.
    else:
        logexp = int(np.floor(np.log10(abs(level))))
        correction = 1./(10**(logexp))
    if abs(logexp) <= threshold:
        logexp = 0.
        correction=1.
    return logexp,correction


def plot_profiles(profiles, cfg= None\
                ,output_file = './Profiles.png'):
    '''This function makes a simple plot of the profiles'''    
    max = 0.
    min = 1000.
    # From the optical profiles we select the first acceptable units and make sure that all 
    # other profiles adhere to these unit. As they are not coupled it is possible that 
    # no profile adheres to the combination of units
   
    first_value_unit = get_accepted_unit(profiles,'values',cfg=cfg)
    first_radii_unit = get_accepted_unit(profiles,'radii',\
        acceptable_units=[u.pc,u.kpc,u.Mpc],cfg=cfg)
    
    second_value_unit = get_accepted_unit(profiles,'values',cfg=cfg
        ,first_value = first_value_unit)
    
    if first_value_unit is None:
        print_log(f'''We cannot find acceptable units in the profiles.
{[profiles[x].print() for x in profiles]}
This is not acceptable for the output.
''',cfg,case=['main','screen'] )
        raise RunTimeError("No proper units")    
    if first_radii_unit is None:
        print_log(f'''We cannot find acceptable units in the radii in profiles.
The units are not PC, KPC or MPC for any profile.
This is not acceptable for the output.
''',cfg,case=['main'] )
        raise RunTimeError("No proper units")
    tot_opt ={'Profile': [],'Radii': []}
    doubleplot = False
    fig = setup_fig()
    ax = fig.add_subplot(111)
    leg_lines = []
    for name in profiles:
        if profiles[name].values.unit == first_value_unit and\
            profiles[name].radii.unit == first_radii_unit:
            min,max,succes,ax,lineout = plot_individual_profile(profiles[name],
                min,max,ax)

            if succes:
                tot_opt = calculate_total_profile(tot_opt,profiles[name])
        elif profiles[name].values.unit == second_value_unit and\
            profiles[name].radii.unit == first_radii_unit:
            if not doubleplot:
                secax= ax.twinx()
                secax._get_lines = ax._get_lines            
            mintwo,maxtwo,succes,secax,lineout = plot_individual_profile(profiles[name]
                ,min,max,secax)
        
            doubleplot = True
              
        else:
            print_log(f'''The profile units of {profiles[name].name} are not {first_value_unit} (unit  = {profiles[name].values.unit})
    or the radii units are   not {first_radii_unit} (unit  = {profiles[name].radii.unit})           
    Not plotting this profile.
    ''',cfg,case=['main'])
        leg_lines = leg_lines + lineout 
    if len(tot_opt['Profile']) > 0 and not doubleplot:  
        lineout = ax.plot(tot_opt['Radii'],tot_opt['Profile'], \
                label = 'Total Optical Profile',color='black',linestyle='--')
        if np.nanmax(tot_opt['Profile'].value) > max:
            max =  np.nanmax(tot_opt['Profile'].value)
        leg_lines.append(lineout[0]) 
        #min = np.nanmin(np.array([x for x in tot_opt['Profile'].value if x > 0.]))
    if min <= 0.:
        min=0.001

    max = max*1.1
    ax.set_ylim(min,max)
    #plt.xlim(0,6)
    ax.set_ylabel(select_axis_label(first_value_unit))
    ax.set_xlabel(select_axis_label(first_radii_unit))    
    ax.set_yscale('log')
    if doubleplot:
        secax.set_ylabel(select_axis_label(second_value_unit))        
        secax.set_xlim(ax.get_xlim())
        secax.set_ylim(mintwo,maxtwo)
        secax.set_yscale('log')
  
    labs = [l.get_label() for l in leg_lines]
    ax.legend(leg_lines, labs, loc=0)

    
    plt.savefig(output_file)
   
    plt.close()



def profiles_to_lines(profiles):
    '''Transform the profiles into a set of line by line columns.'''
    profile_columns = []
    profile_units = []
    to_write = []
    for x in profiles:
        if not profiles[x].values is None:
            to_write.append(x)
            single = [f'{profiles[x].name}_RADII',profiles[x].name]
            single_units = [translate_string_to_unit(profiles[x].radii.unit,invert=True),
                            translate_string_to_unit(profiles[x].values.unit,invert=True)]
            if not profiles[x].errors is None:
                single.append(f'{profiles[x].name}_ERR')
                single_units.append(translate_string_to_unit(profiles[x].values.unit,invert=True))
            profile_columns = profile_columns+single
            profile_units = profile_units+single_units
    lines = [' '.join([f'{y:>15s}' for y in profile_columns])]
    lines.append(' '.join([f'{y:>15s}' for y in profile_units]))   
    count = 0
    finished = False
 
    while not finished:
        finished  = True
        line = []
        for x in to_write:
            single = []
            if len(profiles[x].values) > count:
                single = [f'{profiles[x].radii[count].value:>15.5f}',\
                          f'{profiles[x].values[count].value:>15.5f}']
                if not profiles[x].errors is None:
                    single.append(f'{profiles[x].errors[count].value:>15.5f}')
            else:
                single = [f'{"NaN":>15s}',f'{"NaN":>15s}']
                if not profiles[x].errors is None:
                    single.append(f'{"NaN":>15s}')
            line = line+single
      
        if np.all([x.strip() == 'NaN' for x in line]):
            pass
        else:
            finished = False
            count += 1
            lines.append(' '.join(line))
    return lines

def propagate_mean_error(errors):
    n = len(errors)
    combined = np.sum([(x/n)**2 for x in errors])
    sigma = np.sqrt(combined)
    return sigma

def set_limits(value,minv,maxv,debug = False):
    if value < minv:
        return minv
    elif value > maxv:
        return maxv
    else:
        return value

set_limits.__doc__ =f'''
 NAME:
    set_limits
 PURPOSE:
    Make sure Value is between min and max else set to min when smaller or max when larger.
 CATEGORY:
    support_functions

 INPUTS:
    value = value to evaluate
    minv = minimum acceptable value
    maxv = maximum allowed value

 OPTIONAL INPUTS:
    debug = False

 OUTPUTS:
    the limited Value

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''


'''Strip the unit making sure it is the correct unit '''
def strip_unit(value, requested_unit = None, variable_type = None):
    if requested_unit is None and variable_type is None:
        raise InputError(f'You have to request a unit or set a variable type')
    translation_dict = {'radii' : u.kpc,\
                        'density': u.Msun/u.pc**3,
                       'sbr_lum': u.Lsun/u.pc**2,
                       'sbr_dens': u.Msun/u.pc**2,}
    if requested_unit is None:
        try:
            requested_unit = translation_dict[variable_type]
        except:
            raise InputError(f'We do not know how to match {variable_type}')
    else:
        if variable_type in [x for x in translation_dict]:
            print(f'You are overwriting the default for {variable_type}')
    
    if value.unit == requested_unit:
        return value.value
    else:
        raise RunTimeError(f'The value {value} does not have to unit {requested_unit}')

'select a plotting label based on a unit'
def select_axis_label(input):
    #If the input is not a string we need to convert
    if not isinstance(input,str):
        input = translate_string_to_unit(input,invert=True)
     
    translation_dict = {'ARCSEC': r'Radius (")',
                        'ARCMIN': r"Radius (')",
                        'DEGREE': r"$\mathrm{Radius\,\,  (^{\circ})}$",
                        'MPC': r'Radius (Mpc)',
                        'KPC': r'Radius (kpc)',
                        'PC': r'Radius (pc)',
                        'KM/S': r'$\mathrm{Velocity\,\,  (km\,\,  s^{-1})}$',
                        'M/S': r'$\mathrm{Velocity\,\,  (m\,\,  s^{-1})}$',
                        'M_SOLAR': r'Mass $\mathrm{(M_{\odot})}$',
                        'L_SOLAR': r'Luminosty $\mathrm{(L_{\odot})}$',
                        'L_SOLAR/PC^2': r'$\mathrm{Surface\,\, Brightness\,\, (L_{\odot}\,\, pc^{-2})}$',
                        'M_SOLAR/PC^2': r'$\mathrm{Surface\,\,  Density (M_{\odot}\,\,  pc^{-2})}$',
                        'MAG/ARCSEC^2':r'$\mathrm{Surface\,\,  Brightness (Mag\,\,  arsec^{-2})}$',
                        'L_SOLAR/PC^3': r'$\mathrm{Luminosity\,\,  Density (L_{\odot}\,\,  pc^{-3})}$',
                        'M_SOLAR/PC^3': r'$\mathrm{Density\,\,  (M_{\odot}\,\,  pc^{-3})}$',
                        'SomethingIsWrong': None}
    return translation_dict[input]

def setup_fig(size_factor=1.5,figsize= [7,7]):
    Overview = plt.figure(2, figsize=figsize, dpi=300, facecolor='w', edgecolor='k')
#stupid pythonic layout for grid spec, which means it is yx instead of xy like for normal human beings
    try:
        mpl_fm.fontManager.addfont( "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
        font_name = mpl_fm.FontProperties(fname= "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf").get_name()
    except FileNotFoundError:
        font_name= 'Deja Vu'
        
    labelfont = {'family': font_name,
         'weight': 'normal',
         'size': 8*size_factor}
    plt.rc('font', **labelfont)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    return Overview
setup_fig.__doc__ =f'''
 NAME:
    setup_fig
 PURPOSE:
    Setup a figure to plot in
 CATEGORY:
   
 INPUTS:
    
 OPTIONAL INPUTS:

 OUTPUTS:
  
 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    scipy.ndimage.map_coordinates, np.array, np.mgrid

 NOTE:
'''


'''Translate strings to astropy units and vice versa (invert =True)'''
def translate_string_to_unit(input,invert=False):
    translation_dict = {'ARCSEC': u.arcsec,
                        'ARCMIN': u.arcmin,
                        'DEGREE': u.degree,
                        'KPC': u.kpc,
                        'PC': u.pc,
                        'KM/S': u.km/u.s,
                        'M/S': u.m/u.s,
                        'M_SOLAR': u.Msun,
                        'L_SOLAR': u.Lsun,
                        'L_SOLAR/PC^2': u.Lsun/u.pc**2,
                        'M_SOLAR/PC^2': u.Msun/u.pc**2,
                        'L_SOLAR/PC^3': u.Lsun/u.pc**3,
                        'M_SOLAR/PC^3': u.Msun/u.pc**3,
                        'MAG/ARCSEC^2': u.mag/u.arcsec**2,
                        'SomethingIsWrong': None}
    output =False
    if invert:
        if input in list(translation_dict.values()):
            output = list(translation_dict.keys())[list(translation_dict.values()).index(input)]
    else:
        input = input.strip().upper() 
        # If we have string it is easy
        if input in list(translation_dict.keys()):
            output = translation_dict[input]

    if output is False:
        raise InputError(f'The unit {input} is not recognized for a valid translation.')
    else:
        return output


def write_header(profiles,
        output_dir= './', file= 'You_Should_Set_A_File.txt'):

    with open(f'{output_dir}{file}','w') as file:
        names = [profiles[name].name for name in profiles if name[0:3] != 'SKY']
        file.write(f'# This file contains the info for the following profiles: {", ".join(names)}.\n')
        for name in profiles:
            if name[0:3] != 'SKY':
                string = ''
                if profiles[name].profile_type == 'rotation_curve': 
                    string = f'# {profiles[name].name} is a rotation curve constructed with the following parameters. \n'
                elif profiles[name].profile_type == 'sbr_lum':
                    string = f'# {profiles[name].name} is a luminosity constructed with the following parameters. \n'
                elif profiles[name].profile_type == 'sbr_dens':
                    string = f'# {profiles[name].name} is a surface brightness profile constructed with the following parameters. \n'
                elif profiles[name].profile_type == 'density': 
                    string = f'# {profiles[name].name} is a density profile constructed with the following parameters. \n'
                else:
                    string = f'# {profiles[name].name} is a random profile constructed with the following parameters. \n'
               
                string += f'''#{'':9s}We used a distance of {profiles[name].distance:.1f}. 
#{'':9s}The type of is {profiles[name].type} relating to the component {profiles[name].component}. \n'''

                if profiles[name].profile_type in ['density','sbr_dens']: 
                    string += f'''#{'':9s}We used a Mass to Light ratio of {zero_if_none(profiles[name].MLratio):.3f}.
#{'':9s}A height of {zero_if_none(profiles[name].height)}+/-{zero_if_none(profiles[name].height_error)} of type {zero_if_none(profiles[name].height_type)}.                      
'''
                file.write(string)
           


write_header.__doc__ =f'''
 NAME:
    write_header
 PURPOSE:
    Write a header containing the conversion used in  a table with derived products
 CATEGORY:
    support_functions

 INPUTS:


 OPTIONAL INPUTS:
    output_dir= './'

    Directory where to write the file

    file= 'You_Should_Set_A_File_RC.txt'

    file name




 OUTPUTS:

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:

 NOTE:

'''

  
def write_profiles(profiles,output_dir= './',additional_input=None,\
        cfg=None, filename = None):
    write_profiles = {}


    '''Function to write all the profiles to some text files.'''
    if not profiles is None:
        for name in profiles:
            write_profiles[name] = profiles[name]
    
    if not additional_input is None:
        for name in additional_input:
            write_profiles[name] = additional_input[name]   

    if not write_profiles is None:
        lines = profiles_to_lines(write_profiles)
        write_header(write_profiles,output_dir=output_dir,file=filename)
        with open(f'{output_dir}{filename}','a') as file:   
            for line in lines:
                file.write(f'{line} \n')

       
     
def zero_if_none(val):
    if val is None:
        val = 0.
    return val