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


def add_font(file):
    try:
        mpl_fm.fontManager.addfont(file)
        font_name = mpl_fm.FontProperties(fname=file).get_name()
    except FileNotFoundError:
        font_name = 'DejaVu Sans'
    return font_name


def check_arguments():
    argv = sys.argv[1:]

    if '-v' in argv or '--version' in argv:
        print(f"This is version {pyROTMOD.__version__} of the program.")
        sys.exit()

    if '-h' in argv or '--help' in argv:
        print(''' Use pyROTMOD in this way:
pyROTMOD configuration_file=inputfile.yml   where inputfile is a yaml config file with the desired input settings.
pyROTMOD -h print this message
pyROTMOD print_examples=True prints a yaml file (defaults.yml) with the default setting in the current working directory.
in this file values designated ??? indicated values without defaults.

All config parametere can be set directly from the command line by setting the correct parameters, e.g:
pyROTMOD fitting.HALO=ISO to set the pseudothermal halo.
note that list inout should be set in apostrophes in command line input. e.g.:
pyROTMOD 'fitting.MD=[1.4,True,True]'
''')
        sys.exit()

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
                    physical=False,debug = False, quantity= False):

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
''',None,debug =True)
       
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

def check_input(cfg,fitting=False):
    'Check various input values to avoid problems later on.'
    #check the slashes and get the ininitial dir for the output dir
    if cfg.general.output_dir[-1] != '/':
        cfg.general.output_dir = f"{cfg.general.output_dir}/"
    if cfg.general.output_dir[0] == '/':
        first_dir= cfg.general.output_dir.split('/')[1]
    else:
        first_dir= cfg.general.output_dir.split('/')[0]
    #check the slashes and get the ininitial dir for the logging dir
    if cfg.general.log_directory[-1] != '/':
        cfg.general.log_directory = f"{cfg.general.log_directory}/"
    if cfg.general.log_directory[0] == '/':
        cfg.general.log_directory = cfg.general.log_directory[1:]
    first_log= cfg.general.log_directory.split('/')[0]

    # If the first directories in the log dir and the output dir are not the same
    # we assume the log dir should start at the output dir
    if first_dir != first_log:
        cfg.general.log_directory = f'{cfg.general.output_dir}{cfg.general.log_directory}'
    else:
        cfg.general.log_directory = f'/{cfg.general.log_directory}'
    #Create the directories if non-existent   
    if not os.path.isdir(cfg.general.output_dir):
        os.mkdir(cfg.general.output_dir)
    create_directory(cfg.general.log_directory,cfg.general.output_dir)
  
    log = f"{cfg.general.log_directory}{cfg.general.log}"
    #If it exists move the previous Log
    if os.path.exists(log):
        os.rename(log,f"{cfg.general.log_directory}/Previous_Log.txt")

    #Start a new log
    print_log(f'''This file is a log of the modelling process run at {datetime.now()}.
This is version {pyROTMOD.__version__} of the program.
''',log,debug=cfg.general.debug)

    # check the input files
    if not cfg.RC_Construction.gas_file and cfg.RC_Construction.enable:
        print(f'''You did not set the gas file input''')
        cfg.RC_Construction.gas_file = input('''Please add the gas file or tirific output to be evaluated: ''')
    print_log(f'''We are using the input from {cfg.RC_Construction.gas_file} for the gaseous component.
''',log,debug=cfg.general.debug)
    if cfg.RC_Construction.gas_file.split('.')[1].lower() == 'def' and \
        cfg.RC_Construction.gas_scaleheight[1] is None\
        and cfg.RC_Construction.enable:
        cfg.RC_Construction.gas_scaleheight = [0.,None,'ARCSEC','tir']
    
    if not cfg.RC_Construction.optical_file and cfg.RC_Construction.enable:
        cfg.RC_Construction.optical_file = input('''Please add the optical or galfit file to be evaluated: ''')
    print_log(f'''We are using the input from {cfg.RC_Construction.optical_file} for the optical component.
''',log,debug=cfg.general.debug)

    if cfg.general.distance is None:
        raise InputError(f'We cannot model profiles adequately without a distance proper distance')
    
    print_log(f'''We are using the following distance = {cfg.general.distance}.
''',log,debug=cfg.general.debug)
    # return the cfg and log name

    #write the input to the log dir.
    if cfg.general.debug:
        name = f'{cfg.general.log_directory}run_input'
        if fitting:
            name += '_fitting'
        name +='.yml'   
        with open(name,'w') as input_write:
            input_write.write(OmegaConf.to_yaml(cfg))
       

    return cfg,log
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
    #Need to use raw strings here to avoid problems with \
    label_dictionary = {'Gamma_disk':[r'$\mathrm{M/L_{disk}}$', ''],
                         'Gamma_bulge':[r'$\mathrm{M/L_{bulge}}$', ''],
                         'Gamma_gas': [r'$\mathrm{M/L_{gas}}$',''],
                         'ML_stellar':[r'$\mathrm{M/L_{optical}}$',''],
                         'ML_gas':[r'$\mathrm{M/L_{gas}}$',''],
                         'RHO': [r'$\mathrm{\rho_{c}}$',r'$\mathrm{(M_{\odot} \,\, pc^{-3})}$'],
                         'RHO0': [r'$\mathrm{\rho_{c}}$',r'$\mathrm{(M_{\odot} \,\, pc^{-3})}$'],
                         'R_C': [r'$ \mathrm{R_{c}}$','(kpc)'],
                         'C':[r'C',''],
                         'R200':[r'$ \mathrm{R_{200}}$','(kpc)'],
                         'm': [r'Axion Mass','(eV)'],
                         'central': [r'Central SBR',r'$\mathrm{(M_{\odot}\,\,pc^{-2})}$'],
                         'h': [r'Scale Length','(kpc)'],
                         'mass': [r'Total Mass', r'$\mathrm{(M_{\odot})}$'],
                         'hern_length': ['Hernquist length','(kpc)}$'],
                         'effective_luminosity': [r'$\mathrm{L_{e}}$',r'$\mathrm{(M_{\odot})}$'] ,
                         'effective_radius': [r'$\mathrm{R_{e}}}$','(kpc)'] ,
                         'n': [r'Sersic Index',''],
                         'a0': [r'$\mathrm{a_{0}}$',r'\mathrm{$(cm\,\,s^{-2})}$']
                         }
    if par in label_dictionary:
        if par[:5] == 'Gamma':
            string = f'{label_dictionary[par][0]} {no}'
        else:
            string = label_dictionary[par][0] 
        if abs(exponent) >= 1.:
            string += f'$\\times10^{{{exponent}}}' 
        string += f'{label_dictionary[par][1]}'
                     
    else:
        print(f''' The parameter {par} has been stripped
Unfortunately we can not find it in the label dictionary.''')
        raise RunTimeError(f'{par} is not in label dictionary')
    
    return string   



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
    mass = np.sum(quantity_array([x*y for x,y in zip(ringarea,density)],u.Msun))
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


def plot_individual_profile(profile,max,log = None):
    if not profile.values.unit in [u.Lsun/u.pc**2,u.Msun/u.pc**3] :
        print_log(f'''The units of {profile.name} are not L_SOLAR/PC^2 OR M_SOLAR/PC^3 .
Unit = {profile.values.unit}                  
Not plotting this profile.
''',log )
        return max,False
    if profile.radii.unit != u.kpc:
        print_log(f'''The units of {profile.name} are not KPC.
Not plotting this profile.
''',log )
        return max,False
    plt.plot(profile.radii.value,profile.values.value, \
                label = profile.name)
    if np.nanmax(profile.values) > max:
        max =  np.nanmax(profile.values)
    return max,True

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
                      [u.Lsun/u.pc**2,u.Msun/u.pc**3]):
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
     
        if not funit in acceptable_units:
            funit = None
    return funit

def get_exponent(level,threshold = 2.):
    logexp = int(np.floor(np.log10(level)))
    correction = 1./(10**(logexp))
    if abs(logexp) <= threshold:
        logexp = 0.
        correction=1.
    return logexp,correction


def plot_profiles(gas_profiles,optical_profiles, log= None\
                ,output_dir = './',input_profiles = None):
    '''This function makes a simple plot of the optical profiles'''    
    max = 0.
    # From the optical profiles we select the first acceptable units and make sure that all 
    # other profiles adhere to these unit. As they are not coupled it is possible that 
    # no profile adheres to the combination of units
    first_value_unit = get_accepted_unit(optical_profiles,'values')
    first_radii_unit = get_accepted_unit(optical_profiles,'radii',\
        acceptable_units=[u.pc,u.kpc,u.Mpc])
    if first_value_unit is None:
        print_log(f'''We cannot find acceptable units in the optical profiles.
The units are not L_SOLAR/PC^2 or M_SOLAR/PC^3 for any profile.
This is not acceptable for the output
''',log )
        raise RunTimeError("No proper units")    
    if first_radii_unit is None:
        print_log(f'''We cannot find acceptable units in the radii in optical profiles.
The units are not PC, KPC or MPC for any profile.
This is not acceptable for the output
''',log )
        raise RunTimeError("No proper units")
    for name in gas_profiles:
        if gas_profiles[name].values.unit == first_value_unit and\
            gas_profiles[name].radii.unit == first_radii_unit:
            plt.plot(gas_profiles[name].radii.value,gas_profiles[name].values.value,\
                     label = gas_profiles[name].name )
            max = np.nanmax(gas_profiles[name].values)
        else:
            print_log(f'''The profile units of {gas_profiles[name].name} are not {first_value_unit} (unit  = {gas_profiles[name].values.unit})
or the radii units are   not {first_radii_unit} (unit  = {gas_profiles[name].radii.unit})           
Not plotting this profile.
''',log )
    #This is only used here so do not make it a Density Profile        
    tot_opt ={'Profile': [],'Radii': []}
    for x in optical_profiles:
        if optical_profiles[x].name.split('_')[0] == 'SKY':
            continue
        if optical_profiles[x].values.unit == first_value_unit and\
            optical_profiles[x].radii.unit == first_radii_unit:
            max,succes = plot_individual_profile(optical_profiles[x],max)
            if succes:
                tot_opt = calculate_total_profile(tot_opt,optical_profiles[x])
            
        else:
            print_log(f'''The profile units of {optical_profiles[x].name} are not {first_value_unit} (unit  = {optical_profiles[x].values.unit})
or the radii units are   not {first_radii_unit} (unit  = {optical_profiles[x].radii.unit})           
Not plotting this profile.
''',log )
          
       
    plt.plot(tot_opt['Radii'],tot_opt['Profile'], label='Total Optical')
    
    max = np.nanmax(tot_opt['Profile'].value)
    min = np.nanmin(np.array([x for x in tot_opt['Profile'].value if x > 0.]))
   
    plt.ylim(min,max)
    #plt.xlim(0,6)
    plt.ylabel(select_axis_label(first_value_unit))
    plt.xlabel(select_axis_label(first_radii_unit))    
   

    plt.yscale('log')
    plt.legend()
    if first_value_unit == u.Lsun/u.pc**2:
        plt.savefig(f'{output_dir}/Surface_Brightness_Profiles.png')
    else:
        plt.savefig(f'{output_dir}/Density_Profiles.png')
    plt.close()

def print_log(log_statement,log, screen = True,debug = False):
    log_statement = f"{log_statement}"
    if screen or not log:
        print(log_statement)
    if log:
        with open(log,'a') as log_file:
            log_file.write(log_statement)

print_log.__doc__ =f'''
 NAME:
    print_log
 PURPOSE:
    Print statements to log if existent and screen if Requested
 CATEGORY:
    support_functions

 INPUTS:
    log_statement = statement to be printed
    log = log to print to, can be None

 OPTIONAL INPUTS:
    debug = False

    screen = False
    also print the statement to the screen

 OUTPUTS:
    line in the log or on the screen

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    linenumber, .write

 NOTE:
    If the log is None messages are printed to the screen.
    This is useful for testing functions.
'''



def profiles_to_lines(profiles):
    '''Trandform the profiles into a set of line by line columns.'''
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
                single = [f'{profiles[x].radii[count].value:>15.2f}',\
                          f'{profiles[x].values[count].value:>15.2f}']
                if not profiles[x].errors is None:
                    single.append(f'{profiles[x].errors[count].value:>15.2f}')
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
                        'density': u.Msun/u.pc**2}
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
                else:
                    string = f'# {profiles[name].name} is a density profile constructed with the following parameters. \n'
                string += f'''#{'':9s}We used a distance of {profiles[name].distance:.1f}. 
#{'':9s}The type of is {profiles[name].type} relating to the component {profiles[name].component}. \n'''

                if profiles[name].profile_type == 'density_profile': 
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

  
def write_profiles(gas_profile_in,total_rc,optical_profiles = None,output_dir= './',\
             log =None, filename = None,optical_filename='Optical_Mass_Densities.txt'):
    '''Function to write all the profiles to some text files.'''
    if not optical_profiles is None:
        optical_lines = profiles_to_lines(optical_profiles)
        write_header(optical_profiles,output_dir=output_dir,file=optical_filename)
        with open(f'{output_dir}{optical_filename}','a') as opt_file:   
            for line in optical_lines:
                opt_file.write(f'{line} \n')
    if not  filename is None:
        gas_profile = copy.deepcopy(gas_profile_in)
        gas_profile['V_OBS'] = total_rc
        gas_lines =  profiles_to_lines(gas_profile)
        #rc_lines = profiles_to_lines({'V_OBS':total_rc})  

        write_header(gas_profile,output_dir=output_dir,file=filename)
        with open(f'{output_dir}{filename}','a') as file:
            for line in gas_lines:
                    file.write(f'{line} \n')
       
     
def zero_if_none(val):
    if val is None:
        val = 0.
    return val