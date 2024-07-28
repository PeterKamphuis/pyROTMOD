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
from omegaconf import OmegaConf,ListConfig
from datetime import datetime
from pyROTMOD.support.errors import InputError,SupportRunError


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
def check_quantity(value,unit = None):
    if not isinstance(value,u.quantity.Quantity):
        if isiterable(value):
        #if it is an iterable we make sure it it is an numpy array
            if not isinstance(value,np.array) and not value is None:
                value = np.array(value,dtype=float)
        if not value is None:
            value = value * unit
    return value
  

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
    if len(kpc) == 1 and not quantity:
        kpc = float(kpc[0])
    return kpc

def check_input(cfg):
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
    #write the input to the log dir.
    with open(f"{cfg.general.log_directory}run_input.yml",'w') as input_write:
        input_write.write(OmegaConf.to_yaml(cfg))
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
        cfg.RC_Construction.gas_scaleheight[1] is None:

        cfg.RC_Construction.gas_scaleheight = [0.,'tir']
    
    if not cfg.RC_Construction.optical_file and cfg.RC_Construction.enable:
        cfg.RC_Construction.optical_file = input('''Please add the optical or galfit file to be evaluated: ''')
    print_log(f'''We are using the input from {cfg.RC_Construction.optical_file} for the optical component.
''',log,debug=cfg.general.debug)

    if cfg.general.distance is None:
        raise InputError(f'We cannot model profiles adequately without a distance proper distance')
    
    print_log(f'''We are using the following distance = {cfg.general.distance}.
''',log,debug=cfg.general.debug)
    # return the cfg and log name
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
'''
def check_fitting_input(cfg):
    if cfg.fitting.enable:
        for key in dir(cfg.fitting):
            if type(getattr(cfg.fitting,key))==ListConfig:
                tmp = getattr(cfg.fitting,key)
                for i in range(3):
                    try:
                        if tmp[i].lower() != 'none':
                            tmp[i]=float(tmp[i])
                        else:
                            tmp[i]=None
                    except AttributeError:
                        pass

                for i in [3,4]:
                    try:
                        tmp[i]=eval(tmp[i])
                    except TypeError:
                        tmp[i]=bool(tmp[i])
                setattr(cfg.fitting,key,tmp)
    return cfg, log
'''
def get_effective_radius(radii,density,debug=False,log= None):

   
    mass,ringarea = integrate_surface_density(radii,density)
    Cuma_prof = []
    for i,rad in enumerate(radii):
        if i == 0:
            Cuma_prof.append(ringarea[i]*density[i])
        else:
            new = Cuma_prof[-1]+ringarea[i]*density[i]
            Cuma_prof.append(new)
    #now to get the half of the total mass.
    if Cuma_prof[0] > mass/2.:
        print_log(f'''Your effective radius of the bulge is smaller than the first ring.
You should improve the resolution of the bulge profile.''',log)
        out_rad = radii[0]
    else:
        out_rad = radii[np.where(Cuma_prof<mass/2.)[-1][-1]]

    return mass,out_rad



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
            number = splitted[-1]
    except ValueError:
        component = key
       
    return component,number

def integrate_surface_density(radii,density, log=None):

    ringarea= [0 if radii[0] == 0 else np.pi*((radii[0]+radii[1])/2.)**2]
    ringarea = np.hstack((ringarea,
                         [np.pi*(((y+z)/2.)**2-((y+x)/2.)**2) for x,y,z in zip(radii,radii[1:],radii[2:])],
                         [np.pi*((radii[-1]+0.5*(radii[-1]-radii[-2]))**2-((radii[-1]+radii[-2])/2.)**2)]
                         ))
    #radii are in kpc and density in M_sun/pc^2
    ringarea = ringarea*1000.**2
   
    #print(ringarea,density)
    mass = np.sum([x*y for x,y in zip(ringarea,density)])
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


def old_ensure_kpc_radii(in_radii,unit = None, distance= None):
    '''Check that our list is a proper radius only list in kpc. if not try to convert.'''
    if unit is None:
        raise InputError(f'We can not check the units if we have no units')
  
    if unit == 'KPC':
        return in_radii
    elif unit == 'PC':
        correct_rad = copy.deepcopy(in_radii)/1000.
    elif unit in ['ARCSEC','ARCMIN','DEGREE']:
        if distance is None:
            raise InputError(f'We can not convert {unit} to kpc without a distance')
        correct_rad = copy.deepcopy(in_radii)  
        correct_rad = convertskyangle(correct_rad, distance, unit=unit)
    return correct_rad



def plot_individual_profile(profile,max,log = None):
    
    if profile.unit != u.Msun/u.pc**2:
        print_log(f'''The units of {profile.name} are not M_SOLAR/PC^2.
Not plotting this profile.
''',log )
        return max,False
    if profile.radii_unit != u.kpc:
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

    

def plot_profiles(gas_profiles,optical_profiles, log= None\
                ,output_dir = './',input_profiles = None):
    '''This function makes a simple plot of the optical profiles'''    
    max = 0.
    for name in gas_profiles:
        if gas_profiles[name].unit == u.Msun/u.pc**2 and \
            gas_profiles[name].radii_unit == u.kpc:
            plt.plot(gas_profiles[name].radii.value,gas_profiles[name].values.value,\
                     label = gas_profiles[name].name )
            max = np.nanmax(gas_profiles[name].values)
        else:
            print_log(f'''The units of {gas_profiles[name]} are not M_SOLAR/PC^2 or the radii are not in KPC.
Not plotting this profile.
''',log )
    #This is only used here so do not make it a Density Profile        
    tot_opt ={'Profile': [],'Radii': []}
    for x in optical_profiles:
        max,succes = plot_individual_profile(optical_profiles[x],max)
      
        if succes:
            tot_opt = calculate_total_profile(tot_opt,optical_profiles[x])
       
    plt.plot(tot_opt['Radii'],tot_opt['Profile'], label='Total Optical')
    '''
    if not (input_profiles  is None):
        for x in input_profiles:
            if x != 'RADI':
                if input_profiles[x][1] != 'M_SOLAR/PC^2':
                    print_log(f''The units of {input_profiles[x][0]} are not M_SOLAR/PC^2.
Not plotting this profile.
'',log )
                    continue
                plt.plot(radii[2:],np.array( input_profiles[x][2:]), label =  input_profiles[x][0])
    '''     
    max = np.nanmax(tot_opt['Profile'].value)
    min = np.nanmin(np.array([x for x in tot_opt['Profile'].value if x > 0.]))

    plt.ylim(min,max)
    #plt.xlim(0,6)
    plt.ylabel(r'Density (M$_\odot$/pc$^2$)')
    plt.xlabel('Radius (kpc)')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'{output_dir}/Mass_Profiles.png')
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

def propagate_mean_error(errors):
    n = len(values)
    combined = np.sum(errors)
    sigma = combined/n
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
                        'MAG/ARCSEC^2': u.mag/u.arcsec**2}
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



def write_header(distance= None, MLratio= None, opt_scaleheight=None,\
        gas_scaleheight=None, RC = True,opt_disk_type = None,
        output_dir= './', file= 'You_Should_Set_A_File_RC.txt'):

    with open(f'{output_dir}{file}','w') as file:
        if RC:
            file.write('# This file contains the rotation curves derived by pyROTMOD. \n')
        else:
            file.write('# This file contains the mass density profiles derived by pyROTMOD. \n')
        if distance != None:
            file.write(f'# We used a distance of {distance:.1f} Mpc. \n')
        if MLratio != None:
            file.write(f'# We used a Mass to Light ratio for the stars of {MLratio:.3f}. \n')
        if opt_scaleheight != None:
            if opt_scaleheight[0] > 0:
                file.write(f'# We used stellar scale height {opt_scaleheight[0]:.3f}. \n')
            else:
                file.write(f'# We assumed the stellar disk to be infinitely thin. \n')
        if gas_scaleheight != None:
            if gas_scaleheight[0] > 0:
                file.write(f'# We used a gas scale height {gas_scaleheight[0]:.3f}. \n')
            else:
                file.write(f'# We assumed the gas disk to be infinitely thin. \n')



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

def write_RCs(RCs,total_rc,rc_err,log=None,\
        output_dir= './', file= 'You_Should_Set_A_File_RC.txt'):
    #print(RCs)
    with open(f'{output_dir}{file}','a') as opt_file:
        for x in range(len(RCs[0])):

            line = [RCs[i][x] for i in range(len(RCs))]
            line.append(total_rc[x])
            line.append(rc_err[x])

            if x <= 1:
                writel = ' '.join([f'{y:>15s}' for y in line])
            else:
                writel = ' '.join([f'{y:>15.2f}' for y in line])
            writel = f'{writel} \n'
            opt_file.write(writel)
write_RCs.__doc__ =f'''
 NAME:
    write_RCs
 PURPOSE:
    Write the derived RC
 CATEGORY:
    support_functions

 INPUTS:
    RCs = Dictionary with derived RCs
    total_rc = The observed RC
    rc_err = Error on the observed RC

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

def profiles_to_lines(profiles):
    
    profile_columns = []
    profile_units = []
    to_write = []
    for x in profiles:
        if not profiles[x].values is None:
            to_write.append(x)
            single = [f'{profiles[x].name}_RADII',profiles[x].name]
            single_units = [translate_string_to_unit(profiles[x].radii_unit,invert=True),
                            translate_string_to_unit(profiles[x].unit,invert=True)]
            if not profiles[x].errors is None:
                single.append(f'{profiles[x].name}_ERR')
                single_units.append(translate_string_to_unit(profiles[x].unit,invert=True))
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
  
def write_profiles(gas_profile,total_rc,optical_profiles,output_dir= './',\
             log =None):
    '''Function to write all the profiles to some text files.'''
    optical_lines = profiles_to_lines(optical_profiles)
    with open(f'{output_dir}Optical_Mass_Densities.txt','w') as opt_file:
        for line in optical_lines:
            opt_file.write(f'{line} \n')

    gas_lines =  profiles_to_lines(gas_profile)
    rc_lines = profiles_to_lines({'V_OBS':total_rc})   
    with open(f'{output_dir}Gas_Mass_Density_And_RC.txt','w') as file:
        nan_gas_line = ' '.join(f'{"NaN":>15s}' for x in gas_lines[0].split())
        nan_rc_line = ' '.join(f'{"NaN":>15s}' for x in gas_lines[0].split())
        no_lines = np.max([len(gas_lines),len(rc_lines)])
        for x in range(no_lines):
            if x < len(gas_lines):
                line = gas_lines[x]
            else:
                line = nan_gas_line
            if x < len(rc_lines):
                line = f'{line} {rc_lines[x]}'
            else:
                line = f'{line} {nan_rc_line}'
            file.write(f'{line} \n')
     
