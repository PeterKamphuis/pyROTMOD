# -*- coding: future_fstrings -*-

import copy
import numpy as np
import os
import warnings
import time

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt

from astropy import units as u
from omegaconf import OmegaConf,MissingMandatoryValue,ListConfig
from datetime import datetime

class InputError(Exception):
    pass

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

def check_input(cfg,default_output,default_log_directory,version,debug=False):
    if cfg.general.output_dir[-1] != '/':
        cfg.general.output_dir = f"{cfg.general.output_dir}/"

    using_default=False
    if default_output != cfg.general.output_dir:
        if default_log_directory == cfg.general.log_directory:
            using_default =True
            cfg.general.log_directory=f'{cfg.general.output_dir}Logs/{datetime.now().strftime("%H:%M:%S-%d-%m-%Y")}/'

    if not os.path.isdir(cfg.general.output_dir):
        os.mkdir(cfg.general.output_dir)

    if not os.path.isdir(f"{cfg.general.output_dir}/Logs"):
        os.mkdir(f"{cfg.general.output_dir}/Logs")

    if not os.path.isdir(f"{cfg.general.log_directory}"):
        os.mkdir(f"{cfg.general.log_directory}")
    else:
        if using_default:
            time.sleep(1.)
            cfg.general.log_directory=f'{cfg.general.output_dir}Logs/{datetime.now().strftime("%H:%M:%S-%d-%m-%Y")}/'
            os.mkdir(f"{cfg.general.log_directory}")

    #write the input to the log dir.
    with open(f"{cfg.general.log_directory}run_input.yml",'w') as input_write:
        input_write.write(OmegaConf.to_yaml(cfg))


    log = f"{cfg.general.log_directory}{cfg.general.log}"
    #If it exists move the previous Log
    if os.path.exists(log):
        os.rename(log,f"{cfg.general.log_directory}/Previous_Log.txt")


    print_log(f'''This file is a log of the modelling process run at {datetime.now()}.
This is version {version} of the program.
''',log,debug=cfg.general.debug)
    if not cfg.RC_Construction.gas_file and cfg.RC_Construction.enable:
        print(f'''You did not set the gas file input''')
        cfg.RC_Construction.gas_file = input('''Please add the gas file or tirific output to be evaluated: ''')
    print_log(f'''We are using the input from {cfg.RC_Construction.gas_file} for the gaseous component.
''',log,debug=cfg.general.debug)


    if not cfg.RC_Construction.optical_file and cfg.RC_Construction.enable:
        cfg.RC_Construction.optical_file = input('''Please add the optical or galfit file to be evaluated: ''')
    print_log(f'''We are using the input from {cfg.RC_Construction.optical_file} for the optical component.
''',log,debug=cfg.general.debug)

    if not cfg.general.distance:
        cfg.general.distance= input(f'''Please provide the distance (0. will use vsys and the hubble flow from a .def file): ''')

    if cfg.general.distance == 0.:
        try:
            vsys = gas_profiles.load_tirific(cfg.RC_Construction.gas_file, Variables = ['VSYS'])
            if vsys[0] == 0.:
                raise InputError(f'We cannot model profiles adequately without a distance')
            else:
                cfg.general.distance = vsys[0]/c.H_0
        except:
            raise InputError(f'We cannot model profiles adequately without a distance proper distance')
        ######################################## Read the optical profiles ##########################################
    print_log(f'''We are using the following distance = {cfg.general.distance}.
''',log,debug=cfg.general.debug)

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
                    tmp[i]=bool(tmp[i])
                setattr(cfg.fitting,key,tmp)
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

def integrate_surface_density(radii,density, log=None):

    ringarea= [0 if radii[0] == 0 else np.pi*((radii[0]+radii[1])/2.)**2]
    ringarea = np.hstack((ringarea,
                         [np.pi*(((y+z)/2.)**2-((y+x)/2.)**2) for x,y,z in zip(radii,radii[1:],radii[2:])],
                         [np.pi*((radii[-1]+0.5*(radii[-1]-radii[-2]))**2-((radii[-1]+radii[-2])/2.)**2)]
                         ))
    ringarea = ringarea*1000.**2
    #print(ringarea,density)
    mass = np.sum([x*y for x,y in zip(ringarea,density)])
    return mass

def read_columns(filename,optical=True,gas=False,debug=False,log=None):
    with open(filename, 'r') as input_text:
        lines= input_text.readlines()

    input_columns =[x.strip().upper() for x in lines[0].split()]
    units = [x.strip().upper() for x in lines[1].split()]

    possible_radius_units = ['KPC','PC','ARCSEC','ARCMIN','DEGREE',]
    allowed_types = ['EXPONENTIAL','SERSIC','DISK','BULGE']
    possible_units = ['L_SOLAR/PC^2','M_SOLAR/PC^2','MAG/ARCSEC^2','KM/S','M/S']
    allowed_velocities = ['V_OBS','V_OBS_ERR']
    for i,types in enumerate(input_columns):
        if i == 0:
            if types != 'RADI':
                raise InputError(f'Your first column in the input file {filename} is not the RADI')
            if units[i] not in possible_radius_units:
                raise InputError(f'''Your RADI column in the input file {filename} does not have the right units.
Possible units are: {', '.join(possible_radius_units)}. Yours is {units[i]}.''')

        else:
            if types.split('_')[0] not in allowed_types and types not in allowed_velocities:
                raise InputError(f'''Column {types} is not a recognized input.
Allowed columns are {', '.join(allowed_types)} or for the total RC {','.join(allowed_velocities)}''')
            else:
                if types in allowed_velocities and units[i] not in ['KM/S','M/S']:
                    raise InputError(f'''Column {types} has to have units of velocity so either KM/S or M/S''')
                elif units[i] not in possible_units:
                    raise InputError(f'''Column {types} has to have units of {', '.join(possible_units)}
the unit {units[i]} can not be processed.''')

    found_input = {}
    for type in input_columns:
        found_input[type] = []

    for line in lines:
        input = line.split()
        for i,type in enumerate(input_columns):
            found_input[type].append(input[i])

    if found_input['RADI'][1] in ['ARCMIN','DEGREE']:
        increase = 60 if found_input['RADI'][1] == 'ARCMIN' else 3600.
        found_input['RADI'][2:] = [x*increase for x in optical_profiles[0][2:]]
        found_input['RADI'][1] = 'ARCSEC'
    print_log(f'''In {filename} we have found the following columns:
{', '.join([f'{x} ({y})' for x,y in zip(input_columns,units)])}.
''',log)
    return found_input

def ensure_kpc_radii(in_radii,distance=1.,log=None):
    '''Check that our list is a proper radius only list in kpc. if not try to convert.'''

    if in_radii[0] != 'RADI':
        raise InputError('This is list is not marked RADI but you are trying to use it as such.')
    if in_radii[1] == 'KPC':
        return in_radii
    elif in_radii[1] == 'PC':
        correct_rad = copy.deepcopy(in_radii)
        correct_rad[2:] = [x/1000. for x in in_radii[2:]]
        correct_rad[1] = 'KPC'
    elif in_radii[1] in ['ARCSEC','ARCMIN','DEGREE']:
        correct_rad = copy.deepcopy(in_radii)
        correct_rad[2:] = convertskyangle(np.array(correct_rad[2:],dtype=float),\
            float(distance),unit=in_radii[1])
        correct_rad[1] = 'KPC'
    return correct_rad

def plot_profiles(in_radii,gas_profile,optical_profiles,distance = 1., \
                    log= None, errors = [0.],output_dir = './'):
    '''This function makes a simple plot of the optical profiles'''
    radii = ensure_kpc_radii(in_radii, distance=distance)
    opt_radii = ensure_kpc_radii(optical_profiles['RADI'],distance=distance)
    max = 0.
    lower_ind = 0
    if gas_profile[1] == 'M_SOLAR/PC^2':
        plt.plot(radii[2:],np.array(gas_profile[2:]),label = gas_profile[0])
        max = np.nanmax(np.array(gas_profile[2:]))
        lower_ind = np.where(np.array(opt_radii[2:],dtype=float) > float(radii[2])/2.)[0][0]
    else:
        print_log(f'''The units of {gas_profile[0]} are not M_SOLAR/PC^2.
Not plotting this profile.''',log )
    tot_opt = [0 for x in opt_radii[2:]]
    for x in optical_profiles:
        if x != 'RADI':
            if optical_profiles[x][1] != 'M_SOLAR/PC^2':
                print_log(f'''The units of {optical_profiles[x][0]} are not M_SOLAR/PC^2.
Not plotting this profile.''',log )
                continue
            plt.plot(opt_radii[2:],np.array(optical_profiles[x][2:]), label = optical_profiles[x][0])
            if np.nanmax(np.array(optical_profiles[x][2+lower_ind:])) > max:
                max =  np.nanmax(np.array(optical_profiles[x][2+lower_ind:]))
            if len(tot_opt) > 0:
                tot_opt = [old+new for old,new in zip(tot_opt,optical_profiles[x][2:])]
            else:
                tot_opt =  optical_profiles[x][2:]
    plt.plot(opt_radii[2:],np.array(tot_opt), label='Total Optical')
    max = np.nanmax(tot_opt)
    min = np.nanmin(np.array([x for x in tot_opt if x > 0.]))

    plt.ylim(min,max)
    #plt.xlim(0,6)
    plt.ylabel('Density (M$_\odot$/pc$^2$)')
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

def read_RCs(dir= './', file= 'You_Should_Set_A_File_RC.txt'):
    with open(f'{dir}{file}','r') as file:
        lines = file.readlines()

    RCs = []

    count = 0
    start = False
    while not start:
        if lines[count][0] != '#':
            start =True
        else:
            count += 1

    for i in range(count,len(lines)):
        values = [x.strip() for x in lines[i].split()]

        for x in range(len(values)):
            if i == count:
                RCs.append([values[x]])
                if values[x] == 'V_OBS':
                    totalind = x
                if values[x] == 'V_OBS_ERR':
                    errind = x
            elif i == count+1:
                RCs[x].append(values[x])
            else:
                RCs[x].append(float(values[x]))
    totalrc = RCs[totalind]
    del RCs[totalind]
    if totalind < errind:
        errind -= 1
    err_rc = RCs[errind]
    del RCs[errind]
    print(RCs)
    return RCs,totalrc,err_rc
    #           RCs

read_RCs.__doc__ =f'''
 NAME:
    read_RCs
 PURPOSE:
    Read the RCs from a file. The file has to adhere to the pyROTMOD output format
 CATEGORY:
    support_functions

 INPUTS:
    di    directory
    RCs = Dictionary with derived RCs
    total_rc = The observed RC
    rc_err = Error on the observed RC

 OPTIONAL INPUTS:
    dir= './'

    Directory where  the file file is locate

    file= 'You_Should_Set_A_File_RC.txt'

    file name

 OUTPUTS:

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:

 NOTE:

'''
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

def write_profiles(in_radii,gas_profile,total_rc,optical_profiles,distance = 1., \
            errors = [0.],output_dir= './', log =None):
    '''Function to write all the profiles to some text files.'''
    radii = ensure_kpc_radii(in_radii,distance=distance,log=log)
    optical_profiles['RADI'] =  ensure_kpc_radii(optical_profiles['RADI'],distance=distance,log=log)
    with open(f'{output_dir}Optical_Mass_Densities.txt','w') as opt_file:
        for x in range(len(optical_profiles['RADI'])):
            line = [optical_profiles[i][x] for i in optical_profiles]
            if x <= 1:
                writel = ' '.join([f'{y:>15s}' for y in line])
            else:
                writel = ' '.join([f'{y:>15.2f}' for y in line])
            writel = f'{writel} \n'
            opt_file.write(writel)
    with open(f'{output_dir}Gas_Mass_Density_And_RC.txt','w') as file:
        for x in range(len(radii)):
            line = [radii[x],gas_profile[x],total_rc[x]]
            if errors[0] != 0.:
                line.append(errors[x])
            if x <= 1:
                writel = ' '.join([f'{y:>15s}' for y in line])
            else:
                writel = ' '.join([f'{y:>15.2f}' for y in line])
            writel = f'{writel} \n'
            file.write(writel)
