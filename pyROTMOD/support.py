# -*- coding: future_fstrings -*-

import copy
import numpy as np
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt

from astropy import units as u

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

def write_RCs(RCs,total_rc,rc_err,distance = 1., errors = [0.],\
            log=None, output_dir= './'):
    #print(RCs)
    with open(f'{output_dir}/All_RCs.txt','w') as opt_file:
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
