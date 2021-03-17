# -*- coding: future_fstrings -*-


import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt



# function for converting kpc to arcsec and vice versa

def convertskyangle(angle, distance=1., unit='arcsec', distance_unit='Mpc', physical=False,debug = False):
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
    if len(kpc) == 1:
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

def plot_profiles(radii,gas_profile,optical_profiles,distance = 1., errors = [0.],output_dir = './'):
    plt.plot(convertskyangle(radii[2:],distance=distance),np.array(gas_profile[2:]),label = gas_profile[0])
    max = np.nanmax(np.array(gas_profile[2:]))
    lower_ind = np.where(np.array(optical_profiles[0][2:]) > float(radii[2])/2.)[0][0]
    tot_opt = []
    for x in range(1,len(optical_profiles)):
        plt.plot(convertskyangle(optical_profiles[0][2:],distance=distance),np.array(optical_profiles[x][2:]), label = optical_profiles[x][0])
        if np.nanmax(np.array(optical_profiles[x][2+lower_ind:])) > max:
            max =  np.nanmax(np.array(optical_profiles[x][2+lower_ind:]))
        if len(tot_opt) > 0:
            tot_opt = [old+new for old,new in zip(tot_opt,optical_profiles[x][2:])]
        else:
            tot_opt =  optical_profiles[x][2:]
    plt.plot(convertskyangle(optical_profiles[0][2:],distance=distance),np.array(tot_opt), label='Total Optical')
    max = np.nanmax(tot_opt)
    plt.ylim(0.1,max)
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

def write_RCs(RCs,total_rc,rc_err,distance = 1., errors = [0.],output_dir= './'):
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

def write_profiles(radii,gas_profile,total_rc,optical_profiles,distance = 1., errors = [0.],output_dir= './'):
    with open(f'{output_dir}Optical_Mass_Densities.txt','w') as opt_file:
        for x in range(len(optical_profiles[0])):
            line = [optical_profiles[i][x] for i in range(len(optical_profiles))]
            if x == 1:
                line[0] = 'KPC'
            if x > 1:
                line[0] = convertskyangle(float(line[0]), distance)
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
            if x == 1:
                line[0] = 'KPC'
            if x > 1:
                line[0] = convertskyangle(float(line[0]), distance)
            if x <= 1:
                writel = ' '.join([f'{y:>15s}' for y in line])
            else:
                writel = ' '.join([f'{y:>15.2f}' for y in line])
            writel = f'{writel} \n'
            file.write(writel)
