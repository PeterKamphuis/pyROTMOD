# -*- coding: future_fstrings -*-

# This is an attempt at a holistic python version of ROTMOD and ROTMAS using bayesian fitting and such

#from optparse import OptionParser
from omegaconf import OmegaConf,MissingMandatoryValue

import pyROTMOD
from pyROTMOD.conf.config_defaults import RotModConfig
from pyROTMOD.optical.optical import get_optical_profiles
from pyROTMOD.gas.gas import get_gas_profiles
from pyROTMOD.rotmod.rotmod import convert_dens_rc
from pyROTMOD.rotmass.rotmass import the_action_is_go
from pyROTMOD.support import plot_profiles,write_profiles,write_RCs,print_log,integrate_surface_density,convertskyangle
import pyROTMOD.constants as c
import traceback
import warnings
import os
import sys
import numpy as np
from datetime import datetime
class InputError(Exception):
    pass

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


def main(argv):
    if '-h' in argv or '--help' in argv:
        print(''' Use pyROTMOD in this way:
pyROTMOD -c inputfile.yml   where inputfile is a yaml config file with the desired input settings.
pyROTMOD -h print this message
pyROTMOD -e prints a yaml file (defaults.yml) with the default setting in the current working directory.
in this file values designated ??? indicated values without defaults.

All config parametere can be set directly from the command line by setting the correct parameters, e.g:
pyROTMOD rotmass.HALO=ISO to set the pseudothermal halo.
note that list inout should be set in apostrophes in command line input. e.g.:
pyROTMOD 'rotmass.MD=[1.4,True,True]'
''')
        sys.exit()


    #initialize constants
    c.initialize()
    #initialize default settings
    cfg = OmegaConf.structured(RotModConfig)
    # print the default file
    if '-e' in argv:
        with open('default.yml','w') as default_write:
            default_write.write(OmegaConf.to_yaml(cfg))
        print(f'''We have printed the file default.yml in {os.getcwd()}.
Exiting pyROTMOD.''')
        sys.exit()
    default_output=cfg.general.output_dir
    default_log_directory = cfg.general.log_directory
    #Check for yml file
    if '-c' in argv:
        configfile = argv[argv.index('-c')+1]
        inputconf = OmegaConf.load(configfile)
        #merge yml file with defaults
        cfg = OmegaConf.merge(cfg,inputconf)
        argv.remove('-c')
        argv.remove(configfile)
    # read command line arguments anything list input should be set in '' e.g. pyROTMOD 'rotmass.MD=[1.4,True,True]'
    inputconf = OmegaConf.from_cli(argv)
    cfg = OmegaConf.merge(cfg,inputconf)
    if cfg.general.debug:
        warnings.showwarning = warn_with_traceback
    if cfg.general.output_dir == f'{os.getcwd()}/pyROTMOD_products/':
        cfg.general.output_dir = f'{os.getcwd()}/pyROTMOD_products_{cfg.rotmass.HALO}/'
    if cfg.general.output_dir[-1] != '/':
        cfg.general.output_dir = f"{cfg.general.output_dir}/"

    if default_output != cfg.general.output_dir:
        if default_log_directory == cfg.general.log_directory:
            cfg.general.log_directory=f'{cfg.general.output_dir}Logs/{datetime.now().strftime("%H:%M:%S-%d-%m-%Y")}/'

    if not os.path.isdir(cfg.general.output_dir):
        os.mkdir(cfg.general.output_dir)

    if not os.path.isdir(f"{cfg.general.output_dir}/Logs"):
        os.mkdir(f"{cfg.general.output_dir}/Logs")

    if not os.path.isdir(f"{cfg.general.log_directory}"):
        os.mkdir(f"{cfg.general.log_directory}")

    #write the input to the log dir.
    with open(f"{cfg.general.log_directory}run_input.yml",'w') as input_write:
        input_write.write(OmegaConf.to_yaml(cfg))


    log = f"{cfg.general.log_directory}{cfg.general.log}"
    #If it exists move the previous Log
    if os.path.exists(log):
        os.rename(log,f"{cfg.general.log_directory}/Previous_Log.txt")

    with open(log,'w') as write_log:
        write_log.write(f'''This file is a log of the modelling process run at {datetime.now()}.
This is version {pyROTMOD.__version__} of the program.
''')
        try:
            write_log.write(f'''We are using the input from {cfg.galaxy.gas_file} for the gaseous component.
''')
        except MissingMandatoryValue:
            print(f'''You did not set the gas file input''')
            cfg.galaxy.gas_file = input('''Please add the gas file or tirific output to be evaluated: ''')
            print(cfg.galaxy.gas_file)
            write_log.write(f'''We are using the input from {cfg.galaxy.gas_file} for the gaseous component.
''')
        try:
            write_log.write(f'''We are using the input from {cfg.galaxy.optical_file} for the optical component.
''')
        except MissingMandatoryValue:
            cfg.galaxy.optical_file = input('''Please add the optical or galfit file to be evaluated: ''')
            write_log.write(f'''We are using the input from {cfg.galaxy.optical_file} for the optical component.
''')
        try:
            write_log.write(f'''We are using the following distance = {cfg.galaxy.distance}.
''')
        except MissingMandatoryValue:
            cfg.galaxy.distance= input(f'''Please provide the distance (0 will use vsys and the hubble flow): ''')
            write_log.write(f'''We are using the following distance = {cfg.galaxy.distance}.
''')


    if cfg.galaxy.distance == 0.:
        try:
            vsys = gas_profiles.load_tirific(cfg.galaxy.gas_file, Variables = ['VSYS'])
            if vsys[0] == 0.:
                raise InputError(f'We cannot model profiles adequately without a distance')
            else:
                cfg.galaxy.distance = vsys[0]/c.H_0
        except:
            raise InputError(f'We cannot model profiles adequately without a distance')
        ######################################## Read the optical profiles ##########################################

    try:

        optical_profiles,components,galfit_file = get_optical_profiles(cfg.galaxy.optical_file,
                                                    distance= float(cfg.galaxy.distance),exp_time =1.,
                                                    zero_point_flux = float(cfg.galaxy.zero_point_flux),
                                                    MLRatio= float(cfg.galaxy.mass_to_light_ratio), log= log)
    except Exception as e:
        print_log(f" We could not obtain the optical components and profiles because of {e}",log)
        traceback.print_exc()

    print_log(f"We found the following optical components:\n",log)
    for x in components:
            # Components are returned as [type,integrated magnitude,scale parameter in arcsec,sercic index or scaleheight in arcsec, axis ratio]
        if x[0] in ['expdisk','edgedisk']:
            print_log(f'''We have found an exponential disk with the following values.
The total mass of the disk is {x[1]:.2e} M_sol  a central mass density {x[2]:.2f} M_sol/pc^2 with a M/L {float(cfg.galaxy.mass_to_light_ratio)}.
The scale length is {x[3]:.2f} kpc and the scale height {x[4]:.2f} kpc.
The axis ratio is {x[5]:.2f}.
''' ,log,screen= True)
        if x[0] in ['sersic']:
            print_log(f'''We have found a sersic component with the following values.
The total mass is {x[1]:.2e} a central mass density {x[2]:.2f} M_sol/pc^2 with a M/L {float(cfg.galaxy.mass_to_light_ratio)}.
The effective radius {x[3]:.2f} kpc and sersic index = {x[4]:.2f}.
The axis ratio is {x[5]:.2f}.
''' ,log,screen= True)


    ######################################### Read the gas profiles and RC ################################################
    radii,gas_profile, total_rc,total_rc_err,scaleheights  = get_gas_profiles(cfg.galaxy.gas_file,log=log)
    print_log(f'''We have found a gas disk with a total mass of  {integrate_surface_density(convertskyangle(np.array(radii[2:],dtype=float),float(cfg.galaxy.distance)),np.array(gas_profile[2:],dtype=float)):.2e}
and a central mass density {gas_profile[2]:.2f} M_sol/pc^2.
''' ,log,screen= True)



    ########################################## Make a plot with the extracted profiles ######################3
    plot_profiles(radii, gas_profile,optical_profiles,distance =float(cfg.galaxy.distance),output_dir = cfg.general.output_dir)

    ########################################## Make a nice file with all the different components as a column ######################3
    write_profiles(radii, gas_profile,total_rc,optical_profiles,distance =float(cfg.galaxy.distance), errors = total_rc_err ,output_dir = cfg.general.output_dir )
    ######################################### Convert to Rotation curves ################################################
    derived_RCs = convert_dens_rc(radii, optical_profiles, gas_profile,components,distance =float(cfg.galaxy.distance),galfit_file = galfit_file)
    write_RCs(derived_RCs,total_rc,total_rc_err,output_dir = cfg.general.output_dir)

    ######################################### Run our Bayesian interactive fitter thingy ################################################
    if radii[1] == 'ARCSEC':
        radii[2:] = convertskyangle(np.array(radii[2:],dtype=float),float(cfg.galaxy.distance))
        radii[1]  = 'KPC'
    the_action_is_go(radii,derived_RCs, total_rc,total_rc_err,rotmass_settings=cfg.rotmass,log_directory=cfg.general.log_directory,log=log,debug=cfg.general.debug)

if __name__ =="__main__":
    main()
