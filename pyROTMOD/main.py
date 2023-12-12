# -*- coding: future_fstrings -*-

# This is an attempt at a holistic python version of ROTMOD and ROTMAS using bayesian fitting and such

#from optparse import OptionParser


import pyROTMOD
from pyROTMOD.conf.config_defaults import read_config,read_fitting_config
from pyROTMOD.optical.optical import get_optical_profiles
from pyROTMOD.gas.gas import get_gas_profiles
from pyROTMOD.rotmod.rotmod import convert_dens_rc
from pyROTMOD.rotmass.rotmass import rotmass_main
from pyROTMOD.support import plot_profiles,write_profiles,write_RCs,print_log,\
            integrate_surface_density, ensure_kpc_radii,check_input,write_header,\
            read_RCs,add_font
import pyROTMOD.constants as c
import traceback
import warnings
import os
import sys
import numpy as np
class InputError(Exception):
    pass

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


def main(argv):
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

    cfg = read_config(argv)
    if cfg.general.debug:
        warnings.showwarning = warn_with_traceback
 
    default_output=cfg.general.output_dir
    default_log_directory = cfg.general.log_directory


    cfg, log= check_input(cfg,default_output,default_log_directory,pyROTMOD.__version__)
    #Add the requested font and get the name
    font_name = add_font(cfg.general.font)

    if cfg.RC_Construction.enable:
        try:
            optical_profiles,components,galfit_file,optical_vel,original_profiles = \
                get_optical_profiles(cfg.RC_Construction.optical_file,\
                    distance= cfg.general.distance,exposure_time=cfg.RC_Construction.exposure_time,\
                    MLRatio= cfg.RC_Construction.mass_to_light_ratio,band = cfg.RC_Construction.band,\
                    log= log,output_dir=cfg.general.output_dir)

        except Exception as e:
            print_log(f" We could not obtain the optical components and profiles because of {e}",log,debug=cfg.general.debug,screen=False)
            traceback.print_exc()
            raise InputError(f'We failed to retrieved the optical components from {cfg.RC_Construction.optical_file}')


        print_log(f"We found the following optical components:\n",log,debug=cfg.general.debug)
        print(components)
        for x in components:
                # Components are returned as [type,integrated magnitude,scale parameter in arcsec,sercic index or scaleheight in arcsec, axis ratio]
            if x['Type'] in ['expdisk','edgedisk']:
                print_log(f'''We have found an exponential disk with the following values.
''',log,debug=cfg.general.debug)
            elif x['Type'] in ['sersic']:
                print_log(f'''We have found a sersic component with the following values.
''',log,debug=cfg.general.debug)
            elif x['Type'] in ['hernquist']:
                print_log(f'''We have found a hernquist component with the following values.
''',log,debug=cfg.general.debug)
            elif x['Type'] in ['random_disk','random_bulge']:
                print_log(f'''We have found a unparameterized component with the following values.
''',log,debug=cfg.general.debug)
            else:
                print_log(f'''We have found a {x['Type']} component with the following values.
''',log,debug=cfg.general.debug)
            print_log(f'''The total mass of the disk is {x['Total SB']}   a central mass density {x['Central SB']}  with a M/L {float(cfg.RC_Construction.mass_to_light_ratio)}.
The scale length is {x['scale length']}  and the scale height {x['scale height']}.
The axis ratio is {x['axis ratio']}.
''' ,log,debug=cfg.general.debug)



        ######################################### Read the gas profiles and RC ################################################
        radii,gas_profile, total_rc,total_rc_err,scaleheights  = get_gas_profiles(cfg.RC_Construction.gas_file,log=log,debug =cfg.general.debug)

        if gas_profile[1] == 'M_SOLAR/PC^2':
            correct_rad = ensure_kpc_radii(radii,distance=cfg.general.distance)
            print_log(f'''We have found a gas disk with a total mass of  {integrate_surface_density(correct_rad[2:],np.array(gas_profile[2:],dtype=float)):.2e}
    and a central mass density {gas_profile[2]:.2f} M_sol/pc^2.
    ''' ,log,screen= True)


        ########################################## Make a plot with the extracted profiles ######################3
        plot_profiles(radii, gas_profile,optical_profiles,\
            distance =float(cfg.general.distance),\
            output_dir = cfg.general.output_dir, log=log,\
            input_profiles=original_profiles)

        ########################################## Make a nice file with all the different components as a column ######################3
        write_profiles(radii, gas_profile,total_rc,optical_profiles,\
            distance =float(cfg.general.distance), errors = total_rc_err ,\
            output_dir = cfg.general.output_dir, log=log)
        ######################################### Convert to Rotation curves ################################################
        if cfg.RC_Construction.scaleheight[0] == 0.:
            opt_h_z = [cfg.RC_Construction.scaleheight[0],None]
        else:
            opt_h_z = cfg.RC_Construction.scaleheight
        if cfg.RC_Construction.gas_scaleheight[1]:
            cfg.RC_Construction.gas_scaleheight[1]=cfg.RC_Construction.gas_scaleheight[1].lower()

        if cfg.RC_Construction.gas_scaleheight[1] == 'tir':
            gas_hz = [np.nanmean(scaleheights[0,1]),scaleheights[2]]
        elif cfg.RC_Construction.gas_scaleheight[0] != 0:
            gas_hz = cfg.RC_Construction.gas_scaleheight
        else:
            gas_hz= [0.,None]

        derived_RCs = convert_dens_rc(radii, optical_profiles, gas_profile,\
                components,distance =cfg.general.distance,log=log,
                galfit_file = galfit_file,opt_h_z = opt_h_z,
                gas_scaleheight=gas_hz,output_dir=cfg.general.output_dir)

        write_header(distance=cfg.general.distance,MLratio=cfg.RC_Construction.mass_to_light_ratio,\
            opt_scaleheight=opt_h_z,gas_scaleheight=gas_hz,\
            output_dir = cfg.general.output_dir, file= cfg.general.RC_file)
        write_RCs(derived_RCs,total_rc,total_rc_err, log = log, \
                    output_dir = cfg.general.output_dir, file= cfg.general.RC_file)
    else:
        print_log(f'We start to read the RCs.\n',log)
        derived_RCs,total_rc,total_rc_err = read_RCs(dir=cfg.general.output_dir, file=cfg.general.RC_file)
        print_log(f'We managed to read the RCs.\n',log)

    ######################################### Run our Bayesian interactive fitter thingy ################################################
    baryonic_components = [x[0] for x in derived_RCs if x[0] != 'RADI']
  
    cfg = read_fitting_config(cfg,baryonic_components)
    cfg, log= check_input(cfg,default_output,default_log_directory,pyROTMOD.__version__)    
   
    if cfg.fitting_general.enable:
        if not os.path.isdir( f'{cfg.general.output_dir}{cfg.fitting_general.HALO}/'):
            os.mkdir( f'{cfg.general.output_dir}{cfg.fitting_general.HALO}/')
        radii = ensure_kpc_radii(derived_RCs[0],distance=cfg.general.distance,log=log )
        rotmass_main(radii,derived_RCs, total_rc,total_rc_err,\
            out_dir = f'{cfg.general.output_dir}{cfg.fitting_general.HALO}/',\
            rotmass_settings=cfg.fitting_general,log_directory=cfg.general.log_directory,\
            rotmass_parameter_settings = cfg.fitting_parameters,\
            results_file = cfg.fitting_general.results_file,\
            log=log,debug=cfg.general.debug, font=font_name)

if __name__ =="__main__":
    main()
