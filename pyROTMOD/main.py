# -*- coding: future_fstrings -*-

# This is an attempt at a holistic python version of ROTMOD and ROTMAS using bayesian fitting and such


from pyROTMOD.conf.config_defaults import read_config,read_fitting_config
from pyROTMOD.rotmod.rotmod import obtain_RCs, read_RCs
from pyROTMOD.rotmass.rotmass import rotmass_main
from pyROTMOD.support.minor_functions import check_arguments, check_input, \
    add_font,print_log
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


def main():
    'The main should be simple'
    argv = check_arguments()
    cfg = read_config(argv)
    if cfg.general.debug:
        warnings.showwarning = warn_with_traceback
 
    cfg, log= check_input(cfg)
    #Add the requested font and get the font name name
    font_name = add_font(cfg.general.font)
   
    if cfg.RC_Construction.enable:
        print_log(f'We start to derive the RCs.\n',log)
        derived_RCs,total_rc = obtain_RCs(cfg,log=log)
        print_log(f'We managed to derive  the RCs.\n',log)
    else:
        #If we have run before we can simple read from the RC file
        print_log(f'We start to read the RCs.\n',log)
        derived_RCs,total_rc = read_RCs(dir=cfg.general.output_dir, file=cfg.general.RC_file)
        print_log(f'We managed to read the RCs.\n',log)

    ######################################### Run our Bayesian interactive fitter thingy ################################################
    baryonic_components = [x[0] for x in derived_RCs if x[0] != 'RADI']
    
    #We need to reset the configuration to include the profiles and the parameters to be fitted.
    cfg = read_fitting_config(cfg,baryonic_components)
    cfg, log= check_input(cfg)   

   
    if cfg.fitting_general.enable:
        if not os.path.isdir( f'{cfg.general.output_dir}{cfg.fitting_general.HALO}/'):
            os.mkdir( f'{cfg.general.output_dir}{cfg.fitting_general.HALO}/')
        #radii = ensure_kpc_radii(derived_RCs[0],distance=cfg.general.distance,log=log )
        rotmass_main(derived_RCs, total_rc,\
            out_dir = f'{cfg.general.output_dir}{cfg.fitting_general.HALO}/',\
            rotmass_settings=cfg.fitting_general,log_directory=cfg.general.log_directory,\
            rotmass_parameter_settings = cfg.fitting_parameters,\
            results_file = cfg.fitting_general.results_file,\
            log=log,debug=cfg.general.debug, font=font_name)

if __name__ =="__main__":
    main()
