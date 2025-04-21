# -*- coding: future_fstrings -*-

# This is an attempt at a holistic python version of ROTMOD and ROTMAS using bayesian fitting and such


from pyROTMOD.conf.config_defaults import read_config,read_fitting_config
from pyROTMOD.rotmod.rotmod import obtain_RCs, read_RCs
from pyROTMOD.rotmass.rotmass import rotmass_main
from pyROTMOD.support.minor_functions import check_input, \
    add_font
from pyROTMOD.support.log_functions import print_log
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
    cfg = read_config()
    if cfg.output.debug:
        warnings.showwarning = warn_with_traceback
   
    cfg = check_input(cfg)
    #Add the requested font and get the font name name
    font_name = add_font(cfg.input.font)
   
    if cfg.RC_Construction.enable:
        print_log(f'We start to derive the RCs.\n',cfg,case=['main','screen'])
        derived_RCs,total_rc = obtain_RCs(cfg)
        print_log(f'We managed to derive  the RCs.\n',cfg,case=['main','screen'])
    else:
        #If we have run before we can simple read from the RC file
        print_log(f'We start to read the RCs.\n',cfg,case=['main','screen'])  
        total_rc, derived_RCs = read_RCs(file=f'{cfg.output.output_dir}{cfg.output.RC_file}'\
            ,cfg=cfg)
        print_log(f'We managed to read the RCs.\n',cfg,case=['main','screen'])

    #If there are profiles that are not proper we remove them
    names = [name for name in derived_RCs] 
    for name in names:
        if name[0:3] in ['SKY']:
            del derived_RCs[name]

    ######################################### Run our Bayesian interactive fitter thingy ################################################
    
    #We need to reset the configuration to include the profiles and the parameters to be fitted.
    cfg = read_fitting_config(cfg,derived_RCs)
    cfg= check_input(cfg,fitting =True)   

   
    if cfg.fitting_general.enable:
        if not os.path.isdir( f'{cfg.output.output_dir}{cfg.fitting_general.HALO}/'):
            os.mkdir( f'{cfg.output.output_dir}{cfg.fitting_general.HALO}/')
        #radii = ensure_kpc_radii(derived_RCs[0],distance=cfg.general.distance,log=log )
        rotmass_main(cfg, derived_RCs, total_rc)


if __name__ =="__main__":
    main()
