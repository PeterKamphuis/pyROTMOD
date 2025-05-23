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
import logging
class InputError(Exception):
    pass

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
def main_trace():
    from viztracer import VizTracer
    with VizTracer(output_file="pyROTMOD_Run_Viztracer.json",min_duration=1000) as tracer:
        main()

def main():
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.font').disabled = True
    'The main should be simple'
    cfg = read_config()
    if cfg.output.debug:
        warnings.showwarning = warn_with_traceback
    # Let's assume that reading the config file is not going to be a problem
    try:
        cfg = check_input(cfg)
        #Need to set the number of threads for XLA else it will regularly see only one
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cfg.input.ncpu}"
        # Let's assume that reading the config file is not going to be a problem
        #Add the requested font and get the font name name
    
        if cfg.RC_Construction.enable:
            print_log(f'We start to derive the RCs.\n',cfg,case=['main'])
            derived_RCs,total_rc = obtain_RCs(cfg)
            print_log(f'We managed to derive  the RCs.\n',cfg,case=['main'])
        else:
            #If we have run before we can simple read from the RC file
            print_log(f'We start to read the RCs.\n',cfg,case=['main'])  
            total_rc, derived_RCs = read_RCs(file=f'{cfg.output.output_dir}{cfg.output.RC_file}'\
                ,cfg=cfg)
            print_log(f'We managed to read the RCs.\n',cfg,case=['main'])

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
    except KeyboardInterrupt:
        traceback.print_exception(*sys.exc_info())
        print_log(f'KeyboardInterrupt: The program was interrupted by the user.\n',cfg,case=['main','screen'])
        pass
    except Exception as e:
        traceback.print_exception(*sys.exc_info())
        print_log(f'Error: {e}\n',cfg,case=['main'])
        print_log(f'The program was interrupted by an error.\n',cfg,case=['main','screen'])
        pass
    


if __name__ =="__main__":
    main()
