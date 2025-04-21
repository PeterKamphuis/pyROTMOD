# -*- coding: future_fstrings -*-
# !!!!!!!!!!!!!!!!!!!!!!!! Do not use hydra, it is a piece of shit in organizing the output!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from dataclasses import dataclass,field
from omegaconf import OmegaConf,open_dict
from typing import List,Optional
from datetime import datetime
from pyROTMOD.support.minor_functions import get_uncounted
import os
import sys
import psutil
import pyROTMOD.rotmass.potentials as potentials
import pyROTMOD

@dataclass
class Input:
    try:
        ncpu: int = len(psutil.Process().cpu_affinity())
    except AttributeError:
        ncpu: int = psutil.cpu_count()-1
    RC_file: Optional[str] = None
    distance: Optional[float] = None  #This uses the vsys from the gas input file
    font: str = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"

@dataclass
class Output:
    RC_file: str = f'RCs_For_Fitting.txt'
    out_base: str = 'Final_Results'
    log: str = 'log.txt'
    output_dir: str = f'{os.getcwd()}/pyROTMOD_products/'
    log_directory: str = f'{output_dir}Logs/{datetime.now().strftime("%H:%M:%S-%d-%m-%Y")}/'
    debug: bool = False
    debug_functions: List = field(default_factory=lambda: ['ALL'])
    verbose: bool = False
    chain_data: bool = False #If True we save the chain data in a file
   
@dataclass
class RCConstruction:
    enable: bool = True
    out_base: str = ''
    optical_file: Optional[str] = None
    gas_file: Optional[str] = None
    #Scale height for the optical profiles provided as  [value, error, unit, type]
    #If value = 0 or or type = None use infinitely thin disks (inf_thin),
    #Other type options are ['exp', 'sech-sq','sech', 'constant', 'lorentzian']. If a galfit fit provides a scale height this takes precedence
    # 0 for a bulge profile will be a spherical bulge
    # units can be 'PC', "KPC', "ARCSEC', 'ARCMIN', 'DEGREE'
    scaleheight: List = field(default_factory=lambda: [0., None, 'KPC', 'inf_thin']) 
    # truncation radius at which point the profile will gaussian tapered to 0.
    # given as radius, softening_length as fraction of scalelength, unit
    truncation_radius: List = field(default_factory=lambda: [None, 0.2,  'KPC']) 
    # Same for gas, if read from def file this takes precedence
    gas_scaleheight: List = field(default_factory=lambda: [0., None,  'KPC', 'inf_thin']) 
    gas_truncation_radius: List = field(default_factory=lambda: [None, 0.2, 'KPC']) 
    axis_ratio: float = 1.
    exposure_time: float = 1.
    mass_to_light_ratio: float = 1.0
    keep_random_profiles: bool = False #If we have random profiles in Lsun/pc^2
    # we keep them and assume they are SBR_Profile when multiplied with MLratio, 
    # if set to false we attempt to fit a profile to them with known functions.
    band: str='SPITZER3.6'
    gas_band: str = '21cm'

@dataclass
class Fitting:
    enable: bool = True
    negative_values: bool = False
    initial_minimizer: str = 'differential_evolution' #Not  functioning for numpyro backend
    HALO: str = 'NFW'
    single_stellar_ML: bool = True
    fixed_stellar_ML: bool = True # If set to false individual settings in an input yaml still take precedence
    fixed_gas_ML: bool = True # If set to false individual settings in an input yaml still take precedence
    single_gas_ML: bool = False
    mcmc_steps: int= 2000 #Amount of steps per parameter
    burn: int = 500 #Number of steps to discard in MCMC chain per   free parameter
    numpyro_chains: Optional[int] = None # If not set it will be set to the number of available cpus
    use_gp: bool = True    #numpyro uses tingp and lmfit uses sklearn
    gp_kernel_type: str = 'RBF'
    backend : str = 'numpyro' # lmfit or numpyro
    max_iterations: int = 5
    
@dataclass
class ExtendConfig:
    print_examples: bool=False
    configuration_file: Optional[str] = None
    input: Input = field(default_factory = Input)
    output: Output = field(default_factory = Output)
    RC_Construction: RCConstruction = field(default_factory = RCConstruction)
    fitting_general: Fitting = field(default_factory = Fitting) 

@dataclass
class ShortConfig:
    print_examples: bool=False
    configuration_file: Optional[str] = None
    input: Input = field(default_factory = Input)
    output: Output = field(default_factory = Output)
    RC_Construction: RCConstruction = field(default_factory = RCConstruction)
    input_config: Optional[dict] = None
    file_config: Optional[dict] = None
    fitting_general: Fitting = field(default_factory = Fitting)

def add_dynamic(in_dict,in_components, halo = 'NFW'):
    halo_config = getattr(potentials,halo)
    with open_dict(in_dict):
        dict_elements = []
        for name in in_components:
            component,no = get_uncounted(in_components[name].name)
            if 'GAS' in component.upper():
                dict_elements.append([f'{in_components[name].name}',
                    [1.33, None, None,not in_dict['fitting_general']['fixed_gas_ML'],True]])
            else:
                dict_elements.append([f'{in_components[name].name}',
                    [1., None, None,not in_dict['fitting_general']['fixed_stellar_ML'],True]])  
            
        for key in halo_config.parameters:          
            dict_elements.append([f'{key}',halo_config.parameters[key]]) 
        
        in_dict.fitting_parameters = {}
        for ell in dict_elements:
            in_dict.fitting_parameters[ell[0]] = ell[1]
    return in_dict

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




def create_masked_copy(cfg_out,cfg_in): 
   
    mask = []
    
    try:
        for key in cfg_in.__dict__['_content']:
            if key != 'fitting_parameters':
                mask.append(key)
        cfg_file = OmegaConf.masked_copy(cfg_in ,\
                            mask)
        with open_dict(cfg_file):
            if 'fitting_parameters' in cfg_in.__dict__['_content']:
                cfg_file.fitting_parameters = {}
                for key in cfg_in.__dict__['_content']['fitting_parameters']:
                    if key in cfg_out.__dict__['_content']['fitting_parameters']:
                        array_unchecked = cfg_in.fitting_parameters[key]
                        array_correct = cfg_out.fitting_parameters[key]
                        for i,element in enumerate(array_unchecked):
                            if not isinstance(element,type(array_correct[i])):
                                array_unchecked[i] = correct_type(array_unchecked[i],type(array_correct[i]))
                        cfg_file.fitting_parameters[key] =  array_unchecked
    except AttributeError:
        cfg_file = {}
    cfg_out = OmegaConf.merge(cfg_out,cfg_file )
    return cfg_out


def correct_type(var,ty):
    if ty == int:
        var = int(var)
    elif ty == bool:
        if isinstance(var,str):  
            if var[0].lower() == 't':
                var = True
            elif var[0].lower() == 'f':
                var = False
            else:
                var = bool(var)
        else:
            var = bool(var)
    elif ty in [float]:
        var = float(var)
    elif ty == type(None):
        if var is None:
            pass
        else:
            try:
                var = float(var)
            except ValueError:
                var = None
    return var    


def read_config():
    argv = check_arguments()
    cfg = OmegaConf.structured(ShortConfig)
    # print the default file
    inputconf = OmegaConf.from_cli(argv)
    short_inputconf = OmegaConf.masked_copy(inputconf,\
                ['print_examples','configuration_file','input','output','RC_Construction'])
    cfg_input = OmegaConf.merge(cfg,short_inputconf)
   

    if cfg_input.configuration_file:
        try:
            yaml_config = OmegaConf.load(cfg_input.configuration_file)
            short_yaml_config =  OmegaConf.masked_copy(yaml_config,\
                    ['print_examples','configuration_file','input','output','RC_Construction'])
                                                        
    #merge yml file with defaults
          
        except FileNotFoundError:
            cfg_input.configuration_file = input(f'''
You have provided a config file ({cfg_input.configuration_file}) but it can't be found.
If you want to provide a config file please give the correct name.
Else press CTRL-C to abort.
configuration_file = ''')
    else:
        yaml_config = {}
        short_yaml_config = {}
    cfg.input_config = inputconf
    cfg.file_config = yaml_config
    if cfg_input.print_examples:
        cfg = read_fitting_config(cfg,'',print_examples=True)
        with open('ROTMOD-default.yml','w') as default_write:
            default_write.write(OmegaConf.to_yaml(cfg))
        print(f'''We have printed the file ROTMOD-default.yml in {os.getcwd()}.
Exiting pyROTMOD.''')
        sys.exit()


    # read command line arguments anything list input should be set in '' e.g. pyROTMOD 'rotmass.MD=[1.4,True,True]'

   
    cfg = OmegaConf.merge(cfg,short_yaml_config)
    cfg = OmegaConf.merge(cfg,short_inputconf) 
    return cfg



def read_fitting_config(cfg,baryonic_RCs,print_examples=False):
    halo = 'NFW' 
    try:
        halo = cfg.file_config.fitting_general.HALO
    except:
        pass
    try:
        halo = cfg.input_config.fitting_general.HALO
    except:
        pass
    halo = halo.upper()
    if halo == 'MOND':
        halo = 'MOND_CLASSIC'
    if print_examples:
        baryonic_components = {'DISK_GAS_1': [],'EXPONENTIAL_1':{},'HERNQUIST_1': []}
        cfg.fitting_general.HALO = halo
    halo_conf = f'{halo}_config'
    cfg_new = OmegaConf.structured(ExtendConfig)
    cfg_new = add_dynamic(cfg_new,baryonic_RCs,halo = halo_conf)
    cfg_new = create_masked_copy(cfg_new,cfg.file_config)
    cfg_new = create_masked_copy(cfg_new,cfg.input_config) 
    cfg_new.fitting_general.HALO = halo
    return cfg_new
