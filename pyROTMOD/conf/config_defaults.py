# -*- coding: future_fstrings -*-
# !!!!!!!!!!!!!!!!!!!!!!!! Do not use hydra, it is a piece of shit in organizing the output!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from dataclasses import dataclass,field
from omegaconf import OmegaConf,open_dict,MissingMandatoryValue,ListConfig, MISSING
from typing import List,Optional
from datetime import datetime
from pyROTMOD.support import get_uncounted
import os
import numpy as np
import pyROTMOD.rotmass.potentials as potentials
@dataclass
class General:
    ncpu: int = 6
    log: str = 'log.txt'
    output_dir: str = f'{os.getcwd()}/pyROTMOD_products/'
    RC_file: str = f'RCs_For_Fitting.txt'
    log_directory: str = f'{output_dir}Logs/{datetime.now().strftime("%H:%M:%S-%d-%m-%Y")}/'
    debug: bool = False
    distance: Optional[float] = None  #This uses the vsys from the gas input file
    font: str = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"

@dataclass
class Galaxy_Settings:
    enable: bool = True
    optical_file: Optional[str] = None
    gas_file: Optional[str] = None
    scaleheight: List = field(default_factory=lambda: [0., None]) #If 0 use infinitely thin disks, if vertical mode is none as wel #vertical options are  ['exp', 'sech-sq','sech'] These do not apply to parametrized functions.
    gas_scaleheight: List = field(default_factory=lambda: [0., None]) #If 0 use infinitely thin disks, if vertical mode is none as wel #vertical options are  ['exp', 'sech-sq',''sech-simple'] set secon value to 'tir' to use tirific values
    axis_ratio: float = 1.
    exposure_time: float = 1.
    mass_to_light_ratio: float = 1.0
    band: str='SPITZER3.6'

@dataclass
class Rotmass:
    enable: bool = True
    negative_values: bool = False
    HALO: str = 'NFW'
    optical_lock: bool = True
    gas_lock: bool = True
    mcmc_steps: int= 2000 #Amount of steps per parameter, burn is a quarter
    results_file: str = 'Final_Results'

@dataclass
class ExtendConfig:
    print_examples: bool=False
    configuration_file: Optional[str] = None
    general: General = General()
    RC_Construction: Galaxy_Settings = Galaxy_Settings()
    fitting_general: Rotmass= Rotmass() 


@dataclass
class ShortConfig:
    print_examples: bool=False
    configuration_file: Optional[str] = None
    general: General = General()
    RC_Construction: Galaxy_Settings = Galaxy_Settings()
    input_config: Optional[dict] = None
    file_config: Optional[dict] = None
    fitting_general: Rotmass= Rotmass() 

def read_config(argv):
    
    cfg = OmegaConf.structured(ShortConfig)
    # print the default file
    inputconf = OmegaConf.from_cli(argv)
    short_inputconf = OmegaConf.masked_copy(inputconf,\
                ['print_examples','configuration_file','general','RC_Construction'])
    cfg_input = OmegaConf.merge(cfg,short_inputconf)
   

    if cfg_input.configuration_file:
      
        try:
            yaml_config = OmegaConf.load(cfg_input.configuration_file)
            short_yaml_config =  OmegaConf.masked_copy(yaml_config,\
                    ['print_examples','configuration_file','general','RC_Construction'])
                                                        
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
        exit()


    # read command line arguments anything list input should be set in '' e.g. pyROTMOD 'rotmass.MD=[1.4,True,True]'

   
    cfg = OmegaConf.merge(cfg,short_yaml_config)
    cfg = OmegaConf.merge(cfg,short_inputconf)

   
    
    return cfg


def read_fitting_config(cfg,baryonic_components,print_examples=False):
    halo = 'NFW'
    try:
        halo = cfg.file_config.fitting_general.HALO
    except:
        pass
    try:
        halo = cfg.input_config.fitting_general.HALO
    except:
        pass
    if print_examples:
        baryonic_components = {'DISK_GAS': [],'EXPONENTIAL_1':{},'HERNQUIST_1': []}
        cfg.fitting_general.HALO = halo
    halo_conf = f'{halo}_config'
    cfg_new = OmegaConf.structured(ExtendConfig)
    cfg_new = add_dynamic(cfg_new,baryonic_components,halo = halo_conf)
    cfg_new = create_masked_copy(cfg_new,cfg.file_config)
    cfg_new = create_masked_copy(cfg_new,cfg.input_config) 
    cfg_new.fitting_general.HALO = halo
    return cfg_new

def add_dynamic(in_dict,in_components, halo = 'NFW'):
    halo_config = getattr(potentials,halo)
    with open_dict(in_dict):
        dict_elements = []
        for component_full in in_components:
            component,no = get_uncounted(component_full)
            if component in ['DISK_GAS']:
                dict_elements.append([f'{component_full}',[1.33, None, None,True,True]])
            elif component in ['DISK_STELLAR','EXPONENTIAL', 'SERSIC','BULGE_STELLAR','BULGE', 'HERNQUIST']:
                dict_elements.append([f'{component_full}',[1., None, None,True,True]])  
            
        for key in halo_config.parameters:          
            dict_elements.append([f'{key}',halo_config.parameters[key]]) 
        
        in_dict.fitting_parameters = {}
        for ell in dict_elements:
            in_dict.fitting_parameters[ell[0]] = ell[1]
    return in_dict

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