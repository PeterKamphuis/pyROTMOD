# -*- coding: future_fstrings -*-
# !!!!!!!!!!!!!!!!!!!!!!!! Do not use hydra, it is a piece of shit in organizing the output!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from dataclasses import dataclass,field
import omegaconf
from omegaconf import MISSING
from typing import List,Optional
from datetime import datetime
import os


@dataclass
class General:
    ncpu: int = 6
    log: str = 'log.txt'
    output_dir: str = f'{os.getcwd()}/pyROTMOD_products/'
    log_directory: str = f'{output_dir}Logs/{datetime.now().strftime("%H:%M:%S-%d-%m-%Y")}/'
    debug: bool = False

@dataclass
class Galaxy_Settings:
    optical_file: Optional[str] = None
    gas_file: Optional[str] = None
    distance: Optional[float] = None  #This uses the vsys from the gas input file
    scaleheight: List = field(default_factory=lambda: [0., None]) #If 0 use infinitely thin disks, if vertical mode is none as wel #vertical options are  ['exp', 'sech-sq','sech'] These do not apply to parametrized functions.
    gas_scaleheight: List = field(default_factory=lambda: [0., None]) #If 0 use infinitely thin disks, if vertical mode is none as wel #vertical options are  ['exp', 'sech-sq',''sech-simple'] set secon value to 'tir' to use tirific values
    exposure_time: float = 1.
    mass_to_light_ratio: float = 0.6
    band: str='SPITZER3.6'

@dataclass
class Rotmass:
    enable: bool = True
    MG: List = field(default_factory=lambda: [1.4, None, None,True,True]) #Initial guess, minimum (-1 unset), maximum (-1 unset), fit, included
    MD: List = field(default_factory=lambda: [1., None, None, True,True])
    MB: List = field(default_factory=lambda: [1., None, None, True,True])
    RHO_0: List = field(default_factory=lambda: [None, None, None, True,True])
    R_C: List = field(default_factory=lambda: [None, None, None, True,True])
    C: List = field(default_factory=lambda: [None, None, None, True,True])
    R200: List = field(default_factory=lambda: [None, None, None, True,True])
    negative_values: bool = False
    HALO: str = 'NFW'
    mcmc_steps: int= 2000 #Amount of steps per parameter, burn is a quarter

@dataclass
class RotModConfig:
    print_examples: bool=False
    configuration_file: Optional[str] = None
    general: General = General()
    galaxy: Galaxy_Settings = Galaxy_Settings()
    fitting: Rotmass= Rotmass()
