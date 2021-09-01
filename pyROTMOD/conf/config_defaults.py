# -*- coding: future_fstrings -*-
# !!!!!!!!!!!!!!!!!!!!!!!! Do not use hydra, it is a piece of shit in organizing the output!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from dataclasses import dataclass,field
import omegaconf
from omegaconf import MISSING
from typing import List
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
    optical_file: str = MISSING
    gas_file: str = MISSING
    distance: float = 0.   #This uses the vsys from the gas input file
    zero_point_flux: float = 280.9 # This is actually the spitzer zero point flux from https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/17/
    mass_to_light_ratio: float = 0.6
# For WISE
        # input_parameters.zero_point_flux =309.504                       #Jy From the WISE photometry website http://wise2.ipac.caltech.edu/docs/release/prelim/expsup/sec4_3g.html

@dataclass
class Rotmass:
    MG: List = field(default_factory=lambda: [1.4, True,True])
    MD: List = field(default_factory=lambda: [1., True,True])
    MB: List = field(default_factory=lambda: [1., True,True])
    negative_ML: bool = False
    HALO: str = 'NFW'

@dataclass
class RotModConfig:
    general: General = General()
    galaxy: Galaxy_Settings = Galaxy_Settings()
    rotmass: Rotmass= Rotmass()
