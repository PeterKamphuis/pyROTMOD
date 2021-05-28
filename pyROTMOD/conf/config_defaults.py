# -*- coding: future_fstrings -*-
#import hydra
from dataclasses import dataclass,field
import omegaconf
from omegaconf import MISSING
from typing import List
import os

@dataclass
class General:
    ncpu: int = 6
    log: str = 'log.txt'
    output_dir: str = f'{os.getcwd()}/pyROTMOD_products'

@dataclass
class Galaxy_Settings:
    optical_file: str = MISSING
    gas_file: str = MISSING
    distance: str = MISSING
    zero_point_flux: float = 280.9 # This is actually the spitzer zero point flux from https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/17/
    mass_to_light_ratio: float = 0.6
@dataclass
class Rotmass:
    MG: List = field(default_factory=lambda: [1.4, 1.,1.])
    MD: List = field(default_factory=lambda: [1., True,True])
    MB: List = field(default_factory=lambda: [1., False,False])
    HALO: str = 'NFW'

@dataclass
class RotModConfig:
    general: General = General()
    galaxy: Galaxy_Settings = Galaxy_Settings()
    rotmass: Rotmass= Rotmass()
