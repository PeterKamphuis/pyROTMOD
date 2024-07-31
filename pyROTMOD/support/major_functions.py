# -*- coding: future_fstrings -*-

import copy
import numpy as np
import os
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as mpl_fm

from astropy import units as u

from pyROTMOD.support.errors import InputError
from pyROTMOD.support.classes import Density_Profile, Rotation_Curve
from pyROTMOD.support.minor_functions import print_log, convertskyangle,\
    translate_string_to_unit
#from pyROTMOD.optical.conversions import mag_to_lum



'''Read a text file with columns into a density profile'''
def read_columns(filename,optical=True,gas=False,debug=False,log=None):
    with open(filename, 'r') as input_text:
        lines= input_text.readlines()

    input_columns =[x.strip().upper() for x in lines[0].split()]
    units = [x.strip().upper() for x in lines[1].split()]
   
    possible_radius_units = ['KPC','PC','ARCSEC','ARCMIN','DEGREE',]
    allowed_types = ['RADII','EXPONENTIAL','SERSIC','DISK','BULGE','DENSITY','HERNQUIST']
    possible_units = ['L_SOLAR/PC^2','M_SOLAR/PC^2','MAG/ARCSEC^2','KM/S','M/S']
    allowed_velocities = ['V_OBS', 'V_ROT']


    same_radii = False
    if 'RADII' in input_columns:
        same_radii = True
        radii = []
        radii_unit = units[input_columns.index('RADII')]
      
    
    found_input = {}
  
    for type in input_columns:
        if type[-5:] != 'RADII' and  type[-3:] != 'ERR':
            if units[input_columns.index(type)] in ['KM/S','M/S']:
                found_input[type] = Rotation_Curve(type=type.split('_')[0]\
                                                   ,name=type)
            elif units[input_columns.index(type)] in ['L_SOLAR/PC^2','M_SOLAR/PC^2','MAG/ARCSEC^2']:
                found_input[type] = Density_Profile(type,type)
                   
      

    for line in lines[2:]:
        input = line.split()
        for i,type in enumerate(input_columns):
            if type[-5:] == 'RADII':
                if units[i] not in possible_radius_units:
                    raise InputError(f'''Your RADI column in the input file {filename} does not have the right units.
Possible units are: {', '.join(possible_radius_units)}. Yours is {units[i]}.''')   
                if same_radii:
                    radii.append(input[i]) 
                else:
                    if found_input[type[:-6]].radii is None:
                        found_input[type[:-6]].radii = [input[i]]
                    else:  
                        found_input[type[:-6]].radii.append(input[i])  
                    if found_input[type[:-6]].radii_units is None:
                        found_input[type[:-6]].radii_units =\
                            translate_string_to_unit(units[input_columns.index(type)])
            elif type[-3:] == 'ERR':
                if units[i] != units[input_columns.index(type[:-4])]:
                    raise InputError(f'''The units of your profile {type[:-4]} and your errors ({type}) differ.''')
                if found_input[type[:-4]].errors is None:
                    found_input[type[:-4]].errors = [input[i]]
                else:  
                    found_input[type[:-4]].errors.append(input[i])  
            else:
                if type.split('_')[0] not in allowed_types and type not in allowed_velocities:
                    raise InputError(f'''Column {type} is not a recognized input.
Allowed columns are {', '.join(allowed_types)} or for the total RC {','.join(allowed_velocities)}''')
                if type in allowed_velocities and units[i] not in ['KM/S','M/S']:
                    raise InputError(f'''Column {type} has to have units of velocity so either KM/S or M/S''')
                elif units[i] not in possible_units:
                    raise InputError(f'''Column {type} has to have units of {', '.join(possible_units)}
the unit {units[i]} can not be processed.''')

                if found_input[type].values is None:
                    found_input[type].values = [input[i]]
                else:
                    found_input[type].values.append(input[i])  
                if found_input[type].units is None:
                    found_input[type].units =\
                            translate_string_to_unit(units[input_columns.index(type)])
    # Check that all profiles have a radii
  
    for type in found_input:
        if same_radii:
            # Check that all profiles have a radii
            if found_input[type].radii is None:
                found_input[type].radii = radii
                found_input[type].radii_units = translate_string_to_unit(radii_unit)


    print_log(f'''In {filename} we have processed the following columns:
{', '.join([f'{x} ({y})' for x,y in zip(input_columns,units)])}.
''',log)
    return found_input
