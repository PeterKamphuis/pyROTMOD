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
from pyROTMOD.support.profile_classes import Luminosity_Profile, Rotation_Curve,\
    SBR_Profile
from pyROTMOD.support.log_functions import print_log
from pyROTMOD.support.minor_functions import translate_string_to_unit
#from pyROTMOD.optical.conversions import mag_to_lum



'''Read a text file with columns into a density profile'''
def read_columns(filename,cfg=None):
    with open(filename, 'r') as input_text:
        lines= input_text.readlines()
    start = 0
    for lin in lines:
        if lin[0] == '#':
            start += 1
        else:
            break
    input_columns =[x.strip().upper() for x in lines[start].split()]
    units = [x.strip().upper() for x in lines[start+1].split()]
    possible_radius_units = ['KPC','PC','ARCSEC','ARCMIN','DEGREE',]
    allowed_types = ['EXPONENTIAL','SERSIC','DISK','BULGE','DENSITY','HERNQUIST']
    possible_units = ['L_SOLAR/PC^2','M_SOLAR/PC^2','MAG/ARCSEC^2','KM/S','M/S']
    allowed_velocities = ['V_OBS', 'V_ROT','VOBS','VROT']

    #First we select the columns that are legit
    column_indications = {}
    found_input = {}
    for i,column in enumerate(input_columns):
        split_column = column.split('_')
        if split_column[-1].upper() in ['RADII','ERR']:
            continue           
        if split_column[0].upper() not in allowed_types and column.upper() not in allowed_velocities:
            print_log(f'''Column {column} is not a recognized input.
Allowed columns are {', '.join(allowed_types)} or for the total RC {','.join(allowed_velocities)}
''',cfg,case=['main'])
            continue
        column_out = column
        if column[0].upper() == 'V':
            if column[1] != '_':
                column_out = f'V_{column[1:]}'
       
        column_indications[column_out] = {'index':input_columns.index(column)}
        if units[i] in ['KM/S','M/S']:
            found_input[column_out] = Rotation_Curve(name=column_out)
        elif units[i] in ['L_SOLAR/PC^2','MAG/ARCSEC^2']:
            found_input[column_out] = Luminosity_Profile(name=column_out)
        elif units[i] in ['M_SOLAR/PC^2']:
            found_input[column_out] = SBR_Profile(name=column_out)
        elif units[i] in ['M_SOLAR/PC^3']:
            found_input[column_out] = SBR_Profile(name=column_out)
            found_input[column_out].profile_type='density'
        found_input[column_out].values = []
        if f'{column}_ERR' in input_columns:
            column_indications[column_out]['err_index'] = input_columns.index(\
                f'{column}_ERR')
            found_input[column_out].errors = []
        else:
            column_indications[column_out]['err_index'] = None
        if f'{column}_RADII' in input_columns:
            column_indications[column_out]['radii_index'] = input_columns.index(\
                f'{column}_RADII')
            found_input[column_out].radii = []
        elif 'RADII' in input_columns:
            column_indications[column_out]['radii_index'] = input_columns.index(f'RADII')
            found_input[column_out].radii = []
        else:
            raise InputError(f'No radii column found for {column}')
          
    
    #Now we read the columns
    for line in lines[start+2:]:
        input = line.split()
        for column in found_input:
            found_input[column].values.append(float(\
                input[column_indications[column]['index']]))
            if column_indications[column]['err_index'] is not None:
                found_input[column].errors.append(\
                    float(input[column_indications[column]['err_index']]))
            if column_indications[column]['radii_index'] is not None:
                found_input[column].radii.append(\
                    float(input[column_indications[column]['radii_index']]))
    #Check that the columns are not empty
    to_remove  = []
    for incolumn in found_input:
        #Remove nan values
        vals = np.array(found_input[incolumn].values, dtype=float)
        found_input[incolumn].values = vals[~np.isnan(vals)]
        if found_input[incolumn].errors is not None:
            errs =  np.array(found_input[incolumn].errors,dtype=float)
            found_input[incolumn].errors = errs[~np.isnan(vals)]
        radii =  np.array(found_input[incolumn].radii,dtype=float)
        found_input[incolumn].radii = radii[~np.isnan(vals)]

        #Check that the columns are not empty
        if np.sum(found_input[incolumn].values) == 0.:
            print_log(f'The column {incolumn} is empty. We will ignore it.',cfg,case=['main'])
            input_columns.remove(incolumn)
            to_remove.append(incolumn)
    if len(to_remove) > 0:
        for incolumn in to_remove:      
            del found_input[incolumn]
    for type in found_input:
     
        found_input[type].radii = np.array(found_input[type].radii,dtype=float)\
                *translate_string_to_unit(units[column_indications[type]['radii_index']])
                #found_input[type].radii_units = translate_string_to_unit(radii_unit)
        if not found_input[type].values is None:
            found_input[type].values =  np.array(found_input[type].values,dtype=float)\
                *translate_string_to_unit(units[column_indications[type]['index']])
        if not found_input[type].errors is None:
            found_input[type].errors =   np.array(found_input[type].errors,dtype=float)\
                *translate_string_to_unit(units[column_indications[type]['err_index']])
      


    print_log(f'''In {filename} we have processed the following columns:
{', '.join([f'{x}'  for x in found_input])}.
''',cfg,case=['main'])
   
    return found_input
