# -*- coding: future_fstrings -*-
import numpy as np
from pyROTMOD.support.major_functions import read_columns
from pyROTMOD.support.minor_functions import print_log, propagate_mean_error,\
    translate_string_to_unit

from pyROTMOD.support.errors import InputError
from pyROTMOD.support.profile_classes import SBR_Profile,Rotation_Curve
from astropy import units as unit
#Function to convert column densities

def columndensity(levels,systemic = 100.,beam=[1.,1.],channel_width=1.,column= False,arcsquare=False,solar_mass_input =False,solar_mass_output=False, debug = False,log=None):
    if debug:
        print_log(f'''COLUMNDENSITY: Starting conversion from the following input.
{'':8s}Levels = {levels}
{'':8s}Beam = {beam}
{'':8s}channel_width = {channel_width}
''',log,debug =True)
    beam=np.array(beam)
    f0 = 1.420405751786E9 #Hz rest freq
    c = 299792.458 # light speed in km / s
    pc = 3.086e+18 #parsec in cm
    solarmass = 1.98855e30 #Solar mass in kg
    mHI = 1.6737236e-27 #neutral hydrogen mass in kg
    if debug:
                print_log(f'''COLUMNDENSITY: We have the following input for calculating the columns.
{'':8s}COLUMNDENSITY: level = {levels}, channel_width = {channel_width}, beam = {beam}, systemic = {systemic})
''',log,debug=debug)
    if systemic > 10000:
        systemic = systemic/1000.
    f = f0 * (1 - (systemic / c)) #Systemic frequency
    if arcsquare:
        HIconv = 605.7383 * 1.823E18 * (2. *np.pi / (np.log(256.)))
        if column:
            # If the input is in solarmass we want to convert back to column densities
            if solar_mass_input:
                levels=levels*solarmass/(mHI*pc**2)
            #levels=levels/(HIconv*channel_width)
            levels = levels/(HIconv*channel_width)
        else:

            levels = HIconv*levels*channel_width
            if solar_mass_output:
                levels=levels*mHI/solarmass*pc*pc
    else:
        if beam.size <2:
            beam= [beam,beam]
        b=beam[0]*beam[1]
        if column:
            if solar_mass_input:
                levels=levels*solarmass/(mHI*pc**2)
            TK = levels/(1.823e18*channel_width)
            levels = TK/(((605.7383)/(b))*(f0/f)**2)
        else:
            TK=((605.7383)/(b))*(f0/f)**2*levels
            levels = TK*(1.823e18*channel_width)
    if ~column and solar_mass_input:
        levels = levels*mHI*pc**2/solarmass
    return levels
columndensity.__doc__ = '''
;+
; NAME:
;       columndensity(levels,systemic = 100.,beam=[1.,1.],channel_width=1.,column= False,arcsquare=False,solar_mass_input =False,solar_mass_output=False)
;
; PURPOSE:
;       Convert the various surface brightnesses to other values
;
; CATEGORY:
;       Support
;
; INPUTS:
;       levels = the values to convert
;       systemic = the systemic velocity of the source
;        beam  =the beam in arcse
;       channelwidth = width of a channel in km/s
;     column = if true input is columndensities
;
;
; OPTIONAL INPUTS:
;
;
; KEYWORD PARAMETERS:
;       -
;
; OUTPUTS:
;
; OPTIONAL OUTPUTS:
;       -
;
; MODULES CALLED:
;
;
; EXAMPLE:
;

'''

def get_individual_tirific_disk(disk,filename,log=None, debug =False):

    if disk == 1:
        RC = Rotation_Curve(type='random_disk')
        sbr = SBR_Profile(type='random_disk')
        ext1 =''
    else:
        sbr = SBR_Profile(type='random_disk')
        ext1=f'_{disk}'
    ext2 = f'_{disk+1}'
    sbr.profile_type = 'sbr_dens'
  
    if disk == 1: 
        Variables =  ['RADI','VROT','VROT_2','VROT_ERR',\
                      'VROT_2_ERR']
    else:
        Variables = []
    
    Variables =Variables + [f'SBR{ext1}',f'SBR{ext2}',f'Z0{ext1}',f'Z0{ext2}'\
                    ,f'LTYPE{ext1}',f'LTYPE{ext2}',f'VSYS{ext1}',\
                    f'SBR{ext1}_ERR',f'SBR{ext2}_ERR',f'Z0{ext1}_ERR',\
                    f'Z0{ext2}_ERR']
    
    value_dictionary = load_tirific(filename, Variables=Variables, dict=True)  
  
    sbr.height = np.mean(value_dictionary[f'Z0{ext1}']+value_dictionary[f'Z0{ext2}'])*unit.arcsec
  
    if np.sum(value_dictionary[f'Z0{ext1}_ERR']+value_dictionary[f'Z0{ext2}_ERR']) != 0.:
    
        sbr.height_error = np.mean(value_dictionary[f'Z0{ext1}_ERR']+\
                                value_dictionary[f'Z0{ext2}_ERR'])*unit.arcsec
                                
   
    if  value_dictionary[f'LTYPE{ext1}'][0] != value_dictionary[f'LTYPE{ext2}'][0] :
        print_log(f'''Your def file has different vertical functions. 
We always assume the function used in the first disk ({value_dictionary[f"LTYPE{ext1}"][0]}) instead of the second ({value_dictionary[f"LTYPE{ext2}"][0]})''',log)
    if value_dictionary[f'LTYPE{ext1}'][0] == 0.:
        sbr.height_type = 'constant'
    elif value_dictionary[f'LTYPE{ext1}'][0] == 1.:
        sbr.height_type = 'gaussian'
    elif value_dictionary[f'LTYPE{ext1}'][0] == 2:
        sbr.height_type = 'sech-sq'
    elif value_dictionary[f'LTYPE{ext1}'][0] == 3:
        sbr.height_type = 'exp'
    elif value_dictionary[f'LTYPE{ext1}'][0] == 4:
        sbr.height_type = 'lorentzian'
    
    av = np.array([(x1+x2)/2.*1000. for x1,x2 in zip(\
        value_dictionary[f'SBR{ext1}'],value_dictionary[f'SBR{ext2}'])],dtype=float)
   
    sbr.values = np.array(columndensity(av,arcsquare = True , solar_mass_output = True,\
                         systemic= value_dictionary[f'VSYS{ext1}'][0]),dtype=float)*unit.Msun/unit.pc**2
    if np.sum(value_dictionary[f'SBR{ext1}_ERR']+value_dictionary[f'SBR{ext2}_ERR']) != 0.:
        length = len(value_dictionary[f'SBR{ext1}_ERR']+value_dictionary[f'SBR{ext2}_ERR'])
        av_errors = np.array([(x+y)/2.*1000. for x,y in zip(
                            value_dictionary[f'SBR{ext1}_ERR'],\
                            value_dictionary[f'SBR{ext2}_ERR'] )],dtype=float)
        sbr.errors = np.array(columndensity(av_errors,arcsquare = True , solar_mass_output = True,\
                         systemic= value_dictionary[f'VSYS{ext1}'][0]),dtype=float)*unit.Msun/unit.pc**2
   
    sbr.radii = np.array(value_dictionary['RADI'],dtype=float)*unit.arcsec
    
    if disk == 1:
        RC.radii = np.array(value_dictionary['RADI'],dtype=float)*unit.arcsec
        
        RC.values =  np.array([(x1+x2)/2. for x1,x2 in\
            zip(value_dictionary['VROT'],value_dictionary['VROT_2']) ],\
            dtype=float)*unit.km/unit.s
       
        RC.errors = np.array([(x1+x2)/2. for x1,x2 in\
            zip(value_dictionary['VROT_ERR'],value_dictionary['VROT_2_ERR']) ],\
            dtype=float)*unit.km/unit.s
      
        return  sbr, RC
    else:
        return sbr


      
       


def get_gas_profiles(cfg,log=None, debug =False):
    filename = cfg.RC_Construction.gas_file
  
    print_log(f"Reading the gas density profile from {filename}. \n",log,screen =True )

    gas_density = {}
    if filename.split('.')[1].lower() == 'def':
        #we have a tirific file
        nur = load_tirific(filename, Variables=['NDISKS'])[0]
        disk = 1
        count = 1
        while disk < nur:
            if disk == 1:
                sbr, RC = \
                    get_individual_tirific_disk(disk,filename,\
                                        debug = cfg.output.debug)
                RC.name = 'V_OBS'
            else:
                sbr =  get_individual_tirific_disk(disk,filename,\
                                        debug = cfg.output.debug)
            sbr.name = f'DISK_GAS_{count}'
            gas_density[f'DISK_GAS_{count}']= sbr    
          
            count += 1
            disk = disk+2
    else:
        all_profiles = read_columns(filename,debug = cfg.output.debug,\
                                    log=log)

        for type in all_profiles:
            if type in ['V_OBS','V_ROT']:
                RC = all_profiles[type]
            else:
                gas_density[type] =  all_profiles[type]
                gas_density[type].height = 0.
                gas_density[type].height_type = 'inf_thin'
  
    RC.distance = cfg.input.distance * unit.Mpc
    RC.band = cfg.RC_Construction.band   
    RC.component = 'All' 
    RC.extend = RC.radii[-1]   
    RC.check_profile()
    for name in gas_density:
        gas_density[name].component = 'gas'
        gas_density[name].band = cfg.RC_Construction.gas_band  
        gas_density[name].distance = cfg.input.distance * unit.Mpc
        if  gas_density[name].height is None:
            if cfg.RC_Construction.gas_scaleheight[3] == 'tir':
                raise InputError(f'The height for {gas_density[name].name} should have been set by the tirific file but it was not')
            gas_density[name].height = cfg.RC_Construction.gas_scaleheight[0]\
                *translate_string_to_unit(cfg.RC_Construction.gas_scaleheight[2])
            if not cfg.RC_Construction.scaleheight[1] is None:
                gas_density[name].height_error = cfg.RC_Construction.gas_scaleheight[1]\
                    *translate_string_to_unit(cfg.RC_Construction.gas_scaleheight[2])
        if gas_density[name].height_type is None:
            gas_density[name].height_type = cfg.RC_Construction.gas_scaleheight[3]
        if  gas_density[name].truncation_radius is None:
            if not cfg.RC_Construction.gas_truncation_radius[0] is None:
                gas_density[name].truncation_radius = cfg.RC_Construction.gas_truncation_radius[0]*\
                    translate_string_to_unit(cfg.RC_Construction.gas_truncation_radius[2])
                gas_density[name].softening_length = cfg.RC_Construction.gas_truncation_radius[1]
       
        gas_density[name].check_profile()  
        gas_density[name].calculate_attr() 
        gas_density[name].extend = gas_density[name].radii[-1] 
      
    return gas_density, RC


def load_tirific(def_input,Variables = None,array = False,\
        ensure_rings = False ,dict=False):
    #Cause python is the dumbest and mutable objects in the FAT_defaults
    # such as lists transfer
    if Variables == None:
        Variables = ['BMIN','BMAJ','BPA','RMS','DISTANCE','NUR','RADI',\
                     'VROT','Z0', 'SBR', 'INCL','PA','XPOS','YPOS','VSYS',\
                     'SDIS','VROT_2',  'Z0_2','SBR_2','INCL_2','PA_2','XPOS_2',\
                     'YPOS_2','VSYS_2','SDIS_2','CONDISP','CFLUX','CFLUX_2']


    # if the input is a string we first load the template
    if isinstance(def_input,str):
        def_input = tirific_template(filename = def_input )

    out = []
    for key in Variables:

        if key[-3:] == 'ERR':
            key =f'# {key}'

        try:
            out.append([float(x) for x  in def_input[key].split()])
        except KeyError:
            out.append([])
        except ValueError:
            out.append([x for x  in def_input[key].split()])

    #Because lists are stupid i.e. sbr[0][0] = SBR[0], sbr[1][0] = SBR_2[0] but  sbr[:][0] = SBR[:] not SBR[0],SBR_2[0] as logic would demand

    if array:
        tmp = out
        #We can ensure that the output has the same number of values as there are rings
        if ensure_rings:
            length=int(def_input['NUR'])
        else:
            #or just take the longest input as the size
            length = max(map(len,out))
        #let's just order this in variable, values such that it unpacks properly into a list of variables
        out = np.zeros((len(Variables),length),dtype=float)
        for i,variable in enumerate(tmp):
            if len(variable) > 0.:
                out[i,0:len(variable)] = variable[0:len(variable)]

    if dict:
        tmp = {}
        for i,var in enumerate(Variables):
            tmp[var] = out[i]
        out = tmp
    elif len(Variables) == 1:
        out= out[0]
    #print(f'''LOAD_TIRIFIC: We extracted the following profiles from the Template.
#{'':8s}Requested Variables = {Variables}
#{'':8s}Extracted = {out}
#''')
    #Beware that lists are stupid i.e. sbr[0][0] = SBR[0], sbr[1][0] = SBR_2[0] but  sbr[:][0] = SBR[:] not SBR[0],SBR_2[0] as logic would demand
    # However if you make a np. array from it make sure that you specify float  or have lists of the same length else you get an array of lists which behave just as dumb

    return out
load_tirific.__doc__ =f'''
 NAME:
    load_tirific

 PURPOSE:
    Load values from variables set in the tirific files

 CATEGORY:
    common_functions

 INPUTS:
    def_input = Path to the tirific def file or a FAT tirific template dictionary

 OPTIONAL INPUTS:
    Variables = ['BMIN','BMAJ','BPA','RMS','DISTANCE','NUR','RADI','VROT',
                 'Z0', 'SBR', 'INCL','PA','XPOS','YPOS','VSYS','SDIS','VROT_2',  'Z0_2','SBR_2',
                 'INCL_2','PA_2','XPOS_2','YPOS_2','VSYS_2','SDIS_2','CONDISP','CFLUX','CFLUX_2']


    array = False
        Specify that the output should be an numpy array with all varables having the same length

    ensure_rings =false
        Specify that the output array should have the length of the NUR parameter in the def file

    dict = False
        Return the output as a dictionary with the variable names as handles
 OUTPUTS:
    outputarray list/array/dictionary with all the values of the parameters requested

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
    This function has the added option of a dictionary compared to pyFAT
'''


def tirific_template(filename = ''):
    if filename == '':
        raise InputError(f'Tirific_Template does not know a default')
    else:
        with open(filename, 'r') as tmp:
            template = tmp.readlines()
    result = {}
    counter = 0
    # Separate the keyword names
    for line in template:
        key = str(line.split('=')[0].strip().upper())
        if key == '':
            result[f'EMPTY{counter}'] = line
            counter += 1
        else:
            result[key] = str(line.split('=')[1].strip())
    return result
tirific_template.__doc__ ='''
 NAME:
    tirific_template

 PURPOSE:
    Read a tirific def file into a dictionary to use as a template.
    The parameter ill be the dictionary key with the values stored in that key

 CATEGORY:
    read_functions

 INPUTS:
    filename = Name of the def file

 OPTIONAL INPUTS:
    filename = ''
    Name of the def file, if unset the def file in Templates is used



 OUTPUTS:
    result = dictionary with the read file

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
      split, strip, open

 NOTE:
'''

'''
def load_tirific(filename,Variables = ['BMIN','BMAJ','BPA','RMS','DISTANCE','NUR','RADI','VROT',
                 'Z0', 'SBR', 'INCL','PA','XPOS','YPOS','VSYS','SDIS','VROT_2',  'Z0_2','SBR_2',
                 'INCL_2','PA_2','XPOS_2','YPOS_2','VSYS_2','SDIS_2','CONDISP','CFLUX','CFLUX_2'],
                 unpack = True , debug = False ):
    if debug:
        print_log(f''LOAD_TIRIFIC: Starting to extract the following paramaters:
{'':8s}{Variables}
'',None,screen=True, debug = True)
    Variables = np.array([e.upper() for e in Variables],dtype=str)
    numrings = []
    while len(numrings) < 1:
        with open(filename, 'r') as tmp:
            numrings = [int(e.split('=')[1].strip()) for e in tmp.readlines() if e.split('=')[0].strip().upper() == 'NUR']



    #print(numrings)tmp
    outputarray=np.zeros((numrings[0],len(Variables)),dtype=float)
    with open(filename, 'r') as tmp:
        unarranged = tmp.readlines()
    # Separate the keyword names
    for line in unarranged:
        var_concerned = str(line.split('=')[0].strip().upper())
        #if debug:
        #    print_log(f''LOAD_TIRIFIC: extracting line
#{'':8s}{var_concerned}.
#'',None,screen=False, debug = True)
        if len(var_concerned) < 1:
            var_concerned = 'xxx'
        varpos = np.where(Variables == var_concerned)[0]
        if varpos.size > 0:
            tmp =  np.array(line.split('=')[1].rsplit(),dtype=float)
            if len(outputarray[:,0]) < len(tmp):
                tmp_out=outputarray
                outputarray = np.zeros((len(tmp), len(Variables)), dtype=float)
                outputarray[0:len(tmp_out),:] = tmp_out
            outputarray[0:len(tmp),int(varpos)] = tmp[0:len(tmp)]
        else:
            if var_concerned[0] == '#':
                varpos = np.where(var_concerned[2:].strip() == Variables)[0]
#                if debug:
#                    print_log(f''LOAD_TIRIFIC: comparing {var_concerned[2:].strip()} to the variables.
#{'':8s}Found {varpos}.
#'',None,screen=True, debug = True)
                if varpos.size > 0:
                    tmp = np.array(line.split('=')[1].rsplit(),dtype=float)
                    if len(outputarray[:, 0]) < len(tmp):
                        tmp_out = outputarray
                        outputarray = np.zeros((len(tmp), len(Variables)), dtype=float)
                        outputarray[0:len(tmp_out), :] = tmp_out
                    outputarray[0:len(tmp),int(varpos)] = tmp[:]
    if unpack:
        return (*outputarray.T,)
    else:
        return outputarray
'''