# -*- coding: future_fstrings -*-
import numpy as np
from pyROTMOD.support import print_log,read_columns
#Function to convert column densities
# levels should be n mJy/beam when flux is given
class InputError(Exception):
    pass
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

def get_individal_tirific_disk(disk,filename):

    if disk == 1:
        ext1 =''
    else:
        ext1=f'_{disk}'
    ext2 = f'_{disk+1}'

    if disk == 1: 
        Variables =  ['RADI','VROT','VROT_2','VROT_ERR',\
                      'VROT_2_ERR']
    else:
        Variables = []
    
    Variables =Variables + [f'SBR{ext1}',f'SBR{ext2}',f'Z0{ext1}',f'Z0{ext2}'\
                      ,f'LTYPE{ext1}',f'LTYPE{ext2}',f'VSYS{ext1}']
    
    value_dictionary = load_tirific(filename, Variables=Variables, dict=True)  
    scaleheight=[np.mean(value_dictionary[f'Z0{ext1}']),\
                 np.mean(value_dictionary[f'Z0{ext2}'])]
   
    if  value_dictionary[f'LTYPE{ext1}'][0] != value_dictionary[f'LTYPE{ext2}'][0] :
        print_log(f'''Your def file has different vertical functions. 
We always assume the function used in the first disk ({value_dictionary[f"LTYPE{ext1}"][0]}) instead of the second ({value_dictionary[f"LTYPE{ext2}"][0]})''',log)
    if value_dictionary[f'LTYPE{ext1}'][0] == 0.:
        scaleheight.append('constant')
    elif value_dictionary[f'LTYPE{ext1}'][0] == 1.:
        scaleheight.append('gaussian')
    elif value_dictionary[f'LTYPE{ext1}'][0] == 2:
        scaleheight.append('sech-sq')
    elif value_dictionary[f'LTYPE{ext1}'][0] == 3:
        scaleheight.append('exp')
    elif value_dictionary[f'LTYPE{ext1}'][0] == 4:
        scaleheight.append('lorentzian')
    outdisk = int((disk+1)/2.)    
    gas_density = [f'DISK_GAS_{outdisk}','M_SOLAR/PC^2']
    av = np.array([(x1+x2)/2.*1000. for x1,x2 in zip(\
        value_dictionary[f'SBR{ext1}'],value_dictionary[f'SBR{ext2}'])],dtype=float)
    tmp = columndensity(av,arcsquare = True , solar_mass_output = True,\
                         systemic= value_dictionary[f'VSYS{ext1}'][0])
    for x in tmp:
        gas_density.append(x)    
    if disk == 1:
        radii = ['RADI','ARCSEC']+list(value_dictionary['RADI'])
        total_RC = ['V_OBS','KM/S']
        for x1,x2 in zip(value_dictionary['VROT'],value_dictionary['VROT_2']):
            total_RC.append((x1+x2)/2.)
        total_RC_Err = ['V_OBS_ERR','KM/S']
      
        for x1,x2 in zip(value_dictionary['VROT_ERR'],value_dictionary['VROT_2_ERR']):
            total_RC_Err.append((x1+x2)/2.)
        return  radii, total_RC,total_RC_Err,scaleheight,gas_density
    else:
        return scaleheight,gas_density


      
       


def get_gas_profiles(filename,log=None, debug =False):
    print_log(f"Reading the gas density profile from {filename}. \n",log,screen =True )
    if filename.split('.')[1].lower() == 'def':
        #we have a tirific file
        nur = load_tirific(filename, Variables=['NDISKS'])[0]
        disk = 1
        while disk < nur:
            if disk == 1:
                radii, total_RC,total_RC_Err,scaleheight,gas_density = \
                    get_individal_tirific_disk(disk,filename)
                scaleheight= [scaleheight]
                gas_density= [gas_density] 
            
            else:
                tmp_scaleheight,tmp_gas_density =  get_individal_tirific_disk(disk,filename)
                scaleheight.append(tmp_scaleheight)
                gas_density.append(tmp_gas_density)
            disk = disk+2
    else:
        all_profiles = read_columns(filename,debug=debug,log=log)
        found = False
        gas_density =[]
        total_RC = []
        total_RC_Err = []
        for type in all_profiles:
            if all_profiles[type][0] == 'RADI':
                radii = all_profiles[type][:2]+[float(x) for x in all_profiles[type][2:]]
            elif all_profiles[type][0].split('_')[0] == 'DISK' and \
                all_profiles[type][0].split('_')[1] in ['G','GAS']:
                if len(gas_density) < 1:
                    gas_density = [all_profiles[type][:2]+[float(x) for x in all_profiles[type][2:]]]
                else:
                    if all_profiles[type][1] == gas_density[-1][1]:
                        gas_density.append([all_profiles[type][:2]+[float(x) for x in all_profiles[type][2:]]])
                        #gas_density[2:] = [float((x+y)/2.) for x,y in (all_profiles[type][2:],gas_density[2:])]
                    else:
                        raise InputError(f'We do not know how to mix units for the gas disk')
            elif all_profiles[type][0] == 'V_OBS':
                if len(total_RC) < 1:
                    total_RC = all_profiles[type][:2]+[float(x) for x in all_profiles[type][2:]]
                else:
                    if all_profiles[type][1] == gas_density[1]:
                        total_RC[2:] = [float((x+y)/2.) for x,y in (all_profiles[type][2:],total_RC[2:])]
                    else:
                        raise InputError(f'We do not know how to mix units for the gas disk')
            elif all_profiles[type][0] == 'V_OBS_ERR':
                if len(total_RC_Err) < 1:
                    total_RC_Err = all_profiles[type][:2]+[float(x) for x in all_profiles[type][2:]]
                else:
                    if all_profiles[type][1] == gas_density[1]:
                        total_RC_Err[2:] = [float((x+y)/2.) for x,y in (all_profiles[type][2:],total_RC_Err[2:])]
                    else:
                        raise InputError(f'We do not know how to mix units for the gas disk')
        scaleheight = [[0.,0.,None]]
        gas_density = [gas_density]
    print()
    return radii, gas_density, total_RC,total_RC_Err,scaleheight


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