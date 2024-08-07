# -*- coding: future_fstrings -*-

import numpy as np
import warnings

from pyROTMOD.support.minor_functions import print_log,translate_string_to_unit
from pyROTMOD.support.major_functions import read_columns
from pyROTMOD.optical.conversions import mag_to_lum
from pyROTMOD.support.errors import InputError, BadFileError
from pyROTMOD.support.classes import Density_Profile

from astropy import units as unit

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt






def get_optical_profiles(cfg,log=None):
   
    #filename,distance = 0.*unit.Mpc,band = 'SPITZER3.6',exposure_time=1.,\
    #                        MLRatio = 0.6, log =None,debug=False, scale_height=None,
    #                        output_dir='./'):
     #get_optical_profiles(cfg.RC_Construction.optical_file,\
    #           ,exposure_time=cfg.RC_Construction.exposure_time,\
    #           ,band = cfg.RC_Construction.band,\
    #            log= log,output_dir=cfg.general.output_dir)
    '''Read in the optical Surface brightness profiles or the galfit file'''
    # as we do a lot of conversions in the optical module we make distance a quantity with unit Mpc
    distance = cfg.general.distance * unit.Mpc
    MLRatio = cfg.RC_Construction.mass_to_light_ratio*unit.Msun/unit.Lsun
    print_log(f"GET_OPTICAL_PROFILES: We are reading the optical parameters from {cfg.RC_Construction.optical_file}. \n",log, screen =True)
    if distance.value == 0.:
        raise InputError(f'We cannot convert profiles adequately without a distance.')
    with open(cfg.RC_Construction.optical_file) as file:
        input = file.readlines()

    firstline = input[0].split()
    correctfile  = False
    try:
        if firstline[0].strip().lower() == 'radi':
            correctfile = True
    except:
        pass
    galfit_file = False

    # If the first line and first column is not correct we assume a Galfit file
    if not correctfile:
        optical_profiles, galfit_info = read_galfit(input,log=log,debug=cfg.general.debug)
        galfit_info['exposure_time'] =cfg.RC_Construction.exposure_time*unit.second
        
        galfit_file = True
        

    else:
        optical_profiles = read_columns(cfg.RC_Construction.optical_file\
                            ,debug=cfg.general.debug,log=log)
   
    for name in optical_profiles:
        optical_profiles[name].band = cfg.RC_Construction.band
        optical_profiles[name].distance = distance
        optical_profiles[name].MLratio = MLRatio  
        optical_profiles[name].component = 'stars' 
     
        if optical_profiles[name].height == None:
            optical_profiles[name].height = cfg.RC_Construction.scaleheight[0]\
                *translate_string_to_unit(cfg.RC_Construction.scaleheight[2])
            #optical_profiles[name].height_unit = cfg.RC_Construction.scaleheight[2]
            if not cfg.RC_Construction.scaleheight[1] is None:
                optical_profiles[name].height_error = cfg.RC_Construction.scaleheight[1]\
                    *translate_string_to_unit(cfg.RC_Construction.scaleheight[2])
        if optical_profiles[name].height_type == None:
            optical_profiles[name].height_type = cfg.RC_Construction.scaleheight[3]

        if galfit_file:
    
            #for the expdisk profiles  we apparently need to deproject the totalSB
            if  optical_profiles[name].type in ['expdisk']:
                IntLum = mag_to_lum(optical_profiles[name].total_SB, \
                                    band =optical_profiles[name].band , distance=distance)
                 # and transform to a face on total magnitude (where does this come from?)
                optical_profiles[name].total_SB =  IntLum/optical_profiles[name].axis_ratio 
            optical_profiles[name].create_profile()
        else:
            optical_profiles[name].calculate_components()
    
  
    print_log(f"We found the following optical components:\n",log,debug=cfg.general.debug)
    for name in optical_profiles:
        # Components are returned as [type,integrated magnitude,scale parameter in arcsec,sercic index or scaleheight in arcsec, axis ratio]
        if optical_profiles[name].type in ['expdisk','edgedisk']:
            print_log(f'''We have found an exponential disk with the following values.
''',log,debug=cfg.general.debug)
        elif optical_profiles[name].type in ['sersic']:
            print_log(f'''We have found a sersic component with the following values.
''',log,debug=cfg.general.debug)
        elif optical_profiles[name].type in ['hernquist']:
            print_log(f'''We have found a hernquist component with the following values.
''',log,debug=cfg.general.debug)
        elif optical_profiles[name].type in ['random_disk','random_bulge']:
            print_log(f'''We have found a unparameterized component with the following values.
''',log,debug=cfg.general.debug)
        else:
            print_log(f'''We have found a {optical_profiles[name].type} component with the following values.
''',log,debug=cfg.general.debug)
        print_log(f'''The total mass of the disk is {optical_profiles[name].total_SB}   a central mass density {optical_profiles[name].central_SB}  with a M/L {optical_profiles[name].MLratio}.
The scale length is {optical_profiles[name].scale_length}  and the scale height {optical_profiles[name].height}.
The axis ratio is {optical_profiles[name].axis_ratio}.
''' ,log,debug=cfg.general.debug)


    return optical_profiles,galfit_file

get_optical_profiles.__doc__ =f'''
 NAME:
    get_optical_profiles

 PURPOSE:
    Read in the optical file provided and convert the input to homogenuous profiles to be used by rotmod.

 CATEGORY:
    optical

 INPUTS:
    filename = name of the file to be read.

 OPTIONAL INPUTS:
    distance = 0.
        Distance to the galaxy for converting flux parameters required for a galfit file or when input is in magnitude/arcsec^2

    exposure_time = 1.
        certain galfit components require a exposure time from the header of the image.

    band = 'SPITZER3.6'
        band used for the observation
    MLRatio = 0.6
        mass to light ratio used for light profile. Note that this is used to multiply the luminosity profile.
        Hence when doing the mass decomposition this factor is incorporated. Set to 1. if you want to get  the MD and MB from the decomposition.

    log = None
    debug = False
 OUTPUTS:

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''

def organize_profiles(profiles):
    organized = {}
    all_radii = False
    if 'RADI' in [x for x in profiles]:
        all_radii = True
    for type in profiles:
        if type[-4:] != 'RADI':
            organized[type] = {'Profile': np.array([x for x in\
                                    profiles[type][2:]],dtype=float),\
                    'Profile_Unit': profiles[type][1]}
            if all_radii:
                organized[type]['Radii'] = np.array([x for x in\
                                    profiles['RADI'][2:]],dtype=float)
                organized[type]['Radii_Unit'] = profiles['RADI'][1]
            else:
                organized[type]['Radii'] = np.array([x for x in\
                                        profiles[f'{type}_RADI'][2:]],dtype=float)
                organized[type]['Radii_Unit'] = profiles[f'{type}_RADI'][1]

organize_profiles.__doc__ =f'''
 NAME:
    organize_profiles

 PURPOSE:
    Transform a list of profile with the old [''Name', 'Unit', profile[0:]
    To a dictionary withe entries of name {{'Profile': profile[0:], 
    'Profile_Unit': unit, 'Radii': individual radius, 'Radii_Unit': radius unit }}
    This is to allow for storage of individual radii for each profile.

 CATEGORY:
    optical

 INPUTS:
    profiles = old profiles

 OPTIONAL INPUTS:
    
 OUTPUTS:
    dict_profiles = New dictionary
 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''


def old_plot_profile(in_radii,density, exponential,name='generic',\
                    output_dir='./',red_chi = None, count = '1'):
    '''This function makes a simple plot of the optical profiles'''
    figure = plt.figure(figsize=(10,7.5) , dpi=300, facecolor='w', edgecolor='k')
    ax = figure.add_subplot(1,1,1)
    ax.plot(in_radii,density, label='Input Profile')
    ax.plot(in_radii,exponential, label='Fitted Profile')


    #plt.xlim(0,6)
    ax.set_ylabel(r'Density (M$_\odot$/pc$^2$)')
    ax.set_xlabel('Radius (kpc)')
    if red_chi:
        ax.text(0.5,1.05,f'''Red. $\\chi^{{2}}$ = {red_chi:.4f}''',rotation=0, va='bottom',ha='center', color='black',\
            bbox=dict(facecolor='white',edgecolor='white',pad=0.5,alpha=0.),\
            zorder=7, backgroundcolor= 'white',fontdict=dict(weight='bold',size=16),transform=ax.transAxes)
    ax.set_yscale('log')
    plt.legend()
    if int(count) > 1:
        name = f'{name}_{count}'
    plt.savefig(f'{output_dir}/{name}_Profiles.png')
    plt.close()

old_plot_profile.__doc__ =f'''
 NAME:
     plot_profile(in_radii,density, exponential,name='generic',\
                        output_dir='./',red_chi = None,bulge =False, count = 1):
 PURPOSE:
    plot the fitted function over the input profile

 CATEGORY:
    optical

 INPUTS:
    in_radii = radii
    density = original profile
    exponential = fitted profile

 OPTIONAL INPUTS:
    name = 'generic'
        Name of the fitted function to be used in the file name

    output_dir = './'
        Destination directory for filename

    red_chi = None
        Reduced Chi square

    count = '1'
        number of function fitting used in file name

 OUTPUTS:
    png plot
 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''
'''
def process_read_profile(optical_profiles,cleaned_components,\
                    output_dir = './',debug =False, log = None):
    optical_profiles_out = {}
    component_out  = []
    exponen_count = 1
    hern_count = 1
    for type in optical_profiles:
        prof= type.split('_')
        if prof[0] == 'EXPONENTIAL':
            if int(prof[1]) > int(exponen_count):
                exponen_count = int(prof[1])
        if prof[0] == 'HERNQUIST':
            if int(prof[1]) > int(hern_count):
                hern_count = int(prof[1])

    ori_count= 0
    for i,type in enumerate(optical_profiles):
        if type == 'RADI':
            optical_profiles_out[type] = optical_profiles[type]
        elif optical_profiles[type][1] in ['KM/S']:
            if type[:3] in ['EXP','DEN','DIS','SER']:
                cleaned_components[i-1]['Type'] = 'random_disk'
            else:
                cleaned_components[i-1]['Type'] = 'random_bulge'
            optical_profiles_out[type] = optical_profiles[type]
            component_out.append(cleaned_components[i-1])
        elif optical_profiles[type][1] in ['M_SOLAR/PC^2'] and \
            type[:3] in ['EXP','HER','SER','DEN'] :
            result,profiles,components = fit_profile(optical_profiles['RADI'][2:],optical_profiles[type][2:],\
                    cleaned_components[i-1],function=type, output_dir = output_dir\
                    ,debug = debug, log = log)
            if result == 'process':
                optical_profiles_out[f'EXPONENTIAL_{exponen_count}'] = \
                    [f'EXPONENTIAL_{exponen_count}','M_SOLAR/PC^2'] + list(profiles[1])
                component_out.append(components[1])
                optical_profiles_out[f'HERNQUIST_{hern_count}'] = \
                    [f'HERNQUIST_{hern_count}','M_SOLAR/PC^2'] + list(profiles[0])
                component_out.append(components[0])
                exponen_count += 1
                hern_count += 1
            elif result == 'ok':

                if components['Type'] == 'expdisk':
                    optical_profiles_out[f'EXPONENTIAL_{exponen_count}'] = \
                        [f'EXPONENTIAL_{exponen_count}','M_SOLAR/PC^2'] + list(profiles)
                    exponen_count += 1
                elif components['Type'] == 'hernquist':
                    optical_profiles_out[f'HERNQUIST_{hern_count}'] = \
                        [f'HERNQUIST_{hern_count}','M_SOLAR/PC^2'] + list(profiles)
                    hern_count += 1
                else:
                    optical_profiles_out[type] = optical_profiles[type]
                component_out.append(components)
            else:
                optical_profiles_out[type] = optical_profiles[type]
                component_out.append(cleaned_components[i-1])
        elif type[:3] == 'DIS':
            cleaned_components[i-1]['Type'] = 'random_disk'
            optical_profiles_out[type] = optical_profiles[type]
            component_out.append(cleaned_components[i-1])
        else:
            print_log(f''PROCESS_READ_PROFILES: We do not know how convert the profile {optical_profiles[type][0]} with the units {optical_profiles[type][1]} to velocities
'',log)
            raise InputError(f''PROCESS_READ_PROFILES: We do not know how convert the profile {optical_profiles[type][0]} with the units {optical_profiles[type][1]} to velocities'')
    return optical_profiles_out,component_out
process_read_profile.__doc__ ='''
f'''
 NAME:
    process_read_profile(optical_profiles,cleaned_components,fit_random = False,\
                        output_dir = './',debug =False, log = None):
 PURPOSE:
    Fit the components of the read profile with their specified function
    in case the densities are read from file and not a galfit file

 CATEGORY:
    optical

 INPUTS:
    optical_profiles = the read input profiles dictionary
    cleaned_components = list of components to match the profiles
                        (this runs i-1 compared to the profiles as the radii doesn't have components )



 OPTIONAL INPUTS:
    fit_random = False
        if input is a radom density disk (DISK_) do we want to fit it?
    output_dir = './'
        standard for directory name where to put comparison plots
    log = None
    debug = False

 OUTPUTS:
    updated components list

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE: For now we can not model the sersic profile into a density distribution so
       sersic profiles are converted to hernquist or exponential profiles in rotmod
        but only if 0.75 < n < 1.25 (to exponential) or 3.75 < n < 4.25 (hernquist)
'''
def read_galfit(lines,log=None,debug=False):

    
    recognized_components = ['expdisk','sersic','edgedisk','sky','devauc']
    output = ['EXPONENTIAL','HERNQUIST','SERSIC','SKY']
    counter = [0 for x in output]
    #This dictionary relates the profile to the potential (Except for the sersic)
    trans_dict = {'expdisk': 'EXPONENTIAL',
                  'sersic': 'SERSIC',
                  'edgedisk': 'EXPONENTIAL',
                  'sky': 'SKY',
                  'devauc':'HERNQUIST'}
    mag_zero = []
    plate_scale = []
    read_component = False
    components = {}
    max_radius = 0.
    for line in lines:
        tmp = [x.strip().lower() for x in line.split()]

        if len(tmp) > 0:
            if tmp[0].lower() == 'j)':
                mag_zero = [float(tmp[1])]
            if tmp[0].lower() == 'k)':
                plate_scale = [float(tmp[1]), float(tmp[2])] # [arcsec per pixel]
            if tmp[0].lower() == 'z)':
                read_component = False
            if len(tmp) > 1:
                if tmp[1] == 'component':
                    read_component = True
                    continue
            if read_component:
                if tmp[0] == '0)':
                    current_component = tmp[1]
                    if current_component not in recognized_components:
                        print_log(f'''pyROTMOD does not know how to process {current_component} not reading it
    ''',log)
                        read_component = False
                    else:
                       
                        counter[output.index(trans_dict[current_component])] += 1
                        current_name = f'{trans_dict[current_component]}_{counter[output.index(trans_dict[current_component])]}'
                        components[current_name] = Density_Profile(\
                            type=current_component,name=current_name)
                                  
                        #if current_component in ['expdisk','sersic','devauc']:
                        #    components[current_name].scale_height = 0. * unit.kpc
                        #    components[current_name].scale_height_type = 'inf_thin'
                if current_component in ['sky']:
                    if tmp[0] == '1)':
                        components[current_name].background = float(tmp[1])    
                    elif tmp[0] == '2)':
                        components[current_name].dx = float(tmp[1])*unit.pix
                    elif tmp[0] == '3)':
                        components[current_name].dy = float(tmp[1])*unit.pix
                else:
                    if tmp[0] == '1)':
                        components[current_name].central_position =\
                              [float(tmp[1]),float(tmp[2])]*unit.pix  
                    elif tmp[0] == '3)':
                        if  current_component in ['edgedisk']:
                            components[current_name].central_SB = float(tmp[1])\
                                *unit.mag/unit.arcsec**2
                        else:
                            components[current_name].total_SB = float(tmp[1])*unit.mag
                    elif tmp[0] == '4)':
                        if current_component in ['sersic','devauc']:
                            components[current_name].R_effective = float(tmp[1])\
                            *np.mean(plate_scale)*unit.arcsec
                            if max_radius < 5* float(tmp[1]): 
                                max_radius = 5 * float(tmp[1])
                        if current_component in ['edgedisk']:
                            components[current_name].scale_height = float(tmp[1])\
                                *np.mean(plate_scale)*unit.arcsec
                            components[current_name].scale_height_type = 'sech-sq'
                        if current_component in ['expdisk']:
                            components[current_name].scale_length = float(tmp[1])\
                                *np.mean(plate_scale)*unit.arcsec
                            #components[current_name].scale_height = 0.*unit.arcsec
                            if max_radius < 10 * float(tmp[1]): 
                                max_radius = 10 * float(tmp[1])
                    elif tmp[0] == '5)'  and\
                        current_component in ['sersic','edgedisk','devauc']:
                        if current_component in ['sersic','devauc']:
                            components[current_name].sersic_index = float(tmp[1])
                        elif current_component in ['edgedisk']:
                            if max_radius < 10 * float(tmp[1]): 
                                max_radius = 10 * float(tmp[1])
                            components[current_name].scale_length = float(tmp[1])\
                                *np.mean(plate_scale)*unit.arcsec
                    elif tmp[0] == '9)' and current_component in ['expdisk','sersic','devauc']:
                        components[current_name].axis_ratio = float(tmp[1])
                    elif tmp[0] == '10)':
                        components[current_name].PA = float(tmp[1])*unit.degree
    
    if len(plate_scale) == 0 or len(mag_zero) == 0:
        raise BadFileError(f'Your file  is not recognized by pyROTMOD')
    
    for d in components:
        # add radii
        components[d].radii= np.linspace(0,max_radius,int(max_radius/2.))*\
            np.mean(plate_scale)*unit.arcsec # in arcsec
        components[d].radii_unit = unit.arcsec
    


    galfit_info = {}
    galfit_info['radii'] = np.linspace(0,max_radius,int(max_radius/2.))*\
        np.mean(plate_scale)*unit.arcsec # in arcsec
    galfit_info['plate_scale'] = plate_scale*unit.arcsec #[arcsec per pixel]
    galfit_info['magnitude_zero'] = mag_zero*unit.mag

    return components,galfit_info

read_galfit.__doc__ =f'''
 NAME:
    read_galfit

 PURPOSE:
    Read in the galfit file and extract the parameters for each component in there

 CATEGORY:
    optical

 INPUTS:
    lines = the string instance of an opened file

 OPTIONAL INPUTS:
    log = None
    debug = False

 OUTPUTS:
   components = set of parameters with units for each component as well as some global components



 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''



