# -*- coding: future_fstrings -*-

import numpy as np

from pyROTMOD.support.minor_functions import print_log,translate_string_to_unit,get_uncounted
from pyROTMOD.support.major_functions import read_columns
from pyROTMOD.optical.conversions import mag_to_lum
from pyROTMOD.support.errors import InputError, BadFileError, UnitError
from pyROTMOD.support.profile_classes import SBR_Profile,Component,Luminosity_Profile,copy_attributes

from astropy import units as unit

#Convert on sky profiles to in plane SBR profiles
def convert_luminosity_profile(profile_in,cfg=None):
    transfer=Component()
    profiles_out = SBR_Profile(distance=profile_in.distance\
        ,radii=profile_in.radii,MLratio=profile_in.MLratio,
        component='Stars',band=profile_in.band)
    for attr, value in transfer.__dict__.items():
        if attr not in ['central_SB','values']:
            setattr(profiles_out,attr,getattr(profile_in,attr))  
   
    profiles_out.create_profile() 
    
    if 'random' in profile_in.type:
        profiles_out.values = profile_in.values*profile_in.MLratio  
        profiles_out.radii = profile_in.radii 
    if profiles_out.values.unit not in [unit.Msun/unit.kpc**2,unit.Msun/unit.pc**2] and\
        profiles_out.profile_type == 'sbr_dens:':
        raise UnitError(f'The unit {profiles_out.unit} is not recognized for the surface brightness profile {profiles_out.name}')
    if profiles_out.values.unit not in [unit.Msun/unit.kpc**3,unit.Msun/unit.pc**3] and\
        profiles_out.profile_type == 'density:':
        raise UnitError(f'The unit {profiles_out.unit} is not recognized for the density profile {profiles_out.name}')
 
    return profiles_out    

def get_optical_profiles(cfg):
    '''Read in the optical Surface brightness profiles or the galfit file'''
    # as we do a lot of conversions in the optical module we make distance a quantity with unit Mpc
    distance = cfg.input.distance * unit.Mpc
    MLRatio = cfg.RC_Construction.mass_to_light_ratio*unit.Msun/unit.Lsun
    print_log(f"GET_OPTICAL_PROFILES: We are reading the optical parameters from {cfg.RC_Construction.optical_file}. \n"\
        ,cfg, case = ['main','screen'])
    if distance.value == 0.:
        raise InputError(f'We cannot convert profiles adequately without a distance.')
    with open(cfg.RC_Construction.optical_file) as file:
        input_lines = file.readlines()

    firstline = input_lines[0].split()
    correctfile  = False
    try:
        if firstline[0].strip().lower() == 'radii':
            correctfile = True
    except:
        pass
    galfit_file = False

    # If the first line and first column is not correct we assume a Galfit file
    if not correctfile:
        optical_profiles, galfit_info = read_galfit(input_lines,cfg=cfg)
        galfit_info['exposure_time'] =cfg.RC_Construction.exposure_time*unit.second
        
        galfit_file = True

    else:
        optical_profiles = read_columns(cfg.RC_Construction.optical_file\
                            ,cfg=cfg)
    split = {'names': []}
    for name in optical_profiles:
       
        print_log(f"GET_OPTICAL_PROFILES: We are processing the optical parameters for {name}. \n"\
            ,cfg, case = ['main','screen'])
        optical_profiles[name].band = cfg.RC_Construction.band
        optical_profiles[name].distance = distance
        optical_profiles[name].MLratio = MLRatio  
        optical_profiles[name].component = 'Stars' 
        if optical_profiles[name].height is None:
            optical_profiles[name].height = cfg.RC_Construction.scaleheight[0]\
                *translate_string_to_unit(cfg.RC_Construction.scaleheight[2])
            #optical_profiles[name].height_unit = cfg.RC_Construction.scaleheight[2]
            if not cfg.RC_Construction.scaleheight[1] is None:
                optical_profiles[name].height_error = cfg.RC_Construction.scaleheight[1]\
                    *translate_string_to_unit(cfg.RC_Construction.scaleheight[2])
        if optical_profiles[name].height_type is None:
            optical_profiles[name].height_type = cfg.RC_Construction.scaleheight[3]
        if optical_profiles[name].truncation_radius is None:
            if not cfg.RC_Construction.truncation_radius[0] is None:
                optical_profiles[name].truncation_radius = cfg.RC_Construction.truncation_radius[0]*\
                    translate_string_to_unit(cfg.RC_Construction.truncation_radius[2])
                optical_profiles[name].softening_length = \
                    cfg.RC_Construction.truncation_radius[1]*unit.dimensionless_unscaled
          
    
        if galfit_file:
    
            #for the expdisk profiles  we apparently need to deproject the totalSB
            if  optical_profiles[name].type in ['expdisk']:
                IntLum = mag_to_lum(optical_profiles[name].total_luminosity, \
                                    band =optical_profiles[name].band , distance=distance)
                 # and transform to a face on total magnitude (where does this come from?)
                optical_profiles[name].total_luminosity =  IntLum/optical_profiles[name].axis_ratio 
            
           
            optical_profiles[name].check_radius()
            #This should be laste
            optical_profiles[name].create_profile()    
            
        else:
            print_log(f"GET_OPTICAL_PROFILES: We are calculating the optical parameters for {name}. \n"\
                ,cfg, case = ['main','screen'])
            optical_profiles[name].calculate_attr(cfg=cfg)
            if optical_profiles[name].type =='hernq+expdisk':
               split['names'].append(name)
            
    for name in split['names']:
        split_double_profile(optical_profiles,name,cfg=cfg)
        
  
    print_log(f"We found the following optical components:\n",cfg, case = ['main','screen'])
    for name in optical_profiles:
        
        # Components are returned as [type,integrated magnitude,scale parameter in arcsec,sercic index or scaleheight in arcsec, axis ratio]
        if optical_profiles[name].type in ['expdisk','edgedisk']:
            print_log(f'''We have found an exponential disk with the following values.
''',cfg, case = ['main'])
        elif optical_profiles[name].type in ['sersic']:
            print_log(f'''We have found a sersic component with the following values.
''',cfg, case = ['main'])
        elif optical_profiles[name].type in ['hernquist']:
            print_log(f'''We have found a hernquist component with the following values.
''',cfg, case = ['main'])
        elif optical_profiles[name].type in ['random_disk','random_bulge']:
            print_log(f'''We have found a unparameterized component with the following values.
''',cfg, case = ['main'])
        else:
            print_log(f'''We have found a {optical_profiles[name].type} component with the following values.
''',cfg, case = ['main'])
        print_log(f'''The total mass of the disk is {optical_profiles[name].total_mass}   a central mass density {optical_profiles[name].central_SB}  with a M/L {optical_profiles[name].MLratio}.
The scale length is {optical_profiles[name].scale_length}  and the scale height {optical_profiles[name].height}.
The axis ratio is {optical_profiles[name].axis_ratio}.
''' ,cfg, case = ['main'])
    for name in optical_profiles:
        print_log(f'''The profile for {name} has been created and checked.
''',cfg, case = ['main'])
    return optical_profiles

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

def read_galfit(lines,cfg=None):
    '''Read in the galfit file and extract the parameters for each component in there'''
    recognized_components = ['expdisk','sersic','edgedisk','sky','devauc']
    output = ['EXPONENTIAL','HERNQUIST','SERSIC','SKY']
    counter = [0 for x in output]
    #This dictionary relates the profile to the potential (Except for the sersic)
    trans_dict = {'expdisk': 'EXPONENTIAL',
                  'sersic': 'SERSIC',
                  'edgedisk': 'EXPONENTIAL',
                  'sky': 'SKY',
                  'devauc':'SERSIC'}
    mag_zero = []
    plate_scale = []
    read_component = False
    components = {}
    max_radius = 0.*unit.arcsec
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
    ''',cfg, case = ['main'])
                        read_component = False
                    else:
                       
                        counter[output.index(trans_dict[current_component])] += 1
                        current_name = f'{trans_dict[current_component]}_{counter[output.index(trans_dict[current_component])]}'
                        components[current_name] = Luminosity_Profile(\
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
                            components[current_name].total_luminosity = float(tmp[1])*unit.mag
                    elif tmp[0] == '4)':
                        if current_component in ['sersic','devauc']:
                            components[current_name].R_effective = float(tmp[1])\
                            *np.mean(plate_scale)*unit.arcsec
                            if max_radius < 5* components[current_name].R_effective: 
                                max_radius = 5 * components[current_name].R_effective
                        if current_component in ['edgedisk']:
                            components[current_name].scale_height = float(tmp[1])\
                                *np.mean(plate_scale)*unit.arcsec
                            components[current_name].scale_height_type = 'sech-sq'
                        if current_component in ['expdisk']:
                            components[current_name].scale_length = float(tmp[1])\
                                *np.mean(plate_scale)*unit.arcsec
                            #components[current_name].scale_height = 0.*unit.arcsec
                            if max_radius < 10*components[current_name].scale_length: 
                                max_radius = 10 * components[current_name].scale_length
                    elif tmp[0] == '5)'  and\
                        current_component in ['sersic','edgedisk','devauc']:
                        if current_component in ['sersic','devauc']:
                            components[current_name].sersic_index = float(tmp[1])
                        elif current_component in ['edgedisk']:
                            components[current_name].scale_length = float(tmp[1])\
                                *np.mean(plate_scale)*unit.arcsec
                            if max_radius < 10*components[current_name].scale_length: 
                                max_radius = 10 * components[current_name].scale_length
                            
                    elif tmp[0] == '9)' and current_component in ['expdisk','sersic','devauc']:
                        components[current_name].axis_ratio = float(tmp[1])
                    elif tmp[0] == '10)':
                        components[current_name].PA = float(tmp[1])*unit.degree
    
    if len(plate_scale) == 0 or len(mag_zero) == 0:
        raise BadFileError(f'Your input ({input}) is not recognized by pyROTMOD')
    
    for d in components:
        # add radii
        components[d].radii= np.linspace(0,max_radius.value,int(max_radius.value))*\
            max_radius.unit
              
       
        #components[d].radii_unit = unit.arcsec
    


    galfit_info = {}
    galfit_info['radii'] = np.linspace(0,max_radius.value,int(max_radius.value/2.))*\
            max_radius.unit# in arcsec
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

def split_double_profile(profiles,original_name,cfg=None):
    names = [name for name in profiles]
    profile_type = profiles[original_name].type
    if  profile_type == 'hernq+expdisk':
        hcounter = 1
        ecounter = 1
        for name in names:
            strip_name = get_uncounted(name)[0]
            if strip_name.lower() == 'hernquist':
                hcounter += 1
            if strip_name == 'exponential':
                ecounter += 1
        new_profile_name = [f'HERNQUIST_{hcounter}',f'EXPONENTIAL_{ecounter}']
        profile_type = ['hernquist','expdisk']
        for i,name in enumerate(new_profile_name):
            profiles[name] = Luminosity_Profile(type=profile_type[i],name=name)
            exclude = ['type','name','values','total_mass']
            if i == 0:
                exclude =exclude+ ['central_SB','scale_length']
            elif i == 1:
                exclude =exclude+ ['total_luminosity','hernquist_scale_length'] 
            
            copy_attributes(profiles[original_name],profiles[name],
                exclude=exclude)
           
            profiles[name].create_profile()
          
  
    del profiles[original_name]