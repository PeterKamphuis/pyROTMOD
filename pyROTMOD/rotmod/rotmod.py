# -*- coding: future_fstrings -*-

from galpy.potential import MN3ExponentialDiskPotential as MNP, RazorThinExponentialDiskPotential as EP,\
                            TriaxialHernquistPotential as THP
from scipy.integrate import quad
#from astropy import units
from astropy import units as unit
from pyROTMOD.support.minor_functions import integrate_surface_density,\
                        print_log,set_limits,plot_profiles,write_profiles
from pyROTMOD.support.major_functions import read_columns                     
from pyROTMOD.optical.optical import get_optical_profiles
from pyROTMOD.gas.gas import get_gas_profiles

from pyROTMOD.support.errors import InputError, UnitError, RunTimeError
from pyROTMOD.support.classes import Rotation_Curve
import pyROTMOD.support.constants as c
import numpy as np
import traceback
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt


def bulge_RC(Density_In,RC_Out,log=None,debug=None):

    if Density_In.type in ['bulge','sersic','hernquist','devauc']:
        print_log(f'''We have found an hernquist profile with the following values.
The total mass of the disk is {Density_In.total_SB} a central mass density {Density_In.central_SB} .
The scale length is {Density_In.scale_length} and the scale height {Density_In.height}.
The axis ratio is {Density_In.axis_ratio}.
''' ,log,debug=debug)
        RC_radii = check_radius(Density_In,RC_Out)
       
        RC = hernquist_parameter_RC(RC_radii,Density_In,log = log, \
                    truncation_radius = RC_Out.truncation_radius )
        RC_Out.radii = RC_radii
        RC_Out.values = RC

    else:
        raise InputError(f'We have not yet implemented the disk modelling of {Density_In.type}.')

def combined_rad_sech_square(z,x,y,h_z):
    return interg_function(z,x,y)*norm_sechsquare(z,h_z)

def combined_rad_exp(z,x,y,h_z):
    return interg_function(z,x,y)*norm_exponential(z,h_z)

def combined_rad_sech_simple(z,x,y,h_z):
    return interg_function(z,x,y)*simple_sech(z,h_z)

def convert_dens_rc( optical_profiles, gas_profiles, cfg = None,\
    log= None, debug =False,output_dir='./'):
    #these should be dictionaries with their individual radii
    #organized in their typical profiles
    
    '''This function converts the mass profiles to rotation curves'''
   
    RCs = {}
    profiles = {}
    #combine the gas and optical
    for name in optical_profiles:
        profiles[f'{name}_opt'] = optical_profiles[name]
    for name in gas_profiles:
        profiles[f'{name}_gas'] = gas_profiles[name]

    ########################### First we convert the optical values and produce a RC at locations of the gas ###################
    for name in profiles:
        #if it is already a Rotation Curve we do not need to Converts
        if isinstance(profiles[name], Rotation_Curve):
            RCs[name] = profiles[name]
            continue
        #Initiate a new Rotation_Curve with the info of the density profile
        RCs[name] = Rotation_Curve(name=profiles[name].name, distance = profiles[name].distance,\
            band= profiles[name].band, type=profiles[name].type,\
            component=profiles[name].component, truncation_radius =\
            profiles[name].truncation_radius )
        
        if profiles[name].type in ['expdisk','edgedisk']:
            print_log(f'We have detected the input to be an disk',log)
            exponential_RC(profiles[name],  RCs[name], log= log,debug=debug)
        elif profiles[name].type in ['random_disk']: 
            print_log(f'This is a random density disk',log)
            random_RC(profiles[name], RCs[name],log= log,debug=debug) 
        elif profiles[name].type in ['sersic','devauc']:
                #This should work fine for a devauc profile which is simply sersic with n= 4
                if  0.75 < profiles[name].sersic_index < 1.25:
                    print_log(f'''We have detected the input to be a sersic profile.
this one is very close to a disk so we will transform to an exponential disk. \n''',log)
                    exponential_RC(profiles[name],  RCs[name], log= log,debug=debug)
                elif 3.75 < profiles[name].sersic_index < 4.25:
                    print_log(f'''We have detected the input to be a sersic profile.
this one is very close to a bulge so we will transform to a hernquist profile. \n''',log)
                    bulge_RC(profiles[name],  RCs[name], log= log,debug=debug)
                else:
                    print_log(f'''We have detected the input to be a density profile for a sersic profile.
This is not something that pyROTMOD can deal with yet. If you know of a good python implementation of Baes & Gentile 2010.
Please let us know and we'll give it a go.''',log)
                    raise InputError('We have detected the input to be a density profile for a sersic profile. pyROTMOD cannot yet process this.')

        elif profiles[name].type in ['random_bulge','hernquist']:
                #if not galfit_file:
                #    found_RC = bulge_RC(kpc_radii,optical_radii,np.array(o))
                #    print_log(f'We have detected the input to be a density profile for a bulge that is too complicated for us',log)
                #else:
                print_log(f'Assuming a classical bulge spherical profile in a Hernquist profile',log)
                bulge_RC(profiles[name],  RCs[name], log= log,debug=debug) 
        else:
                print_log(f'We do not know how to convert the mass density of {profiles[name].type}',log)
    return RCs    
convert_dens_rc.__doc__ =f'''
 NAME:
    convert_dens_rc(radii, optical_profiles, gas_profile,components,\
        distance =1.,opt_h_z = [0.,None], gas_scaleheight= [0.,None], galfit_file =False,\
        log= None, debug =False,output_dir='./'):
 PURPOSE:
    Convert the density profile into RCs for the different baryonic components.

 CATEGORY:
    rotation modelling

 INPUTS:
    radii = radii for which to produce the rotation values
    optical_profiles = The density profiles for the optical profiles
    gas_profile = The density profiles for the gas profiles
    components = the components read from the galfit file. If the profiles are
                read from a file these will be empty except for the type of disk and a 0.

 OPTIONAL INPUTS:
    distance = 1.
        distance to the galaxy
    opt_h_z = [0.,None]
        scaleheight of the optical disk and vertical distribution requested.
        if 0. an infinitely thin disk is assumed

    gas_scaleheight= [0.,None]
        same as opt_h_z but for the gas disk

    galfit_file =False,
        indicator for whether a galfit file is read or the values originate from somewhere else.

    log= None,
        Run log
    debug =False,
        trigger for enhanced verbosity and plotting
    output_dir='./'
        directory for check plots

 OUTPUTS:
    RCs = dictionary with all RCs at their native resolution and their radii

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE: If the parameterizations of the galfit file are now the function uses
        galpy to calculate the rotation curves for the optical contributions
       The optical distributions can be:
       'expdisk','edgedisk' --> in an input file EXPONENTIAL_
       This triggers the Miyamoto Nagai potential if a optical scaleheight is given
       or the RazorThinExponentialDiskPotential if not. If the profile is read from
       an input file a single exponential is fit to the distribution and the fitted
       parameters are based on to the potential.

       'inifite_disk' --> in an input file DISK_
       the density profile is converted to rotational velocities using the cassertano 1983
       descriptions. This is always the case for the gas disk.

       'bulge': ---> BULGE_
       Profile is parameterized and a Speherical Hernquist Potential is used
       to calculate the the RC. If the type is bulge it is assumed that the
       scale length corresponds to the hernquist scale length

       'sersic': --> SERSIC_
       If the galfit components are not present the code will throw an error
       if the galfit n parameters is present it will assume a exponential disk
       when 0.75 < n < 1.25 or a bulge when 3.75 < n < 4.25)


'''

           
def random_RC(Density_In,RC_Out,log=None,debug=None):
    print_log(f'''We are fitting a random density distribution disk following Cassertano 1983.
''',log,debug=debug)
     
   
    #If the RC has already radii we assume we want it on that radii
    RC_radii = check_radius(Density_In,RC_Out)

    RC = random_density_disk(RC_radii,Density_In,truncation_radius =\
        RC_Out.truncation_radius, log=log, debug=debug)
    RC_Out.radii = RC_radii
    RC_Out.values = RC
    

def check_radius(Density_In,RC_Out):
    '''Check which radii we want to use for our output RC'''
      #If the RC has already radii we assume we want it on that radii
 
    if RC_Out.radii != None:
        RC_radii = RC_Out.radii
    else:
        RC_radii = Density_In.radii
    # Make sure it is in kpc

    if RC_radii.unit != unit.kpc:
        raise InputError(f'These radii are not in kpc, they should be by now')
    return RC_radii

def check_height(Density_In):
    '''Check the type of the vertical distributio'''
    if not Density_In.height_type in  ['sech','exp'] and Density_In.height != 0.:
        raise InputError(f'We cannot have {Density_In.height_type} with a thick exponential disk. Use a different disk or pick exp or sech')
    sech = False
    if Density_In.height_type == 'sech':
        sech = True
    return sech

def exponential_RC(Density_In,RC_Out,log=None,debug=None):
    # All the checks on the components should be done previously so if we are missing something
    # this should just fail
    sech = check_height(Density_In)
    #If the RC has already radii we assume we want it on that radii
    RC_radii = check_radius(Density_In,RC_Out)
    print_log(f'''We have found an exponential disk with the following values.
The total mass of the disk is {Density_In.total_SB} a central mass density {Density_In.central_SB} .
The scale length is {Density_In.scale_length} and the scale height {Density_In.height}.
The axis ratio is {Density_In.axis_ratio}.
''' ,log,debug=debug)
    RC = exponential_parameter_RC(RC_radii,Density_In,log = log,sech = sech, \
                    truncation_radius = RC_Out.truncation_radius )
    RC_Out.radii = RC_radii
    RC_Out.values = RC
   
exponential_RC.__doc__ =f'''
 NAME:
    disk_RC

 PURPOSE:
    parametrize the density profile by fitting a single gaussian if no parameters
     are supplied and return the correspondin RC

 CATEGORY:
    rotation modelling

 INPUTS:
    radii = radii at which to evaluate the profile to obtain rotaional velocities
            in (km/s)
    density = mass surface density profile at location of radii, ignored if components[1] != 0.

 OPTIONAL INPUTS:
    h_z = [0.,'exp']
        optical scale height for the disk under consideration and the vertical mode
        Ignored if the components['scale height'] is not none

    components = {{'Type': 'expdisk', scale height': None, 'scale length': None}}
        dictionary with parameterization of the disk.
        if scale ehight is None the optical scale height is used,
        if scale length is none the density distribution is fitted with a single exponentional

    log= None,
        Run log
    debug =False,
        trigger for enhanced verbosity and plotting
    output_dir='./'
        directory for check plots


 OUTPUTS:
    Rotation curve for the specified disk at location of RCs
 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''


def apply_truncation(RC, radii, truncation_radius, Mass ):
    # This is an approximation of the implemantation of a truncation radius
    if truncation_radius[0] < radii[-1]:
     
        if truncation_radius[0].unit != truncation_radius[1].unit:
            raise RunTimeError(f' The units in the truncation radius do not match. Most likely the scale length value is not yet converted.')
        mix =  ( radii - truncation_radius[0].value) / truncation_radius[1].value
        mix[mix > 1.] = 1.
        mix[mix < 0.] = 0.
        RC =RC*(1-mix)+np.sqrt(c.Gsol*float(Mass)/(radii))*mix

    return RC

def exponential_parameter_RC(radii,parameters, sech =False, log= False,\
        truncation_radius = [None], debug=False):
    #print(f'This is the total mass in the parameterized exponential disk {float(parameters[1]):.2e}')
  
    if parameters.height_type != 'inf_thin':
        #This is not very exact for getting a 3D density
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            area = 2.*quad(sechsquare,0,np.inf,args=parameters.height.value)[0]*unit.pc
        central = parameters.central_SB/(1000.*area)

        exp_disk_potential = MNP(amp=central,hr=parameters.scale_length,hz=parameters.height,sech=sech)
    else:
        exp_disk_potential = EP(amp=parameters.central_SB,hr=parameters.scale_length)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        RC = [exp_disk_potential.vcirc(x*radii.unit) for x in radii.value]
    if not truncation_radius[0] is None:
        RC = apply_truncation(RC,radii,truncation_radius,parameters.total_SB)
       
    if np.isnan(RC[0]) and radii[0] == 0:
        RC[0] = 0.
    if RC[0] <= 0.:
        RC[0] = 0.
    RC =np.array(RC,dtype=float)*unit.km/unit.s
    return RC

exponential_parameter_RC.__doc__ =f'''
 NAME:
    exponential_parameter__RC

 PURPOSE:
    match the parameterized profile to the potential from galpy

 CATEGORY:
    rotation modelling

 INPUTS:
    radii = radii at which to evaluate the profile to obtain rotaional velocities
            in (km/s)
    parameters = dictionary with the required input -->
                corresponds to the components dictionary. In order to abvoid errors
                the values in the dictionary are quantities. i.e with astropy units
 OPTIONAL INPUTS:
    sech =False,
        indicator to use sech vertical distribution instead of exponential.

    truncation_radius = [None]
        location of trucation of the disk

    log= None,
        Run log
    debug =False,
        trigger for enhanced verbosity and plotting


 OUTPUTS:
    Rotation curve for the specified disk at location of RCs
 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''



def get_rotmod_scalelength(radii,density):
    s=0.
    sx =0.0
    sy =0.
    sxy = 0.
    sxx = 0.
    rcut =radii[-1]
    delt = rcut-radii[-2]
    for i in range(len(radii)):
        #This is the rotmod method
        if density[i] > 0.:
            s += 1.
            sx +=radii[i]
            sxx += radii[i]**2
            sy += np.log(density[i])
            sxy += radii[i]*np.log(density[i])
        det = s*sxx-sx*sx
    h = det/(sx*sy-s*sxy)
    if h > 0.5*rcut:
        h = 0.5*rcut
    if h < 0.1*rcut:
        h = 0.1*rcut
    dens = np.exp((sy*sxx-sx*sxy)/det)
    return h,dens

def hernquist_parameter_RC(radii,parameters, truncation_radius = [None],\
                           log= False, debug=False):
    '''This assumes a Hernquist potential where the scale length should correspond to hernquist scale length'''
    #The two is specified in https://docs.galpy.org/en/latest/reference/potentialhernquist.html?highlight=hernquist
    #It is assumed this hold with the the triaxial potential as well.
    bulge_potential = THP(amp=2.*parameters.total_SB,a= parameters.scale_length ,b= 1.,c = parameters.axis_ratio)
    #bulge_potential = THP(amp=2.*parameters['Total SB'],a= parameters['scale length'])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        RC = [float(bulge_potential.vcirc(x*radii.unit)) for x in radii.value]

    if not truncation_radius[0] is None:
        RC = apply_truncation(RC,radii,truncation_radius,parameters.total_SB)
 
    if np.isnan(RC[0]) and radii[0] == 0:
        RC[0] = 0.
    if RC[0] <= 0.:
        RC[0] = 0.
    return np.array(RC,dtype=float)*unit.km/unit.s

hernquist_parameter_RC.__doc__ =f'''
 NAME:
    hernquist_parameter__RC

 PURPOSE:
    match the parameterized profile to the potential from galpy

 CATEGORY:
    rotation modelling

 INPUTS:
    radii = radii at which to evaluate the profile to obtain rotaional velocities
            in (km/s)
    parameters = dictionary with the required input -->
                corresponds to the components dictionary. In order to avoid errors
                the values in the dictionary are quantities. i.e with astropy units
 OPTIONAL INPUTS:

    truncation_radis = []
                    location of trucation of the disk

    log= None,
        Run log
    debug =False,
        trigger for enhanced verbosity and plotting


 OUTPUTS:
    Rotation curve for the specified bulge at the location of the RCs
 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''


#this function is a carbon copy of the function in GIPSY's rotmod (Begeman 1989, vd Hulst 1992)
# Function to integrate from Cassertano 1983
def interg_function(z,x,y):
    xxx = (x**2 + y**2 + z**2)/(2.*x*y)
    rrr = (xxx**2-1.)
    ppp = 1.0/(xxx+np.sqrt(rrr))
    fm = 1.-ppp**2
    el1 = ( 1.3862944 + fm * ( 0.1119723 + fm * 0.0725296 ) ) - \
          ( 0.5 + fm * ( 0.1213478 + fm * 0.0288729 ) ) * np.log( fm )
    el2 = ( 1.0 + fm * ( 0.4630151 + fm * 0.1077812 ) ) - \
          ( fm * ( 0.2452727 + fm * 0.0412496 ) ) * np.log( fm )
    r = ( ( 1.0 - xxx * y / x ) * el2 / ( rrr ) + \
        ( y / x - ppp ) * el1 / np.sqrt( rrr ) ) /np.pi
    return r * np.sqrt( x / ( y * ppp ) )

def integrate_v(radii,density,R,rstart,h_z,step,iterations=200,mode = None ):
    # we cannot actually use quad or some such as the density profile is not a function but user supplied

    vsquared = 0.
    eval_radii = [rstart + step * i for i in range(iterations)]
    eval_density = np.interp(np.array(eval_radii),np.array(radii),np.array(density))
    weights = [4- 2*((i+1)%2) for i in range(iterations)]
    weights[0], weights[-1] = 1., 1.
    weights = np.array(weights,dtype=float)*step
    # ndens = the number of integrations
    for i in range(iterations):
        if 0 < eval_radii[i] < radii[len(density)-1] and eval_density[i] > 0 and R > 0.:
            zdz = integrate_z(R, eval_radii[i], h_z,mode)
            vsquared += 2.* np.pi*c.Grotmod.value/3. *zdz*eval_density[i]*weights[i]
    return vsquared

def integrate_z(radius,x,h_z,mode):
    if h_z != 0.:
       
        vertical_function = select_vertical(mode)
        # We integrated out to 250*h_z otherwise we get overflow problems
        # in principle it should be infinite (np.inf)
        zdz = quad(vertical_function,0.,250.*h_z,args=(radius,x,h_z))[0]
        
    else:
        if not (radius == x) and not radius == 0. and not x == 0. and \
                    (radius**2 + x**2)/(2.*radius*x) > 1:
                    zdz = interg_function(h_z,radius, x)
        else:
            zdz = 0.
    return zdz

def norm_exponential(radii,h):
    return np.exp(-1.*radii/h)/h
def norm_sechsquare(radii,h):
    return 1./(np.cosh(radii/h)**2*h)

'''This functions is the global function that creates and reads the RCs if the are not all read from file.
Basically if the RC_Construction is enabled this function is called.'''
def obtain_RCs(cfg,log=None):
   
    ######################################### Read the gas profiles and RC ################################################
    gas_profiles, total_rc  =\
        get_gas_profiles(cfg,log=log)
     ######################################### Read the optical profiles  #########################################
    try:
        optical_profiles,galfit_file = get_optical_profiles(cfg, extend = total_rc.extend,log=log)
    except Exception as e:
        print_log(f" We could not obtain the optical components and profiles because of {e}",log,debug=cfg.general.debug,screen=False)
        traceback.print_exc()
        raise InputError(f'We failed to retrieved the optical components from {cfg.RC_Construction.optical_file}')


   
   
    for profile in gas_profiles:
        print_log(f'''We have found a gas disk with a total mass of  {integrate_surface_density(gas_profiles[profile].radii,gas_profiles[profile].values)[0]:.2e}
and a central mass density {gas_profiles[profile].values[0]:.2f}.
''' ,log,screen= True)
    

  
    ########################################## Make a plot with the extracted profiles ######################3
    plot_profiles(gas_profiles,optical_profiles,\
        output_dir = cfg.general.output_dir, log=log)
   

    ########################################## Make a nice file with all the different components as a column ######################3
    write_profiles(gas_profiles,total_rc,optical_profiles= optical_profiles,\
        output_dir = cfg.general.output_dir, log=log, filename='Gas_Mass_Density_And_RC.txt')
 
    ######################################### Convert to Rotation curves ################################################
   
    derived_RCs = convert_dens_rc(optical_profiles, gas_profiles, cfg=cfg,\
            output_dir=cfg.general.output_dir)
   
    write_profiles(derived_RCs,total_rc,\
        output_dir = cfg.general.output_dir, log=log, filename=cfg.general.RC_file)
    

    return derived_RCs, total_rc

def random_density_disk(radii,density_profile,truncation_radius = None, debug= None,log= None):
    print_log(f'''We are calculating the random disk with:
h_z = {density_profile.height} and vertical mode = {density_profile.height_type}
''',log)
    
    if density_profile.values.unit != unit.Msun/unit.pc**2:
        raise UnitError(f'Your radius has to be in kpc for a rand density disk')
    else:
        density = np.array(density_profile.values.value) *1e6 # Apparently this is done in M-solar/kpc^2
    
    
    if radii.unit != unit.kpc:
        raise UnitError(f'Your radius has to be in kpc for a random density disk')
    else:
        rad_unit = radii.unit
        radii = radii.value
   
    ntimes =50.
    mode = density_profile.height_type
    if density_profile.height.unit != unit.kpc:
        raise UnitError(f'The scale height is not in kpc, it should be by now. (height = {density_profile.height}, name = {density_profile.name})')
    h_z = density_profile.height.value
    h_r,dens0 = get_rotmod_scalelength(radii,density)
    if truncation_radius[0] is None:
        rcut = radii[-1]+5.
    else:
        if truncation_radius[0].unit != rad_unit:
            raise UnitError(f'Your truncation radius has to be in kpc for a random density disk') 
        rcut = truncation_radius[0].value
    if truncation_radius[1] is None:
        delta = 0.2*h_r
    else:
        delta = truncation_radius[1]*h_r
    RC = []
   
    h_z1 = set_limits(h_z,0.1*h_r,0.3*h_r)

   
    for i,r in enumerate(radii):
        #looping throught the radii
        vsquared = 0.
        r1 = set_limits(r - 3.0 * h_z1, 0, np.inf)
        r2 = 0.
        if r1 < rcut +2.*delta:
            r2 = r+(r-r1)
            #This is very optimized
            ndens = int(6 * ntimes +1)
            step = (r2-r1)/(ndens -1)
            rstart = r1
            
            vsquared += integrate_v(radii,density,r,rstart,h_z,step,iterations=ndens,mode=mode)
            
            if r1 > 0.:
                ndens = int(r1*ntimes/h_r)
                ndens =int(2 * ( ndens / 2 ) + 3)
                step = r1 / ( ndens - 1 )
                rstart = 0.
                vsquared += integrate_v(radii,density,r,rstart,h_z,step,iterations=ndens,mode=mode)
        if r2 < ( rcut + 2.0 * delta ):
            ndens = ( rcut + 2.0 * delta - r2 ) * ntimes / h_r
            ndens = int(2 * ( ndens / 2. ) + 3.)
            step = ( rcut + 2.0 * delta - r2 ) / ( ndens - 1 )
            rstart = r2
            vsquared += integrate_v(radii,density,r,rstart,h_z,step,iterations=ndens,mode=mode)
        
        if vsquared < 0.:

            RC.append(-np.sqrt(-vsquared))
        else:
            RC.append(np.sqrt(vsquared))
   
    RC = np.array(RC,dtype=float)*unit.km/unit.s
    return RC

def sechsquare(x,b):
    '''Sech square function '''
    #sech(x) = 1./cosh(x)
    return 1./np.float64(np.cosh(x/b))**2

# Any vertical function goes as long a integral 0 --> inf (vert_func*dz) = 1.
def select_vertical(mode):
    if mode == 'sech-sq':
        return combined_rad_sech_square
    elif mode == 'exp':
        return combined_rad_exp
    elif mode == 'sech-simple':
        return combined_rad_sech_simple
    else:
        if mode:
            raise InputError('This vertical mode is not yet implemented')
        else:
            return None

#This appears to be unused
def selected_vertical_dist(mode):
    if mode == 'sech-sq':
        return norm_sechsquare
    elif mode == 'exp':
        return norm_exponential
    elif mode == 'sech-simple':
        return simple_sech
    else:
        return None

def simple_sech(radii,h):
    return 2./h/np.pi/np.cosh(radii/h)

def read_RCs(dir= './', file= 'You_Should_Set_A_File_RC.txt',debug=None,log=None):
    #read the file
    all_RCs = read_columns(f'{dir}{file}',debug=debug,log=log)

    #split out the totalrc
    input_RCs  ={}
    totalrc = None

    for name in all_RCs:
        if name in ['V_OBS','VROT']:
            totalrc = all_RCs[name]
            totalrc.component='All'
        else:
            input_RCs[name] = all_RCs[name]

    return input_RCs,totalrc
    #           RCs

read_RCs.__doc__ =f'''
 NAME:
    read_RCs
 PURPOSE:
    Read the RCs from a file. The file has to adhere to the pyROTMOD output format
 CATEGORY:
    support_functions

 INPUTS:
    di    directory
    input RCs = Dictionary with derived RCs
    total_rc = The observed RC
   

 OPTIONAL INPUTS:
    dir= './'

    Directory where  the file file is locate

    file= 'You_Should_Set_A_File_RC.txt'

    file name

 OUTPUTS:

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:

 NOTE:

'''


