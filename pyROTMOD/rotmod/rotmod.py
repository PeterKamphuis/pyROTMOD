# -*- coding: future_fstrings -*-

from galpy.potential import MN3ExponentialDiskPotential as MNP, RazorThinExponentialDiskPotential as EP,\
                            TriaxialHernquistPotential as THP
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.stats import hypsecant
from scipy.integrate import quad
#from astropy import units
from astropy import units as unit
from pyROTMOD.support import convertskyangle,integrate_surface_density,\
                        ensure_kpc_radii,print_log
from pyROTMOD.optical.optical import fit_profile,exponential,hernquist_profile
import pyROTMOD.constants as c
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt

class InputError(Exception):
    pass

def convert_dens_rc(radii, optical_profiles, gas_profiles,components,\
    distance =1.,opt_h_z = [0.,None], gas_scaleheights= [0.,None], galfit_file =False,\
    log= None, debug =False,output_dir='./'):
    #these should be dictionaries with their individual radii
    #organized in their typical profiles
    print(optical_profiles,gas_profiles)
    exit()
    '''This function converts the mass profiles to rotation curves'''
    kpc_radii = np.array(ensure_kpc_radii(radii,distance=distance,log=log)[2:],\
        dtype=float)
    optical_radii = np.array(ensure_kpc_radii(optical_profiles['RADI'],\
        distance=distance,log=log)[2:],dtype=float)
    RCs = [ensure_kpc_radii(radii,distance=distance,log=log)]

    ########################### First we convert the optical values and produce a RC at locations of the gas ###################
    for x,type in enumerate(optical_profiles):
        if optical_profiles[type][0] != 'RADI' and optical_profiles[type][1] != 'KM/S':
            #create an interpolated profile of in profile
            #tmp = CubicSpline(optical_radii,np.array(optical_profiles[type][2:]),extrapolate=True)
            #if components[x-1]['Type'] == 'infinite_disk':
            #    print_log(f'We have detected the input to be a random density profile hence we extrapolate from that to get the rotation curve',log)
            #    found_RC= random_density_disk(kpc_radii,tmp(kpc_radii),h_z = opt_h_z[0],mode = opt_h_z[1], log=log)
            if components[x-1]['Type'] in ['expdisk','edgedisk','random_disk']:
                print_log(f'We have detected the input to be an disk',log)
                found_RC = disk_RC(optical_radii,np.array(optical_profiles[type][2:]),out_radii=kpc_radii,\
                            h_z=opt_h_z,components = components[x-1],\
                            debug=debug,log=log, output_dir =output_dir)
                if components[x-1]['Type'] in ['random_disk']:
                    optical_profiles[type][0] = 'DISK_STELLAR'
            elif components[x-1]['Type'] in ['sersic','devauc']:
                #This should work fine for a devauc profile which is simply sersic with n= 4
                if  0.75 < components[x-1]['sersic index'] < 1.25:
                    print_log(f'''We have detected the input to be a sersic profile.
this one is very close to a disk so we will transform to an exponential disk. \n''',log)
                    found_RC = disk_RC(optical_radii,np.array(optical_profiles[type][2:]),out_radii=kpc_radii,\
                            h_z=opt_h_z,components = components[x-1],\
                            debug=debug,log=log, output_dir =output_dir)
                elif 3.75 < components[x-1]['sersic index'] < 4.25:
                    print_log(f'''We have detected the input to be a sersic profile.
this one is very close to a bulge so we will transform to a hernquist profile. \n''',log)
                    found_RC = bulge_RC(optical_radii,np.array(optical_profiles[type][2:]),out_radii=kpc_radii,\
                            h_z=opt_h_z,components = components[x-1],\
                            debug=debug,log=log, output_dir =output_dir)
                else:
                    found_RC = None
                    print_log(f'''We have detected the input to be a density profile for a sersic profile.
This is not something that pyROTMOD can deal with yet. If you know of a good python implementation of Baes & Gentile 2010.
Please let us know and we'll give it a go.''',log)
                    raise InputError('We have detected the input to be a density profile for a sersic profile. pyROTMOD cannot yet process this.')

            elif components[x-1]['Type'] in ['random_bulge','hernquist']:
                #if not galfit_file:
                #    found_RC = bulge_RC(kpc_radii,optical_radii,np.array(o))
                #    print_log(f'We have detected the input to be a density profile for a bulge that is too complicated for us',log)
                #else:
                print_log(f'Assuming a classical bulge spherical profile in a Hernquist profile',log)
                found_RC = bulge_RC(optical_radii,np.array(optical_profiles[type][2:]),out_radii=kpc_radii,\
                        h_z=opt_h_z,components = components[x-1],\
                        debug=debug,log=log, output_dir =output_dir)
            else:
                found_RC = [None,None,None,None]
                print_log(f'We do not know how to convert the mass density of {components[x-1]["Type"]}',log)
            if np.any(found_RC[2:]):
                try:
                    type =optical_profiles[type][0].split('_')
                    found_RC[0] = f'''{found_RC[0]}_{'_'.join(type[1:])}'''
                except:
                    pass

                RCs.append(found_RC)
        else:
            if type != 'RADI':
                tmp = CubicSpline(optical_radii,np.array(optical_profiles[type][2:]),extrapolate=True)
                RCs.append(optical_profiles[type][:2]+tmp(kpc_radii))

    ########################### and last the gas which we do not interpolate  ###################
    for i,gas_profile in enumerate(gas_profiles):
        if gas_profile[1] != 'KM/S':
            found_RC = random_density_disk(kpc_radii,gas_profile[2:],h_z = gas_scaleheights[i][0],mode = gas_scaleheights[i][1])
            RCs.append([gas_profile[0],'KM/S']+list(found_RC))
        else:
            RCs.append(gas_profile)

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
def bulge_RC(radii,density,h_z = [0.,'exp'],components = {'Type': 'sersic', 'axis ratio': None, 'scale length': None}, \
            log = None, debug=False, output_dir = './', truncated = -1.,out_radii = None):
    # First we need to get the total mass in the disk
    #print(f'We are using this vertical distributions {vertical_distribution}')
    #print(radii,density)

    if components['Total SB'] == None:
        components['Total SB'] = integrate_surface_density(radii,density)*unit.Msun

    if components['R effective'] == None:
        # if this is specified as an exponential disk without the parameters defined we fit a expoenential
        ring_area = integrate_surface_density(radii,density,calc_ring_area=True)
        scale = radii[-1]
        for i in range(1,len(radii)):
            current_mass = np.sum(ring_area[:i]*density[:i])*unit.Msun
            if current_mass < components['Total SB']/2.:
                components['scale length'] = radii[i]/1.8153
                break
    else:
        components['scale length']  = float(components['R effective'].value/1.8153)

    #try:
    #if components['Type'] == 'sersic':
    #    result,profile,components = fit_profile(radii,optical_profiles[type][2:],\
    #            cleaned_components[i-1],function='HERNQUIST_1', output_dir = output_dir\
    #            ,debug = debug, log = log)

    # let's see if our fit has a reasonable reduced chi square

    if components['Type'] in ['bulge','sersic','hernquist','devauc']:
        print_log(f'''We have found an hernquist profile with the following values.
The total mass of the disk is {components['Total SB']} a central mass density {components['Central SB']} .
The scale length is {components['scale length']} kpc and the scale height {components['scale height']} kpc.
The axis ratio is {components['axis ratio']}.
''' ,log,debug=debug)
        if out_radii[0] != None:
            RC_radii =out_radii
        else:
            RC_radii = radii
        RC = hernquist_parameter_RC(RC_radii,components,log = log, \
                    truncated = truncated )
        RC = [f'HERNQUIST','KM/S']+list(RC)

    else:
        raise InputError(f'We have not yet implemented the disk modelling of {components["Type"]}.')

    return RC


def bulge_RC_old(radii,opt_rad,density,debug=False,log=None):
    density =np.array(density)
    density[np.where(density < 0.)] = 0.
    mass,effective_radius=get_effective_radius(opt_rad,density,debug=debug,log=log)
    hern_scale = float(effective_radius)/1.8153  #Equation 38 in Hernquist 1990
    ##B anc are to x ratio of the profile i.e. one means spherical
    bulge_potential = THP(amp=2.*float(mass)*unit.Msun,a= hern_scale*unit.kpc,b= 1.,c = 1.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        RC = [bulge_potential.vcirc(x*unit.kpc) for x in radii]

    if np.isnan(RC[0]) and radii[0] == 0:
        RC[0] = 0.
    if RC[0] <= 0.:
        RC[0] = 0.
        # take from the density profile
    return RC

def hernquist_parameter_RC(radii,parameters, truncated =-1.,log= False, debug=False):
    '''This assumes a Hernquist potential where the scale length should correspond to hernquist scale length'''
    #print(f'This is the total mass in the parameterized exponential disk {float(parameters[1]):.2e}') 
    parameters['scale length'] = None
    if parameters['scale length'] is None:
        #This is a guess really
        central_dens = parameters['Central SB']* 1./(1*unit.pc)
        #I don't understand why astropy is not convvert the pc **3 to pc
        parameters['scale length'] = (parameters['Total SB']/(2.*np.pi*central_dens))**1/3*1./(unit.pc**2)
     

    if parameters['axis ratio']:
        axis_ratio = parameters['axis ratio']
    else:
        if parameters['scale height'] != None:
            axis_ratio = parameters['scale height']/parameters['scale length']
            axis_ratio = axis_ratio.value
        else:
            axis_ratio =1.
     
    #The two is specified in https://docs.galpy.org/en/latest/reference/potentialhernquist.html?highlight=hernquist
    #It is assumed this hold with the the triaxial potential as well.
    bulge_potential = THP(amp=2.*parameters['Total SB'],a= parameters['scale length'] ,b= 1.,c = axis_ratio)
    #bulge_potential = THP(amp=2.*parameters['Total SB'],a= parameters['scale length'])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        RC = [float(bulge_potential.vcirc(x*unit.kpc)) for x in radii]
    if truncated > 0.:
        ind =  np.where(radii > truncated)
        RC[ind] = np.sqrt(c.Gsol*float(parameters['Total SB'])/(radii[ind]*1000.*c.pc/(100.*1000.)))
    if np.isnan(RC[0]) and radii[0] == 0:
        RC[0] = 0.
    if RC[0] <= 0.:
        RC[0] = 0.
    return RC

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

    truncated =-1.
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


def disk_RC(radii,density,h_z = [0.,'exp'],components = {'Type': 'expdisk', 'scale height': None, 'scale length': None}, \
            log = None, debug=False, output_dir = './', truncated = -1.,out_radii = None):
    # First we need to get the total mass in the disk
    #print(f'We are using this vertical distributions {vertical_distribution}')
    #print(radii,density)
    if components['Type'] == 'sersic':
        fit_profile(radii,density,components,function='EXPONENTIAL')

        # let's see if our fit has a reasonable reduced chi square

    try:
        tmp = len(exp_profile)
    except UnboundLocalError:
        exp_profile = density

    if components['Total SB'] == None:
    #print(exp_profile)
        components['Total SB'] = integrate_surface_density(radii,exp_profile)*unit.Msun
    sech =False
    if components['scale height'] == None:
        components['scale height'] = h_z[0]*unit.kpc
        if h_z[1] == 'sech':
            sech = True
        if h_z[1] not in ['sech','exp'] and components['scale height'] != 0.:
            raise InputError(f'We cannot have {h_z[1]} with a thick exponential disk. Use a different disk or pick exp or sech')
    if out_radii[0] != None:
        RC_radii =out_radii
    else:
        RC_radii = radii
    if components['Type'] in ['expdisk','edgedisk']:
        print_log(f'''We have found an exponential disk with the following values.
The total mass of the disk is {components['Total SB']} a central mass density {components['Central SB']} .
The scale length is {components['scale length']} and the scale height {components['scale height']}.
The axis ratio is {components['axis ratio']}.
''' ,log,debug=debug)
        RC = exponential_parameter_RC(RC_radii,components,log = log,sech = sech, \
                    truncated = truncated )
        RC = [f'EXPONENTIAL','KM/S']+list(RC)
    elif components['Type'] in ['random_disk']:
        print_log(f'''We are fitting a random density distribution disk following Cassertano 1983.
''',log,debug=debug)

        RC = random_density_disk(RC_radii,density,h_z = h_z[0],\
            mode = h_z[1], log=log)
        RC = [f'DISK','KM/S']+list(RC)
    else:
        raise InputError(f'We have not yet implemented the disk modelling of {components["Type"]}.')

    return RC
disk_RC.__doc__ =f'''
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


def get_effective_radius(radii,density,debug=False,log= None):
    #The effective radius is where the integrated profile reaches half of the total

    ringarea= [0 if radii[0] == 0 else np.pi*((radii[0]+radii[1])/2.)**2]
    ringarea = np.hstack((ringarea,
                         [np.pi*(((y+z)/2.)**2-((y+x)/2.)**2) for x,y,z in zip(radii,radii[1:],radii[2:])],
                         [np.pi*((radii[-1]+0.5*(radii[-1]-radii[-2]))**2-((radii[-1]+radii[-2])/2.)**2)]
                         ))
    ringarea = ringarea*1000.**2

    mass = np.sum([x*y for x,y in zip(ringarea,density)])

    Cuma_prof = []
    for i,rad in enumerate(radii):
        if i == 0:
            Cuma_prof.append(ringarea[i]*density[i])
        else:
            new = Cuma_prof[-1]+ringarea[i]*density[i]
            Cuma_prof.append(new)
    #now to get the half of the total mass.
    if Cuma_prof[0] > mass/2.:
        print_log(f'''Your effective radius of the bulge is smaller than the first ring.
You should improve the resolution of the bulge profile.''',log)
        out_rad = radii[0]
    else:
        out_rad = radii[np.where(Cuma_prof<mass/2.)[-1][-1]]

    return mass,out_rad




def sersic_parameter_RC(radii,parameters,sech=True):
    # There is no parameterized version for a potential of all sersic indices as there is no deprojected surface density profile
    # However for n = 4 we can use the hernquist profile, and n =1 the exponential disk, else we need to go through the parameterization of a deprojected profile of Baes & Gentile 2010

    if 0.75 < float(parameters['sersic index']) < 1.25:
        # assume this to be close enough to a infinitely thin exponential
        hr =  float(parameters[3])/1.678
        exp_disk_potential = EP(amp=float(parameters[2])*unit.Msun/unit.pc**2,hr=hr*unit.kpc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            RC = exp_disk_potential.vcirc(radii*unit.kpc)
    elif 3.75 < float(parameters['sersic index']) < 4.25:
        # take this to be a bulge and we use a hernequist potential
        hern_scale = float(parameters[3])/1.8153  #Equation 38 in Hernquist 1990
        bulge_potential = THP(amp=2.*float(parameters[1])*unit.Msun,a= hern_scale*unit.kpc,b= 1.,c = float(parameters[5]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            RC = [bulge_potential.vcirc(x*unit.kpc) for x in radii]
    else:
        print("This sersic  profile cannot be parameterized.")
    if np.isnan(RC[0]) and radii[0] == 0:
        RC[0] = 0.
    if RC[0] <= 0.:
        RC[0] = 0.
        # take from the density profile
    return RC

def exponential_parameter_RC(radii,parameters, sech =False, truncated =-1.,log= False, debug=False):
    #print(f'This is the total mass in the parameterized exponential disk {float(parameters[1]):.2e}')
    if parameters['scale height'] == None:
        parameters['scale height'] = 0.*unit.kpc
    if parameters['scale height'].value > 0.:
        #This is not very exact for getting a 3D density
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            area = 2.*quad(sechsquare,0,np.inf,args=(parameters['scale height'].value))[0]*unit.pc
        central = parameters['Central SB']/(1000.*area)

        exp_disk_potential = MNP(amp=central,hr=parameters['scale length'],hz=parameters['scale height'],sech=sech)
    else:
        exp_disk_potential = EP(amp=parameters['Central SB'],hr=parameters['scale length'])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        RC = exp_disk_potential.vcirc(radii*unit.kpc)
    if truncated > 0.:
        ind =  np.where(radii > truncated)
        RC[ind] = np.sqrt(c.Gsol*float(parameters['Total SB'])/(radii[ind]*1000.*c.pc/(100.*1000.)))
    if np.isnan(RC[0]) and radii[0] == 0:
        RC[0] = 0.
    if RC[0] <= 0.:
        RC[0] = 0.
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

    truncated =-1.
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

def sechsquare(x,b):
    '''Sech square function '''
    #sech(x) = 1./cosh(x)
    return 1./np.float64(np.cosh(x/b))**2

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


def random_density_disk(radii,density,h_z = 0.,mode = None, log= None):
    print_log(f'''We are calculating the random disk with:
h_z = {h_z} and vertical mode = {mode}
''',log)
    density = np.array(density) *1e6 # Apparently this is done in M-solar/kpc^2
    rcut = radii[-1]
    delta = 0.2
    ntimes =50.
    h_r,dens0 = get_rotmod_scalelength(radii,density)
    #print(h_r,dens0)
    RC = []
    h_z1 = set_limits(h_z,0.1*h_r,0.3*h_r)
    for i,r in enumerate(radii):
        #looping throught the radii
        vsquared =0.
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
    return RC


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
            vsquared += 2.* np.pi*c.Grotmod/3. *zdz*eval_density[i]*weights[i]
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

def selected_vertical_dist(mode):
    if mode == 'sech-sq':
        return norm_sechsquare
    elif mode == 'exp':
        return norm_exponential
    elif mode == 'sech-simple':
        return simple_sech
    else:
        return None

def combined_rad_sech_square(z,x,y,h_z):
    return interg_function(z,x,y)*norm_sechsquare(z,h_z)
def combined_rad_exp(z,x,y,h_z):
    return interg_function(z,x,y)*norm_exponential(z,h_z)
def combined_rad_sech_simple(z,x,y,h_z):
    return interg_function(z,x,y)*simple_sech(z,h_z)
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

def norm_exponential(radii,h):
    return np.exp(-1.*radii/h)/h
def norm_sechsquare(radii,h):
    return 1./(np.cosh(radii/h)**2*h)
def simple_sech(radii,h):
    return 2./h/np.pi/np.cosh(radii/h)

def set_limits(value,minv,maxv,debug = False):
    if value < minv:
        return minv
    elif value > maxv:
        return maxv
    else:
        return value

set_limits.__doc__ =f'''
 NAME:
    set_limits
 PURPOSE:
    Make sure Value is between min and max else set to min when smaller or max when larger.
 CATEGORY:
    support_functions

 INPUTS:
    value = value to evaluate
    minv = minimum acceptable value
    maxv = maximum allowed value

 OPTIONAL INPUTS:
    debug = False

 OUTPUTS:
    the limited Value

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''
