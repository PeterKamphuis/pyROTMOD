# -*- coding: future_fstrings -*-

from galpy.potential import MiyamotoNagaiPotential as MNP
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from astropy import units
from pyROTMOD.support import convertskyangle
import pyROTMOD.constants as c
import numpy as np
def convert_dens_rc(radii, optical_profiles, gas_profile,components,distance =1.,gas_scaleheight=0., galfit_file =False):

    kpc_radii = convertskyangle(radii[2:],distance=distance)
    optical_radii = convertskyangle(np.array(optical_profiles[0][2:]),distance=distance)
    #print(components, [optical_profiles[i][0] for i in range(len(optical_profiles))])
    RCs = [['RADII','KPC',]]
    for rad in kpc_radii:
        RCs[0].append(rad)

    ########################### First we convert the optical values and produce a RC at locations of the gas ###################
    for x in range(1,len(optical_profiles)):
        if components[x-1][0] in ['expdisk','edgedisk']:
            if not galfit_file:
                print(f'We have detected the input to be a density profile hence we extropolate from that to get the rotation curve')
                tmp = CubicSpline(optical_radii,np.array(optical_profiles[x][2:]),extrapolate=True)
                found_RC = exponential_RC(kpc_radii,tmp(kpc_radii), float(components[x-1][3]))
            else:
                print(f'We have detected the input to be a galfit file hence we extropolate from that to get the rotation curve')
                found_RC = exponential_parameter_RC(kpc_radii,components[x-1])

        elif components[x-1][0] in ['sersic']:
            #if not galfit_file:
            print(f'We have detected the input to be a density profile hence we extropolate from that to get the rotation curve for sersic')
            tmp = CubicSpline(optical_radii,np.array(optical_profiles[x][2:]),extrapolate=True)
            found_RC = sersic_RC(kpc_radii,tmp(kpc_radii),eff_radius =  float(components[x-1][2]), axis_ratio = float(components[x-1][4]))
            #else:
            #    print(f'We have detected the input to be a density profile hence we extropolate from that to get the rotation curve')
            #    found_RC = sersic_parameter_RC(kpc_radii,components[x-1])
            #found_RC = sercic_RC(optical_radii,np.array(optical_profiles[x][2:]), [components[x-1][3]])
        else:
            print(f'We do not know how to convert the mass density of {components[x][0]}')
        RCs.append([optical_profiles[x][0], 'KM/S'])
        for vel in found_RC:
            RCs[x].append(vel)

    ########################### and last the gas which we do not interpolate  ###################
    gas_scaleheight = 0.4
    found_RC = exponential_RC(kpc_radii,gas_profile[2:], gas_scaleheight)
    RCs.append(['DISK_G', 'KM/S'])
    for vel in found_RC:
        RCs[-1].append(vel)

    return RCs


def sersic_RC(radii,density,axis_ratio = 0., eff_radius = 0.):
    # The problem here is that the scale in the bulge is not necessarily the one in the Miyago-Nagai potential hence we estimate it for each point from the density distribution
    if eff_radius ==0.:
        eff_radius = radii[-1]/2.

    scalelength,central,Mass_ind,Mass = get_individual_parameters(radii,density,initial_estimates =[density[0],eff_radius])

    RC = []
    for i in range(len(radii)):
        if scalelength[i] ==0. and central[i] == 0.:
            #print('When the sersic is a point mass')
            #print(c.Gsol,radii[i],c.pc, Mass)
            RC.append(np.sqrt(c.Gsol*Mass/(radii[i]*1000.*c.pc/(100.*1000.))))
        else:
            sersic_potential = MNP(amp=float(Mass_ind[i])*units.Msun,a=float(scalelength[i])*units.kpc,b=float(scalelength[i]*axis_ratio)*units.kpc)
            RC.append(sersic_potential.vcirc(radii[i]*units.kpc))
    #print(f'This is what we derive')
    #for i,rad in enumerate(radii):
    #    print(f'rad = {rad}, velocity = {RC[i]}')

    return RC

def sercic_parameter_RC(radii,parameters):
    print(radii,density,vertical_distribution)
    RC = 1.
    return RC

def get_individual_parameters(radii,density,fit_bound=[[-np.inf,-np.inf],[np.inf,np.inf]],initial_estimates = [0.,0.]):
    extended_profile = CubicSpline(radii,density,extrapolate = True,bc_type ='natural')
    scalelength = []
    central = []
    Mass_ind = []
    Mass = get_mass(radii,density)
    for i in range(len(radii)):
        # The area in the ring
        if i == 0.:
            if radii[i] == 0.:
                pair0 = [-2*radii[i+1],extended_profile(-2*radii[i+1])]
                pair1 = [-1*radii[i+1],extended_profile(-1*radii[i+1])]

            else:
                pair0 = [-1*radii[i+1],extended_profile(-1*radii[i+1])]
                pair1 = [0,extended_profile(0.)]

            pair3 = [radii[i+1],density[i+1]]
            pair4 = [radii[i+2],density[i+2]]
        if i == 1.:
            if radii[i-1] == 0.:

                pair0 = [-1*radii[i+1],extended_profile(-1*radii[i+1])]
            else:
                pair0 = [0,extended_profile(0.)]
            pair1 = [radii[i-1],density[i-1]]
            pair3 = [radii[i+1],density[i+1]]
            pair4 = [radii[i+2],density[i+2]]
        elif i == len(radii)-2:
            pair0 =  [radii[i-2],density[i-2]]
            pair1= [radii[i-1],density[i-1]]
            pair3 = [radii[i+1],density[i+1]]
            pair4 = [2.*radii[i+1]-radii[i],extended_profile(2.*radii[i+1]-radii[i])]
        elif i == len(radii)-1:
            pair0 =  [radii[i-2],density[i-2]]
            pair1= [radii[i-1],density[i-1]]
            outer_extra = 2.*radii[i]-radii[i-1]
            outer_extra2 = outer_extra+(radii[i]-radii[i-1])
            pair3 = [outer_extra,extended_profile(outer_extra)]
            pair4 = [outer_extra2,extended_profile(outer_extra2)]
        else:
            pair0 =  [radii[i-2],density[i-2]]
            pair1= [radii[i-1],density[i-1]]
            pair3 = [radii[i+1],density[i+1]]
            pair4 = [radii[i+2],density[i+2]]
        pair2 = [radii[i],density[i]]

        fit_radii = [pair0[0],pair1[0],pair2[0],pair3[0],pair4[0]]
        fit_density = [pair0[1],pair1[1],pair2[1],pair3[1],pair4[1]]
        #fit_radii = [pair1[0],pair2[0],pair3[0]]
        #fit_density = [pair1[1],pair2[1],pair3[1]]
        for i,par in enumerate(fit_density):
            if par < 1e-8:
                fit_density[i] = 0.
        #print('This is what well fit')
        #print(fit_radii,fit_density)

        if np.sum(fit_density) == 0.:
            exp_parameters =[0.,0.]
        else:
            try:
                exp_parameters, exp_covariance = curve_fit(exponential, fit_radii, fit_density,p0=initial_estimates,bounds=fit_bound)
            except RuntimeError:
                exp_parameters =[0.,0.]
        scalelength.append(exp_parameters[1])
        central.append(exp_parameters[0])
        exp_profile =exponential(radii,*exp_parameters)
        print(exp_profile)
        Mass_ind.append(get_mass(radii,exp_profile))
    print(f'This is the total mass in the exponential disk {Mass:.2e}')
    return scalelength,central,Mass_ind,Mass

def get_mass(radii,profile):
    Mass= 0.
    for i,rad in enumerate(radii):
        if i == 0.:
            inner = radii[i]/2.
            outer = (radii[i]+radii[i+1])/2.
        elif i == len(radii)-1:
            inner = (radii[i-1]+radii[i])/2.
            outer = radii[i]+(radii[i-1]-radii[i])/2.
        else:
            inner = (radii[i-1]+radii[i])/2.
            outer = (radii[i]+radii[i+1])/2.
        area = (np.pi*outer**2-np.pi*inner**2)*1000**2.
        Mass += area*profile[i]
    return Mass
# Obtain the velocities of a density profile where the vertical distribution is a exponential disk.
def exponential_RC(radii,density,vertical_distribution):
    # First we need to get the total mass in the disk
    #print(f'We are using this vertical distributions {vertical_distribution}')
    #print(radii,density)
    tot_parameters, tot_covariance = curve_fit(exponential, radii, density,p0=[3.,5])

    fit_bound = [[0.,-np.inf],[tot_parameters[0]+1.,np.inf]]
    #fit_bound = [[-np.inf,-np.inf],[np.inf,np.inf]]
    #print(f'for the total profile we find h = {tot_parameters[1]} and cent = {tot_parameters[0]}')

    scalelength,central,Mass,Mass_tot = get_individual_parameters(radii,density,fit_bound=fit_bound,initial_estimates = tot_parameters)


    print('These are the scalelengths and and central densities')
    print(scalelength,central)
    print('These are the individual masses')
    print([f'{x:.2e}' for x in Mass])

    # Now to caluclate the vcirc for each of these parameters
    RC = []

    for i in range(len(radii)):
        if scalelength[i] ==0. and central[i] == 0.:
            RC.append(np.sqrt(c.Gsol*Mass_tot/(radii[i]*1000.*c.pc/(100.*1000.))))
        else:
            exp_disk_potential = MNP(amp=float(Mass_tot)*units.Msun,a=float(scalelength[i])*units.kpc,b=float(vertical_distribution)*units.kpc)
            RC.append(exp_disk_potential.vcirc(radii[i]*units.kpc))
    #print(f'This is what we derive')
    #for i,rad in enumerate(radii):
    #    print(f'rad = {rad}, velocity = {RC[i]}')
    # Take  MiyamotoNagaiPotential from galpy and set scale length between neighbouring points
    # Interpolate with cubic (Akima) spline and fit exponential scale length at each point and calculate V_Circ with given scale height
    # If this works it will allow automatically allow for incorporrating flaring

    # Following Binney and Tremaine (eq 2.29)Centripal force is vc^2/r which means vc =sqrt(r*|F|)
    #   F = dPHI/dr with PHI the gravitational potential
    #vc^2 =

    #RC = np.sqrt(-r*Force)
    #print(radii,density,vertical_distribution)
    #RC = 1.
    return RC


def exponential_parameter_RC(radii,parameters,truncated =-1.):
    print(f'This is the total mass in the parameterized exponential disk {float(parameters[1]):.2e}')
    exp_disk_potential = MNP(amp=float(parameters[1])*units.Msun,a=float(parameters[2])*units.kpc,b=float(parameters[3])*units.kpc)
    RC = exp_disk_potential.vcirc(radii*units.kpc)
    if truncated > 0.:
        ind =  np.where(radii > truncated)
        RC[ind] = np.sqrt(c.Gsol*float(parameters[1])/(radii[ind]*1000.*c.pc/(100.*1000.)))

    return RC
def exponential(radii,central,h):
    return central*np.exp(-1.*radii/h)
