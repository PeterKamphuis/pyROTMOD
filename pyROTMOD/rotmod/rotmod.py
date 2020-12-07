# -*- coding: future_fstrings -*-


from scipy.interpolate import CubicSpline
import numpy as np
def convert_dens_rc(radii, optical_profiles, gas_profile,components):

    optical_radii = np.array(optical_profiles[0][2:])
    #print(components, [optical_profiles[i][0] for i in range(len(optical_profiles))])
    optical_Rcs = []

    ########################### First we convert the optical values and produce a RC at locations of the gas ###################
    for x in range(1,len(optical_profiles)):
        if components[x-1][0] in ['expdisk','edgedisk']:
            found_RC = exponential_RC(optical_radii,np.array(optical_profiles[x][2:]), [components[x-1][3]])
        elif components[x-1][0] in ['sersic']:
            found_RC = sercic_RC(optical_radii,np.array(optical_profiles[x][2:]), [components[x-1][3]])
        else:
            print(f'We do not know how to convert the mass density of {components[x][0]}')
        tmp = CubicSpline(optical_radii,found_RC,extrapolate=True)
        optical_RCs.append([optical_profiles[x][0], 'KM/S'])
        for vel in tmp(radii):
            optical_RCs[x].append(vel)


    ########################### and last the gas which we do not interpolate  ###################
    gas_RC = exponential_RC(radii[2:],gas_profile[2:], 0.)

    return optical_RCs , gas_RC


def sercic_RC(radii,density,vertical_distribution):
    print(radii,density,vertical_distribution)
    RC = 1.
    return RC

# Obtain the velocities of a density profile where the vertical distribution is a exponential disk.
def exponential_RC(radii,density,vertical_distribution):
    
    RC = np.sqrt(-r*Force)
    print(radii,density,vertical_distribution)
    RC = 1.
    return RC
