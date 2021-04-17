# -*- coding: future_fstrings -*-

# This is an attempt at a holistic python version of ROTMOD and ROTMAS using bayesian fitting and such

from optparse import OptionParser
import pyROTMOD
from pyROTMOD.optical.optical import get_optical_profiles
from pyROTMOD.gas.gas import get_gas_profiles
from pyROTMOD.rotmod.rotmod import convert_dens_rc
from pyROTMOD.rotmass.rotmass import the_action_is_go
from pyROTMOD.support import plot_profiles,write_profiles,write_RCs,print_log,integrate_surface_density,convertskyangle
import pyROTMOD.constants as c
import traceback
import warnings
import os
import numpy as np
from datetime import datetime
class InputError(Exception):
    pass


def main(argv):
    c.initialize()
    #Random
    ############################# Handle the arguments that are entered to the program ########################
    parser  = OptionParser()
    parser.add_option('-c','--cf','--configuration_file', action ="store" ,dest = "configfile", default = 'FAT_INPUT.config', help = 'Define the input configuration file.',metavar='CONFIGURATION_FILE')
    parser.add_option('-o','--of','--optical_file', action ="store" ,dest = "optical_file", default = None, help = 'Provide a file with the optical distributions. Either a file with columns named RADI DISK BULGE or a galfit file.',metavar='OPTICAL_FILE')
    parser.add_option('-d','--distance', action ="store" ,dest = "distance", default = 0., help = 'A Distance to galaxy, this does not work without it.',metavar='DISTANCE')
    parser.add_option('-f','--zero_point_flux', action ="store" ,dest = "zero_point_flux", default = 0., help = 'Zero point flux (Jy) for convurting magnitude to flux. Default is the WISE zero point fluz',metavar='DISTANCE')
    parser.add_option('-g','--gf','--gas_file', action ="store" ,dest = "gas_file", default = None, help = 'Provide a file with the gas distributions. Ether a file with the columns RADI SBR VROT (with SBR in Jy/beam*km.s) or a tirific.def file.',metavar='GAS_FILE')
    parser.add_option('-m','--ml','--mass_to_light', action ="store" ,dest = "MLRatio", default = 0.45, help = 'Mass to Light Ratio to use for optical conversion to profile',metavar='MASS_TO_LIGHT')
    parser.add_option('-n','--ncpu', action ="store" ,dest = "ncpu", default = 6, help = 'Number of CPUs to use.')
    parser.add_option('-l','--log', action ="store" ,dest = "log", default ='Log.txt', help = 'Log file to write information to.')
    parser.add_option('-p','--output_dir', action ="store" ,dest = "output", default ='pyROTMOD_products', help = 'Directory to write our output to')
    input_parameters,args = parser.parse_args()
    output_dir =  input_parameters.output
    if output_dir[-1] != '/':
        output_dir = f"{output_dir}/"
    output_dir = f"{os.getcwd()}/{output_dir}"

    log = f"{output_dir}{input_parameters.log}"
    print(log)
    if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        #If it exists move the previous Log
    if os.path.exists(log):
        os.rename(log,f"{output_dir}/Previous_Log.txt")

    with open(log,'w') as write_log:
        write_log.write(f'''This file is a log of the modelling process run at {datetime.now()}.
This is version {pyROTMOD.__version__} of the program.
''')

    if input_parameters.distance == 0.:
        try:
            vsys = gas_profiles.load_tirific(input_parameters.gas_file, Variables = ['VSYS'])
            if vsys[0] == 0.:
                raise InputError(f'We cannot model profiles adequately without a distance')
            else:
                input_parameters.distance = vsys[0]/c.H_0
        except:
            raise InputError(f'We cannot model profiles adequately without a distance')
    # if the user does not provide a zero point flux we assume a WISE 3.4 image
    if input_parameters.zero_point_flux == 0:
        # This is actually the spitzer zero point flux from https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/17/
        input_parameters.zero_point_flux = 280.9 # jy
        # For WISE
        # input_parameters.zero_point_flux =309.504                       #Jy From the WISE photometry website http://wise2.ipac.caltech.edu/docs/release/prelim/expsup/sec4_3g.html
    ######################################## Read the optical profiles ##########################################

    try:

        optical_profiles,components,galfit_file = get_optical_profiles(input_parameters.optical_file,
                                                    distance= float(input_parameters.distance),exp_time =1.,
                                                    zero_point_flux = float(input_parameters.zero_point_flux),
                                                    MLRatio= float(input_parameters.MLRatio), log= log)
    except Exception as e:
        print_log(f" We could not obtain the optical components and profiles because of {e}",log)
        traceback.print_exc()

    print_log(f"We found the following optical components",log)
    for x in components:
            # Components are returned as [type,integrated magnitude,scale parameter in arcsec,sercic index or scaleheight in arcsec, axis ratio]
        if x[0] in ['expdisk','edgedisk']:
            print_log(f'''We have found an exponential disk with the following values.
The total mass of the disk is {x[1]:.2e} M_sol  a central mass density {x[2]:.2f} M_sol/pc^2 with a M/L {float(input_parameters.MLRatio)}.
The scale length is {x[3]:.2f} kpc and the scale height {x[4]:.2f} kpc.
The axis ratio is {x[5]:.2f}.
''' ,log,screen= True)
        if x[0] in ['sersic']:
            print_log(f'''We have found a sersic component with the following values.
The total mass is {x[1]:.2e} a central mass density {x[2]:.2f} M_sol/pc^2 with a M/L {float(input_parameters.MLRatio)}.
The effective radius {x[3]:.2f} kpc and sersic index = {x[4]:.2f}.
The axis ratio is {x[5]:.2f}.
''' ,log,screen= True)


    ######################################### Read the gas profiles and RC ################################################
    radii,gas_profile, total_rc,total_rc_err,scaleheights  = get_gas_profiles(input_parameters.gas_file,log=log)
    print_log(f'''We have found a gas disk with a total mass of  {integrate_surface_density(convertskyangle(np.array(radii[2:],dtype=float),float(input_parameters.distance)),np.array(gas_profile[2:],dtype=float)):.2e}
and a central mass density {gas_profile[2]:.2f} M_sol/pc^2.
''' ,log,screen= True)



    ########################################## Make a plot with the extracted profiles ######################3
    plot_profiles(radii, gas_profile,optical_profiles,distance =float(input_parameters.distance),output_dir = output_dir )

    ########################################## Make a nice file with all the different components as a column ######################3
    write_profiles(radii, gas_profile,total_rc,optical_profiles,distance =float(input_parameters.distance), errors = total_rc_err ,output_dir = output_dir )
    ######################################### Convert to Rotation curves ################################################
    derived_RCs = convert_dens_rc(radii, optical_profiles, gas_profile,components,distance =float(input_parameters.distance),galfit_file = galfit_file)
    write_RCs(derived_RCs,total_rc,total_rc_err,output_dir = output_dir)

    ######################################### Run our Bayesian interactive fitter thingy ################################################
    if radii[1] == 'ARCSEC':
        radii[2:] = convertskyangle(np.array(radii[2:],dtype=float),float(input_parameters.distance))
        radii[1]  = 'KPC'
    the_action_is_go(radii,derived_RCs, total_rc,total_rc_err)
