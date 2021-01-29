# -*- coding: future_fstrings -*-

# This is an attempt at a holistic python version of ROTMOD and ROTMAS using bayesian fitting and such

from optparse import OptionParser
from pyROTMOD.optical.optical import get_optical_profiles
from pyROTMOD.gas.gas import get_gas_profiles
from pyROTMOD.rotmod.rotmod import convert_dens_rc
from pyROTMOD.rotmass.rotmass import the_action_is_go
from pyROTMOD.support import plot_profiles,write_profiles,write_RCs
import pyROTMOD.constants as c

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
    input_parameters,args = parser.parse_args()

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


    optical_profiles,components,galfit_file = get_optical_profiles(input_parameters.optical_file,
                                                    distance= float(input_parameters.distance),
                                                    zero_point_flux = float(input_parameters.zero_point_flux),
                                                    MLRatio= float(input_parameters.MLRatio))


    ######################################### Read the gas profiles and RC ################################################
    radii,gas_profile, total_rc,total_rc_err,scaleheights  = get_gas_profiles(input_parameters.gas_file)

    ########################################## Make a plot with the extracted profiles ######################3
    plot_profiles(radii, gas_profile,optical_profiles,distance =float(input_parameters.distance))

    ########################################## Make a nice file with all the different components as a column ######################3
    write_profiles(radii, gas_profile,total_rc,optical_profiles,distance =float(input_parameters.distance), errors = total_rc_err )
    ######################################### Convert to Rotation curves ################################################
    print(galfit_file)
    derived_RCs = convert_dens_rc(radii, optical_profiles, gas_profile,components,distance =float(input_parameters.distance),galfit_file = galfit_file)
    write_RCs(derived_RCs,total_rc,total_rc_err)

    ######################################### Run our Bayesian interactive fitter thingy ################################################

    the_action_is_go(radii,derived_RCs, total_rc,total_rc_err)
