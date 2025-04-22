The Configuration File
=====

Introduction
----

pyROTMOD uses OmegaConf (https://github.com/omry/omegaconf) to handle the input settings. pyROTMOD can be ran with default settings or settings from a yml configuration file or with command line input.

In a run pyROTMOD first checks the defaults, then the configuration yaml and finally the command line input. This mean that if a value is set in all three input methods the one from the command line is used.

The yml file has six main sections individuals keywords, input, output, RC_Construction, fitting_general and fitting_parameters that contain several parameters that can adjust how pyROTMOD runs. All these  parameters are described in detail below.

Individual Keywords
----

Individual Keywords
 --------
*No specifier*

**print_examples**:

  *bool, optional, default = False*

  Print an example input yaml file and an example catalogue.

**configuration_file**:

  *str, optional, default = None*

  input configuration file

Input Keywords
--------
*Specified with input*

**ncpu**:

  *int, optional, default = assigned or available in hardware - 1*

  Number of CPUs used for threaded parts of pyROTMOD. In numpyro this determines the number of chains used in the mcmc fitting. In galfit this is the number of threads used for rotation curve fitting.


**RC_File.txt**

  *str, optional, default = 'RCs_For_Fitting.txt'*

  File where RCs can be given to the code without deriving their density profiles.
  They need to be in the pyROTMOD format. This file is only read when RC_Construction is enabled to allow for a mixture of density profiles and derived RCs. In case one wants to continue from the output of a previous run the RC file in output is read.

**distance**: 

  *float, required, default = None*

  Distance to the galaxy. In case the gas_file is a tirific file the default is vsys from that file. In case of a table no default exists.
  This is a required parameter and the code will exit if not provided.

 **font**: 

  *str, optional, default = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"*


Output Keywords
--------
*Specified with output*

**RC_File.txt**:

  *str, optional, default = 'RCs_For_Fitting.txt'*

  File where the all the derived RCs will be written in the pyROTMOD format. \n
  !! If RC_Contrustion is enabled this file will be overwritten !!!

**out_base**:

  *str, optional, default = 'Final_Results'*

  Base name of the output files. 

**output_dir**:

  *str, optional, default = f'{os.getcwd()}/pyROTMOD_products/'*

  Directory where to place all the output products.

**log_directory**:

  *str, optional, default = f'{output_dir}Logs/{datetime.now().strftime("%H:%M:%S-%d-%m-%Y")}/'*

  Name of the directory where log files will be stored.

**log**:

  *str, optional, default = log.txt*

  Name of the log file where all print messages are produced.

**debug**:

  *bool, optional, default =False*

  Switch for printing debug messages in the log. If you are posting an issue with a log on the github please run once with this turned on.

**debug_functions**:

  *List, optional, default = ['ALL']*

  List that can specify for which functions debug messages should be printed. This is a list of strings and can contain any function name.
  For example to print debug messages in rotmass_main output.debug_functions=['ROTMASS_MAIN']

**verbose**:

  *bool, optional, default =False*

  Print more messages then usual to the screen. 

RC_Construction Keywords
--------
*Specified with RC_Construction*

**enable**

  *bool, optional, default =True*

  Whether to construct the RC from the density files. If this is disabled the code will only fit the RCs as listed in output directory output RC file.
**out_base**:

  *str, optional, default = ''*
  Base name of the output files. Additional base_name for output produced in this section

**optical_file**: null

  *str, required, no default*

  The file containing the optical light distributions. This can either be a galfit file or a table in a text file. In case of the latter the first column should be 'RADI' and specify the radii of the profile in kpc or arcsec.
  The units of every column should be specified on the second row.  Acceptable units are  'KM/S', 'M/S', 'M_SOLAR/PC^2', 'M_SOLAR/PC^2', 'MAG/ARCSEC^2' where the velocity profiles can not be mixed with luminosity or mass profiles.
  Other columns can be EXPONTIAL_#,HERNQUIST_#, SERSIC_#,  DENSITY_#, DISK_# ,BULGE_#  where the # indicates an integer number.
  IF the input is in velocities the input is transfer directly to the RC fitting else the following scheme is maintained:
  The profiles EXPONENTIAL and HERNQUIST are parameterized with their respective functions.
  The SERSIC profiles are parameterized with an exponential when 0.75 < n < 1.25 and an hernquist profile when  3.75 < n < 4.25
  The DISK and BULGE profiles are assumed to be unparameterized disk profiles
  With the DENSITY an attempt will be made to fit the exponential or hernquist profiles or both and if it will be split according to the best reduced chi square.
  In case the parameterisation fails a random disk will be assumed (EXP,SER,DENS) or the profile will be removed from the RC fitting (HERN, BULG)
  In the RC fitting all components are combined and only a disk component (EXPONENTIAL,DISK,SERSIC) and a bulge component (HERNQUIST, BULGE) are used.
  Note that we cannot yet convert random BULGE profiles hence they need to be in velocity to be included.

**gas_file**: null

  *str, required, no default*

  The file containing the total rotation curve and the gas distribution. This can be a tirific .def file or a table in a text file. In case of the latter it should be arranged as the optical file.
  The RADI can be different and the gas disk should be indicated with DISK_G_# the observed RC as V_OBS  with V_OBS_ERR as its error. In case of a tirific file every pair of even-uneven disks are combined into a single disk under the assumption the def indicates different values of the approaching and receding side that should be averaged.
  The first pair of disks is assumed to be V_OBS. 

**scaleheight**

  *list, optional, default = [0., None, 'KPC', 'inf_thin']*

  scale height and vertical mode of the optical disks. If 0. or vertical mode = None infinitely thin disks are used.
  vertical mode options are  ['exp', 'sech-sq','sech', 'constant', 'lorentzian']. Anything in galfit file supersedes this input.

**truncation_radius**:

  *list, optional, default = [None, 0.2, 'KPC']*

  Truncation radius and softening length at which the density will be tapered to zero. The softening length is as faction of scale length.
  The last item is the units of the truncation radius.

**gas_scaleheight**

  *list, optional, default =  [0., None, 'KPC', 'inf_thin']*

  scale height and vertical mode of the gas  disks. If read from tirific def file that takes precedence.

**gas_truncation_radius**:

  *list, optional, default = [None, 0.2, 'KPC']*

  same as from optical.

**axis_ratio**

  * float, optional, default = 1.*

  axis ratio of the disks. Anything in galfit file supersedes this input.

**exposure_time**:

  *float, optional, default = 1.*

  Exposure time of the optical image. Certain galfit components (edge,sersic) take this into account. For the fit with galfit this should be set in the header of the image.


**mass_to_light_ratio**:

  *float, optional, default = 1.0*

  Mass to light ratio to be used for converting the optical luminosity profiles to mass profiles.
  Be aware that also in the fitting of the RCs a mass to light component is implemented. and that if this is set 
  to a different value than one any variations in RC fitting would reflect from this value. 
  

**keep_random_profiles**:

  *bool, optional, default = False*

  If we have random profiles in Lsun/pc^2 we do not fit one of the known functions to them keepand assume they are SBR_Profile when multiplied with MLratio, 
  if set to false we attempt to fit a profile to them with known functions.

**band**:

  *str, optional , default = SPITZER3.6*

  Band to be used for magnitude to flux/luminosity conversion. This is only used if the input file is in MAG/ARCSEC^2 or when the input file is a galfit file.
  currently available bands are SPITZER3.6, WISE3.4 

**gas_band**:

  *str, optional , default = 21cm*

  For future use



General Fitting Keywords
--------
*Specified with fitting_general*

**enable**

  *bool, optional, default =True*

  Run the Bayesian Fitting of RCs.

**negative_values**:

  *Bool, optional, default = False*

  Allow for negative values for the parameters or Not.

**initial_minimizer**:
  
  *str, optional, default = 'differential_evolution'

  The minimizer used in the initial estimates. This can be any lmfit minimizer. This has no effect when using numpyro as the fitting method. 
  The default is differential_evolution which is a global minimizer.

**HALO**:

  *str, optional, default =NFW'*

  The requested DM halo potential. For now this is NFW, ISO, BURKERT for the NFW profile, the pseudo isothermal halo profile and a Burkert profile.
  MOND is also an option to fit the classic implementation of MOND.

**single_stellar_ML**:

  *bool, optional, default = True*

  If set to True the code will assume a single mass to light ratio for all stellar components. If set to False the code will 
  fit a different mass to light ratio for each stellar component.
  This is not recommended as it can easily lead to degeneracies.

**single_gas_ML**:
  *bool, optional, default = False*

  If set to True the code will assume a single mass to light ratio for all gas components. 

**fixed_stellar_ML**:

  *bool, optional, default = True*

  If set to True the code will assume a fixed mass to light ratio for all stellar components.
  The individual mass to light ratios set in the fitting_parameters section take precedence. 
  
**fixed_gas_ML**:

  *bool, optional, default = True*

  If set to True the code will assume a fixed mass to light ratio for all gas components.


  
**mcmc_steps**:

  *int, optional, default= 2000*

  Number of integration steps per parameter for the emcee fitting.

**burn**: 

  *int, optional, default = 500* 
  
  Number of steps to discard in MCMC chain per   free parameter
    
**numpyro_chains**: 

  *int, optional, default = number of avalaible cpus*
  
  no effect in case of lmfit
    
**use_gp**: 

  *bool, optional, default = True*
  
  use Gaussian Processes as described in https://arxiv.org/abs/2211.06460 or not (False).
  Note that that the test in that article only converges due to the boundaries set and the problem is actually unconstrained due to the disk-halo degeneracy. 
  As such it is not clear to me yet if Gaussian Processes truly change things. Please use this code to further investigate this issue.
  In the code lmfit uses sklearn and numpyro uses tinygp to implement the Gaussian Processes.

**gp_kernel**:

  *str, optional, default = 'RBF'*

  The kernel used in the Gaussian Processes. For now this ahas no effect.
 
**backend**:
  *str, optional, default = 'numpyro'*

  The backend used for the fitting. This can be numpyro or lmfit. Numpyro is the default and is much faster than lmfit.
  
**max_iterations**:

  pyROTMOD will run the fitting unitl the limits of the parameters converge to 5*stdev of the parameter. 
  This ensures that each parameter is well constrained. However, in case of unconstrained fits this it can be that the parameters do not converge
  and the code will run for a long time. If the code has done max_iterations it will assume that the problem diverges.

Fitting Keywords
--------
*Specified with fitting_parameters*


The following specify the initial guesses and limits for the parameters that are fitted in the mass modelling.
The list is build up from five parameters

  1. Initial guess (float)
  2. Lower limit (float). if set to null no lower limit is imposed
  3. Upper limit (float). if set to null no upper limit is imposed
  4. Fit parameter (bool). If this item is false the initial guess is fixed in the fitting.
  5. Include parameter (bool). If this item is set to false the parameter is not included in the final function to be fitted. E.g., if for gas disk this is False the gas disk is not considered in the fitting at all.

For the parameters of the DM they must be included in the definition of the DM Halo and they will always be included into the final function.
These parameters are set dynamically and when not included in the final fitting equation they are ignored in the evealution of the curve (5. is False).
Parameters for the baryonic curves should correspond to their type name with a counter (e.g. DISK_GAS, EXPONENTIAL_1,EXPONENTIAL_2, BULGE_1, HERNQUIST_1)
For the DM halo they should correspond to the parameter being fitted. The code can deal with multiple instance of of optical and gas disk. However, these can quickly become degenerate.
For complex models it is probably better to run the fitting first and then copy the tyaml file from the log directory to get all parameters.

**DISK_GAS_#**

  *List, optional, default = [1.33,null,null,true,true]*

  The mass to light ratio for the gas disk used in the fitting. If the lower limit is unset it is the initial guess divided by 10. If the upper limit is unset is is the intial guess *20.
  Input for multiple gas disks is allowed, the M/L for these disks is by default assumed to be different but can be set to a single parameter.

**EXPONENTIAL_#**:

  *List, optional, default = [1.0,null,null,true,true]*

  The mass to light ratio for the optical exponential disk used in the fitting. If the lower limit is unset it is the initial guess divided by 10. If the upper limit is unset is is the intial guess *20.


**HENRQUIST_#**:

  *List, optional, default = [1.0,null,null,true,true]*

  The mass to light ratio for the optical bulge used in the fitting. If the lower limit is unset it is the initial guess divided by 10. If the upper limit is unset is is the intial guess *20.

**RHO_0**:

  *List, optional, default = [null,null,null,true,true]*

  Used in BURKERT and ISO halo profile. If the initial guess is null a random value between the lower and upper limit is taken. if these are unset they are 1 and 1000,respectively


**R_C**:

  *List, optional, default = [null,null,null,true,true]*

  Used in BURKERT and ISO halo profile. If the initial guess is null a random value between the lower and upper limit is taken. if these are unset they are 1 and 1000,respectively

**C**:

  *List, optional, default = [null,null,null,true,true]*

  Used in NFW halo profile. If the initial guess is null a random value between the lower and upper limit is taken. if these are unset they are 1 and 1000,respectively

**R200**:

  *List, optional, default = [null,null,null,true,true]*

  Used in NFW halo profile. If the initial guess is null a random value between the lower and upper limit is taken. if these are unset they are 1 and 1000,respectively
