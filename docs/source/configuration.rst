#The Configuration File
=====

Introduction
----

pyROTMOD uses OmegaConf (https://github.com/omry/omegaconf) to handle the input settings. pyROTMOD can be ran with default settings or settings from a yml configuration file or with command line input.

In a run pyROTMOD first checks the defaults, then the configuration yaml and finally the command line input. This mean that if a value is set in all three input methods the one from the command line is used.

The yml file has four main sections individuals keywords, general, galaxy and rotmass that contain several parameters that can adjust how pyROTMOD runs. All these  parameters are described below.

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

  configuration input file

General Keywords
--------
*Specified with general*

**ncpu**:

  *int, optional, default = 6*

  Number CPUs used for threaded parts of pyROTMOD. This is not implemented yet.

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

**RC_File.txt**

  *str, optional, default = 'RCs_For_Fitting.txt'*

  File with all the input RCs in the pyROTMOD format. If RC_Contrustion is enabled this is where the RCs are written to if only fitting is enabled this is where the RCs are obtained from.

RC_Construction Keywords
--------
*Specified with RC_Construction*

**enable**

  *bool, optional, default =True*

  Whether to construct the RC from the density files

**optical_file**: null

  *str, required, no default*

  The file containing the optical light distributions. This can either be a galfit file or a table in a text file. In case of the latter the first column should be 'RADI' and specify the radii of the profile in kpc or arcsec.
  The units of every column should be specified on the second row.  Acceptable units are  'KM/S', 'M/S', 'M_SOLAR/PC^2', 'M_SOLAR/PC^2', 'MAG/ARCSEC^2' where the velocity profiles can not be mixed with luminosity or mass profiles.
  Other columns can be EXPONTIAL_# or BULGE_# where the # indicates an integer number. In case multiple profiles are specified for EXPONENTIAL or BULGE their individual RCs

**gas_file**: null

  *str, required, no default*

  The file containg the total rotation curve and the gas distribution. This can be a tirific .def file or a table in a text file. In case of the latter it should be arranged as the optical file.
  The RADI can be different and the gas disk should be indicated with DISK_G. the observed RC as V_OBS  with V_OBS_ERR as its error.

**distance**: null

  *float, optional, default = vsys*

  Distance to the galaxy. In case the gas_file is a tirific file the default is vsys from that file. In case of a table no default exists.

**exposure_time**:

  *float, optional, default = 1.*

  Exposure time of the optical image. Certain galfit components (edge,sersic) take this into account. For the fit with galfit this should be set in the header of the image.

**mass_to_light_ratio**:

  *float, optional, default = 0.6*

  Mass to light ratio to be used for converting the optical luminosity profiles to mass profiles

**band**: SPITZER3.6

  Band to be used for magnitude to flux/luminosity conversion. This is only used if the input file is in MAG/ARCSEC^2 or when the input file is a galfit file.


Fitting Keywords
--------
*Specified with fitting*

**enable**

  *bool, optional, default =True*

  Run the Bayesian Fitting.

**negative_values**:

  *Bool, optional, default = False*

  Allow for negative values for the parameters.

**HALO**:

   *str, optional, default =NFW'*

   The requested DM halo potential. For now this is NFW, ISO, BURKERT for the NAFW profile, the pseudo isothermal halo profile and a Burkert profile.

**mcmc_steps**:

    *int,optional, default= 2000*

    Number of integration steps per parameter for the emcee fitting.


The following specify the initial guesses and limits for the parameters that are fitted in the mass modelling.
The list is build up from five parameters

  1. Initial guess (float)
  2. Lower limit (float). if set to null no lower limit is imposed
  3. Upper limit (float). if set to null no lower limit is imposed
  4. Fit parameter (bool). If this item is false the initial guess is fixed in the fitting.
  4. Include parameter (bool). If this item is set to false the parameter is not included in the final function to be fitted. E.g., if for MG this is False the gas disk is not added to the gas disk.

For the parameters of  the DM they must be included in the definition of the DM Halo and they will always be included into the final function.

**MG**

  *List, optional, default = [1.4,null,null,true,true]*

  The mass to light ratio for the gas disk used in the fitting. If the lower limit is unset it is the initial guess divided by 10. If the upper limit is unset is is the intial guess *20.

**MD**:

  *List, optional, default = [1.0,null,null,true,true]*

  The mass to light ratio for the optical exponential disk used in the fitting. If the lower limit is unset it is the initial guess divided by 10. If the upper limit is unset is is the intial guess *20.


**MB**:

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
