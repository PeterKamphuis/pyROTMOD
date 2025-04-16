from pyROTMOD.support.errors import InputError,RunTimeError,UnitError
from pyROTMOD.support.minor_functions import check_quantity,convertskyangle\
      ,get_uncounted,isquantity
from pyROTMOD.support.log_functions import print_log
from pyROTMOD.optical.conversions import mag_to_lum
from pyROTMOD.optical.profiles import exponential_luminosity,edge_luminosity,\
      sersic_luminosity,sersic_profile,edge_profile,exponential_profile,\
      hernquist_profile,hernquist_luminosity
from pyROTMOD.optical.profile_functions import fit_profile,\
      truncate_density_profile
from pyROTMOD.optical.calculate_profile_components import calculate_central_SB,\
      calculate_total_mass,calculate_R_effective,calculate_L_effective,\
      calculate_scale_length,calculate_axis_ratio,calculate_hernquist_scale_length,\
      calculate_total_luminosity
import astropy.units as u 
import numpy as np
import copy
import warnings
import inspect

class Component:
      #These should be set with a unit
      def __init__(self, type = None, name = None, central_SB = None,\
            R_effective = None, scale_length = None,\
            height_type = None, height = None,L_effective=None,\
            hernquist_scale_length = None,\
            height_error = None ,sersic_index = None, central_position = None, \
            axis_ratio = None, PA = None ,background = None, dx = None,\
            dy = None, truncation_radius = None, extend = None,
            softening_length =None,total_mass=None,total_luminosity=None):
            self.name = name
            self.type = type
            self.central_SB = central_SB 
            self.total_mass = total_mass
            self.total_luminosity = total_luminosity  
            self.R_effective = R_effective
            self.L_effective = L_effective
            self.extend = extend
            self.scale_length = scale_length
            self.hernquist_scale_length = hernquist_scale_length
            self.truncation_radius = truncation_radius
            self.softening_length = softening_length
            self.height_type = height_type
            self.height = height
            self.height_error = height_error
            self.sersic_index = sersic_index
            self.central_position = central_position
            self.axis_ratio = axis_ratio
            self.PA = PA
            self.background = background
            self.dx = dx
            self.dy = dy
            self.unit_dictionary = {'scale_length': u.kpc,\
                              'hernquist_scale_length': u.kpc,\
                              'total_mass': u.Msun,\
                              'total_luminosity': u.Lsun,
                              'height': u.kpc,\
                              'height_error': u.kpc,\
                              'central_SB': {'sbr_lum':u.Lsun/u.pc**2,\
                                    'sbr_dens': u.Msun/u.pc**2,\
                                    'density':u.Msun/u.pc**3},\
                              'PA': u.degree,\
                              'R_effective': u.kpc,\
                              'L_effective': u.Lsun/u.pc**2,\
                              'extend': u.kpc,\
                              'truncation_radius': u.kpc,\
                              'softening_length': u.kpc,\
                              'central_position': u.pix,\
                              'dx': u.pix,\
                              'dy': u.pix}
                              #Astrpoy does not have suitable units for the background
      def fill_empty(self):
            for attr, value in self.__dict__.items():
                  if value is None:
                        if attr in ['radii','values','unit','errors','radii','radii_unit','component','band','distance','MLratio']:
                              continue
                              #raise InputError(f'We cannot derive {attr} please set it')
                        elif attr in ['axis_ratio']:
                              calculate_axis_ratio(self)    
                        elif attr in ['central_SB']:
                              calculate_central_SB(self)
                        elif attr in ['R_effective']:
                              calculate_R_effective(self)
                        elif attr in ['L_effective']:
                              calculate_L_effective(self)
                        elif attr in ['scale_length']:
                              calculate_scale_length(self)
                        elif attr in ['hernquist_scale_length']:
                              calculate_hernquist_scale_length(self)
                        elif attr in ['total_mass']:
                              calculate_total_mass(self)
                        elif attr in ['total_luminosity']:
                              calculate_total_luminosity(self)
      def print(self):
            print_class(self)
 
class SBR_Profile(Component):
      '''These are in plane the surface brightness profiles
      The Tilted ring Model provides these directly but for fits made to the 
      sky images they first have to be obtained by conversion the difference 
      between a 3d density and 2d surface brightness profile is determined by the units
      type name translation indicates the required units
      '''
      allowed_units = ['L_SOLAR/PC^2','M_SOLAR/PC^2','M_SOLAR/PC^3']
      allowed_radii_unit = ['KPC','PC']
      #allowed_types = ['EXPONENTIAL','SERSIC','DISK','BULGE','HERNQUIST']
      type_name_translation = {'random_disk': ['DISK',u.Msun/u.pc**2], # --> to be converted with cassertano 
                              'random_bulge': ['BULGE',u.Msun/u.pc**2], # --> to be converted with cassertano 
                              'expdisk': ['EXPONENTIAL',u.Msun/u.pc**2], # --> to be converted with Nagai-Miyamoto
                              'edgedisk': ['EXPONENTIAL',u.Msun/u.pc**2], # --> to be converted with Nagai-Miyamoto       
                              'sersic': ['SERSIC',u.Msun/u.pc**2], #to be converted with NM if n == 0.9 < n 1.1 n, hernquist if 3.9 < n < 4.1, and Einosto if else    
                              'devauc': ['HERNQUIST',u.Msun/u.pc**2], # To be converted with hernquist
                              }
      allowed_height_types = ['constant','gaussian','sech_sq','lorentzian','exp','inf_thin']
      def __init__(self, values = None, errors = None, radii = None,
                  type = None, height = None,\
                  height_type = None, band = None, \
                  height_error =None ,name =None, MLratio= None, 
                  distance = None,component = None, central_SB = None,\
                  total_mass = None, total_luminosity = None, R_effective = None,L_effective=None,\
                  scale_length = None, truncation_radius = None,\
                  sersic_index = None, central_position = None, \
                  axis_ratio = None, PA = None ,background = None,\
                  dx = None, dy = None,softening_length =None,
                  hernquist_scale_length = None):
            super().__init__(type = type,name = name,
                  height = height, height_type=height_type,
                  height_error = height_error, central_SB = central_SB,
                  total_mass = total_mass,total_luminosity=total_luminosity,
                  R_effective = R_effective, scale_length = scale_length,
                  sersic_index = sersic_index, central_position = central_position, 
                  axis_ratio = axis_ratio, PA = PA ,background = background, 
                  dx = dx, dy = dy ,truncation_radius=truncation_radius,
                  softening_length = softening_length,hernquist_scale_length=hernquist_scale_length)
            self.values = values
            self.errors = errors  
            #The density profiles can be sbr_dens or density
            self.profile_type = 'sbr_dens'
            self.radii = radii
            self.component = component # stars, gas or DM
            self.band = band
            self.distance = distance
            self.MLratio= MLratio
            self.original_values = None
            for attr in ['values','errors']:
                  self.unit_dictionary[attr] = {'density':u.Msun/u.pc**3,\
                                    'sbr_lum':u.Lsun/u.pc**2,\
                                    'sbr_dens':u.Msun/u.pc**2}
            self.unit_dictionary['radii'] = u.kpc
            self.unit_dictionary['distance'] = u.Mpc
            self.unit_dictionary['MLratio'] = u.Msun/u.Lsun

      
            
    
     
      def check_attr(self):
            
            #The unit_dictionary contain all attr in the Component section
            if not self.radii is None:
                 set_requested_units(self,'radii')

            if self.type is None:
                  set_type(self)
            for attr, value in Component().__dict__.items():
                  value = getattr(self,attr)
                  if not value is None:
                        set_requested_units(self,attr)
            self.fill_empty()
          
            if not self.softening_length is None:
                  if self.softening_length.unit == u.dimensionless_unscaled\
                        and not self.scale_length is None:
                        self.softening_length *= self.scale_length
                  

      def check_component(self):
            self.component = check_profile_component(self.component,self.name)
      
      # Check that all profiles are quantities in the right units                             
      # i.e. M_SOLAR/PC^2 and KPC
      def check_profile(self):
            self.values = check_quantity(self.values)
            self.errors = check_quantity(self.errors)
            self.radii = check_quantity(self.radii)
            for attr in ['radii','values']:
                  set_requested_units(self,attr)
           
      def check_radius(self):
            self.radii = check_radii(self)
           

    
      def create_profile(self):
            self.check_radius()
            self.check_attr()
      
            #Changing the units in the profiles important that you only trust values with quantities
          
            if self.type in ['expdisk']:
                  self.values = exponential_profile(self)
            elif self.type in ['edgedisk']:
                  self.values = edge_profile(self)
                  self.profile_type='density'
            elif self.type in ['sersic','devauc']:
                  self.values = sersic_profile(self)
                  self.profile_type='density'
            elif self.type in ['bulge','hernquist']:
                  self.values = hernquist_profile(self)
                  self.profile_type='density'
            if not self.type in ['sky','psf']: 
                  self.check_profile()
                  truncate_density_profile(self)

class Luminosity_Profile(SBR_Profile):
      '''These are on sky luminosity profiles they have to be converted to density or SBR profiles'''
      allowed_units = ['L_SOLAR/PC^2','MAG/ARCSEC^2','JY/BEAM']
      allowed_radii_unit = ['ARCSEC','ARCMIN','DEGREE','KPC','PC']
      #allowed_types = ['EXPONENTIAL','SERSIC','DISK','BULGE','HERNQUIST']
      type_name_translation = {'random_disk': 'DISK', # --> no conversion has to be fitted
                              'random_bulge': 'BULGE', # -->  no conversion has to be fitted
                              'expdisk': 'EXPONENTIAL', # --> converted with an exponential
                              'edgedisk': 'EXPONENTIAL', # --> converted with an exponential
                              'sersic': 'SERSIC', #to be converted with exponential if n == 0.9 < n 1.1 n, hernquist if 3.9 < n < 4.1, and baas & gentile an else    
                              'devauc': 'HERNQUIST', # To be converted with hernquist
                              'sky': 'SKY', # No conversion
                              'psf': 'PSF' # No conversion
                              }
      def __init__(self, values = None, errors = None, radii = None,
                  type = None, height = None,\
                  height_type = None, band = None, \
                  height_error =None ,name =None, MLratio= None, 
                  distance = None,component = None, central_SB = None,\
                  total_mass = None, total_luminosity= None, R_effective = None,\
                  scale_length = None,L_effective=None,\
                  sersic_index = None, central_position = None, \
                  axis_ratio = None, PA = None ,background = None,\
                  dx = None, dy = None,softening_length =None
                   ,hernquist_scale_length = None):
            super().__init__(type = type,name = name, values = values,\
                  errors = errors, radii = radii, band = band,\
                  height = height, height_type=height_type, component= component,\
                  height_error = height_error, central_SB = central_SB,\
                  total_mass = total_mass, total_luminosity=total_luminosity,
                  R_effective = R_effective,\
                  scale_length = scale_length, distance =distance,\
                  MLratio=MLratio,L_effective=L_effective,\
                  sersic_index = sersic_index, central_position = central_position, \
                  axis_ratio = axis_ratio, PA = PA ,background = background, \
                  dx = dx, dy = dy,softening_length = softening_length,
                  hernquist_scale_length=hernquist_scale_length) 
            self.profile_type = 'sbr_lum'   

     
      def create_profile(self):
            
            self.check_attr()
           
            #Changing the units in the profiles important that you only trust values with quantities
            if self.type in ['expdisk']:
                  self.values = exponential_luminosity(self)
            elif self.type in ['edgedisk']:
                  self.values = edge_luminosity(self)
            elif self.type in ['sersic','devauc']:
                  self.values = sersic_luminosity(self)
            elif self.type in ['hernquist']:
                  self.values = hernquist_luminosity(self)
            if not self.type in ['sky','psf']: 
                  self.check_profile()
                  truncate_density_profile(self)

      def calculate_attr(self,cfg = None): 
            '''Calculate the various attributes from fitting a profile 
            If apply = True we replace values with the fitted profile to
            '''
            #First check that the calues are appropriate
            if self.name is None:
                  if not self.type is None:     
                        self.name = f'{self.type}' 
            print_log(f'First we check the profile.',cfg ,case=['debug_add'])
            self.check_profile()

            #Fit the profile
            print_log(f'Fitting profile {self.name}',cfg ,case=['debug_add'])
            succes = fit_profile(self, cfg=cfg)
            #If the fit is succesful copy the new attributes and 
            # if apply the profil
          
            self.check_attr()

class Rotation_Curve:
      '''Class for storing rotation Curves'''
      allowed_units = ['KM/S', 'M/S'] 
      allowed_radii_unit = ['ARCSEC','ARCMIN','DEGREE','KPC','PC']
      def __init__(self, values = None, errors = None, radii = None, band=None,\
                  type = None, name= None,truncation_radius =[None],\
                  distance = None, component= None, fitting_variables = None,\
                  extend = None,softening_length = None):
            self.type = type
            self.profile_type = 'rotation_curve'
            self.band = band
            self.distance = distance
            self.name = name
            self.component = component # stars, gas or DM or All
            self.values = values
            self.matched_values = None
            self.calculated_values = None
            self.radii = radii
            self.truncation_radius = truncation_radius
            self.softening_length = softening_length
            self.extend = extend
            self.matched_radii = None
            self.errors = errors
            self.matched_errors = None
            self.calculated_errors = None
            self.fitting_variables = fitting_variables
            self.curve = None
            self.individual_curve = None
            self.include = True
            self.unit_dictionary = {'distance': u.Mpc,\
                              'values': u.km/u.s,\
                              'matched_values':  u.km/u.s,\
                              'calculated_values': u.km/u.s,\
                              'radii': u.kpc,\
                              'truncation_radius': u.kpc,\
                              'softening_length': u.kpc,\
                              'extend': u.kpc,\
                              'matched_radii': u.kpc,\
                              'errors': u.km/u.s,\
                              'matched_errors': u.km/u.s,\
                              'calculated_errors': u.km/u.s,\
                            }
      
      def print(self):
            print_class(self)
      def calculate_RC(self):
            #Note that the units are assumed to be correct from the function
            if not self.numpy_curve is None:
                  sets,ranges =  set_variables_and_ranges(self)
                  if sets is None:
                        #if there are Nones in the sets we set the calculated values to None
                        self.calculated_values = None
                        return
                  with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                       
                        self.calculated_values = self.numpy_curve['function']\
                              (*sets)*self.values.unit
                        #if self.values[0] == 0.:
                        #      self.calculated_values[0] = 0.
                  #Note that this is a confidence area, the actual errors would be mean(values-errors[0],errors[1]-values)
                  cf_area =  calculate_confidence_area(\
                        self,ranges = ranges)
                  count = 0.
                  for calc_area in cf_area:
                        if np.array_equal(calc_area,self.calculated_values.value):
                              count += 1
                  if count == 2:
                        cf_area = 0.
                  if np.sum(cf_area) == 0.:
                        self.calculated_errors = None
                  else:
                        self.calculated_errors = cf_area*self.values.unit

            else:
                  raise InputError(f'We can not calculate a curve for {self.name} if the function is not defined')



      def check_component(self):
            self.component = check_profile_component(self.component,self.name)
            
      def check_radius(self):
            self.radii = check_radii(self)

      #Make sure that if we want a single ml all Gamma parameters are set 
      def check_unified(self,single_stars,single_gas):
            found_variables =copy.deepcopy(self.fitting_variables)
            if single_stars and self.component in ['stars']:

                  for variable in   found_variables:
                        if variable in [f'Gamma_disk_{get_uncounted(self.name)[1]}', \
                              f'Gamma_bulge_{get_uncounted(self.name)[1]}',
                              f'Gamma_sersic_{get_uncounted(self.name)[1]}']:
                              self.fitting_variables['ML_stellar'] =\
                                    self.fitting_variables[variable]
                              del self.fitting_variables[variable]
                      
            if single_gas and  self.component in ['gas']:
                  for variable in   found_variables:
                        if variable == f'Gamma_gas_{get_uncounted(self.name)[1]}':
                              self.fitting_variables['ML_gas'] = self.fitting_variables[variable]
                              del self.fitting_variables[variable]

     
      def check_profile(self):
            self.values = check_quantity(self.values)
            self.errors = check_quantity(self.errors)
            self.radii = check_quantity(self.radii)
            for attr in ['radii','values','radii']:
                  set_requested_units(self,attr)
            
      
      def match_radii(self,to_match):
            if self.radii.unit != to_match.radii.unit:
                  raise InputError(f'The units of the radii in {self.name} and {to_match.name} are not the same in profile {self.name}.')
            if not self.errors is None:
                  self.matched_errors = np.interp(to_match.radii.value,\
                        self.radii.value,self.errors.value)*self.errors.unit
            else:
                  self.matched_errors = None
            self.matched_radii= to_match.radii
            if not self.values is None:
                  self.matched_values = np.interp(to_match.radii.value,\
                        self.radii.value,self.values.value)*self.values.unit
            else:
                  self.matched_values = None
            


'''Determine the component of the profile. It seems this is open to interpretation best to set it when invoking'''
def check_profile_component(component,name):
      if component is None:
            if not name is None:
                  if get_uncounted(name)[0] in\
                              ['EXPONENTIAL','BULGE','SERSIC','DISK','HERNQUIST']:
                        component = 'stars'
                  elif get_uncounted(name)[0] in ['DISK_GAS']:
                        component = 'gas'
                  elif get_uncounted(name)[0] in ['V_OBS','VROT']:
                        component = 'All'
                  else:
                        component = 'DM'
      return component


# check that the radii extend far enough
def check_radii(self):
      un = self.radii.unit
      tmp_radii = list(self.radii.value)
     
      if not self.extend is None:
            max_rad = self.extend
      elif not self.truncation_radius is None:
            max_rad = self.truncation_radius+self.softening_length
      else:
            max_rad = self.radii[-1]
      if max_rad.unit != self.radii.unit:
            raise UnitError(f'The unit in the radii ({self.radii.unit}) does not match the extend we compare to ({max_rad.unit}) in profile {self.name}')      

    #And that it covers the full length that we require
      if self.radii[-1] < max_rad:  
            ring_width = tmp_radii[-1]-tmp_radii[-2]
            while tmp_radii[-1]*un < max_rad:
                  tmp_radii.append(tmp_radii[-1]+ring_width)
      return np.array(tmp_radii,dtype=float)*un
#print all attributes of class
def print_class(self):
      for attr, value in self.__dict__.items():
            print(f' {attr} = {value} \n')  
#Has to be here to avoid circular imports
def set_requested_units(self,req_attr):
      #If value is None skip
      #make sure that we not feed back
    
      value = getattr(self,req_attr) 
     
      try:
            requested_unit = self.unit_dictionary[req_attr]
      except KeyError:
            requested_unit = None
     
      if isinstance(requested_unit,dict):
            requested_unit = requested_unit[self.profile_type]
   
      #if the value is none or the reuested unit is unknown we leave
      if requested_unit is None or\
            value is None:
            return
      if not isquantity(value):
            raise UnitError(f'{req_attr} in {self.name} has no unit, please assign a unit when filling it (value = {value})')
  
      new_value = None
    
      while new_value is None:      
            #If the unit is already the required one return the value 
            if value.unit == requested_unit:
                  new_value = value
                  break
            # see if we can simply convert
            try: 
                  new_value = value.to(requested_unit)
                  break
            except:
                  pass
            # Now it gets more tricky
            # if we have a dimensionless unit we can convert it to the requested unit
           
            requested = 1.*requested_unit
            if req_attr == 'softening_length' and value.unit == u.dimensionless_unscaled:
                  if not self.scale_length is None:
                        new_value = self.softening_length *self.scale_length
                  else:
                        new_value = value
                        requested_unit = u.dimensionless_unscaled
                  break
      
            if (value.unit in [u.kpc, u.pc] and \
                  requested.unit in [u.arcsec,u.arcmin.u.degree]):
                  new_value = convertskyangle(value,self.distance, quantity =True,\
                              physical=True).to(requested_unit)
                  break
           
       
            if (value.unit in [u.arcsec,u.arcmin,u.degree] and \
                  requested.unit in [u.kpc, u.pc]):
                  new_value = convertskyangle(value,self.distance, \
                        quantity =True).to(requested_unit)
                  break

            #if our input unit is in mag and we wnat to go L_solar
            #These we do not want to break as it is possible to convert mag to Msun
            if requested_unit in [u.Lsun, u.Msun] and value.unit == u.mag:
                  new_value = mag_to_lum(value,distance=self.distance,\
                        band=self.band)
            elif requested_unit in [u.Lsun/u.pc**2, u.Msun/u.pc**2] and \
                  value.unit == u.mag/u.arcsec**2:
                  new_value = mag_to_lum(value,band=self.band)
            if not new_value is None:
                  value = new_value
            # If we get here we want to converts Lsun or Lsun/pc**2 to Msun
            if (value.unit == u.Lsun and requested_unit == u.Msun) or \
                  (value.unit == u.Lsun/u.pc**2 and requested_unit == u.Msun/u.pc**2) or \
                  (value.unit == u.Lsun/u.pc**3 and requested_unit == u.Msun/u.pc**3):
                  new_value = self.MLratio*value
                  break

            new_value = 1.*requested_unit**3
               
      if new_value.unit != requested_unit:
            raise InputError(f'For {req_attr} in {self.name} we do not know how convert {value.unit} to {requested.unit}')
      else:
            setattr(self,req_attr,new_value) 

      
def set_type(profile):
      bare_name = get_uncounted(profile.name)[0]                
      for name in profile.type_name_translation:
            if bare_name.upper() == profile.type_name_translation[name][0]:
                  profile.type = name
                  return
      profile.type = 'random'
      

def calculate_confidence_area(RC, ranges = [None]): 
    if any(x is None for x in ranges):
        return [np.zeros(len(RC.radii)),np.zeros(len(RC.radii))]
    if RC.numpy_curve == None:
        raise RunTimeError(f'We can not calculate a curve without a function ({RC.name} is lacking)')
    all_possible_curve = []
    all_sets=create_all_combinations(ranges)
    for set_in in all_sets:
            set_in = [np.array(x,dtype=float) for x in set_in]
            with warnings.catch_warnings():
                  warnings.simplefilter("ignore")

               
                  curve_calc = RC.numpy_curve['function'](*set_in)
                  #if np.isnan(curve_calc[0]):
                  #      curve_calc[0] = 0.
            all_possible_curve.append(curve_calc)
    all_possible_curve=np.array(all_possible_curve,dtype=float)
   
    minim = np.argmin(all_possible_curve,axis=0)
    maxim =  np.argmax(all_possible_curve,axis=0)
    #print(all_possible_curve)
    confidence_area = np.array([[all_possible_curve[loc,i] for i,loc in enumerate(minim) ],\
                       [all_possible_curve[loc,i] for i,loc in enumerate(maxim)]],dtype=float)
    
    return confidence_area

calculate_confidence_area.__doc__ =f'''
 NAME:
    calculate_confidence_area

 PURPOSE:
    calculate the minimum and maximum possible for every point based on the 1 sigma errors in fit

 CATEGORY:
    rotmass

 INPUTS:

 OPTIONAL INPUTS:

 OUTPUTS:

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
'''
def create_all_combinations(ranges):
    '''As mesh grid tries to grid all the arrays they need to be replaced and substituded back in'''
    all_sets =[]   
    replace_dict = {} 
    for i in range(len(ranges)):
        try:
            if len(ranges[i][0]) > 1:
                replace_dict[f'z{i}'] = ranges[i][0]
                replace_dict[f'o{i}'] = ranges[i][1]
                ranges[i][0] = f'z{i}'
                ranges[i][1] = f'o{i}'
        except:
            pass
    all_sets = [list(x) for x in np.array(np.meshgrid(*ranges)).T.reshape(-1,len(ranges))]
    for i in range(len(all_sets)):
        for j in range(len(all_sets[i])):
            if all_sets[i][j] in  replace_dict:
                all_sets[i][j] = replace_dict[all_sets[i][j]]
    return all_sets

def set_variables_and_ranges(RC):
      ranges = []
      sets = []    
      collected_variables = []
      req_var = list(copy.deepcopy(RC.numpy_curve['variables']))
      for variable in RC.numpy_curve['variables']:
            if variable == 'r':
                  collected_variables.append('r')  
                  sets.append(RC.radii.value)
                  ranges.append([RC.radii.value, RC.radii.value])
            elif variable.split('_')[0] == 'V':
                  collected_variables.append(variable)
                  sets.append(RC.values.value)
                  if not RC.errors is None:
                        err = RC.errors.to(RC.values.unit).value
                  else:
                        err =0.
                  ranges.append([(RC.values.value-err), RC.values.value+err])
            elif variable not in [str(x) for x in 
                  inspect.signature(RC.numpy_curve['function']).parameters]: 
                  if variable.lower() in ['amplitude','lgamplitude','length_scale']:
                        req_var.remove(variable)
                  else:
                        print(f'Variable |{variable}| is not in the function signature')
                        print([f'|{str(x)}|' for x in 
                              inspect.signature(RC.numpy_curve['function']).parameters])       
            else:
                  sets.append(RC.fitting_variables[variable].value)
                  collected_variables.append(variable)
                  range_isbadname = [RC.fitting_variables[variable].value,RC.fitting_variables[variable].value]
                  if RC.fitting_variables[variable].variable:
                        if not RC.fitting_variables[variable].stddev is None:
                              #Too accomadate non ngeative we make sure the lower range is always >= min
                              if (RC.fitting_variables[variable].value-
                                    RC.fitting_variables[variable].stddev < 
                                    RC.fitting_variables[variable].min):
                                    range_isbadname[0] = RC.fitting_variables[variable].min
                              else:
                                    range_isbadname[0] = RC.fitting_variables[variable].value-\
                                          RC.fitting_variables[variable].stddev
                              if (RC.fitting_variables[variable].value+
                                    RC.fitting_variables[variable].stddev > 
                                    RC.fitting_variables[variable].max):
                                    range_isbadname[1] = RC.fitting_variables[variable].max
                              else:
                                    range_isbadname[1] = RC.fitting_variables[variable].value+\
                                          RC.fitting_variables[variable].stddev
                        else:
                              if not RC.fitting_variables[variable].min is None:
                                    range_isbadname[0] = RC.fitting_variables[variable].min
                              if not RC.fitting_variables[variable].max is None:
                                    range_isbadname[1] = RC.fitting_variables[variable].max
                  ranges.append(range_isbadname)              
                
            
      for x in sets:
            if x is None:
                  sets = None

    
            '''
            match_variable = variable
            fits_variable = None
            if variable == 'ML':
                match_variable = ML_Variable
            for x in function_variable_settings:
                if function_variable_settings[x]['Variables'][0] == match_variable:
                    fits_variable = x
            if fits_variable is not None:  
                collected_variables.append(variable)
                sets.append(function_variable_settings[fits_variable]['Settings'][0])
                ranges.append([function_variable_settings[fits_variable]['Settings'][0],\
                                  function_variable_settings[fits_variable]['Settings'][0] ])
                if function_variable_settings[fits_variable]['Settings'][3]:
                    if function_variable_settings[fits_variable]['Settings'][1] != None:
                        ranges[-1][0] = function_variable_settings[fits_variable]['Settings'][1] 
                    if function_variable_settings[fits_variable]['Settings'][2] != None:
                        ranges[-1][1] = function_variable_settings[fits_variable]['Settings'][2] 
            '''
   
      if not np.array_equal(req_var,collected_variables):
            print(f'''We have messed up the collection of variables for the curve {RC.numpy_curve['function'].__name__}
requested variables = {req_var}
collected variables = {collected_variables}''' )
            raise RunTimeError(f'Ordering Error in variables collection')
      return sets,ranges

def copy_attributes(source, target, exclude=None):
    """
    Copy all attributes from the source object to the target object.

    Args:
        source: The object to copy attributes from.
        target: The object to copy attributes to.
        exclude: A list of attribute names to exclude from copying.
    """
    if exclude is None:
        exclude = []
    for attr, value in vars(source).items():
        if attr not in exclude:
            setattr(target, attr, value)


