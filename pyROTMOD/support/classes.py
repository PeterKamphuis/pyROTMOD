from pyROTMOD.support.errors import InputError,RunTimeError
from pyROTMOD.support.minor_functions import check_quantity,convertskyangle\
      ,get_uncounted
from pyROTMOD.optical.conversions import mag_to_lum
from pyROTMOD.optical.profiles import exponential_luminosity,edge_luminosity,\
      sersic_luminosity,fit_profile,calculate_central_SB,calculate_total_SB,\
      calculate_R_effective,calculate_scale_length,calculate_axis_ratio

import astropy.units as u 
import numpy as np
import copy
import warnings

class Component:
      #These should be set with a unit
      def __init__(self, type = None, name = None, central_SB = None,\
            total_SB = None, R_effective = None, scale_length = None,\
            height_type = None, height = None,\
            height_error = None ,sersic_index = None, central_position = None, \
            axis_ratio = None, PA = None ,background = None, dx = None, dy = None):
            self.name = name
            self.type = type
            self.central_SB = central_SB 
            self.total_SB = total_SB
            self.R_effective = R_effective
            self.scale_length = scale_length
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
                              'type' : None,\
                              'name' : None,
                              'total_SB': u.Msun,\
                              'height': u.kpc,\
                              'height_error': u.kpc,\
                              'central_SB': u.Msun/u.pc**2,\
                              'PA': u.degree,\
                              'R_effective': u.kpc,\
                              'height_type': None,\
                              'sersic_index': None,\
                              'central_position': u.pix,\
                              'axis_ratio': None,\
                              'background': None,\
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
                        elif attr in ['total_SB']:
                              calculate_total_SB(self)
                        elif attr in ['R_effective']:
                              calculate_R_effective(self)
                        elif attr in ['scale_length']:
                              calculate_scale_length(self)
      def print(self):
            for attr, value in self.__dict__.items():
                  print(f' {attr} = {value} \n')  



class Density_Profile(Component):
      allowed_units = ['L_SOLAR/PC^2','M_SOLAR/PC^2','MAG/ARCSEC^2']
      allowed_radii_unit = ['ARCSEC','ARCMIN','DEGREE','KPC','PC']
      #allowed_types = ['EXPONENTIAL','SERSIC','DISK','BULGE','HERNQUIST']
      type_name_translation = {'random_disk': 'DISK', # --> to be converted with cassertano 
                              'random_bulge': 'BULGE', # --> to be converted with cassertano 
                              'expdisk': 'EXPONENTIAL', # --> to be converted with Nagai-Miyamoto
                              'edgedisk': 'EXPONENTIAL', # --> to be converted with Nagai-Miyamoto       
                              'sky': 'SKY', # Not to be converted
                              'sersic': 'SERSIC', #to be converted with NM if n == 0.75 < n 1.25 n, hernquits if 3.75 < n < 4.25, baars and gentile else   
                              'devauc': 'HERNQUIST', # To be converted with hernquist
                              'sersic_disk': 'SERSIC_DISK', #Disk instance of a sersic  n == 0.75 < n <1.25
                              'sersic_bulge': 'SERSIC_BULGE' #Bulge instance of a sersic  n == 3.75 < n <4.25
                              }
      allowed_height_types = ['constant','gaussian','sech_sq','lorentzian','exp','inf_thin']
      def __init__(self, values = None, errors = None, radii = None,
                  type = None, height = None,\
                  height_type = None, band = None, \
                  height_error =None ,name =None, MLratio= None, 
                  distance = None,component = None, central_SB = None,\
                  total_SB = None, R_effective = None,\
                  scale_length = None,\
                  sersic_index = None, central_position = None, \
                  axis_ratio = None, PA = None ,background = None,\
                  dx = None, dy = None ):
            super().__init__(type = type,name = name,\
                  height = height, height_type=height_type,\
                  height_error = height_error, central_SB = central_SB,\
                  total_SB = total_SB, R_effective = R_effective, scale_length = scale_length,\
                  sersic_index = sersic_index, central_position = central_position, \
                  axis_ratio = axis_ratio, PA = PA ,background = background, dx = dx, dy = dx )
            self.values = values
            self.errors = errors  
            self.profile_type = 'density_profile'
            self.radii = radii
            self.component = component # stars, gas or DM
            self.band = band
            self.distance = distance
            self.MLratio= MLratio
          
       
      def print(self):
            for attr, value in self.__dict__.items():
                  print(f' {attr} = {value} \n')
                    
     
                        
                        
                              

                        
                        
           
            
          
      
          
        
            
      def calculate_attr(self,debug = False, log = None, apply =False):
           
            self.check_profile()
           
            components = Component()
            for attr, value in components.__dict__.items():
                  setattr(components,attr, getattr(self,attr))

            succes, components, profile = fit_profile(self.radii,self.values,\
                        name=self.type, debug=debug,log=log, components = components)
          
            if succes is True:
                  for attr, value in components.__dict__.items():
                        setattr(self,attr, value)
                  if apply:

                        self.values = profile 

            elif succes == 'process':
                  for attr, value in components[0].__dict__.items():
                        value = [value, getattr(components[1],attr)]
                        setattr(self,attr, value)
                  if apply:
                        self.type = 'EXP+HERN'
                        self.values = profile 
                  #This means the profiles is best fitted with a bulge + exp profile    
            else:
                  if self.name.split('_')[0] in ['EXPONENTIAL','DENSITY','DISK','SERSIC']:
                        self.type = 'random_disk'
                  else:
                        self.type =  'random_bulge'
           
            self.check_attr()
     
      def check_attr(self):
            
            #The unit_dictionary contain all attr in the Component section
            if not self.radii is None:
                  self.radii = set_requested_units(self.radii,requested_unit=u.kpc,\
                        distance=self.distance)

        
            for attr in self.unit_dictionary:
                  value = getattr(self,attr)
                  if not value is None:
                        value = set_requested_units(value,requested_unit=self.unit_dictionary[attr],\
                                          distance=self.distance,band=self.band,MLratio=self.MLratio)
                        setattr(self,attr, value)
            self.fill_empty()


      def check_component(self):
            self.component = check_profile_component(self.component,self.name)
      
      # Check that all profiles are quantities in the right units                             
      # i.e. M_SOLAR/PC^2 and KPC
      def check_profile(self):
            self.values = check_quantity(self.values)
            self.errors = check_quantity(self.errors)
            self.radii = check_quantity(self.radii)
            self.values = set_requested_units(self.values,requested_unit=u.Msun/u.pc**2,\
                        distance=self.distance,band=self.band,MLratio=self.MLratio)
            self.errors = set_requested_units(self.errors,requested_unit=u.Msun/u.pc**2,\
                        distance=self.distance,band=self.band,MLratio=self.MLratio)
            self.radii = set_requested_units(self.radii,requested_unit=u.kpc,\
                        distance=self.distance)
     
   

    
      def create_profile(self):
            self.check_attr()
            #Changing the units in the profiles important that you only trust values with quantities
            if self.type in ['expdisk']:
                  self.values = exponential_luminosity(self)
            elif self.type in ['edgedisk']:
                  self.values = edge_luminosity(self)
            elif self.type in ['sersic','devauc']:
                  self.values = sersic_luminosity(self)
            if not self.type in ['sky','psf']: 
                  self.check_profile()
     
   
     

class Rotation_Curve:
      allowed_units = ['KM/S', 'M/S'] 
      allowed_radii_unit = ['ARCSEC','ARCMIN','DEGREE','KPC','PC']
      def __init__(self, values = None, errors = None, radii = None, band=None,\
                  type = None, name= None,\
                  distance = None, component= None, fitting_variables = None):
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
            self.matched_radii = None
            self.errors = errors
            self.matched_errors = None
            self.calculated_errors = None
            self.fitting_variables = fitting_variables
            self.curve = None
            self.individual_curve = None
      

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
            


      #Make sure that if we want a single ml all Gamma parameters are set 
      def check_unified(self,single_stars,single_gas):
            found_variables =copy.deepcopy(self.fitting_variables)
            if single_stars and self.component in ['stars']:

                  for variable in   found_variables:
                        if variable in [f'Gamma_disk_{get_uncounted(self.name)[1]}', \
                              f'Gamma_bulge_{get_uncounted(self.name)[1]}']:
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
            self.values = set_requested_units(self.values,requested_unit=u.km/u.s,\
                        distance=self.distance,band=self.band)
            self.errors = set_requested_units(self.errors,requested_unit=u.km/u.s,\
                        distance=self.distance,band=self.band)
          
            self.radii = set_requested_units(self.radii,requested_unit=u.kpc,\
                        distance=self.distance)
      
      def match_radii(self,to_match):
            if self.radii.unit != to_match.radii.unit:
                  raise InputError(f'The units of the radii in {self.name} and {to_match.name} are not the same.')
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

#Has to be here to avoid circular imports
def set_requested_units(value_in,requested_unit = None,distance = None, \
                       band = None, MLratio = None):
      #If value is None skip
      #make sure that we not feed back
      value = copy.deepcopy(value_in) 
      
      if value is None:
            return None
      
      if requested_unit is None:
            try:
                  return value.value
            except AttributeError:
                  return value
      
      #If the unit is already the required one return the value 
      if value.unit == requested_unit:
            return value
     
      # see if we can simply convert
      try: 
            return value.to(requested_unit)
      except:
            pass
      # Now it gets more tricky
      requested = 1.*requested_unit
           
      
      if (value.unit in [u.kpc, u.pc] and \
            requested.unit in [u.arcsec,u.arcmin.u.degree]):
            new_value = convertskyangle(value,distance, quantity =True,\
                              physical=True)
            return new_value.to(requested_unit)
       
      if (value.unit in [u.arcsec,u.arcmin,u.degree] and \
            requested.unit in [u.kpc, u.pc]):
            new_value = convertskyangle(value,distance, quantity =True)  
            return new_value.to(requested_unit)

    #if our input unit is in mag and we wnat to go L_solar
      
      if requested_unit in [u.Lsun, u.Msun] and value.unit == u.mag:
            value = mag_to_lum(value,distance=distance,band=band)
      elif requested_unit in [u.Lsun/u.pc**2, u.Msun/u.pc**2] and \
            value.unit == u.mag/u.arcsec**2:
            value = mag_to_lum(value,band=band)
      
    # If we get here we want to converts Lsun or Lsun/pc**2 to Msun

      
      if (value.unit == u.Lsun and requested_unit == u.Msun) or \
            (value.unit == u.Lsun/u.pc**2 and requested_unit == u.Msun/u.pc**2):
                  new_value = MLratio*value
                  return new_value
      if value.unit != requested.unit:
            raise InputError(f'We do not know how convert {value.unit} to {requested.unit}')
      return value
          


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
            else:
                  sets.append(RC.fitting_variables[variable][0])
                  ranges.append([RC.fitting_variables[variable][0],RC.fitting_variables[variable][0]])
                  collected_variables.append(variable)
                  if RC.fitting_variables[variable][3]:
                        if RC.fitting_variables[variable][1] != None:
                              ranges[-1][0] = RC.fitting_variables[variable][1] 
                        if RC.fitting_variables[variable][2] != None:
                              ranges[-1][1] = RC.fitting_variables[variable][2] 
            
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
   
      if not np.array_equal(RC.numpy_curve['variables'],collected_variables):
            print(f'''We have messed up the collection of variables for the curve {RC.numpy_curve['function'].__name__}
requested variables = {RC.numpy_curve['variables']}
collected variables = {collected_variables}''' )
            raise RunTimeError(f'Ordering Error in variables collection')
      return sets,ranges
            
            