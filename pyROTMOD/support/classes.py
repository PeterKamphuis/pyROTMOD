from pyROTMOD.support.errors import InputError
from pyROTMOD.support.minor_functions import check_quantity,convertskyangle
from pyROTMOD.optical.conversions import mag_to_lum
from pyROTMOD.optical.profiles import exponential_luminosity,edge_luminosity,\
      sersic_luminosity,fit_profile,calculate_central_SB

import astropy.units as u 
import numpy as np
import copy

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
                              'devauc': 'HERNQUIST' # To be converted with hernquist
                              }
      allowed_height_types = ['constant','gaussian','sech_sq','lorentzian','exp','inf_thin']
      def __init__(self, values = None, errors = None, radii = None,
                  unit = None, radii_unit = None, type = None, height = None,\
                  height_type = None, band = None, \
                  height_error =None ,name =None, MLratio= None, 
                  distance = None,component = None ):
            super().__init__(type = type,name = name,\
                  height = height, height_type=height_type,\
                  height_error = height_error )
            self.values = values
            self.errors = errors  
            self.radii = radii
            self.component = component # stars, gas or DM
            self.band = band
            self.distance = distance
            self.MLratio= MLratio
          
       
      def print(self):
            for attr, value in self.__dict__.items():
                  print(f' {attr} = {value} \n')  
      def fill_empty(self):
            for attr, value in self.__dict__.items():
                  if value is None:
                        if attr in ['radii','values','unit','errors','radii','radii_unit','component','band','distance','MLratio']:
                              continue
                              #raise InputError(f'We cannot derive {attr} please set it')
                        elif attr in ['axis_ratio']:
                              if not self.height in [None,0.] and \
                                    not self.scale_length is None:
                                    self.axis_ratio = self.height/self.scale_length
                        elif attr in ['central_SB']:
                              self.central_SB = calculate_central_SB(self)
                              

                        
                        
      # Check that all profiles are quantities in the right units                             
      # i.e. M_SOLAR/PC^2 and KPC
      def check_profile(self):
            self.fill_empty()
            self.values = check_quantity(self.values)
            self.errors = check_quantity(self.errors)
            self.radii = check_quantity(self.radii)
            self.values,self.unit = set_requested_units(self.values,requested_unit=u.Msun/u.pc**2,\
                        distance=self.distance,band=self.band,MLratio=self.MLratio)
            self.errors, tmp = set_requested_units(self.errors,requested_unit=u.Msun/u.pc**2,\
                        distance=self.distance,band=self.band,MLratio=self.MLratio)
            self.radii, self.radii_unit = set_requested_units(self.radii,requested_unit=u.kpc,\
                        distance=self.distance)
          
       # Check that all profiles are quantities in the right units                             
      # i.e. M_SOLAR/PC^2 and KPC
      def check_components(self):
            
            #The unit_dictionary contain all attr in the Component section
            if not self.radii is None:
                  self.radii, self.radii_unit = set_requested_units(self.radii,requested_unit=u.kpc,\
                        distance=self.distance)

        
            for attr in self.unit_dictionary:
                  value = getattr(self,attr)
                  if not value is None:
                        value,unit = set_requested_units(value,requested_unit=self.unit_dictionary[attr],\
                                          distance=self.distance,band=self.band,MLratio=self.MLratio)
                        setattr(self,attr, value)
            self.fill_empty()
          
        
            
      def calculate_components(self,debug = False, log = None):
            self.check_profile()
            components = Component()
            succes, components, profile = fit_profile(self.radii,self.values,\
                        function=self.type, debug=debug,log=log, output = components)
            if succes is True:
                  for attr, value in components.__dict__.items():
                        setattr(self,attr, value)
                  self.values = profile 

            elif succes == 'process':
                  for attr, value in components[0].__dict__.items():
                        value = [value, getattr(components[1],attr)]
                        setattr(self,attr, value)
                  self.name = 'EXP+HERN'
                  self.values = profile 
                  #This means the profiles is best fitted with a bulge + exp profile    
            else:
                  if self.name.split('_')[0] in ['EXPONENTIAL','DENSITY','DISK','SERSIC']:
                        self.type = 'random_disk'
                  else:
                        self.type =  'random_bulge'
            self.check_components()

   


      def create_profile(self):
            self.check_components()
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
                  unit = None, radii_unit = None,type = None, name= None,\
                  distance = None, component= None):
            self.type = type
            self.band = band
            self.distance = distance
            self.name = name
            self.component = component # stars, gas or DM or All
            self.values = values
            self.radii = radii
            self.errors = errors  
            self.unit = unit  
            self.radii_unit = radii_unit 

      
      def fill_empty(self):
            
           
            if self.type is None:
                  if not self.name is None:
                        if self.name.split('_')[0] in ['EXPONENTIAL','DENSITY','DISK','SERSIC']:
                              self.type = 'random_disk'
                        else:
                              self.type =  'random_bulge'
                  else:
                        print(f'Without a name we cannot determine the type')      
     
      def check_profile(self):
            self.fill_empty()
            self.values = check_quantity(self.values)
            self.errors = check_quantity(self.errors)
            self.radii = check_quantity(self.radii)
            self.values, self.unit = set_requested_units(self.values,requested_unit=u.km/u.s,\
                        distance=self.distance,band=self.band)
            self.errors, tmp = set_requested_units(self.errors,requested_unit=u.km/u.s,\
                        distance=self.distance,band=self.band)
          
            self.radii, self.radii_unit = set_requested_units(self.radii,requested_unit=u.kpc,\
                        distance=self.distance)
           
            



#Has to be here to avoid circular imports
def set_requested_units(value_in,requested_unit = None,distance = None, \
                       band = None, MLratio = None):
      #If value is None skip
      #make sure that we not feed back
      value = copy.deepcopy(value_in) 
      
      if value is None:
            return None,requested_unit
      
      if requested_unit is None:
            try:
                  return value.value,None
            except AttributeError:
                  return value,None
      
      #If the unit is already the required one return the value 
      if value.unit == requested_unit:
            return value, requested_unit
      # see if we can simply convert
      try: 
            return value.to(requested_unit), requested_unit
      except:
            pass
      # Now it gets more tricky
      requested = 1.*requested_unit
           
      
      if (value.unit in [u.kpc, u.pc] and \
            requested.unit in [u.arcsec,u.arcmin.u.degree]):
            new_value = convertskyangle(value,distance, quantity =True,\
                              physical=True)
            return new_value.to(requested_unit), requested_unit
       
      if (value.unit in [u.arcsec,u.arcmin,u.degree] and \
            requested.unit in [u.kpc, u.pc]):
            new_value = convertskyangle(value,distance, quantity =True)  
            return new_value.to(requested_unit), requested_unit
      

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
                  return new_value, requested_unit
      if value.unit != requested.unit:
            raise InputError(f'We do not know how convert {value.unit} to {requested.unit}')
      return value, requested_unit
          
 
            
            