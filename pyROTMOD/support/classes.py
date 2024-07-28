from pyROTMOD.support.errors import InputError
from pyROTMOD.support.minor_functions import check_quantity,convertskyangle
from pyROTMOD.optical.conversions import mag_to_lum
from pyROTMOD.optical.profiles import exponential_luminosity,edge_luminosity,\
      sersic_luminosity,fit_profile

import astropy.units as u 
import numpy as np
import copy

class Component:
      def __init__(self, type = None, name = None):
            self.name = name
            self.type = type
            self.central_SB = None 
            self.total_SB = None 
            self.R_effective = None 
            self.scale_height = None
            self.scale_length = None
            self.sersic_index = None
            self.central_position = None
            self.axis_ratio = None
            self.PA = None
            self.background = None
            self.dx = None
            self.dy = None
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
                  height_unit = None, height_type = None, band = None, \
                  height_error =None ,name =None, MLratio= None, 
                  distance = None ):
            super().__init__(type = type,name = name)
            self.values = values
            self.radii = radii
            self.errors = errors  
            self.unit = unit  
            self.band = band
            self.distance = distance
            self.MLratio= MLratio
            self.radii_unit = radii_unit
            self.height_unit = height_unit
            self.height_type = height_type 
            self.height = height 
            self.height_error = height_error
      def print(self):
            for attr, value in self.__dict__.items():
                  print(f' {attr} = {value} \n')  

      def fill_empty(self):
            for attr, value in self.__dict__.items():
                  if value is None:
                        if attr in ['radii']:
                              raise InputError(f'We cannot derive {attr} please set it')
                        elif attr in ['values']:
                              raise InputError(f' To create the profile we need the distance and band please use create_profile')
                        elif attr in ['axis_ratio']:
                              if not self.scale_height is None and \
                                    not self.scale_length is None:
                                    self.axis_ratio = self.scale_height/self.scale_length
                              else:
                                   print(f'We cannot derive an axis ratio as we have no scale length or scale height')
      # Check that all profiles are quantities in the right units                             
      # i.e. M_SOLAR/PC^2 and KPC
      def check_profile(self):
            self.fill_empty()
            self.values = check_quantity(self.values,unit =self.unit)
            self.errors = check_quantity(self.errors,unit =self.unit)
            self.radii = check_quantity(self.radii,unit =self.radii_unit)
            self.values,self.unit = set_requested_units(self.values,requested_unit=u.Msun/u.pc**2,\
                        distance=self.distance,band=self.band,MLratio=self.MLratio)
            self.errors, tmp = set_requested_units(self.errors,requested_unit=u.Msun/u.pc**2,\
                        distance=self.distance,band=self.band,MLratio=self.MLratio)
            self.radii, self.radii_unit = set_requested_units(self.radii,requested_unit=u.kpc,\
                        distance=self.distance)
          

            
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

   


      def create_profile(self):
            #Changing the units in the profiles important that you only trust values with quantities
            if self.type in ['expdisk']:
                  self.values = exponential_luminosity(self,band=self.band,distance=self.distance)
            elif self.type in ['edgedisk']:
                  self.values = edge_luminosity(self,band=self.band, distance= self.distance)
            elif self.type in ['sersic','devauc']:
                  self.values = sersic_luminosity(self,band = self.band,distance= self.distance)
            if not self.type in ['sky','psf']: 
           
                  self.check_profile()
     
   
     

class Rotation_Curve:
      allowed_units = ['KM/S', 'M/S'] 
      allowed_radii_unit = ['ARCSEC','ARCMIN','DEGREE','KPC','PC']
      def __init__(self, values = None, errors = None, radii = None, band=None,\
                  unit = None, radii_unit = None,type = None, name= None,\
                  distance = None):
            self.type = type
            self.band = band
            self.distance = distance
            self.name = name
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
            self.values = check_quantity(self.values,unit =self.unit)
            self.errors = check_quantity(self.errors,unit =self.unit)
            self.radii = check_quantity(self.radii,unit =self.unit)
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
          
 
            
            