import numpy as np

class Parameter:
      def __init__(self, name = None, value = None, stddev = None, unit = None,
            min = None, max = None, variable = False, include = True,fixed_boundaries = False):
            self.name = name
            self.value = value
            self.stddev = stddev
            self.unit = unit
            self.min = min
            self.max = max
            self.fixed_boundaries = fixed_boundaries
            self.variable = variable
            self.include = include
      def print(self):
            for attr, value in self.__dict__.items():
                  print(f' {attr} = {value} \n')  
      def fill_empty(self):
            #If the value is None we set it to a random number between min and max
            #if the min and max are None we set them to 0.1 and 1000.
            if self.min is None:
                  if self.value is not None and self.value != 0.:
                        self.min = self.value/5.
                  else:      
                        self.min = 0.1
            if self.max is None:
                  if self.value is not None and self.value != 0.:
                        self.max = self.value*5.
                  else:
                        self.max = 1000.
            if self.min == self.max:
                  self.min = self.min*0.9
                  self.max = self.max*0.9
           
            if self.stddev is None:
                  self.stddev = (self.max-self.min)/5.
            if self.value is None:
                  #no_input = True
                  self.value = float(np.random.rand()*\
                        (self.max-self.min)+self.min)
def set_parameter_from_cfg(var_name,var_settings):
      # Using a dictionary make the parameter always to be added
      if not var_settings[1] is None and not var_settings[2] is None:
            fixed_bounds = True
      else:
            fixed_bounds = False
      return Parameter(name=var_name,
            value = var_settings[0],
            stddev = None,
            min=var_settings[1],
            max=var_settings[2],
            variable=var_settings[3],
            include=var_settings[4],
            fixed_boundaries=fixed_bounds)