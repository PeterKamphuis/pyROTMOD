import numpy as np
import copy
class Parameter:
      def __init__(self, name = None, value = None, stddev = None, unit = None,
            variable = False, include = True, log =None,
            fixed_boundaries = [False, False], original_boundaries = [None, None],
            original_value = None):
            self.name = name
            self.value = value
            self.stddev = stddev
            self.unit = unit
            self.log = log
            self.boundaries = [None, None] #min and max
            self.previous_boundaries = [None, None]
            self.original_boundaries = original_boundaries
            self.original_value = original_value
            self.fit_stats = None
            self.stat_use = 'Mean' #Trigger to use a the Mean, Median or Max of the occurence max occurence  instead of the mean
            self.fit_direction = [None, None]  #Following stage of the fit 'intitial_guess', diverging, 'stable' [value,boundaries] set in check_boundaries
            self.fixed_boundaries = fixed_boundaries #Boolean to indicate if the parameter has fixed boundaries [min,max]
            self.variable = variable
            self.include = include
            if name is not None:
                  if name[0:2] == 'lg':      
                        self.log = True
                  else:
                        self.log = False
            else:
                  self.log = None
      def print(self):
            for attr, value in self.__dict__.items():
                  print(f' {attr} = {value} \n')  
      def fill_empty(self):
          
            #If the value is None we set it to a random number between min and max
            #if the min and max are None we set them to 0.1 and 1000.
            if self.log is None:
                  if self.name[0:2] == 'lg':      
                        self.log = True
                  else:
                        self.log = False

            for i in [0,1]:
                  if not self.original_boundaries[i] is None and self.boundaries[i] is None:
                        #existing stuff is made log in a different function   
                        self.boundaries[i] = copy.deepcopy(self.original_boundaries[i])
                  elif self.boundaries[i] is None:
                        if self.value is not None and self.value != 0.:
                              if self.log:
                                    self.boundaries[i] = self.value+(i*2-1)*np.log10(5.)
                              else:      
                                    self.boundaries[i] = self.value*5.**(i*2-1)
                        
                        else: 
                              if self.log:
                                    self.boundaries[i] = (i*6.-3.)
                              else:      
                                    self.boundaries[i] = 10**(i*6.-3.)
           
            if self.boundaries[0] == self.boundaries[1]:
                  self.boundaries[0] = self.boundaries[0]*0.9
                  self.boundaries[1] = self.boundaries[1]*1.1
                 
            if self.stddev is None:
                  self.stddev = (self.boundaries[1]-self.boundaries[0])/5.
            if self.value is None:
                  #no_input = True
                  self.value = float(np.random.rand()*\
                        (self.boundaries[1]-self.boundaries[0])+self.boundaries[0])
           
def set_parameter_from_cfg(var_name,var_settings,adapt_boundaries = True):
      # Using a dictionary make the parameter always to be added
      if adapt_boundaries:
            fixed_bounds = [False,False]
      else:
            fixed_bounds = [True,True]
      original_boundaries = [None, None]
      if not var_settings[1] is None:
            original_boundaries[0] = var_settings[1]
      else:
            #if the boundary is unset we have to adapt regardless of adapt_boundaries
            fixed_bounds[0] = False  
      if not var_settings[2] is None:
            original_boundaries[1] = var_settings[2]
      else:
            #if the boundary is unset we have to adapt regardless of adapt_boundaries
            fixed_bounds[1] = False  
    
      return Parameter(name=var_name,
            value = var_settings[0],
            original_value = var_settings[0],
            original_boundaries = original_boundaries,
            stddev = None,
            variable=var_settings[3],
            include=var_settings[4],
            fixed_boundaries=fixed_bounds)