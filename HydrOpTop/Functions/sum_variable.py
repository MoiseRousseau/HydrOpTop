# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
from .Base_Function_class import Base_Function


class Sum_Variable(Base_Function):
  """
  Sum a output variable
  argument:
  - variable: the name of the variable to sum
  - ids_to_consider: the ids to sum
  """
  def __init__(self, variable, ids_to_consider=None):
    super(Sum_Variable, self).__init__()
    self.name = "Sum " + variable
    self.variable = variable
    self.variables_needed = [variable]
    if ids_to_consider is None:
      self.ids_to_sum = None
    else:
      self.ids_to_sum = np.array(ids_to_consider)-1
    return
  
    
  ### COST FUNCTION ###
  def evaluate(self, p):
    if self.ids_to_sum is None:
      s = np.sum(self.inputs[self.variable])
    else:
      s = np.sum(self.inputs[self.variable][self.ids_to_sum])
    return s
  
  
  ### PARTIAL DERIVATIVES ###
  def d_objective(self, var, p):
    if var == self.variable:
      if self.ids_to_sum is not None: 
        deriv = np.zeros(len(self.inputs[self.variable]),dtype='f8')
        deriv[self.ids_to_sum] = 1.
      else:
        deriv = np.ones(len(self.inputs[self.variable]),dtype='f8')
      return deriv
    else:
      return 0.
                      
