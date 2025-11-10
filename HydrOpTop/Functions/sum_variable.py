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
    self.cell_ids = ids_to_consider
    self.indexes = self.cell_ids
    return
  
    
  ### COST FUNCTION ###
  def evaluate(self, p):
    s = np.sum(self.inputs[self.variable])
    return s
  
  
  ### PARTIAL DERIVATIVES ###
  def d_objective(self, var, p):
    if var == self.variable:
      deriv = np.ones(len(self.inputs[self.variable]),dtype='f8')
      return deriv
    else:
      return 0.


  @classmethod
  def sample_instance(cls):
    # sample cell_ids
    res1 = cls(variable="ABC")
    res1.set_inputs({"ABC":np.random.normal(size=50)})
    return [res1]
