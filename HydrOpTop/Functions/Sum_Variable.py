# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
from .Base_Function_class import Base_Function


class Sum_Variable(Base_Function):
  """
  Sum a output variable (solved or not)
  argument:
  - variable: the name of the variable to sum
  - solved: is this variable solved by the solver ?
  - ids_to_consider: the ids to sum
  """
  def __init__(self, variable, solved=True, ids_to_consider="everywhere"):
    super(Sum_Variable, self).__init__()
    self.name = "Sum " + variable
    self.solved = solved
    if solved:
      self.solved_variables_needed = [variable]
      self.input_variables_needed = []
    else:
      self.solved_variables_needed = []
      self.input_variables_needed = [variable]
      
    self.initialized = False
    
    self.var = None
    return
  
  def set_inputs(self, inputs):
    """
    Method required by the problem crafter to pass the solver output
    variables to the objective
    Inputs argument have the same size and in the same order given in
    "self.output_variable_needed".
    Note that the inputs will be passed one time only (during the
    initialization), and will after be changed in-place, so that function
    will never be called again...
    """
    self.var = inputs[0]
    return
  
  def get_inputs(self):
    """
    Return the current input values
    """
    return [self.var]
    
  ### COST FUNCTION ###
  def evaluate(self, p):
    """
    Evaluate the mechanical compliance f^T * u
    """
    return np.sum(self.var)
  
  
  ### PARTIAL DERIVATIVES ###
  def d_objective_dY(self,p): 
    """
    Derivative according to solved variable
    """
    if self.solved:
      return [np.ones(len(self.var),dtype='f8')]
    else:
      return [0.]
  
  
  def d_objective_dX(self,p):
    """
    Derivative according to input variable
    """
    if self.solved:
      return [0.]
    else:
      return [np.ones(len(self.var),dtype='f8')]
  
                      
