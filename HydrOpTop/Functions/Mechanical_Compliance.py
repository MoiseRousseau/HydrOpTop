# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
from .Base_Function_class import Base_Function


class Mechanical_Compliance(Base_Function):
  """
  Base function class which implement
  """
  def __init__(self, ids_to_consider="everywhere"):
    super(Mechanical_Compliance, self).__init__()
    self.name = "Mechanical Compliance"
    self.solved_variables_needed = ["DISPLACEMENTS"]
    self.input_variables_needed = ["MECHANICAL_LOAD"]
    self.initialized = False
    
    self.u = None
    self.f = None
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
    self.u = inputs[0]
    self.f = inputs[1]
    return
  
  def get_inputs(self):
    """
    Return the current input values
    """
    return [self.u, self.f]
    
  ### COST FUNCTION ###
  def evaluate(self, p):
    """
    Evaluate the mechanical compliance f^T * u
    """
    return np.sum(self.u*self.f)
  
  
  ### PARTIAL DERIVATIVES ###
  def d_objective_dY(self,p): 
    """
    Derivative according to solved variable (displacement)
    """
    return [self.f]
  
  
  def d_objective_dX(self,p):
    """
    Derivative according to input variable (load)
    """
    return [self.u]
  
                      
