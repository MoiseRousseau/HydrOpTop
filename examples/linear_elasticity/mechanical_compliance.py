# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
from HydrOpTop.Functions import Base_Function


class Mechanical_Compliance(Base_Function):
  r"""
  Description:
    Calculate the mechanical compliance defined as:
    
    .. math::
       
       f = F^T \cdot u
    
    Where :math:`F^T` is the transpose of the load vector :math:`F` and :math:`u` the displacement vector
    
  Parameters:
    No parameters. Compliance is calculation considering the whole computational domain
  
  Output variable needed:
    ``DISPLACEMENTS`` and ``MECHANICAL_LOAD``.
  
  """
  def __init__(self, ids_to_consider=None):
    super(Mechanical_Compliance, self).__init__()
    self.indexes = ids_to_consider
    self.name = "Mechanical Compliance"
    self.variables_needed = ["DISPLACEMENTS", "MECHANICAL_LOAD"]
    self.inputs = {}
    return
  
    
  ### COST FUNCTION ###
  def evaluate(self, p):
    """
    Evaluate the mechanical compliance f^T * u
    """
    u = self.inputs["DISPLACEMENTS"]
    f = self.inputs["MECHANICAL_LOAD"]
    return np.sum(u*f)
  
  
  ### PARTIAL DERIVATIVES ###
  def d_objective(self, var, p): 
    """
    Derivative according to solved variable (displacement)
    """
    if var == "MECHANICAL_LOAD":
      return self.inputs["DISPLACEMENTS"]
    if var == "DISPLACEMENTS":
      return self.inputs["MECHANICAL_LOAD"]
    else:
      return 0.
                      
