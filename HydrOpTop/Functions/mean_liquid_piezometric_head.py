# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
from .Base_Function_class import Base_Function

class Mean_Liquid_Piezometric_Head(Base_Function):
  r"""
  The `Mean_Liquid_Piezometric_Head` function compute the mean of the piezometric
  head in the given cell ids:

  Required PFLOTRAN outputs ``LIQUID_PRESSURE`` and ``VOLUME``.
  
  :param ids_to_sum: Cell ids to compute the mean piezometric head
  :type ids_to_sum: iterable
  :param power: the penalizing power `n` above
  :type power: float
  :param gravity: norm of the gravity vector `g`
  :type gravity: float
  :param density: fluid density `\rho`
  :type density: float
  :param ref_pressure: reference pressure in PFLOTRAN simulation
  :type ref_pressure: float   
  """
  def __init__(self, ids_to_sum = "everywhere", penalizing_power = 1,
                     gravity=9.8068, density=997.16, reference_pressure=101325.):
                     
    #objective argument
    if isinstance(ids_to_sum, str) and \
             ids_to_sum.lower() == "everywhere":
      self.ids_to_sum = None
    else:
      self.ids_to_sum = ids_to_sum
    if int(penalizing_power) != penalizing_power:
      print("Penalizing power need to be integer")
      raise(ValueError)
    else:
      self.penalizing_power = penalizing_power
      
    super(Mean_Liquid_Piezometric_Head, self).__init__()

    #inputs for function evaluation 
    self.head = None
    self.volume = None
    
    #required for problem crafting
    self.variables_needed = ["LIQUID_HEAD", "VOLUME"]
    self.name = "Mean PZ Head"
    self.initialized = None
    return
    
  def set_ids_to_sum(self, x):
    self.ids_to_sum = x
    return
  
  def set_penalizing_power(self, x):
    if int(x) != x:
      print("Penalizing power need to be integer")
      raise(ValueError)
    else:
      self.penalizing_power = x
    return
    
  def get_ids_to_sum(self):
    return self.ids_to_sum
    
  def set_inputs(self, inputs):
    self.head = inputs["LIQUID_HEAD"]
    self.volume = inputs["VOLUME"]
    return

  
  ### COST FUNCTION ###
  def evaluate(self, p):
    """
    Evaluate the cost function
    Return a scalar of dimension [L]
    """
    if not self.initialized: self.__initialize__()
    pz_head = self.head * self.volume
    if self.ids_to_sum is None:
      return np.sum(pz_head**self.penalizing_power) / self.V_tot
    else: 
      return np.sum(pz_head[self.ids_to_sum-1]**self.penalizing_power) / self.V_tot
  
  ### PARTIAL DERIVATIVES ###
  def d_objective(self, var, p):
    if var == "LIQUID_HEAD":
      pz_head = self.head * self.volume
      deriv = self.volume
      if self.ids_to_sum is None:
        dobj = self.penalizing_power * deriv * \
	                   pz_head**(self.penalizing_power-1) / self.V_tot
      else:
        dobj = np.zeros(len(self.head), dtype='f8')
        dobj[self.ids_to_sum-1] = (
          self.penalizing_power * deriv * pz_head**(self.penalizing_power-1)
	      )[self.ids_to_sum-1] / self.V_tot
    elif var == "VOLUME":
	    raise NotImplementedError()
    return dobj
  
  def __initialize__(self):
    self.initialized = True
    if self.ids_to_sum is None:
      self.V_tot = np.sum(self.volume)
    else:
      self.V_tot = np.sum(self.volume[self.ids_to_sum-1])
    return
                      
