# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
import h5py


class Sum_Liquid_Piezometric_Head:
  """
  Class that define the objective function as the sum of the liquid piezometric head
  with a penalizing power in a given region.
  Primarly intended to be used in saturated mode.
  Take as input:
  - the liquid pizeometric head (numpy array)
  - the cell ids where to compute the head sum (default all)
  - the penalizing power (default 1.)
  """
  def __init__(self, pz_head_array, ids_to_sum = None, penalizing_power = 1):
    #pflotran output file
    self.pz_head = pz_head_array
    #cell ids to compute the objective function
    if ids_to_sum is None:
      self.ids_to_sum = np.arange(1,len(self.pz_head)+1)
    else:
      self.ids_to_sum = region_to_opt
    
    if int(penalizing_power) != penalizing_power:
      print("Penalizing power need to be integer")
      raise(ValueError)
    self.penalizing_power = penalizing_power
    self.gravity = 9.80655 #m2/s
    self.density = 997.16 #kg/m3
    self.reference_pressure = 101325 #Pa
    return
    
  def get_ids_to_optimize(self):
    return self.reg_ids
    
  def debug_mode(self, x):
    self.finite_difference_debug = x
    return
  
  ### COST FUNCTION AND ITS DERIVATIVE
  def evaluate(self):
    """
    Evaluate the cost function
    Return a scalar of dimension [L]
    """
    if self.ids_to_sum is None: 
      return np.sum(self.pz_head**self.penalizing_power)
    else: 
      return np.sum(self.pz_head[self.ids_to_sum-1]**self.penalizing_power)
  
  def d_objective_dh(self, out=None):
    """
    Evaluate the derivative of the cost function according to the head
    If a numpy array is provided, derivative will be copied in this array
    Else create a new numpy array
    Derivative have no dimension [-]
    """
    if out is None:
      out = np.zeros(len(self.pz_head), dtype='f8')
    else:
      out[:] = 0.
    if self.penalizing_power == 1:
      if self.ids_to_sum is None: 
        out[:] = 1.
      else:
        out[self.ids_to_sum-1] = 1.
    else:
      if self.ids_to_sum is None: 
        out[:] = self.penalizing_power * X**(self.penalizing_power-1)
      else:
        out[self.ids_to_sum-1] = 1. * self.penalizing_power * \
                                 self.pz_head **(self.penalizing_power-1)
    return out

  #def d_objective_dk(self): 
  #  return np.zeros(len(X),dtype='f8') #cost function does not depend on perm
    


class Flux_through_surface:
  def __init__(self, surface_file):
    self.surface_file = surface_file
  
  def evaluate(self):
    return
  
  def d_objective_dX(self):
    return
    
  def d_objective_dk(self):
    return

                      
