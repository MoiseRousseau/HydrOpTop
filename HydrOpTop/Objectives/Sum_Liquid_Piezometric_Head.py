# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
import h5py


class Sum_Liquid_Piezometric_Head:
  """
  Class that define the objective function as the sum of the liquid piezometric head
  with a penalizing power in a given region.
  Primarly intended to be used in saturated mode. Assumed a constant water density
  Take as input:
  - the cell ids where to compute the piezometric head sum (default all)
  - the penalizing power (default 1.)
  - gravity: the gravity (constant) (default=9.80665)
  - density: water liquid density (constant) (default=997.16)
  - reference_pressure: reference pressure for h_pz=0 (default=101325.)
  """
  def __init__(self, ids_to_sum = None, penalizing_power = 1,
                     gravity=9.80655, density=997.16, reference_pressure=101325.):
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
    
    self.head = None
    self.z = None
    
    self.gravity = gravity #m2/s
    self.density = density #kg/m3
    self.mat_props_dependance = []
    self.reference_pressure = reference_pressure #Pa
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
    self.pressure = inputs[0]
    self.z = inputs[1]
    return

  
  ### COST FUNCTION AND ITS DERIVATIVE
  def evaluate(self, pz_head=None):
    """
    Evaluate the cost function
    Return a scalar of dimension [L]
    """
    if pz_head is None: 
      pz_head = (self.pressure-self.reference_pressure) / \
                             (self.gravity * self.density) - self.z 
    if self.ids_to_sum is None: 
      return np.sum(pz_head**self.penalizing_power)
    else: 
      return np.sum(pz_head[self.ids_to_sum-1]**self.penalizing_power)
   
  def d_objective_dP(self, out=None): 
    """
    Evaluate the derivative of the cost function according to the piezometric head
    If a numpy array is provided, derivative will be copied in this array
    Else create a new numpy array
    Derivative have no dimension [-]
    """
    if out is None:
      out = np.zeros(len(self.z), dtype='f8')
    deriv = 1. / (self.gravity * self.density)
    if self.penalizing_power == 1:
      if self.ids_to_sum is None: 
        out[:] = deriv
      else:
        out[self.ids_to_sum-1] = deriv
    else:
      pz_head = (self.pressure-self.reference_pressure) / \
                                 (self.gravity * self.density) - self.z 
      if self.ids_to_sum is None: 
        out[:] = self.penalizing_power / (self.gravity * self.density) * \
                         pz_head**(self.penalizing_power-1)
      else:
        out[self.ids_to_sum-1] = (self.penalizing_power / (self.gravity * self.density) * \
                         pz_head**(self.penalizing_power-1))[self.ids_to_sum-1]
    return out
  
  def d_objective_d_inputs(self, out=None):
    # Does not depend on other variable
    # TODO: add density dependance ?
    return None
    
  
  def __get_PFLOTRAN_output_variable_needed__(self):
    return ["LIQUID_PRESSURE", "Z_COORDINATE"]
  def __is_steady_state__(self): 
    return True
  def __depend_of_mat_props__(self, var=None):
    if var is None: return self.mat_props_dependance
    if var in self.mat_props_dependance: return True
    else: return False

                      
