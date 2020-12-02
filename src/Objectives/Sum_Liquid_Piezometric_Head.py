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
  """
  def __init__(self, ids_to_sum = None, penalizing_power = 1
                     gravity=9.80655, density=997.16):
    #pflotran output file
    #self.pz_head = pz_head_array
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
    #self.reference_pressure = 101325 #Pa
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
  
  def set_liquid_head(self, h):
    self.head = h
    return
    
  def set_z_coordinate(self, z):
    self.z = z
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
      pz_head = self.pressure / (self.gravity * self.density) - self.z 
    if self.ids_to_sum is None: 
      return np.sum(pz_head**self.penalizing_power)
    else: 
      return np.sum(pz_head[self.ids_to_sum-1]**self.penalizing_power)
  
  def d_objective_d_inputs(self, pz_head=None, outs=None):
    """
    Evaluate the derivative of the cost function according to the piezometric head
    If a numpy array is provided, derivative will be copied in this array
    Else create a new numpy array
    Derivative have no dimension [-]
    """
    if pz_head is None: 
      pz_head = self.pressure / (self.gravity * self.density) - self.z 
      
    if outs is None:
      out = [np.zeros(len(self.z), dtype='f8'), 0.]
    else:
      #outs[0][:] = 0. #initialize to zeros
      out[1] = 0.
    if self.penalizing_power == 1:
      if self.ids_to_sum is None: 
        outs[0][:] = 1.
      else:
        out[0][self.ids_to_sum-1] = 1.
    else:
      if self.ids_to_sum is None: 
        out[0][:] = self.penalizing_power * X**(self.penalizing_power-1)
      else:
        out[0][self.ids_to_sum-1] = 1. * self.penalizing_power * \
                                 self.pz_head **(self.penalizing_power-1)
    return outs
    
  
  def __get_PFLOTRAN_output_variable_needed__(self):
    return ["LIQUID_PRESSURE", "Z_COORDINATE"]
  def __is_steady_state__(self): 
    return True

                      
