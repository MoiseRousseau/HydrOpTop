# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
import h5py


class Mean_Liquid_Piezometric_Head:
  """
  Function which return the sum of the liquid piezometric head
  with a penalizing power in a given region.
  Assumed a constant water density (personalizable)
  Take as input:
  - the cell ids where to compute the piezometric head sum (default all)
  - the penalizing power (default 1.)
  - gravity: the gravity (constant) (default=9.80665)
  - density: water liquid density (constant) (default=997.16)
  - reference_pressure: reference pressure for h_pz=0 (default=101325.)
  """
  def __init__(self, ids_to_sum = None, penalizing_power = 1,
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
    
    #inputs for function evaluation 
    self.pressure = None
    self.z = None
    #argument from pflotran simulation
    self.gravity = gravity #m2/s
    self.density = density #kg/m3
    self.reference_pressure = reference_pressure #Pa
    
    #function derivative for adjoint
    self.dobj_dP = None
    self.dobj_dmat_props = None
    self.dobj_dp_partial = None
    self.adjoint = None
    self.filter = None
    
    #required for problem crafting
    self.output_variable_needed = ["LIQUID_PRESSURE", "Z_COORDINATE", "VOLUME"]
    self.name = "Head Sum"
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
    self.pressure = inputs[0]
    self.z = inputs[1]
    self.volume = inputs[2]
    return
    
  def get_inputs(self):
    [self.pressure, self.z, self.volume]
  
  def set_p_to_cell_ids(self,p_ids):
    self.p_ids = p_ids
    return
    
  def set_filter(self, filter):
    self.filter = filter
    return
  
  def set_adjoint_problem(self, x):
    self.adjoint = x
    return

  
  ### COST FUNCTION ###
  def evaluate(self, p):
    """
    Evaluate the cost function
    Return a scalar of dimension [L]
    """
    if not self.initialized: self.__initialize__()
    pz_head = (self.pressure-self.reference_pressure) / \
                             (self.gravity * self.density) + self.z 
    pz_head *= self.volume
    if self.ids_to_sum is None: 
      return np.sum(pz_head**self.penalizing_power) / self.V_tot
    else: 
      return np.sum(pz_head[self.ids_to_sum-1]**self.penalizing_power) / self.V_tot
  
  
  ### PARTIAL DERIVATIVES ###
  def d_objective_dP(self,p): 
    """
    Evaluate the derivative of the function according to the pressure.
    If a numpy array is provided, derivative will be copied 
    in this array, else create a new numpy array.
    Derivative have unit m/Pa
    """
    if not self.initialized: self.__initialize__()
    if self.dobj_dP is None:
      self.dobj_dP = np.zeros(len(self.z), dtype='f8')
    else:
      self.dobj_dP[:] = 0.
    pz_head = (self.pressure-self.reference_pressure) / \
                             (self.gravity * self.density) + self.z 
    deriv = self.volume / (self.gravity * self.density * self.V_tot)
    if self.ids_to_sum is None: 
      self.dobj_dP[:] = self.penalizing_power * deriv * \
                         pz_head**(self.penalizing_power-1)
    else:
      self.dobj_dP[self.ids_to_sum-1] = \
                       (self.penalizing_power * deriv * \
                       pz_head**(self.penalizing_power-1))[self.ids_to_sum-1]
    return
  
  
  def d_objective_d_mat_props(self,p):
    # Does not depend on other variable
    # TODO: add density dependance ?
    self.dobj_dmat_props = [0., 0.]
    return None
  
  
  def d_objective_dp_partial(self,p): 
    """
    Evaluate the PARTIAL derivative of the function according to the density
    parameter p.
    """
    self.dobj_dp_partial = 0.
    return None
  
  
  
  ### TOTAL DERIVATIVE ###
  def d_objective_dp_total(self, p, out=None): 
    """
    Evaluate the TOTAL derivative of the function according to the density
    parameter p. If a numpy array is provided, derivative will be copied 
    in this array, else create a new numpy array.
    Derivative have a length dimension [L]
    """
    if out is None:
      out = np.zeros(len(p), dtype='f8')
    self.d_objective_dP(p) #update objective derivative wrt pressure
    self.d_objective_d_mat_props(p) #update objective derivative wrt mat prop
    self.d_objective_dp_partial(p)
    out[:] = self.adjoint.compute_sensitivity(p, self.dobj_dP, 
               self.dobj_dmat_props, self.output_variable_needed) + self.dobj_dp_partial
    return out
  
  
  ### WRAPPER FOR NLOPT ###
  def nlopt_optimize(self,p,grad):
    cf = self.evaluate(p)
    print(f"Current {self.name}: {cf:.6e}")
    if grad.size > 0:
      self.d_objective_dp_total(p,grad)
      print(f"Min gradient: {np.min(grad):.6e} at cell id {np.argmin(grad)}")
      print(f"Max gradient: {np.max(grad):.6e} at cell id {np.argmax(grad)}")
    return cf
  
  def __initialize__(self):
    self.initialized = True
    if self.ids_to_sum is None:
      self.V_tot = np.sum(self.volume)
    else:
      self.V_tot = np.sum(self.volume[self.ids_to_sum-1])
    return
  
  
  ### REQUIRED FOR CRAFTING ###
  def __require_adjoint__(self): return "RICHARDS"
  def __get_PFLOTRAN_output_variable_needed__(self):
    return self.output_variable_needed
  def __get_name__(self): return self.name
                      
