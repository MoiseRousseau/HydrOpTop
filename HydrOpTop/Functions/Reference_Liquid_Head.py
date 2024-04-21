# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
from .Base_Function_class import Base_Function

class Reference_Liquid_Head(Base_Function):
  r"""
  Compute the difference between the head at the given cells and the head in the simulation.
  
  Required PFLOTRAN outputs ``LIQUID_PRESSURE``, ``ELEMENT_CENTER_Z``.
  
  :param head: Reference head to compute the difference with
  :type head: iterable
  :param cell_ids: The corresponding cell ids of the head. If None, consider head[0] for cell id 1, head[1] for cell id 2, ...
  :type cell_ids: iterable
  :param norm: Norm to compute the difference (i.e. 1 for sum of head error, 2 for MSE, inf for max difference
  :type norm: int
  :param gravity: norm of the gravity vector `g`
  :type gravity: float
  :param density: fluid density `\rho`
  :type density: float
  :param ref_pressure: reference pressure in PFLOTRAN simulation
  :type ref_pressure: float 
  """
  def __init__(
    self,
    head,
    cell_ids=None,
    norm = 1,
    gravity=9.8068,
    density=997.16,
    reference_pressure=101325.
  ):
    
    self.set_error_norm(norm)
    self.ref_head = np.array(head)
    if cell_ids is not None: self.cell_ids = np.array(cell_ids)
    
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
    
    #required for problem crafting
    self.solved_variables_needed = ["LIQUID_PRESSURE"]
    self.input_variables_needed = ["ELEMENT_CENTER_Z"]
    self.name = "Reference Head"
    self.initialized = None
    return
  
  def set_error_norm(self, x):
    if int(x) != x or x <= 0:
      raise ValueError("Error norm need to be a positive integer")
    self.norm = x
    return
    
  def set_inputs(self, inputs):
    self.pressure = inputs[0]
    self.z = inputs[1]
    return
    
  def get_inputs(self):
    [self.pressure, self.z]

  
  ### COST FUNCTION ###
  def evaluate(self, p):
    """
    Evaluate the cost function
    Return a scalar of dimension [L]
    """
    if not self.initialized: self.__initialize__()
    pz_head = (self.pressure-self.reference_pressure) / \
                             (self.gravity * self.density) + self.z
    if self.cell_ids is None: 
      return np.sum((pz_head-self.ref_head)**self.norm)
    else: 
      return np.sum((pz_head[self.cell_ids-1]-self.ref_head)**self.norm)
  
  
  ### PARTIAL DERIVATIVES ###
  def d_objective_dY(self,p): 
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
    deriv = 1 / (self.gravity * self.density)
    if self.cell_ids is None: 
      self.dobj_dP[:] = self.norm * deriv * \
                         (pz_head-self.ref_head)**(self.norm-1)
    else:
      self.dobj_dP[self.cell_ids-1] = \
        self.norm * deriv * (pz_head[self.cell_ids-1]-self.ref_head)**(self.norm-1)
    return [self.dobj_dP]
  
  
  def __initialize__(self):
    self.initialized = True
    return
                      
