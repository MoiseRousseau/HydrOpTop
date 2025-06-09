# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
from .Base_Function_class import Base_Function

class Reference_Liquid_Head(Base_Function):
  r"""
  Compute the difference between the head at the given cells and the head in the simulation.
  
  Required ``LIQUID_HEAD`` output.
  
  :param head: Reference head to compute the difference with
  :type head: iterable
  :param cell_ids: The corresponding cell ids of the head. If None, consider head[0] for cell id 1, head[1] for cell id 2, ...
  :type cell_ids: iterable
  :param observation_name: If observation points have name in the simulator, provide it here
  :type observation_name: list of str
  :param norm: Norm to compute the difference (i.e. 1 for sum of head error, 2 for MSE, inf for max difference
  :type norm: int
  """
  def __init__(
    self,
    head,
    cell_ids=None,
    observation_name=None,
    norm = 1,
  ):
    super(Reference_Liquid_Head, self).__init__()
    
    self.set_error_norm(norm)
    self.ref_head = np.array(head)
    self.cell_ids = np.array(cell_ids)
    
    #inputs for function evaluation 
    self.inputs = None
    self.observation_name = observation_name
    
    #function derivative for adjoint
    self.dobj_dP = None
    self.dobj_dmat_props = None
    self.dobj_dp_partial = None
    self.adjoint = None
    
    #required for problem crafting
    self.variables_needed = ["LIQUID_HEAD"]
    #if self.observation_name is not None:
    # 	self.solved_variables_needed = ["LIQUID_HEAD_AT_OBSERVATION"]
    self.name = "Reference Head"
    self.initialized = None
    return
  
  def set_error_norm(self, x):
    if int(x) != x or x <= 0:
      raise ValueError("Error norm need to be a positive integer")
    self.norm = x
    return

  
  ### COST FUNCTION ###
  def evaluate(self, p):
    """
    Evaluate the cost function
    Return a scalar of dimension [L]
    """
    self.head = self.inputs["LIQUID_HEAD"]
    if not self.initialized: self.__initialize__()
    if self.observation_name is not None:
      r = np.array([
        self.head[x] - h for x,h in zip(self.observation_name, self.ref_head)
      ])
      return np.sum(r**self.norm)
    if self.cell_ids is None: 
      return np.sum((self.head-self.ref_head)**self.norm)
    else:
      return np.sum((self.head[self.cell_ids-1]-self.ref_head)**self.norm)
  
  
  ### PARTIAL DERIVATIVES ###
  def d_objective_dY(self,p): 
    """
    Evaluate the derivative of the function according to the pressure.
    If a numpy array is provided, derivative will be copied 
    in this array, else create a new numpy array.
    Derivative have unit m/m
    """
    self.head = self.inputs["LIQUID_HEAD"]
    if not self.initialized: self.__initialize__()
    if self.dobj_dP is None:
      self.dobj_dP = np.zeros(len(self.head), dtype='f8')
    self.dobj_dP[:] = 0.
    if self.observation_name is not None:
      r = np.array([
        self.head[x] - h for x,h in zip(self.observation_name, self.ref_head)
      ])
      self.dobj_dP[:] = self.norm * r**(self.norm-1)
    elif self.cell_ids is None: 
      self.dobj_dP[:] = self.norm * \
                         (self.head-self.ref_head)**(self.norm-1)
    else:
      self.dobj_dP[self.cell_ids-1] = \
        self.norm * (self.head[self.cell_ids-1]-self.ref_head)**(self.norm-1)
    return [self.dobj_dP]
  
  
  def __initialize__(self):
    self.initialized = True
    return
                      
