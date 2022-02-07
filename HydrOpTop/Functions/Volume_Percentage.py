import numpy as np
from .Base_Function_class import Base_Function

class Volume_Percentage(Base_Function):
  r"""
  Description:
    ``Volume_Percentage`` function compute the ratio of the volume of material
    designed by `p=1` on a prescribed domain :math:`D`:

    .. math::
       
       f = \frac{1}{V_D} \sum_{i \in D} p_i V_i

  Parameters:
    ``ids_to_sum_volume`` (iterable): a list of cell ids on which to compute
    the volume percentage 

    ``volume_of_p0`` (bool): set to ``True``, switch the material and rather
    calculate the volume fraction of the material designed by `p=0`. In this 
    case :math:`p_i` is remplaced by :math:`p'_i = 1-p_i`.

  Require PFLOTRAN output:
    ``VOLUME``.
    
  """
  def __init__(self, ids_to_sum_volume="parametrized_cell", max_volume_percentage=0.2,
                     volume_of_p0=False):
    if isinstance(ids_to_sum_volume, str):
      if ids_to_sum_volume.lower() == "parametrized_cell":
        self.ids_to_consider = None
      else:
        print("Error! Non-recognized option for ids_to_sum_volume: " + ids_to_sum_volume)
        exit(1)
    else:
      self.ids_to_consider = ids_to_sum_volume
    self.max_v_frac = max_volume_percentage
    self.vp0 = volume_of_p0 #boolean to compute the volume of the mat p=1 (False) p=0 (True)
    
    #function inputs
    self.V = None
    
    #quantities derived from the input calculated one time
    self.initialized = False
    self.V_tot = None
    
    #function derivative for adjoint
    self.dobj_dP = 0.
    self.dobj_dmat_props = [0.]
    self.dobj_dp_partial = None
    self.adjoint = None
    
    self.solved_variables_needed = []
    self.input_variables_needed = ["VOLUME"] 
    self.name = "Volume"
    return
  
  
  def set_inputs(self, inputs):
    self.V = inputs[0]
    return
  
  def get_inputs(self):
    return [self.V]
    
  def set_p_to_cell_ids(self, cell_ids):
    self.p_ids = cell_ids
    return
  
  
  ### COST FUNCTION ###
  def evaluate(self,p):
    """
    Evaluate the cost function
    Return a scalar of dimension [L**3]
    """
    if not self.initialized: self.__initialize__()
    if self.vp0: p_ = 1-p
    else: p_ = p
    if self.ids_to_consider is None:
      #sum on all parametrized cell
      cf = np.sum(self.V[self.p_ids-1]*p_)/self.V_tot - self.max_v_frac
    else:
      cf = np.sum((self.V[self.ids_to_consider-1]*p_))/self.V_tot - self.max_v_frac
    return cf

  
  def d_objective_dp_partial(self,p): 
    if self.dobj_dp_partial is None:
      self.dobj_dp_partial = np.zeros(len(p),dtype='f8')
    if self.vp0: factor = -1.
    else: factor = 1.
    if not self.initialized: self.__initialize__()
    if self.ids_to_consider is None:
      self.dobj_dp_partial[:] = factor * self.V[self.p_ids-1]/self.V_tot
    else:
      self.dobj_dp_partial[:] = factor * self.V[self.ids_to_consider-1]/self.V_tot
    return self.dobj_dp_partial
  
  ### INITIALIZER FUNCTION ###
  def __initialize__(self):
    """
    Initialize the derived quantities from the inputs that must be compute one 
    time only.
    """
    self.initialized = True
    if self.ids_to_consider is None:
      self.V_tot = np.sum(self.V[self.p_ids-1])
    else:
      self.V_tot = np.sum(self.V[self.ids_to_consider-1])
    return
  
  
  ### REQUIRED FOR CRAFTING ###
  def __get_constraint_tol__(self): return self.max_v_frac

