import numpy as np
from .Base_Function_class import Base_Function

class Volume_Percentage(Base_Function):
  r"""
  Integrate the volume ponderated value of the primary optimization variable `p` over the domain :math:`D` (i.e., compute the ratio of volume of material designed by `p=1`):

  .. math::
       
      f = \frac{1}{V_D} \sum_{i \in D} p_i V_i
  
  Require PFLOTRAN output ``VOLUME``.

  :param ids_to_sum_volume: a list of cell ids on which to compute the volume percentage 
  :type ids_to_sum_volume: iterable
  :param volume_of_p0: if set to ``True``, switch the material and rather calculate the volume fraction of the material designed by `p=0`. In this case, :math:`p_i` is remplaced by :math:`p'_i = 1-p_i`.
  """
  def __init__(self, ids_to_sum_volume="parametrized_cell",
                     volume_of_p0=False):

    super(Volume_Percentage, self).__init__()
    
    if isinstance(ids_to_sum_volume, str):
      if ids_to_sum_volume.lower() == "parametrized_cell":
        self.ids_to_consider = None
      else:
        print("Error! Non-recognized option for ids_to_sum_volume: " + ids_to_sum_volume)
        exit(1)
    else:
      self.ids_to_consider = ids_to_sum_volume
      
    self.vp0 = volume_of_p0 #boolean to compute the volume of the mat p=1 (False) p=0 (True)
    
    #function inputs
    self.V = None
    
    #quantities derived from the input calculated one time
    self.initialized = False
    self.V_tot = None
    
    self.variables_needed = ["VOLUME"]
    self.name = "Volume"
    self.inputs = {}
    return
  
  
  ### COST FUNCTION ###
  def evaluate(self,p):
    """
    Evaluate the cost function
    Return a scalar of dimension [L**3]
    """
    V = self.inputs["VOLUME"]
    if not self.initialized: self.__initialize__()
    if self.vp0: p_ = 1-p
    else: p_ = p
    if self.ids_to_consider is None:
      #sum on all parametrized cell
      cf = np.sum(V[self.p_ids-1]*p_)/self.V_tot
    else:
      cf = np.sum((V[self.ids_to_consider-1]*p_))/self.V_tot
    return cf

  
  def d_objective_dp_partial(self,p): 
    """
    Derivative according to the density parameter which is the percentage of material in the cell
    """
    V = self.inputs["VOLUME"]
    res = np.zeros(len(p),dtype='f8')
    if self.vp0: 
      factor = -1.
    else:
      factor = 1.
    if not self.initialized: 
      self.__initialize__()
    if self.ids_to_consider is None:
      res[:] = factor * V[self.p_ids-1]/self.V_tot
    else:
      res[:] = factor * V[self.ids_to_consider-1]/self.V_tot
    return res
  
  
  ### INITIALIZER FUNCTION ###
  def __initialize__(self):
    """
    Initialize the derived quantities from the inputs that must be compute one 
    time only.
    """
    self.initialized = True
    if self.ids_to_consider is None:
      self.V_tot = np.sum(self.inputs["VOLUME"][self.p_ids-1])
    else:
      self.V_tot = np.sum(self.inputs["VOLUME"][self.ids_to_consider-1])
    return

  @classmethod
  def sample_instance(cls):
    res1 = cls(ids_to_sum_volume=[2,4,3,8],volume_of_p0=False)
    res1.set_inputs({"VOLUME":np.random.rand(20)[res1.cell_ids]*100})
    res1.deriv_var_to_skip = ["VOLUME"]
    # with volume of p0
    res2 = cls(ids_to_sum_volume=[2,4,3,8],volume_of_p0=True)
    res2.set_inputs({"VOLUME":np.random.rand(20)[res2.cell_ids]*100})
    res2.deriv_var_to_skip = ["VOLUME"]
    return [res1,res2]
