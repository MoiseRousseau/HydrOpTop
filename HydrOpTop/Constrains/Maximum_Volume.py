import numpy as np

class Maximum_Volume:
  """
  Define a maximum volume constrain for the material represented by
  p=1. Constrain is defined as a percentage of the material p=1 on the
  given optimization domain.
  Inputs:
  - max_volume_percentage: the maximum volume percentage
  - cell_volume (numpy array): the volume of the cell being optimized 
                               (indexed the same way that the density 
                               parameter p used by the evaluate method)
  """
  def __init__(self, ids_to_sum_volume, max_volume_percentage):
    if isinstance(ids_to_sum_volume, str) and \
               ids_to_sum_volume.lower() == "everywhere":
      self.ids_to_consider = None
    else:
      self.ids_to_consider = ids_to_sum_volume
    self.max_v_frac = max_volume_percentage
    self.V = None
    self.V_tot = None
    self.p_duplicate = None
    return
  
  def set_inputs(self, inputs):
    self.V = inputs[0]
    self.p = np.zeros(len(self.V), dtype='f8')
    return
    
  def set_p_cell_ids(self, cell_ids):
    self.p_cell_ids = cell_ids
    return
  
  def evaluate(self, p, grad):
    self.p[self.p_cell_ids-1] = p
    if self.ids_to_consider is None:
      V_tot = np.sum(self.V)
      constrain = np.sum(self.V*self.p)/V_tot - self.max_v_frac
    else:
      V_tot = np.sum(self.V[self.ids_to_consider-1])
      constrain = np.sum((self.V*self.p)[self.ids_to_consider-1])/V_tot - self.max_v_frac
    if grad.size > 0:
      if self.ids_to_consider is None:
        grad[:] = self.V/V_tot
      else:
        grad[:] = self.V[self.ids_to_consider-1]/V_tot
    print(f"Current volume constrain: {constrain+self.max_v_frac}")
    return constrain
  
  def __get_PFLOTRAN_output_variable_needed__(self):
    return ["VOLUME"]
  def __need_p_cell_ids__(self): return True
  

