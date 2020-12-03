

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
    self.ids_to_consider = ids_to_sum_volume
    self.max_v_frac = max_volume_percentage
    self.p = None
    return
  
  def set_density_parameter(self, p):
    self.p = p #a reference to the p array
    return
  
  def set_cell_volume(self, V):
    self.V = V
    self.V_tot = np.sum(self.V)
    return
  
  def evaluate(self, p=None):
    if p is None: p = self.p
    constrain = np.sum(V*p)/self.V_tot - self.max_v_frac
    return constrain
    
  def d_constrain_dp(self):
    return self.V/V_tot

