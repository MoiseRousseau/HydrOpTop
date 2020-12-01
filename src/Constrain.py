

class Maximum_volume_constrain:
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
  def __init__(self, density_parameter, cells_volume, max_volume_percentage):
    self.max_v_frac = max_volume_percentage
    self.V = cells_volume
    self.V_tot = np.sum(self.V)
    self.p = density_parameter
    return
  
  def evaluate(self):
    constrain = np.sum(V*self.p)/self.V_tot - self.max_v_frac
    return constrain
    
  def d_constrain_dp(self):
    return self.V/V_tot

