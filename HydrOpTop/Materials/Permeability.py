

class Permeability:
  """
  !Link the density parameter to the permeability by a standard \
  SIMP parametrization.
  Input:
  @param bound: permeability bound
  @param power: the penalizing power (integer)
  !When p=0 -> k=bound[0]
  !When p=1 -> k=bound[1]
  """
  def __init__(self, cell_ids, bound, power=3):
    self.cell_ids = cell_ids
    self.min_K = bound[0]
    self.max_K = bound[1]
    self.power = power
    return
  
  def get_cell_ids_to_parametrize(self):
    return self.cell_ids
    
  def convert_p_to_mat_properties(self, p, out=None):
    if out is None:
      K = self.min_K + (self.max_K-self.min_K) * p**self.power
      return K
    else:
      out[:] = self.min_K + (self.max_K-self.min_K) * p**self.power
      return
  
  def d_mat_properties(self, p, out=None):
    if out is None:
      return self.power * (self.max_K-self.min_K) * p**(self.power-1)
    else:
      out[:] = self.power * (self.max_K-self.min_K) * p**(self.power-1)
      return
  
  def get_name(self):
    return "PERMEABILITY"
    


