

class Permeability:
  """
  !Link the density parameter to the permeability by a standard \
  SIMP parametrization. \n
  Input:
  @param bound: permeability bound
  @param power: the penalizing power (integer)
  !When p=0 -> k=bound[0]
  !When p=1 -> k=bound[1]
  """
  def __init__(self, density_parameter, bound, power=1):
    self.min_K = bound[0]
    self.max_K = bound[1]
    self.power = power
    self.p = density_parameter
    return
    
  def mat_properties(self):
    K = self.min_K + (self.max_K-self.min_K) * self.p**self.power
    return K
  
  def d_mat_properties(self):
    return self.power * (self.max_K-self.min_K) * self.p**(self.power-1)
    


class Permeability_SourceSink:
  """
  Link the density parameter to the permeability and a source
  sink term to simulate water release from the material at p=1
  """
  def __init__(self, k_bound, k_ss, power):
    return
  
  def mat_properties(self, p):
    return
  
  def d_mat_properties(self,p):
    return
