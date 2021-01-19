import numpy as np

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
  def __init__(self, bound, cell_ids_to_parametrize=None, power=3, name="PERMEABILITY"):
    if isinstance(cell_ids_to_parametrize, str) and \
             cell_ids_to_parametrize.lower() == "everywhere":
      self.cell_ids = None
    else:
      self.cell_ids = cell_ids_to_parametrize
    self.min_K = bound[0]
    self.max_K = bound[1]
    self.power = power
    self.name= name #note: Permeability class could be used to
                    # parametrize PERMEABILITY_X, PERMEABILITY_Y, ...
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
    """
    Return the derivative of the material properties according to 
    material parameter p.
    """
    if out is None:
      return self.power * (self.max_K-self.min_K) * p**(self.power-1)
    else:
      out[:] = (self.power * (self.max_K-self.min_K) * p**(self.power-1))
      return
      
  def convert_mat_properties_to_p(self, mat_prop_val):
    if np.min(mat_prop_val) >= self.min_K and \
          np.max(mat_prop_val) <= self.max_K :
      return ( (mat_prop_val - self.min_K) / ( self.max_K - self.min_K) ) ** (1/self.power)
    else:
      print("Min and max permeability value not in the range of material \
             properties")
      return None
  
  def get_name(self):
    return "PERMEABILITY"
    


