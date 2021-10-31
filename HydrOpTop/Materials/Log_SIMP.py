import numpy as np

class Log_SIMP:
  """
  !SIMP parametrization of material properties in logarithmic scale:
  !X(p) = 10^( log(X0) + (log(X1)-log(X0))*p^n )
  !When p=0 -> X=X0
  !When p=1 -> X=X1
  Input:
  @param cell_ids_to_parametrize: cell to parametrize with the given law
  @param bound: material properties bound [min, max]
  @param power: the penalizing power (integer)
  @param reverse: invert parametrization (p=0 -> X=X1, p=1 -> X=X0)
  """
  def __init__(self, cell_ids_to_parametrize,
                     property_name, bounds, power=3, reverse=False):
    if isinstance(cell_ids_to_parametrize, str) and \
             cell_ids_to_parametrize.lower() == "all":
      self.cell_ids = None
    else:
      self.cell_ids = np.array(cell_ids_to_parametrize)
    self.min_, self.max_ = bounds
    self.reverse = reverse
    self.power = power
    self.name= property_name
    return
  
  
  def get_cell_ids_to_parametrize(self):
    return self.cell_ids
  
  
  def convert_p_to_mat_properties(self, p, out=None):
    if out is None: out = np.zeros(len(p),dtype='f8')
    if self.reverse: p_ = 1-p
    else: p_ = p
    out[:] = 10**(np.log10(self.min_) + 
                    np.log10(self.max_/self.min_)*p_**self.power)
    return out
  
  
  def d_mat_properties(self, p, out=None):
    """
    Return the derivative of the material properties according to 
    material parameter p.
    """
    if out is None: out = np.zeros(len(p),dtype='f8')
    if self.reverse: 
      factor = -1.
      p_ = 1-p
    else: 
      factor = 1.
      p_ = p
    pre = np.log(10) * np.log10(self.max_/self.min_) * \
                                 self.convert_p_to_mat_properties(p)
    out[:] = factor * self.power * pre * p_**(self.power-1)
    return out
  
      
  def convert_mat_properties_to_p(self, mat_prop_val):
    if np.min(mat_prop_val) >= self.min_ and \
          np.max(mat_prop_val) <= self.max_ :
      p = ( np.log10(mat_prop_val/self.min_) / np.log10(self.max_/self.min_) ) ** (1/self.power)
      if self.reverse: p = 1-p
      return p
    else:
      print("Min and max permeability value not in the range of material \
             properties")
      return None
  
  def get_name(self):
    return self.name
  

