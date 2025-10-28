import numpy as np
from .Base_Material_class import Base_Material

class Log_SIMP(Base_Material):
  r"""
  Description:
    SIMP stands for Standard Isotropic Material Parametrization. It applies the 
    following relation between the material properties :math:`X` and the density parameter :math:`p`:

    .. math::
       
      X(p) = 10^{\log{X_0} + \log(\frac{X_1}{X_0}) p^n}

    Where :math:`X_0` and :math:`X_1` is the material property values when :math:`p=0`
    and :math:`p=1`. :math:`n` the penalization power (see X).

  Parameters:
    ``cell_ids_to_parametrize`` (iterable): the cell ids on which to apply the parametrization
    
    ``property_name`` (str): the name of the material parameter (same as defined in the solver IO shield)
    
    ``bounds`` (list): the `X_0` and `X_1` values in a list
    
    ``power`` (float): the penalization power `n`
    
    ``reverse`` (bool): set to ``True``, reverse the bounds.
  
  """
  def __init__(self, cell_ids_to_parametrize,
                     property_name, bounds, power=1, reverse=False):
    super(Log_SIMP, self).__init__()
    if isinstance(cell_ids_to_parametrize, str) and \
             cell_ids_to_parametrize.lower() == "__all__":
      self.cell_ids = None
    else:
      self.cell_ids = np.array(cell_ids_to_parametrize)
    self.min_, self.max_ = bounds
    self.reverse = reverse
    self.power = power
    self.name = property_name
    return
  
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
