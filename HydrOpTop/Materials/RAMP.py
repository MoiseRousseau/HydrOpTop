import numpy as np

class RAMP:
  r"""
  Description:
    RAMP stands for Rational Approximation of of Material Properties. It applies the 
    following relation between the material properties :math:`X` and the density parameter :math:`p`:

    .. math::
       
       X(p) = X_0 + (X_1 - X_0) \frac{p} {1 + q (1-p)}

    Where :math:`X_0` and :math:`X_1` is the material property values when :math:`p=0`
    and :math:`p=1`, and :math:`q` is the RAMP parameter.

  Parameters:
    ``cell_ids_to_parametrize`` (iterable): the cell ids on which to apply the parametrization
    
    ``property_name`` (str): the name of the material parameter (same as defined in the solver IO shield)
    
    ``bounds`` (list): the `X_0` and `X_1` values in a list
    
    ``parameter`` (float): the RAMP parameter `q`
    
    ``reverse`` (bool): set to ``True``, reverse the bounds.
  
  """
  def __init__(self, cell_ids_to_parametrize,
                     property_name, bounds, parameter=3, reverse=False):
    if isinstance(cell_ids_to_parametrize, str) and \
             cell_ids_to_parametrize.lower() == "all":
      self.cell_ids = None
    else:
      self.cell_ids = np.array(cell_ids_to_parametrize)
    self.min_, self.max_ = bounds
    self.reverse = reverse
    self.power = parameter
    self.name= property_name
    return
  
  
  def get_cell_ids_to_parametrize(self):
    return self.cell_ids
  
  
  def convert_p_to_mat_properties(self, p, out=None):
    if out is None: out = np.zeros(len(p),dtype='f8')
    if self.reverse: p_ = 1-p
    else: p_ = p
    out[:] = self.min_ + (self.max_-self.min_) * p_ / (1 + self.power * (1-p_))
    return out
  
  
  def d_mat_properties(self, p, out=None):
    #definition from https://www.sciencedirect.com/science/article/pii/S0045793018301932#bib0004
    if out is None: out = np.zeros(len(p),dtype='f8')
    if self.reverse: 
      factor = -1.
      p_ = 1-p
    else: 
      factor = 1.
      p_ = p
    pre = (self.max_-self.min_)
    out[:] = factor * pre * ( (1 + self.power * (1-p_)) + p_ * self.power) / (1 + self.power * (1-p_))**2
    return out
  
  
  def convert_mat_properties_to_p(self, mat_prop_val):
    #TODO
    return None
  
  
  def get_name(self):
    return self.name

