import numpy as np
from .Base_Filter_class import Base_Filter


class Helmholtz_Density_Filter(Base_Filter):
  r"""
  Description:
    Smooth the density filter using the Helmholtz partial differential equation:

    .. math::
      
      \nabla^T \boldsymbol{K} \: \nabla \bar p + \bar p = 0

    This filter is doing approximately the same than the standard density filter above but
    requires much less memory by not storing the distance matrix. Also more
    robust when the optimization mesh had sharp concave boundary. 
    More detail available in Lazarov and Sigmund (2011).

  Not yet implementated, but planned...
  
  """
  #TODO
  def __init__(self):
    self.initialized = False
    
    self.output_variable_needed = []
    return
  
  def set_p_to_cell_ids(self, p_ids):
    self.p_ids = p_ids #if None, this mean all the cell are parametrized
    return
  
  def set_inputs(self, inputs):
    return
  
  
  def initialize(self):
    self.initialized = True
    return
  
  def get_filtered_density(self, p, p_bar=None):
    if not self.initialized: self.initialize()
    if p_bar is None:
      p_bar = np.zeros(len(p), dtype='f8')
    p_bar[:] = 0.
    return p_bar
  
  def get_filter_derivative(self, p):
    if not self.initialized: self.initialize()
    return out
 
  
  def __get_PFLOTRAN_output_variable_needed__(self):
    return self.output_variable_needed 
  

