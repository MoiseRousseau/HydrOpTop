from .Mesh_NNR import Mesh_NNR
import numpy as np
from scipy.sparse import dia_matrix


class Helmholtz_Density_Filter:
  """
  Filter the density parameter according a Helmholtz type PDE
  from Lazarov and Sigmund (2010):
  https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.3072
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
  

