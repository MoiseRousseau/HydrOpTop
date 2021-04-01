import numpy as np
from scipy.sparse import dia_matrix

class Heavyside_Density_Filter:
  """
  Filter the density paramater using a three field method according to
  https://link.springer.com/article/10.1007/s00158-009-0452-7
  """
  def __init__(self, base_density_filter, cutoff=0.5, steepness = 5):
    self.base_density_filter = base_density_filter
    self.cutoff = cutoff
    self.stepness = steepness
    self.initialized = False
    
    self.output_variable_needed = \
            self.base_density_filter.__get_PFLOTRAN_output_variable_needed__()
    return
    
  def set_p_to_cell_ids(self, p_ids):
    #if None, this mean all the cell are parametrized
    self.base_density_filter.set_p_to_cell_ids(p_ids) 
    return
  
  def set_inputs(self, inputs):
    self.base_density_filter.set_inputs(inputs)
    return
  
  def get_filtered_density(self, p, p_filtered=None):
    if not self.initialized: self.initialize()
    if p_filtered is None:
      p_filtered = np.zeros(len(p), dtype='f8')
    else:
      p_filtered[:] = 0.
    p_bar = self.base_density_filter.get_filtered_density(p)
    cpb = 1 - p_bar / self.cutoff
    pbr = (p_bar-self.cutoff) / (1-self.cutoff)
    p_filtered[:] = np.where(p_bar<=self.cutoff,
        self.cutoff*(np.exp(-self.stepness*cpb) - cpb*np.exp(-self.stepness)),
        (1-self.cutoff)*(1-np.exp(-self.stepness*pbr)+np.exp(-self.stepness)*pbr)+self.cutoff)
    return p_filtered
  
  def get_filter_derivative(self, p):
    if not self.initialized: self.initialize()
    p_bar = self.base_density_filter.get_filtered_density(p)
    d_p_bar = self.base_density_filter.get_filter_derivative(p) #matrix
    d_p_filtered = np.where(p_bar<=self.cutoff,
                     np.exp(-self.stepness * (1-p_bar/self.cutoff) ),
                     np.exp(-self.stepness * (p_bar-self.cutoff) / (1-self.cutoff) ) )
    d_p_filtered *= self.stepness
    d_p_filtered += np.exp(-self.stepness)
    d_p = d_p_bar.dot( dia_matrix((d_p_filtered[np.newaxis,:],0),
                                                   shape=d_p_bar.shape) )
    return d_p
    
  def initialize(self):
    if self.initialized: return
    self.base_density_filter.initialize()
    self.initialized = True
    return
  
  def __get_PFLOTRAN_output_variable_needed__(self):
    return self.output_variable_needed 
  

