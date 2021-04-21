import numpy as np
from scipy.sparse import dia_matrix

class Heavyside_Filter:
  """
  Filter the density paramater using a three field method according to
  a Heavyside function
  """
  def __init__(self, base_density_filter=None, cutoff=0.5, steepness = 5):
    self.base_density_filter = base_density_filter
    self.cutoff = cutoff
    self.stepness = steepness
    self.initialized = False
    
    if base_density_filter is None:
      self.output_variable_needed = []
    else:
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
    if self.base_density_filter is not None:
      p_bar = self.base_density_filter.get_filtered_density(p)
    else: p_bar = p
    a = np.tanh(self.stepness*self.cutoff)
    p_filtered[:] = (a+np.tanh(self.stepness*(p_bar-self.cutoff))) / \
                       (a+np.tanh(self.stepness*(1-self.cutoff)))
    return p_filtered
  
  def get_filter_derivative(self, p):
    if not self.initialized: self.initialize()
    p_bar = self.base_density_filter.get_filtered_density(p)
    d_p_bar = self.base_density_filter.get_filter_derivative(p) #matrix
    d_p_filtered = 1 - np.tanh(self.stepness*(p_bar-self.cutoff))**2
    a = np.tanh(self.stepness*self.cutoff)
    d_p_filtered *= self.stepness / (a + np.tanh(self.stepness*(1-self.cutoff)))
    d_p = d_p_bar.dot( dia_matrix((d_p_filtered[np.newaxis,:],0),
                                                   shape=d_p_bar.shape) )
    return d_p
  
  def plot_filtered_density(self):
    try:
      import matplotlib.pyplot as plt
    except:
      print("Matplotlib is not available on your installation")
      print("Please try 'pip3 install matplotlib' and restart the optimization")
    x = np.linspace(0,1,1000)
    save = self.base_density_filter
    self.base_density_filter = None
    y = self.get_filtered_density(x)
    fig,ax = plt.subplots()
    ax.plot(x,y,'b',label="Filtered parameter")
    ax.set_xlabel("Input Parameter")
    ax.set_ylabel("Filtered Parameter")
    ax.grid()
    plt.show()
    self.base_density_filter = save
    return
    
  def initialize(self):
    if self.initialized: return
    if self.base_density_filter is not None:
      self.base_density_filter.initialize()
    self.initialized = True
    return
  
  def __get_PFLOTRAN_output_variable_needed__(self):
    return self.output_variable_needed 
  

