import numpy as np

class Maximum_Volume:
  """
  Define a maximum volume constrain for the material represented by
  p=1. Constrain is defined as a percentage of the material p=1 on the
  given optimization domain.
  Inputs:
  - max_volume_percentage: the maximum volume percentage
  - cell_volume (numpy array): the volume of the cell being optimized 
                               (indexed the same way that the density 
                               parameter p used by the evaluate method)
  """
  def __init__(self, ids_to_sum_volume, max_volume_percentage):
    if isinstance(ids_to_sum_volume, str) and \
               ids_to_sum_volume.lower() == "everywhere":
      self.ids_to_consider = None
    else:
      self.ids_to_consider = ids_to_sum_volume
    self.max_v_frac = max_volume_percentage
    self.V = None
    self.V_tot = None
    self.filter = None
    return
  
  def set_filter(self, filter):
    self.filter = filter
    return
  
  def set_inputs(self, inputs):
    self.V = inputs[0]
    self.p = np.zeros(len(self.V), dtype='f8')
    return
    
  def set_p_cell_ids(self, cell_ids):
    self.p_cell_ids = cell_ids
    return
  
  
  ### EVALUATION ###
  def evaluate(self,p):
    if self.filter:
      p_bar = self.filter.get_filtered_density(p)
    else:
      p_bar = p
    self.p[self.p_cell_ids] = p_bar #self.p spand over all the domain, 
                                  #but p_bar only on the cell id to optimize
    if self.ids_to_consider is None:
      self.V_tot = np.sum(self.V)
      cf = np.sum(self.V*self.p)/self.V_tot - self.max_v_frac
    else:
      self.V_tot = np.sum(self.V[self.ids_to_consider-1])
      cf = np.sum((self.V*self.p)[self.ids_to_consider-1])/self.V_tot - self.max_v_frac
    return cf
  
  ### TOTAL DERIVATIVE ###
  def d_evaluate(self,p,grad):
    if self.ids_to_consider is None:
      grad[:] = self.V/self.V_tot
    else:
      grad[:] = self.V[self.ids_to_consider-1]/self.V_tot
    if self.filter:
      grad[:] = self.filter.get_filter_derivative(p).transpose().dot(grad)
    return
  
  ### WRAPPER ###
  def nlopt_optimize(self, p, grad):
    cf = self.evaluate(p)
    self.d_evaluate(p, grad)
    print(f"Current volume: {cf+self.max_v_frac}")
    return cf
  
  
  def __get_PFLOTRAN_output_variable_needed__(self):
    return ["VOLUME"]
  def __require_adjoint__(self): return False
  def __depend_of_mat_props__(self, var=None): return False

