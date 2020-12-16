import numpy as np

class My_Constrain:
  """
  Add a descrition of the constrain here
  """
  def __init__(self, ids_to_sum_volume, max_volume_percentage):
    #Initialize the constrain
    
    self.filter = None
    return
  
  def set_filter(self, filter):
    #must be completed
    return
  
  def set_inputs(self, inputs):
    #must be completed
    return
    
  def set_p_cell_ids(self, cell_ids):
    #must be completed if __need_p_cell_ids__ return True
    #i.e. if you need the correspondance between p and the cell ids in 
    #PFLOTRAN simulation
    return
  
  def evaluate(self, p, grad):
    if self.filter:
      p_bar = self.filter.get_filtered_density(p)
    else:
      p_bar = p
    #use p_bar to compute the constrain
    constrain = 0 #value of your constrain: c=f(p_bar)
    if grad.size > 0:
      #compute gradient
      my_grad = 0. #to complete
      if self.filter:
        grad[:] = self.filter.get_filter_derivative(p).transpose().dot(my_grad) 
      else:
        grad[:] = my_grad
    return constrain
  
  
  def __get_PFLOTRAN_output_variable_needed__(self):
    return ["VOLUME"]
  def __need_p_cell_ids__(self): return True #/ False
  

