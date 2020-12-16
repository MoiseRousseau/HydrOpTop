#TODO

class No_Filter:
  """
  A dummy filter that do nothing but show the basic method for a filter
  """
  def __init__(self):
    return
  
  def set_inputs(self, inputs):
    return
  
  def get_filtered_density(self, p, p_bar=None):
    if p_bar is None:
      return np.copy(p)
    else:
      p_bar[:] = p
  
  def get_filter_derivative(self, p, out=None):
    if out is None:
      return 1.
    else:
      out = 1.
  
  def __get_PFLOTRAN_output_variable_needed__(self):
    return []
  def __depend_of_mat_props__(self, var=None):
    if var is None: return []
    else: return False
  

