
class Base_Filter:
  p_ids = None
  input_variables_needed = []
  name = ""
  adjoint = None
  
  def set_p_to_cell_ids(self, p_ids):
    self.p_ids = p_ids
    return
  
  def set_inputs(self, inputs):
    return
  
  def get_filtered_density(self, p):
    return p
  
  def get_filter_derivative(self, p, out=None):
    if out is None:
      return 1.
    else:
      out = 1.
  
  def __get_input_variables_needed__(self):
    return self.input_variables_needed
  def __get_solved_variables_needed__(self):
    return []
  def __get_name__(self):
    return self.name
  

