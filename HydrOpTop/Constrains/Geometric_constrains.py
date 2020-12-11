#TODO

class Geometric_Constrain:
  """
  Implement the geometric constrain defined in Zhou et al. (2015): 
  Minimum length scale in topology optimization by geometric constraints,
  (http://dx.doi.org/10.1016/j.cma.2015.05.003)
  Geometric constrains are used to impose a minimum length scale on the 
  optimized geometry. For more information, see the original publication
  above.
  
  Argument:
  - 
  """
  def __init__(self):
    return
  
  def set_inputs(self, inputs):
    return
  
  def evaluate(self, p, grad):
    
    if grad.size > 0:
      pass
    return constrain
  
  def __get_PFLOTRAN_output_variable_needed__(self):
    return ["VOLUME"]
  def __need_p_cell_ids__(self): return True
  

