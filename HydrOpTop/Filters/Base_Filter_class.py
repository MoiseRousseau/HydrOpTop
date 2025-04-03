import numpy as np

class Base_Filter:
  p_ids = None
  name = ""
  adjoint = None
  input_variables_needed = []
  
  def set_p_to_cell_ids(self, p_ids):
    self.p_ids = p_ids
    return
  
  def set_inputs(self, inputs):
    return
  
  def get_filtered_density(self, p):
    """
    TO DEFINE
    """
    return

  def get_filter_derivative(self, p, eps=1e-6):
    """
    Compute derivative of p_bar relative to p with centered finite difference.

    :param p: Density parameter p
    :param eps: Absolute step for finite difference calculation
    """
    J = np.zeros([len(self.p_ids),len(p)], dtype = np.double)

    for i in range(len(p)):
        x1 = p.copy()
        x2 = p.copy()

        x1[i] += eps
        x2[i] -= eps

        f1 = self.get_filtered_density(x1)
        f2 = self.get_filtered_density(x2)

        J[:,i] = (f1 - f2) / (2 * eps)

    return J
  
  def __get_input_variables_needed__(self):
    return self.input_variables_needed
  def __get_solved_variables_needed__(self):
    return []
  def __get_name__(self):
    return self.name
  

