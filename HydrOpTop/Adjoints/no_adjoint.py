import numpy as np

class No_Adjoint:
  """
  A dummy adjoint class that solve no adjoint...
  """
  def __init__(self, parametrized_mat_props, p_ids):
    self.assign_at_ids = p_ids #in solver format!
    self.initialized = False
    self.parametrized_mat_props = parametrized_mat_props
    self.l0 = None
    
    self.dXi_dp = {m.get_name():None for m in parametrized_mat_props} # dim = [mat_prop] * L * T2 / M
    return
  
  def update_mat_derivative(self, p):
    for mat_prop in self.parametrized_mat_props:
      mat_prop.d_mat_properties(
        p, self.dXi_dp[mat_prop.get_name()]
      )
    return
    
  def compute_sensitivity(self, func, filter_sequence, p):
    """
    Compute the total cost function derivative according to material density
    parameter p.

    :param p: the material parameter
    :param dc_dYi: derivative of the function wrt solved variable (Solver ordering)
    :param dc_dXi: derivative of the function wrt input variable (p ordering)
    :param Xi_name: name of the function input variables
    """
    # filter field
    # Jf is dp_bar / dp, in p ordering
    p_bar, Jf = filter_sequence.filter_derivative(p)

    #create and update structures
    if self.initialized == False: self.__initialize_adjoint__(p)
    else: self.update_mat_derivative(p)

    dobj_dX = {}
    for var in func.__get_variables_needed__():
      # If not solved
      if var in [x.get_name() for x in self.parametrized_mat_props]:
        dfunc = func.d_objective(var, p_bar)
        dobj_dX[var] = np.zeros(self.solver.get_system_size())
        if dfunc.ndim == 2: dobj_dX[var] = dobj_dX[var].repeat(dfunc.shape[1]).reshape(self.solver.get_system_size(),dfunc.shape[1])
        dobj_dX[var][func.indexes-self.solver.cell_id_start_at] = dfunc

    dobj_dp_partial = func.d_objective_dp_partial(p_bar)

    dc_dXi_dXi_dp = 0.
    for name in dobj_dX.keys():
      dc_dXi_dXi_dp += dobj_dX[name][self.assign_at_ids-1]*self.dXi_dp[name]
    return dc_dXi_dXi_dp + dobj_dp_partial
  
  def __initialize_adjoint__(self,p): 
    self.dXi_dp = {
      mat_prop.get_name():mat_prop.d_mat_properties(p) for mat_prop in self.parametrized_mat_props
    }
    self.initialized = True
    return

