import numpy as np

class No_Adjoint:
  """
  A dummy adjoint class that solve no adjoint...
  """
  def __init__(self, parametrized_mat_props, solver, p_ids, ids_p):
    self.p_ids = p_ids #in solver format!
    self.ids_p = ids_p
    self.initialized = False
    self.parametrized_mat_props = parametrized_mat_props
    self.l0 = None
    self.solver = solver
    
    self.dXi_dp = {m.get_name():None for m in parametrized_mat_props} # dim = [mat_prop] * L * T2 / M
    return

  def update_mat_derivative(self, p):
    for i,m in enumerate(self.parametrized_mat_props):
      indexes = self.ids_p[m.cell_ids]
      self.dXi_dp[m.get_name()][indexes] = m.d_mat_properties(p[indexes])
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
    if self.initialized == False: self.__initialize_adjoint__(p_bar)
    self.update_mat_derivative(p_bar)

    # in dobj_dX, the function ask only the p_bar for its indexes
    p_bar_ = np.zeros( max(func.indexes.max(),self.p_ids.max())+1 ) + np.nan
    p_bar_[self.p_ids] = p_bar
    p_bar_ = p_bar_[func.indexes]
    dobj_dX = {}
    for var in func.__get_variables_needed__():
      # If not solved
      if var in [x.get_name() for x in self.parametrized_mat_props]:
        dfunc = func.d_objective(var, p_bar_)
        dobj_dX[var] = np.zeros(self.solver.get_system_size())
        if dfunc.ndim == 2: dobj_dX[var] = dobj_dX[var].repeat(dfunc.shape[1]).reshape(self.solver.get_system_size(),dfunc.shape[1])
        dobj_dX[var][func.indexes-self.solver.cell_id_start_at] = dfunc

    dfunc = func.d_objective_dp_partial(p_bar_)
    dobj_dp_partial = np.zeros(self.solver.get_grid_size())
    if np.any(dfunc):
      dobj_dp_partial[func.indexes-self.solver.cell_id_start_at] = dfunc
    dobj_dp_partial = dobj_dp_partial[self.p_ids-self.solver.cell_id_start_at]

    dc_dXi_dXi_dp = 0.
    for name in dobj_dX.keys():
      dc_dXi_dXi_dp += dobj_dX[name][self.p_ids-self.solver.cell_id_start_at] * self.dXi_dp[name]
    return (dc_dXi_dXi_dp + dobj_dp_partial) @ Jf
  
  def __initialize_adjoint__(self,p):
    self.dXi_dp = {
      mat_prop.get_name():np.zeros_like(p) for mat_prop in self.parametrized_mat_props
    }
    self.initialized = True
    return

