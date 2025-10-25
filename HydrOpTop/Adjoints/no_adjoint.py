class No_Adjoint:
  """
  A dummy adjoint class that solve no adjoint...
  """
  def __init__(self, parametrized_mat_props, p_ids):
    self.assign_at_ids = p_ids #in solver format!
    self.adjoint = None
    self.initialized = False
    self.parametrized_mat_props = parametrized_mat_props
    
    self.dXi_dp = {m.get_name():None for m in parametrized_mat_props} # dim = [mat_prop] * L * T2 / M
    return
  
  def update_mat_derivative(self, p):
    for mat_prop in self.parametrized_mat_props:
      mat_prop.d_mat_properties(
        p, self.dXi_dp[mat_prop.get_name()]
      )
    return
    
  def compute_sensitivity(self, p, dc_dYi, dc_dXi):
    #create or update structures
    if self.initialized == False: self.__initialize_adjoint__(p)
    else: self.update_mat_derivative(p)
    
    dc_dXi_dXi_dp = 0.
    for name in dc_dXi.keys():
      dc_dXi_dXi_dp += dc_dXi[name][self.assign_at_ids-1]*self.dXi_dp[name]
    return dc_dXi_dXi_dp
  
  def __initialize_adjoint__(self,p): 
    self.dXi_dp = {
      mat_prop.get_name():mat_prop.d_mat_properties(p) for mat_prop in self.parametrized_mat_props
    }
    self.initialized = True
    return

