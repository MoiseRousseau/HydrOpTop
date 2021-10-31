
class No_Adjoint:
  def __init__(self, parametrized_mat_props, p_ids):
    self.parametrized_mat_props = parametrized_mat_props
    self.assign_at_ids = p_ids #in solver format!
    self.dXi_dp = None
    self.initialized = False
    return
  
  def update_mat_derivative(self, p):
    for i,mat_prop in enumerate(self.parametrized_mat_props):
      mat_prop.d_mat_properties(p, self.dXi_dp[i])
    return
    
  def compute_sensitivity(self, p, dc_dYi, dc_dXi, Xi_name):
    #create or update structures
    if self.initialized == False:
      self.__initialize_adjoint__(p)
    else:
      self.update_mat_derivative(p)
    
    dc_dXi_dXi_dp = 0.
    for i,mat_prop in enumerate(self.parametrized_mat_props):
      for j,name in enumerate(Xi_name):
        if name == mat_prop.name:
          dc_dXi_dXi_dp += dc_dXi[j][self.assign_at_ids-1]*self.dXi_dp[i]
    return dc_dXi_dXi_dp
  
  
  def __initialize_adjoint__(self,p): 
    self.dXi_dp = [mat_prop.d_mat_properties(p) for mat_prop in self.parametrized_mat_props] 
               # dim = [mat_prop] * L * T2 / M
    self.n_parametrized_props = len(self.dXi_dp)
    
    self.initialized = True
    return
    

