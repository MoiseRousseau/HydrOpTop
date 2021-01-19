import time
from scipy.sparse import coo_matrix, dia_matrix
from .adjoint_solving import solve_adjoint


class Sensitivity_Richards:
  """
  Compute the derivative of the cost function according to the material
  distribution parameter p in Richards mode.
  Arguments:
  Note: vector derivative should be numpy array, and matrix in (I,J,data) 
  format as output by PFLOTRAN.get_sensitivity() method.
  If cost_deriv_mat_prop is None, assume the cost function does not depend on
  the material property distribution.
  """
  def __init__(self, parametrized_mat_props, solver, p_ids):
    
    #vector
    #self.dc_dP = cost_deriv_pressure #dim = [cost] * L * T2 / M
    #self.dc_dXi = cost_deriv_mat_prop #[cost] / [mat_prop]
    self.parametrized_mat_props = parametrized_mat_props
    self.solver = solver
    self.assign_at_ids = p_ids #in PFLOTRAN format!
    
    self.method = 'lu' #adjoint solving method
    self.tol = None #adjoint solver tolerance
    
    self.dXi_dp = None
    self.dR_dXi = None
    self.dR_dP = None
    self.initialized = False
    return
  
  def set_adjoint_solving_algo(self,algo=None,tol=None):
    if algo is not None: self.method = algo
    if tol is not None: self.tol = tol
    return
    
  def update_mat_derivative(self, p):
    for i,mat_prop in enumerate(self.parametrized_mat_props):
      mat_prop.d_mat_properties(p, self.dXi_dp[i])
    return
  
  def update_residual_derivatives(self):
    self.solver.get_sensitivity("LIQUID_PRESSURE", coo_mat=self.dR_dP)
    for i,mat_prop in enumerate(self.parametrized_mat_props):
      self.solver.get_sensitivity(mat_prop.get_name(), coo_mat=self.dR_dXi[i])
    return 
  
  
  def compute_sensitivity(self, p, dc_dP, dc_dXi, Xi_name):
    """
    Compute the total cost function derivative according to material density
    parameter p.
    Argument:
    - p : the material parameter
    - dc_dP : derivative of the function wrt pressure
    - dc_dXi : derivative of the function wrt function inputs
    - Xi_name : name of the function input variables
    Note that every derivative must be given in natural ordering (i.e. PFLOTRAN ordering):
    dc_dXi[0] = dc/dX1 (index 0 = cell 1 in PFLOTRAN)
    dc_dXi[1] = dc/dX2 (index 1 = cell 2 in PFLOTRAN)
    ...
    dc_dXi[j] = dc/dX(j+1) (index j = cell j+1 in PFLOTRAN)
    ...
    """
    #create or update structures
    if self.initialized == False:
      self.__initialize_adjoint__(p)
    else:
      self.update_mat_derivative(p)
      self.update_residual_derivatives()
    
    #compute adjoint
    l = solve_adjoint(self.dR_dP, dc_dP, self.method)
    
    #compute dc/dp_bar
    if self.assign_at_ids is None:
      dR_dXi_dXi_dp = (self.dR_dXi[0]).tocsr().multiply(self.dXi_dp[0])
      if self.n_parametrized_props > 1:
        for i in range(1,self.n_parametrized_props):
          dR_dXi_dXi_dp += (self.dR_dXi[i]).tocsr().multiply(self.dXi_dp[i])
    else:
      dR_dXi_dXi_dp = \
              ((self.dR_dXi[0]).tocsr())[:,self.assign_at_ids-1].multiply(self.dXi_dp[0])
      if self.n_parametrized_props > 1:
        for i in range(1,self.n_parametrized_props):
          dR_dXi_dXi_dp += \
              ((self.dR_dXi[i]).tocsr())[:,self.assign_at_ids-1].multiply(self.dXi_dp[i])
    
    dc_dXi_dXi_dp = 0.
    for i,mat_prop in enumerate(self.parametrized_mat_props):
      for j,name in enumerate(Xi_name):
        if name == mat_prop.name:
          dc_dXi_dXi_dp += dc_dXi[j]*self.dXi_dp[i]
    if self.assign_at_ids is not None and dc_dXi_dXi_dp:
      dc_dXi_dXi_dp = dc_dXi_dXi_dp[self.assign_at_ids-1]
      
    S = dc_dXi_dXi_dp - (dR_dXi_dXi_dp.transpose()).dot(l)
    
    return S
  
  
  def __initialize_adjoint__(self,p): 
    self.dXi_dp = [mat_prop.d_mat_properties(p) for mat_prop in self.parametrized_mat_props] 
               # dim = [mat_prop] * L * T2 / M
    self.n_parametrized_props = len(self.dXi_dp)
      
    self.dR_dXi = [self.solver.get_sensitivity(mat_prop.get_name()) 
                                            for mat_prop in self.parametrized_mat_props]
                      
    self.dR_dP = self.solver.get_sensitivity("LIQUID_PRESSURE")
    
    self.initialized = True
    return
    

