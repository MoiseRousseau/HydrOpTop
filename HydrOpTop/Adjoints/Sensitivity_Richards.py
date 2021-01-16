import time
from scipy.sparse import coo_matrix, dia_matrix
from .adjoint_solving import solve_adjoint


class Sensitivity_Richards:
  """
  Compute the derivative of the cost function according to the material
  distribution parameter p in Richards mode.
  Arguments:
  - cost_deriv_pressure: derivative of the cost function according to the
                         pressure in the domain (dc/dP)
  - mat_prop_deriv_mat_parameter: a list of derivatives of the material 
                                  properties according to the materal 
                                  distribution parameter (dX_i/dp)
  - cost_deriv_mat_prop: a list of derivatives of the cost function according 
                         to the material properties (dc/dX_i). Material 
                         properties must be in the same order than 
                         mat_prop_deriv_mat_parameter list.
  - res_deriv_mat_prop: a list of derivatives of the PFLOTRAN Richards residual
                        according to the material properties (dR_P/dX_i). List
                        must be in the same order as the two previous arguments.
  - res_deriv_pressure: derivative of the residual according to the pressure
                        (i.e. the Jacobian of the linear system).
  Note: vector derivative should be numpy array, and matrix in (I,J,data) 
  format as output by PFLOTRAN.get_sensitivity() method.
  If cost_deriv_mat_prop is None, assume the cost function does not depend on
  the material property distribution.
  """
  def __init__(self, mat_props, solver, p_ids):
    
    #vector
    #self.dc_dP = cost_deriv_pressure #dim = [cost] * L * T2 / M
    #self.dc_dXi = cost_deriv_mat_prop #[cost] / [mat_prop]
    self.mat_props = mat_props
    self.solver = solver
    self.assign_at_ids = p_ids
    
    self.method = 'lu' #adjoint solving method
    
    self.dXi_dp = None
    self.dR_dXi = None
    self.dR_dP = None
    self.initialized = False
    return
  
  def set_adjoint_solving_algo(self,x):
    self.method = x
    return
    
  def update_mat_derivative(self, p):
    for i,mat_prop in enumerate(self.mat_props):
      mat_prop.d_mat_properties(p, self.dXi_dp[i])
    return
  
  def update_residual_derivatives(self):
    self.solver.get_sensitivity("LIQUID_PRESSURE", coo_mat=self.dR_dP)
    for i,mat_prop in enumerate(self.mat_props):
      self.solver.get_sensitivity(mat_prop.get_name(), coo_mat=self.dR_dXi[i])
    return 
  
  
  def compute_sensitivity(self, p, dc_dP, dc_dXi):
    """
    Compute the total cost function derivative according to material density
    parameter p.
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
      if self.n_inputs > 1:
        for i in range(1,self.n_inputs):
          dR_dXi_dXi_dp += (self.dR_dXi[i]).tocsr().multiply(self.dXi_dp[i])
    else:
      dR_dXi_dXi_dp = \
              ((self.dR_dXi[0]).tocsr())[:,self.assign_at_ids].multiply(self.dXi_dp[0])
      if self.n_inputs > 1:
        for i in range(1,self.n_inputs):
          dR_dXi_dXi_dp += \
              ((self.dR_dXi[i]).tocsr())[:,self.assign_at_ids].multiply(self.dXi_dp[i])
        
    if dc_dXi is None: 
      dc_dXi_dXi_dp = 0.
    else:
      dc_dXi_dXi_dp = dc_dXi[0]*self.dXi_dp[0]
      if self.n_inputs > 1:
        for i in range(1,self.n_inputs):
          dc_dXi_dXi_dp += dc_dXi[i]*self.dXi_dp[i]
      
    if self.assign_at_ids is None:
      S = dc_dXi_dXi_dp - (dR_dXi_dXi_dp.transpose()).dot(l)
    else:
      S = (dc_dXi_dXi_dp - (dR_dXi_dXi_dp.transpose()).dot(l))
    return S
  
  
  def __initialize_adjoint__(self,p): 
    self.dXi_dp = [mat_prop.d_mat_properties(p) for mat_prop in self.mat_props] 
               # dim = [mat_prop] * L * T2 / M
    self.n_inputs = len(self.dXi_dp)
      
    self.dR_dXi = [self.solver.get_sensitivity(mat_prop.get_name()) 
                                            for mat_prop in self.mat_props]
                      
    self.dR_dP = self.solver.get_sensitivity("LIQUID_PRESSURE")
    
    self.initialized = True
    return
    

