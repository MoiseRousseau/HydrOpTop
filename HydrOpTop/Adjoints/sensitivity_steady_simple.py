import time
from scipy.sparse import coo_matrix, dia_matrix
from .adjoint_solve import Direct_Sparse_Linear_Solver, Iterative_Sparse_Linear_Solver
import numpy as np


class Sensitivity_Steady_Simple:
  """
  Compute the derivative of the cost function according to the material
  distribution parameter p considering a steady-state simulation with
  only one equality constraint (g(x,p) = 0)
  Arguments:
  Note: vector derivative should be numpy array, and matrix in (I,J,data) 
  format as output by solver.get_sensitivity() method.
  I,J are 0 based indexing
  If cost_deriv_mat_prop is None, assume the cost function does not depend on
  the material property distribution.
  """
  def __init__(self, solved_vars, parametrized_mat_props, solver, p_ids):
    
    #vector
    #self.dc_dP = cost_deriv_pressure #dim = [cost] * L * T2 / M
    #self.dc_dXi = cost_deriv_mat_prop #[cost] / [mat_prop]
    self.solved_vars = solved_vars
    self.parametrized_mat_props = parametrized_mat_props
    self.solver = solver
    self.assign_at_ids = p_ids #in solver format!
    
    self.adjoint = Iterative_Sparse_Linear_Solver()
    
    self.dXi_dp = None
    self.dR_dXi = None
    self.dR_dYi = None
    self.initialized = False
    return
    
  def set_adjoint_problem_algo(self, algo=None):
    if algo is not None: self.adjoint.method = algo
    return
    
  def update_mat_derivative(self, p):
    for i,m in enumerate(self.parametrized_mat_props):
      self.dXi_dp[m.get_name()][m.indexes] = m.d_mat_properties(p[m.indexes])
    return
  
  def update_residual_derivatives(self):
    for i,solved_var in enumerate(self.solved_vars):
      self.solver.get_sensitivity(
        solved_var,
        coo_mat=self.dR_dYi[solved_var]
      )
    mat_prop_unique = set([m.get_name() for m in self.parametrized_mat_props])
    for i,mu in enumerate(mat_prop_unique):
      self.solver.get_sensitivity(
        mu, coo_mat=self.dR_dXi[mu]
      )
    return 
  
  
  def compute_sensitivity(self, p, dc_dYi, dc_dXi):
    """
    Compute the total cost function derivative according to material density
    parameter p.
    
    :param p: the material parameter
    :param dc_dYi: derivative of the function wrt solved variable (Solver ordering)
    :param dc_dXi: derivative of the function wrt input variable (p ordering)
    :param Xi_name: name of the function input variables
    """
    #create or update structures
    if self.initialized == False:
      self.__initialize_adjoint__(p)
    else:
      self.update_mat_derivative(p)
      self.update_residual_derivatives()
    
    #note: dR_dXi in solver ordering, so we convert it to p ordering with assign_at_ids
    # and dXi_dP in p ordering
    #thus: dR_dXi_dXi_dp in p ordering
    assert len(self.dR_dXi) == len(self.dXi_dp)
    var = [x for x in self.dXi_dp.keys()][0]
    n = len(self.dXi_dp[var])
    dR_dXi_dXi_dp = ( (self.dR_dXi[var]).tocsr() )[:,self.assign_at_ids-1].dot(
      dia_matrix( (self.dXi_dp[var],[0]), shape=(n,n) )
    )
    
    if self.n_parametrized_props > 1:
      #TODO
      raise NotImplementedError()
    #if self.n_parametrized_props > 1:
    #  for i in range(1,self.n_parametrized_props):
    #    dR_dXi_dXi_dp += ( (self.dR_dXi[i]).tocsr() )[:,self.assign_at_ids-1].dot( dia_matrix((self.dXi_dp[i], [0]),shape=(n,n)) )
    
    dc_dXi_dXi_dp = 0.
    for name in dc_dXi.keys():
      dc_dXi_dXi_dp += dc_dXi[name][self.assign_at_ids-1]*self.dXi_dp[name]

    #if self.assign_at_ids is not None and isinstance(dc_dXi_dXi_dp,np.ndarray):
    #  dc_dXi_dXi_dp = dc_dXi_dXi_dp[self.assign_at_ids-1]
    
    #compute adjoint
    assert len(self.dR_dYi) == len(dc_dYi)
    var = [x for x in self.dR_dYi.keys()][0]
    l = self.adjoint.solve(self.dR_dYi[var], dc_dYi[var]) #solver ordering
      
    S = dc_dXi_dXi_dp - (dR_dXi_dXi_dp.transpose()).dot(l)
    
    return S
  
  
  def __initialize_adjoint__(self,p):
    self.dXi_dp = {}
    mat_prop_unique = set([m.get_name() for m in self.parametrized_mat_props])
    for mu in mat_prop_unique:
      dXi_dp = np.zeros_like(p)
      for m in self.parametrized_mat_props:
        if m.get_name() != mu: continue
        dXi_dp[m.indexes] = m.d_mat_properties(p[m.indexes])
      self.dXi_dp[mu] = dXi_dp

    self.dR_dXi = {
      mu : self.solver.get_sensitivity(mu) for mu in mat_prop_unique
    }

    self.dR_dYi = {
      solved_var : self.solver.get_sensitivity(
        solved_var
      ) for solved_var in self.solved_vars
    }
    
    self.n_parametrized_props = len(self.dXi_dp)
    self.initialized = True
    return
    

