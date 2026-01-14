from scipy.sparse import coo_matrix, dia_matrix, diags
from .adjoint_solve import Direct_Sparse_Linear_Solver, Iterative_Sparse_Linear_Solver
import numpy as np

DEFAULT_SOLVER_ARGS = {
  "method":"iterative",
  "preconditionner":"ilu0",
  "reorder":True,
  "verbose":True,
}


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
  def __init__(self, solved_vars, parametrized_mat_props, solver, p_ids, ids_p, solver_args={}):
    
    #vector
    #self.dc_dP = cost_deriv_pressure #dim = [cost] * L * T2 / M
    #self.dc_dXi = cost_deriv_mat_prop #[cost] / [mat_prop]
    self.solved_vars = solved_vars
    self.parametrized_mat_props = parametrized_mat_props
    self.solver = solver
    self.p_ids = p_ids #in solver format!
    self.ids_p = ids_p #in solver format!
    
    solver_args_ = DEFAULT_SOLVER_ARGS.copy()
    solver_args_.update(solver_args)
    method = solver_args_.pop("method")
    if method == "direct":
      self.adjoint = Direct_Sparse_Linear_Solver(**solver_args_)
    elif method == "iterative":
      self.adjoint = Iterative_Sparse_Linear_Solver(**solver_args_)
    else:
      raise ValueError(f"{method} should be iterative or direct")
    
    self.dXi_dp = None
    self.dR_dXi = []
    self.dR_dYi = []
    self.initialized = False
    self.l0 = None
    return
    
  def update_mat_derivative(self, p):
    for i,m in enumerate(self.parametrized_mat_props):
      indexes = self.ids_p[m.cell_ids]
      self.dXi_dp[m.get_name()][indexes] = m.d_mat_properties(p[indexes])
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

    #create or update structures
    if self.initialized == False:
      self.__initialize_adjoint__(p_bar)
    else:
      self.update_residual_derivatives()
    self.update_mat_derivative(p_bar)

    # Get derivative of the cost function with p_bar
    # derivative according to solved var
    # in dobj_dXY, the function ask only the p_bar for its indexes
    p_bar_ = np.zeros( max(func.indexes.max(),self.p_ids.max())+1 ) + np.nan
    p_bar_[self.p_ids] = p_bar
    p_bar_ = p_bar_[func.indexes]
    dobj_dY = {}
    dobj_dX = {}
    # Loop over the function variables
    for var in func.__get_variables_needed__():
      # If a solved variable, this go to the adjoint
      if var in self.solver.solved_variables:
        dfunc = func.d_objective(var, p_bar_)
        dobj_dY[var] = np.zeros(self.solver.get_system_size())
        if dfunc.ndim == 2: dobj_dY[var] = dobj_dY[var].repeat(dfunc.shape[1]).reshape(self.solver.get_system_size(),dfunc.shape[1])
        dobj_dY[var][func.indexes-self.solver.cell_id_start_at] = dfunc
      # If not solved
      elif var in [x.get_name() for x in self.parametrized_mat_props]:
        dfunc = func.d_objective(var, p_bar_)
        dobj_dX[var] = np.zeros(self.solver.get_system_size())
        if dfunc.ndim == 2: dobj_dX[var] = dobj_dX[var].repeat(dfunc.shape[1]).reshape(self.solver.get_system_size(),dfunc.shape[1])
        dobj_dX[var][func.indexes-self.solver.cell_id_start_at] = dfunc
    
    dfunc = func.d_objective_dp_partial(p_bar_)
    dobj_dp_partial = np.zeros_like(p_bar)
    if np.any(dfunc):
      dobj_dp_partial[self.ids_p[func.indexes]] = dfunc
    # if dfunc.ndim == 2: dobj_dp_partial = dobj_dp_partial.repeat(dfunc.shape[1]).reshape(self.solver.get_system_size(),dfunc.shape[1])

    #note: dR_dXi in solver ordering, so we convert it to p ordering with assign_at_ids
    # and dXi_dP in p ordering
    #thus: dR_dXi_dXi_dp in p ordering
    assert len(self.dR_dXi) == len(self.dXi_dp)
    var = [x for x in self.dXi_dp.keys()][0]
    n = len(self.dXi_dp[var])
    dR_dXi_dXi_dp = ( (self.dR_dXi[var]).tocsr() )[:,self.p_ids].dot(
      dia_matrix( (self.dXi_dp[var],[0]), shape=(n,n) )
    )

    # build right matrix
    # dR_dXi_dXi_dp in p ordering
    # Jf in p ordering
    R_mat = dR_dXi_dXi_dp @ Jf #.transpose().dot(grad)
    
    if self.n_parametrized_props > 1:
      #TODO
      raise NotImplementedError()
    #if self.n_parametrized_props > 1:
    #  for i in range(1,self.n_parametrized_props):
    #    dR_dXi_dXi_dp += ( (self.dR_dXi[i]).tocsr() )[:,self.assign_at_ids-1].dot( dia_matrix((self.dXi_dp[i], [0]),shape=(n,n)) )
    
    dc_dXi_dXi_dp = 0.
    for name in dobj_dX.keys():
      dc_dXi_dXi_dp += dobj_dX[name][self.p_ids]*self.dXi_dp[name]

    #if self.assign_at_ids is not None and isinstance(dc_dXi_dXi_dp,np.ndarray):
    #  dc_dXi_dXi_dp = dc_dXi_dXi_dp[self.assign_at_ids-1]

    #compute adjoint
    assert len(self.dR_dYi) == len(dobj_dY)
    var = [x for x in self.dR_dYi.keys()][0]
    # Mask NaN along diagonal
    keep = np.argwhere(~np.isnan(self.dR_dYi[var].diagonal())).flatten()
    A = self.dR_dYi[var].tocsr()[keep][:,keep]
    b = dobj_dY[var][keep]
    self.l0 = self.adjoint.solve(A, b) #solver ordering
    # const terms
    S = - (R_mat.transpose().tocsc()[:,keep]).dot(self.l0)

    if isinstance(S, np.ndarray) and S.ndim == 2:
      S += (dc_dXi_dXi_dp + dobj_dp_partial[:,None]) @ Jf
    else:
      S += (dc_dXi_dXi_dp + dobj_dp_partial) @ Jf

    return S
  
  
  def __initialize_adjoint__(self,p_bar):
    self.dXi_dp = {
      mat_prop.get_name():np.zeros_like(p_bar) for mat_prop in self.parametrized_mat_props
    }

    mat_prop_unique = set([m.get_name() for m in self.parametrized_mat_props])
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