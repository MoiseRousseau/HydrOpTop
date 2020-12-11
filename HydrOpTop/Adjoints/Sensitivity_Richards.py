
import time

from scipy.sparse import coo_matrix, dia_matrix
import scipy.sparse.linalg as ssl

try:
  import cupy as cp
  import cupyx.scipy.sparse as cpsparse
  import cupyx.scipy.sparse.linalg as csl
except:
  pass


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
  def __init__(self, cost_deriv_pressure, mat_prop_deriv_mat_parameter, 
               cost_deriv_mat_prop, res_deriv_mat_prop, res_deriv_pressure):
    #vector
    self.dc_dP = cost_deriv_pressure #dim = [cost] * L * T2 / M
    self.dc_dXi = cost_deriv_mat_prop #[cost] / [mat_prop]
    #diag matrix
    self.dXi_dp = [dia_matrix((x,0), shape=(len(x),len(x)), dtype='f8') for x 
                       in mat_prop_deriv_mat_parameter] # dim = [mat_prop] * L * T2 / M
    #matrix
    self.dR_dXi = [ coo_matrix((x[:,2],(x[:,0].astype('i8')-1,x[:,1].astype('i8')-1)))
                      for x in res_deriv_mat_prop] # dim = M / (L2 * T2 * [mat_prop])
    self.dR_dP = coo_matrix((res_deriv_pressure[:,2], 
                             (res_deriv_pressure[:,0].astype('i8')-1,
                              res_deriv_pressure[:,1].astype('i8')-1) ) ) #[L-1]
    
    self.method = 'ilu' #adjoint solving method
    self.n_inputs = len(self.dXi_dp)
    return
  
  def set_adjoint_solving_algo(self,x):
    self.method = x
    return
    
  def compute_sensitivity(self, S=None, assign_at_ids=None):
    """
    Compute the total cost function derivative according to material density
    parameter p.
    """
    l = self.solve_adjoint(self.method)
    
    dR_dXi_dXi_dp = (self.dR_dXi[0]).tocsr().dot(self.dXi_dp[0].tocsr())
    if self.n_inputs > 1:
      for i in range(1,self.n_inputs):
        dR_dXi_dXi_dp += (self.dR_dXi[i]).tocsr().dot(self.dXi_dp[i].tocsr())
        
    if self.dc_dXi is None: dc_dXi_dXi_dp = 0.
    else:
      dc_dXi_dXi_dp = self.dc_dXi[0]*self.dXi_dp[0]
      if self.n_inputs > 1:
        for i in range(1,self.n_inputs):
          dc_dXi_dXi_dp += self.dc_dXi[i]*self.dXi_dp[i]
      #TODO: got a problem with the numpy axis and output size
    if S is None:      
      S = dc_dXi_dXi_dp - (dR_dXi_dXi_dp.transpose()).dot(l)
    elif assign_at_ids is None:
      S[:] = dc_dXi_dXi_dp - (dR_dXi_dXi_dp.transpose()).dot(l)
    else:
      S[:] = (dc_dXi_dXi_dp - (dR_dXi_dXi_dp.transpose()).dot(l))[assign_at_ids-1]
    return S
  
  def solve_adjoint(self,method='spsolve'):
    """
    Solve the adjoint problem of the form A*l=b and return x
    """
    start = time.time()
    #prepare matrix
    b = self.dc_dP 
    if method == 'ilu' or method == 'lu': 
      A = (self.dR_dP.transpose()).tocsc() #[L-1]
    elif method == 'lu_gpu':
      b_gpu = cp.array(b)
      A_gpu = cpsparse.csr_matrix(self.dR_dP.transpose().tocsr())
    else:
      A = (self.dR_dP.transpose()).tocsr()
    #solve
    if method == 'ilu': 
      #be careful! this method lead to false result without personalization
      LU = ssl.spilu(A)
      l = LU.solve(b)
    elif method == 'lu': 
      LU = ssl.splu(A)
      l = LU.solve(b)
    elif method == 'spsolve': 
      l = ssl.spsolve(A, b) 
    elif method == 'bicgstab': 
      l, info = ssl.bicgstab(A, b)
      if info: print("Some error append during BiConjugate Gradient Stabilized solve")
    elif method == 'bicg': 
      l, info = ssl.bicg(A, b)
      if info: print("Some error append during BiConjugate Gradient solve")
    elif method == 'cg': 
      l, info = ssl.cg(A, b)
      if info: print("Some error append during Conjugate Gradient solve")
    elif method == 'cgs': 
      l, info = ssl.cgs(A, b)
      if info: print("Some error append during Conjugate Gradient Squared solve")
    elif method == 'lu_gpu': 
      l = csl.lsqr(A_gpu, b_gpu)[0].get()
    else:
      print("Solving method not recognized, stop...")
      exit(1)
    print(f"Time to solve adjoint using {method}: {(time.time() - start)} s")
    return l
    

