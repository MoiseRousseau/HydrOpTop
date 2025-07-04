import time
import scipy.sparse.linalg as spla
from scipy.sparse import dia_matrix
import scipy.sparse as sp
import numpy as np


def __sp_solve__(A, b, l0=None):
    _A = A.tocsr()
    l = spla.spsolve(_A, b) 
    return

def __lu_solve__(A, b, l0=None):
    damped_factor = [0.,1e-12,1e-8,1e-6,1e-4,1e-3,1e-2,1e-1,1.]
    for df in damped_factor:
        if df != 0.:
            print(f"    Current damping factor: {df}")
        #_A = ( A + sp.diags(A.diagonal()*df) ).tocsc()
        _A = ( A + sp.eye(A.shape[0]) * df * A.mean() ).tocsc()
        try:
            LU = spla.splu(_A)
            l = LU.solve(b)
            break
        except:
            print("LU solve failed, try increasing damping factor")
    return l

def __lu_solve_scaled__(A, b, l0=None):
    A_diag = np.where(A.diagonal() == 0., A.mean() * 1e-8, A.diagonal())
    D_ = dia_matrix((np.sqrt(1/np.abs(A_diag)),[0]), shape=A.shape)
    LU = spla.splu(D_ @ A @ D_)
    l = D_ @ LU.solve(D_ @ b)
    return l

def __bicgstab_solve_old__(A, b, l0=None):
    #always use jacobi preconditioning
    D_ = dia_matrix((np.sqrt(1/A.diagonal()),[0]), shape=A.shape)
    _A = D_ * A * D_
    _b = D_ * b
    x0 = l0 / D_.diagonal() if l0 is not None else None
    l, info = spla.bicgstab(
      _A, _b, x0=x0, 
      rtol=5e-4, atol=1e-40
    ) #do not rely on atol
    if info: 
         raise RuntimeError(f"Some error append during BiConjugate Gradient Stabilized solve, error code: {info}")
    l = D_ * l
    return l


def __bicgstab_solve__(A, b, l0=None, prec="jacobi"):
    # Preconditionning
    if prec == "jacobi":
      M = dia_matrix((np.sqrt(1.0/np.abs(A.diagonal())),[0]), shape=A.shape)
    elif prec == "ilu":
      M_ilu = spla.spilu(A.tocsc(), drop_tol=1e-4, fill_factor=0)
      print(M_ilu)
      M = spla.LinearOperator(A.shape, M_ilu.solve)
    # Solve
    l, info = spla.bicgstab(
        A, b, x0=l0, M=M, 
        rtol=5e-4, atol=1e-40
    ) #do not rely on atol
    if info: 
        raise RuntimeError(f"Some error append during BiConjugate Gradient Stabilized solve, error code: {info}")
    return l

def __lsqr_solve__(A,b, l0=None):
    l = spla.lsqr(A, b, atol=1e-6, btol=1e-6, x0=l0)
    if l[1] == 1: 
        print("Warning, adjoint solution is only an approximate solution")
    #print(l)
    return l[0]


algo = {
    "direct": __lu_solve__,
    "iterative": __bicgstab_solve_old__,
    "lu": __lu_solve__,
    "spsolve": __sp_solve__,
    "bicgstab": __bicgstab_solve_old__,
    "least-square": __lsqr_solve__,
}


class Adjoint_Solve:
    r"""
    Solve matrix equation Ax=B
    """
    def __init__(self, algo=None, algo_kwargs={}):
        if algo is not None: 
          self.algo = algo.lower()
        else: 
          self.algo = None
        self.last_l = None
        
        #default parameter for some algorithm
        self.solve_params = algo_kwargs
        return
    
    def solve(self, A, b):
        #default parameter
        if self.algo is None:
            self.algo = self.select_default_algo(A,b)
        #solve
        print(f"Solve adjoint equation using {self.algo} solver")
        t_start = time.time()
        l = algo[self.algo](A,b,self.last_l)
        if self.last_l is None: self.last_l = l
        else: self.last_l[:] = l
        print(f"Time to solve adjoint: {(time.time() - t_start)} s")
        return l
    
    def select_default_algo(self, A, b):
        if len(b) > 10000: 
            return "bicgstab"
        elif A.shape[0] != A.shape[1]:
            return "least-square"
        else: 
            return "lu"

