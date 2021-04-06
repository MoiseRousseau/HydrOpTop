import time
import scipy.sparse.linalg as spla
from scipy.sparse import dia_matrix
import scipy.sparse as sp
import numpy as np
try:
  from petsc4py import PETSc
except:
  pass



class Adjoint_Solve:
  def __init__(self, method=None, library="SciPy"):
    self.method = None
    if self.method is not None:
      self.method = method.lower()
    self.lib = library.lower()
    self.last_l = None
    
    #default parameter for some algorithm
    self.cg_tol = 5e-4
    self.cg_preconditionner = "jacobi"
    return
  
  def solve(self, A, b):
    #default parameter
    if self.method is None:
      if len(b) > 60000: self.method = "bicgstab"
      else: self.method = "lu"
    print(f"Solve adjoint equation using {self.method}")
    t_start = time.time()
    if self.method == "spsolve":
      l = spla.spsolve(A.tocsr(), b) 
    if self.method == "lu":
      l = self.__lu_solve__(A,b)
    elif self.method == "bicgstab":
      l = self.__bicgstab_solve__(A,b)
    print(f"Time to solve adjoint: {(time.time() - t_start)} s")
    return l
  
  def __lu_solve__(self, A, b):
    if self.lib == "scipy":
      _A = A.tocsc() #[L-1]
      LU = spla.splu(_A)
      l = LU.solve(b)
    elif self.lib == "petsc":
      pass
    return l
  
  def __bicgstab_solve__(self, A, b):
    if self.lib == "scipy":
      #always use jacobi preconditioning
      D_ = dia_matrix((np.sqrt(1/A.diagonal()),[0]), shape=A.shape)
      _A = D_ * A * D_
      _b = D_ * b
      l, info = spla.bicgstab(_A, _b, x0=self.last_l, 
                                tol=self.cg_tol, atol=-1) #do not rely on atol
      #copy for making starting guess for future iteration
      if self.last_l is None: self.last_l = np.copy(l)
      else: self.last_l[:] = l
      if info: 
        print("Some error append during BiConjugate Gradient Stabilized solve")
        print(f"Error code: {info}")
        exit(1)
      l = D_ * l
    elif self.lib == "petsc":
      pass
    return l


