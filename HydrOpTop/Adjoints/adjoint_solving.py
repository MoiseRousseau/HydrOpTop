import time
import scipy.sparse.linalg as spla
import scipy.sparse as sp
import numpy as np
try:
  from petsc4py import PETSc
except:
  pass



class Adjoint_Solve:
  def __init__(self, A, b, method="lu", library="SciPy"):
    self.method = method.lower()
    self.lib = library.lower()
    self.A = A
    self.b = b
    self.last_l = None
    
    #default parameter for some algorithm
    self.cg_tol = 1e-5
    self.cg_preconditionner = "ilu(0)"
    return
  
  def solve(self):
    if self.method == "lu":
      l = self.__lu_solve__()
    elif self.method == "bicgstab":
      l = self.__bicgstab_solve__()
    return l
  
  def __lu_solve__(self):
    if self.lib == "scipy":
      _A = (self.A.transpose()).tocsc() #[L-1]
      LU = spla.splu(_A)
      l = LU.solve(b)
    elif self.lib == "petsc":
      pass
    return l
  
  def __bicgstab_solve__(self):
    if self.lib == "scipy":
      self.last_l[:], info = spla.bicgstab(_A, b, x0=self.last_l, 
                              tol=self.bicgstab_tol, atol=-1) #do not rely on atol
      if info: 
        print(info)
        print("Some error append during BiConjugate Gradient Stabilized solve")
        exit(1)
    elif self.lib == "petsc":
      pass
    #copy for making starting guess for future iteration
    if self.last_l is None: self.last_l = np.copy(l)
    else: self.last_l[:] = l
    return l




def solve_adjoint(A, b, x0=None, method='spsolve', tol=None):
    """
    Solve the adjoint problem of the form A*l=b and return x
    """
    
    print(f"Solve adjoint equation using {method}")
    start = time.time()
    #prepare matrix
    if method in ["ilu", "lu", "gmres_pre"]: 
      _A = (A.transpose()).tocsc() #[L-1]
    else:
      _A = sp.csr_matrix(A)
    #solve
    if method == 'ilu': 
      #be careful! this method lead to false result without personalization
      LU = spla.spilu(_A)
      l = LU.solve(b)
    elif method == 'lu': 
      LU = spla.splu(_A)
      l = LU.solve(b)
    elif method == 'spsolve': 
      l = spla.spsolve(_A, b) 
    elif method == "gmres":
      l, info = spla.gmres(_A, b, x0=x0)
      if info: print("Some error append during GMRES solve")
    elif method == "gmres_pre":
      ilu = spla.spilu(_A)
      M = spla.LinearOperator(_A.shape, ilu.solve)
      l, info = spla.gmres(_A, b, x0=x0, M=M)
      if info: print("Some error append during GMRES solve")
    elif method == "lgmres_pre":
      ilu = spla.spilu(_A)
      M = spla.LinearOperator(_A.shape, ilu.solve)
      l, info = spla.lgmres(_A, b, x0=x0, M=M)
      if info: print("Some error append during LGMRES solve")
    elif method == 'bicgstab_old': 
      l, info = spla.bicgstab(_A, b, x0=x0, tol=1e-4, atol=-1)
      if info: print("Some error append during BiConjugate Gradient Stabilized solve")
    elif method == 'bicgstab':
      #use jacobi preconditionning
      print("Don't work now...")
      D = A.diagonal()
      A_ = A.multiply(1/D)
      l, info = spla.bicgstab(A_, b/np.sqrt(D), x0=x0, tol=0.1e-4, atol=-1)#, M=M)
      if info: print("Some error append during BiConjugate Gradient Stabilized solve")
      l /= np.sqrt(D)
    elif method == 'bicg': 
      l, info = spla.bicg(_A, b, x0=x0)
      if info: print("Some error append during BiConjugate Gradient solve")
    elif method == 'cg': 
      l, info = spla.cg(_A, b, x0=x0)
      if info: print("Some error append during Conjugate Gradient solve")
    elif method == 'cgs': 
      l, info = spla.cgs(_A, b, x0=x0)
      if info: print("Some error append during Conjugate Gradient Squared solve")
    
    elif method == "petsc":
      print(dir(A))
      A_ = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data), comm=PETSc.COMM_WORLD)
      ksp = PETSc.KSP()
      ksp.setType("cg")
      ksp.getPC().setType("ilu")
      l, = A_.getVecs()
      l.set(0)
      ksp.setOperators(A_)
      ksp.solve(b,l)
      
    
    else:
      print("Solving method not recognized, stop...")
      exit(1)
    print(f"Time to solve adjoint: {(time.time() - start)} s")
    return l
