import time
import scipy.sparse.linalg as ssl

try:
  import cupy as cp
  import cupyx.scipy.sparse as cpsparse
  import cupyx.scipy.sparse.linalg as csl
except:
  pass


def solve_adjoint(A, b, method='spsolve'):
    """
    Solve the adjoint problem of the form A*l=b and return x
    """
    
    start = time.time()
    #prepare matrix
    if method == 'ilu' or method == 'lu': 
      _A = (A.transpose()).tocsc() #[L-1]
    elif method == 'lu_gpu':
      b_gpu = cp.array(b)
      A_gpu = cpsparse.csr_matrix(A.transpose().tocsr())
    else:
      _A = (A.transpose()).tocsr()
    #solve
    if method == 'ilu': 
      #be careful! this method lead to false result without personalization
      LU = ssl.spilu(_A)
      l = LU.solve(b)
    elif method == 'lu': 
      LU = ssl.splu(_A)
      l = LU.solve(b)
    elif method == 'spsolve': 
      l = ssl.spsolve(_A, b) 
    elif method == 'bicgstab': 
      l, info = ssl.bicgstab(_A, b)
      if info: print("Some error append during BiConjugate Gradient Stabilized solve")
    elif method == 'bicg': 
      l, info = ssl.bicg(_A, b)
      if info: print("Some error append during BiConjugate Gradient solve")
    elif method == 'cg': 
      l, info = ssl.cg(_A, b)
      if info: print("Some error append during Conjugate Gradient solve")
    elif method == 'cgs': 
      l, info = ssl.cgs(_A, b)
      if info: print("Some error append during Conjugate Gradient Squared solve")
    elif method == 'lu_gpu': 
      l = csl.lsqr(A_gpu, b_gpu)[0].get()
    else:
      print("Solving method not recognized, stop...")
      exit(1)
    print(f"Time to solve adjoint using {method}: {(time.time() - start)} s")
    return l
